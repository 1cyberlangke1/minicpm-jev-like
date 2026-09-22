// 请求构造与发送: 表单 -> 官方 /v1/systemone body -> 响应/错误归一化。
// 输入: 编辑器状态 (见 store.ts); 输出: 结果或人话错误;
// 预期: 前端只做「能不能发」的最小检查, 真校验交给服务端, 错误按官方 422 数组原样摊开。

import type {
  Answer,
  Entry,
  FieldError,
  ModelInfo,
  QuestionPayload,
  SystemOneBody,
  SystemOneResponse,
} from './types'
import type { EditorState, QuestionForm } from '../store'

export interface Problem {
  key: string
  detail?: string
}

export interface Conn {
  base: string
  apiKey: string
}

export type Failure = {
  kind: 'client' | 'http' | 'network'
  status: number
  message: string
  fields: FieldError[]
  elapsedMs: number
}

export type Success = {
  model: string
  answers: Record<string, Answer>
  usage: SystemOneResponse['usage']
  elapsedMs: number
  // 记的是「这次请求实际带出去的预算」, 而不是滑杆当前值 —— 结果出来后用户再拖滑杆, 页脚不该跟着变。
  thinkTokens: number
  raw: string
}

export type RunResult = { ok: true; value: Success } | { ok: false; value: Failure }

/** 把题目表单序列化成官方 question 对象. */
function toPayload(question: QuestionForm): QuestionPayload {
  const instructions = parseEntry(question.instructions)
  const payload: QuestionPayload = { type: question.type, instructions }
  // criteria 的值和 instructions 走同一条规则: 写成 JSON 结构就按结构发, 否则按字符串发。
  // 官方 EntryType 本来就两头都收, 界面上那个描述框因此既能填一句话, 也能填一个对象。
  if (question.type === 'noul') {
    const criteria: Record<string, Entry> = {}
    for (const row of question.criteria) if (row.value.trim()) criteria[row.key] = parseEntry(row.value)
    if (Object.keys(criteria).length) payload.criteria = criteria
    return payload
  }
  if (question.type === 'choice') {
    const criteria: Record<string, Entry> = {}
    for (const row of question.criteria) criteria[row.key] = row.value.trim() ? parseEntry(row.value) : null
    payload.criteria = criteria
    return payload
  }
  payload.criteria = question.criteria.map((row) => parseEntry(row.value))
  return payload
}

/** instructions / criteria 的值: 能按 JSON 解析就按结构发, 否则当字符串发 (官方两种都收). */
function parseEntry(text: string): Entry {
  const trimmed = text.trim()
  if (!trimmed) return ''
  if (/[[{]/.test(trimmed[0])) {
    try {
      return JSON.parse(trimmed) as Entry
    } catch {
      return text
    }
  }
  return text
}

/** 客户端最小检查: 只拦「发了也毫无意义」的几种情况, 返回可翻的键 + 附加信息. */
export function clientErrors(editor: EditorState): Problem[] {
  const problems: Problem[] = []
  if (!editor.model) problems.push({ key: 'run.needModel' })
  if (!editor.questions.length) problems.push({ key: 'run.needQuestions' })
  const names = editor.questions.map((question) => question.name.trim())
  if (names.some((name) => !name)) problems.push({ key: 'q.name' })
  const duplicated = [...new Set(names.filter((name, index) => name && names.indexOf(name) !== index))]
  if (duplicated.length) problems.push({ key: 'run.nameTaken', detail: duplicated.join(', ') })
  return problems
}

export function buildBody(editor: EditorState): SystemOneBody {
  const questions: Record<string, QuestionPayload> = {}
  for (const question of editor.questions) questions[question.name.trim()] = toPayload(question)
  return {
    state: parseEntry(editor.stateText),
    model: editor.model,
    questions,
    think_tokens: editor.thinkTokens,
  }
}

function headers(conn: Conn): Record<string, string> {
  return {
    'content-type': 'application/json',
    ...(conn.apiKey ? { authorization: `Bearer ${conn.apiKey}` } : {}),
  }
}

/** 拉本地模型清单. 失败时抛出 network 类错误 (调用方兜). */
export async function fetchModels(conn: Conn): Promise<ModelInfo[]> {
  const response = await fetch(`${conn.base}/models`, { headers: headers(conn) })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  const payload = (await response.json()) as { models: ModelInfo[] }
  return payload.models
}

/** 发一次决策请求: 网络、HTTP、字段错误都归一成 Failure. */
export async function runSystemOne(editor: EditorState, conn: Conn): Promise<RunResult> {
  const body = buildBody(editor)
  const started = performance.now()
  let response: Response
  try {
    response = await fetch(`${conn.base}/systemone`, {
      method: 'POST',
      headers: headers(conn),
      body: JSON.stringify(body),
    })
  } catch {
    return {
      ok: false,
      value: {
        kind: 'network',
        status: 0,
        message: 'err.network',
        fields: [],
        elapsedMs: Math.round(performance.now() - started),
      },
    }
  }
  const elapsedMs = Math.round(performance.now() - started)
  const text = await response.text()
  if (!response.ok) {
    let detail: unknown
    try {
      detail = JSON.parse(text).detail
    } catch {
      detail = undefined
    }
    const fields = Array.isArray(detail) ? (detail as FieldError[]) : []
    const direct =
      typeof detail === 'object' && detail !== null && 'message' in detail
        ? String((detail as { message: unknown }).message)
        : ''
    // 非契约响应体 (反向代理的 HTML 404 页等) 只留一行人话摘要, 不整段糊到界面上。
    const brief = text.trim().replace(/\s+/g, ' ').slice(0, 160)
    const fallback = brief && !brief.startsWith('<') ? `err.raw: ${brief}` : `err.status: ${response.status}`
    return {
      ok: false,
      value: {
        kind: 'http',
        status: response.status,
        message: direct || fields[0]?.msg || fallback,
        fields,
        elapsedMs,
      },
    }
  }
  // 200 不等于「是 JSON」: 反向代理或静态服务错配时会把 index.html 原样吐回来,
  // 直接 JSON.parse 会抛未捕获异常, 这里同样归一成一条可读错误。
  let payload: SystemOneResponse
  try {
    payload = JSON.parse(text) as SystemOneResponse
  } catch {
    const peek = text.trim().replace(/\s+/g, ' ').slice(0, 160)
    return {
      ok: false,
      value: {
        kind: 'http',
        status: response.status,
        // 整页 HTML 当错误详情贴出来只会糊屏, 摘要只留给非标签体的响应
        message: peek && !peek.startsWith('<') ? `err.notJson: ${peek}` : 'err.notJson',
        fields: [],
        elapsedMs,
      },
    }
  }
  return {
    ok: true,
    value: {
      model: payload.model,
      answers: payload.answers,
      usage: payload.usage,
      elapsedMs,
      thinkTokens: body.think_tokens,
      raw: JSON.stringify(payload, null, 2),
    },
  }
}
