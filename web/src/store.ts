// 全局状态: signals + localStorage 持久化。编辑器表单与连接设置都在这里，
// 组件只读 signals、调 actions，不自己存状态。
// 输入: 无; 输出: signals 与 actions;
// 预期: 刷新页面后回到上一次的草稿 (题名、类型、state、主题、语言、key 都在)。

import { signal, computed } from '@preact/signals'
import { DEFAULT_PRESET, PRESETS, type Preset } from './lib/presets'
import { clientErrors, fetchModels, runSystemOne, buildBody } from './lib/api'
import type { Failure, Problem, Success } from './lib/api'
import type { ModelInfo, QuestionType } from './lib/types'
import { DICT, type Lang } from './lib/i18n'

export interface CritRow {
  id: string
  key: string
  value: string
}

export interface QuestionForm {
  id: string
  name: string
  type: QuestionType
  instructions: string
  criteria: CritRow[]
}

export interface EditorState {
  stateText: string
  model: string
  questions: QuestionForm[]
  thinkTokens: number
}

let counter = 0
const uid = (prefix: string): string => `${prefix}-${Date.now().toString(36)}-${(counter++).toString(36)}`

const TYPE_DEFAULTS: Record<QuestionType, () => CritRow[]> = {
  noul: () => [
    { id: uid('c'), key: 'true', value: '' },
    { id: uid('c'), key: 'false', value: '' },
  ],
  choice: () => [
    { id: uid('c'), key: 'A', value: '' },
    { id: uid('c'), key: 'B', value: '' },
  ],
  score: () => [
    { id: uid('c'), key: '', value: '' },
    { id: uid('c'), key: '', value: '' },
  ],
}

export function blankQuestion(index: number): QuestionForm {
  const type: QuestionType = 'choice'
  return { id: uid('q'), name: `question_${index}`, type, instructions: '', criteria: TYPE_DEFAULTS[type]() }
}

function fromPreset(preset: Preset): EditorState {
  return {
    stateText: preset.state,
    model: model$.peek(),
    questions: preset.questions.map((question) => ({
      id: uid('q'),
      name: question.name,
      type: question.type,
      instructions: question.instructions,
      criteria: question.criteria.map((row) => ({ id: uid('c'), key: row.key, value: row.value })),
    })),
    thinkTokens: 0,
  }
}

const STORAGE_KEY = 'minicpm-jev-web@1'

interface Persisted {
  editor: EditorState
  theme: 'dark' | 'light'
  lang: Lang
  base: string
  apiKey: string
}

function loadPersisted(): Partial<Persisted> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? (JSON.parse(raw) as Partial<Persisted>) : {}
  } catch {
    return {}
  }
}

const saved = loadPersisted()

export const theme = signal<'dark' | 'light'>(saved.theme ?? 'dark')
export const lang = signal<Lang>(saved.lang ?? 'zh')
export const base = signal<string>(saved.base ?? '/v1')
export const apiKey = signal<string>(saved.apiKey ?? '')
export const model$ = signal<string>(saved.editor?.model ?? '')
export const stateText = signal<string>(saved.editor?.stateText ?? DEFAULT_PRESET.state)
export const questions = signal<QuestionForm[]>(
  saved.editor?.questions?.length ? saved.editor.questions : fromPreset(DEFAULT_PRESET).questions,
)
export const thinkTokens = signal<number>(saved.editor?.thinkTokens ?? 0)
export const models = signal<ModelInfo[]>([])
export const running = signal(false)
export const elapsed = signal(0)
export const result = signal<Success | null>(null)
export const failure = signal<Failure | null>(null)
export const rawMode = signal(false)
export const rawText = signal('')
export const toast = signal('')

/** 原始 JSON 模式下的实时解析结果; 解析失败为 null。表单模式恒为 null (不参与)。 */
export const rawQuestions = computed<QuestionForm[] | null>(() =>
  rawMode.value ? questionsFromJson(rawText.value) : null,
)

export const editor = computed<EditorState>(() => ({
  stateText: stateText.value,
  model: model$.value,
  // raw 模式以文本为准, 否则会出现「改了 JSON、发的还是旧表单」的错路。
  questions: rawMode.value ? (rawQuestions.value ?? []) : questions.value,
  thinkTokens: thinkTokens.value,
}))

export const connection = computed(() => ({ base: base.value, apiKey: apiKey.value }))

export const bodyPreview = computed(() => JSON.stringify(buildBody(editor.value), null, 2))

export const problems = computed<Problem[]>(() => {
  if (rawMode.value && !rawQuestions.value) return [{ key: 'q.badJson' }]
  return clientErrors(editor.value)
})
export const blocked = computed(() => problems.value.length > 0)

export const t = computed(() => {
  const table = DICT[lang.value]
  return (key: string, vars?: Record<string, string | number>) => {
    let text = table[key] ?? DICT.zh[key] ?? key
    if (vars) for (const [name, value] of Object.entries(vars)) text = text.replace(`{${name}}`, String(value))
    return text
  }
})

let persistTimer: ReturnType<typeof setTimeout> | undefined
function persist(): void {
  clearTimeout(persistTimer)
  persistTimer = setTimeout(() => {
    const payload: Persisted = {
      editor: { stateText: stateText.peek(), model: model$.peek(), questions: questions.peek(), thinkTokens: thinkTokens.peek() },
      theme: theme.peek(),
      lang: lang.peek(),
      base: base.peek(),
      apiKey: apiKey.peek(),
    }
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
    } catch {
      /* 隐私模式下写不进去，草稿留在内存里就行 */
    }
  }, 250)
}

theme.subscribe(persist)
lang.subscribe(persist)
base.subscribe(persist)
apiKey.subscribe(persist)
stateText.subscribe(persist)
model$.subscribe(persist)
questions.subscribe(persist)
thinkTokens.subscribe(persist)

export function flash(message: string): void {
  toast.value = message
  setTimeout(() => {
    if (toast.peek() === message) toast.value = ''
  }, 2200)
}

export function setTheme(next: 'dark' | 'light'): void {
  theme.value = next
  document.documentElement.dataset.theme = next
}

export function toggleLang(): void {
  lang.value = lang.peek() === 'zh' ? 'en' : 'zh'
}

export function patchQuestion(id: string, patch: Partial<QuestionForm>): void {
  questions.value = questions.peek().map((question) => (question.id === id ? { ...question, ...patch } : question))
}

export function patchCriterion(id: string, patch: Partial<CritRow>): void {
  questions.value = questions.peek().map((question) => ({
    ...question,
    criteria: question.criteria.map((row) => (row.id === id ? { ...row, ...patch } : row)),
  }))
}

export function addCriterion(questionId: string): void {
  questions.value = questions.peek().map((question) => {
    if (question.id !== questionId) return question
    const next = TYPE_DEFAULTS[question.type]()
    const suffix = question.criteria.length
    const row = question.type === 'choice' ? { ...next[0], key: `option_${suffix}` } : next[0]
    return { ...question, criteria: [...question.criteria, row] }
  })
}

export function removeCriterion(questionId: string, rowId: string): void {
  questions.value = questions.peek().map((question) =>
    question.id === questionId ? { ...question, criteria: question.criteria.filter((row) => row.id !== rowId) } : question,
  )
}

export function changeType(questionId: string, type: QuestionType): void {
  questions.value = questions.peek().map((question) =>
    question.id === questionId ? { ...question, type, criteria: TYPE_DEFAULTS[type]() } : question,
  )
}

export function addQuestion(): void {
  questions.value = [...questions.peek(), blankQuestion(questions.peek().length + 1)]
}

export function duplicateQuestion(id: string): void {
  const source = questions.peek().find((question) => question.id === id)
  if (!source) return
  const copy: QuestionForm = {
    ...source,
    id: uid('q'),
    name: `${source.name}_copy`,
    criteria: source.criteria.map((row) => ({ ...row, id: uid('c') })),
  }
  const index = questions.peek().findIndex((question) => question.id === id)
  const list = [...questions.peek()]
  list.splice(index + 1, 0, copy)
  questions.value = list
}

export function removeQuestion(id: string): void {
  questions.value = questions.peek().filter((question) => question.id !== id)
}

export function moveQuestion(id: string, delta: number): void {
  const list = [...questions.peek()]
  const from = list.findIndex((question) => question.id === id)
  const to = from + delta
  if (from < 0 || to < 0 || to >= list.length) return
  const [item] = list.splice(from, 1)
  list.splice(to, 0, item)
  questions.value = list
}

/** 表单 <-> 原始 JSON 文本 (切到 raw 时导出, 切回表单时导入; 解析失败就不切). */
export function questionsToJson(): string {
  return JSON.stringify(
    questions.peek().map((question) => ({
      name: question.name,
      type: question.type,
      instructions: question.instructions,
      criteria: question.criteria.map((row) => ({ key: row.key, value: row.value })),
    })),
    null,
    2,
  )
}

/**
 * 判定标准取值转成表单里那一行文本。
 * 输入: 任意 EntryType; 输出: 字符串;
 * 预期: 对象/数组按紧凑 JSON 落进文本框 (发送时 lib/api.ts 会再解析回结构),
 *       否则 String 直取 —— 直接用 String() 会把对象变成 "[object Object]"。
 */
function criterionText(value: unknown): string {
  if (value === null || value === undefined) return ''
  return typeof value === 'object' ? JSON.stringify(value) : String(value)
}

export function questionsFromJson(text: string): QuestionForm[] | null {
  try {
    const parsed: unknown = JSON.parse(text)
    if (!Array.isArray(parsed)) return null
    return parsed.map((item, index) => {
      const record = item as Record<string, unknown>
      // type 原样保留 (包括非法值): 前端不擅自改写用户输入, 交给服务端按契约判 422。
      const type = (record.type ?? 'choice') as QuestionType
      // criteria 有三种合法写法, 都必须吃得下, 悄悄丢掉就等于把用户改好的候选换成默认值:
      //   1) 界面导出格式 [ {key, value}, ... ]
      //   2) 官方 choice/noul 线上格式 { 候选: 描述 | null }
      //   3) 官方 score 线上格式 [ "档位描述", ... ]
      const raw = record.criteria
      const entries: [unknown, unknown][] = Array.isArray(raw)
        ? raw.map((row, order) =>
            typeof row === 'object' && row !== null
              ? [(row as Record<string, unknown>).key ?? order, (row as Record<string, unknown>).value ?? '']
              : [String(order), row],
          )
        : raw && typeof raw === 'object'
          ? Object.entries(raw as Record<string, unknown>)
          : []
      const defaults = TYPE_DEFAULTS[type]
      return {
        id: uid('q'),
        name: typeof record.name === 'string' ? record.name : `question_${index + 1}`,
        type,
        instructions: typeof record.instructions === 'string' ? record.instructions : JSON.stringify(record.instructions ?? '', null, 2),
        criteria: entries.length
          ? entries.map(([key, value]) => ({ id: uid('c'), key: String(key), value: criterionText(value) }))
          : defaults
            ? defaults()
            : [],
      }
    })
  } catch {
    return null
  }
}

export function toggleRaw(): void {
  if (rawMode.peek()) {
    const parsed = questionsFromJson(rawText.peek())
    if (!parsed) {
      flash(t.peek()('q.badJson'))
      return
    }
    questions.value = parsed
    rawMode.value = false
    return
  }
  rawText.value = questionsToJson()
  rawMode.value = true
}

export function applyPreset(id: string): void {
  const preset = PRESETS.find((item) => item.id === id)
  if (!preset) return
  const next = fromPreset(preset)
  stateText.value = next.stateText
  questions.value = next.questions
  thinkTokens.value = 0
  rawMode.value = false
  result.value = null
  failure.value = null
}

export async function refreshModels(): Promise<void> {
  try {
    const list = await fetchModels(connection.peek())
    models.value = list
    if (!model$.peek() || !list.some((item) => item.name === model$.peek())) model$.value = list[0]?.name ?? ''
  } catch {
    models.value = []
  }
}

export async function run(): Promise<void> {
  if (running.peek()) return
  const blocked = problems.peek()
  if (blocked.length) {
    failure.value = {
      kind: 'client',
      status: 0,
      message: blocked.map((item) => (item.detail ? `${item.key}: ${item.detail}` : item.key)).join(' · '),
      fields: [],
      elapsedMs: 0,
    }
    return
  }
  running.value = true
  failure.value = null
  const tick = setInterval(() => (elapsed.value += 1), 10)
  const started = performance.now()
  const outcome = await runSystemOne(editor.peek(), connection.peek())
  clearInterval(tick)
  elapsed.value = Math.round(performance.now() - started)
  running.value = false
  if (outcome.ok) result.value = outcome.value
  else failure.value = outcome.value
}

setTheme(theme.peek())
