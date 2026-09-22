// 官方 /v1/systemone 契约的 TS 镜像 (逐字段对齐 minicpm_jev.service.schemas)。
// 输入: 无; 输出: 类型定义;
// 预期: 这里只做形状描述, 不做校验 —— 服务端是唯一裁判, 前端只负责把人话错误摊开。

export type QuestionType = 'noul' | 'choice' | 'score'

export type Entry = string | Record<string, unknown> | unknown[] | null

export interface QuestionPayload {
  type: QuestionType
  instructions: Entry
  criteria?: Entry
}

export interface SystemOneBody {
  state: Entry
  model: string
  questions: Record<string, QuestionPayload>
  think_tokens: number
}

export interface NoulAnswer {
  type: 'noul'
  noul: number
}

export interface ChoiceAnswer {
  type: 'choice'
  choice: string
  confidence: number
  probabilities: Record<string, number>
}

export interface ScoreAnswer {
  type: 'score'
  score: number
  confidence: number
  legend: Record<string, string>
  probabilities: Record<string, number>
}

export type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer

export interface Usage {
  input_tokens: number
  output_tokens: number
}

export interface SystemOneResponse {
  model: string
  answers: Record<string, Answer>
  usage: Usage
}

export interface ModelInfo {
  name: string
  description: string
  release_date: string
}

/** 422 的字段错误项 (pydantic v2 形状)。 */
export interface FieldError {
  loc: string[]
  msg: string
}
