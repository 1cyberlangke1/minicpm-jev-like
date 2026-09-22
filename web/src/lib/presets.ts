// 示例请求: 点一下就填进编辑器, 六个场景各覆盖一条契约路径。
// 输入: 无; 输出: PRESETS 列表 (每项带标签键 + 说明键 + state 文本 + 题目表单);
// 预期: 只作为默认内容, 不是特例路径 —— 用户改完之后走的是同一条请求构造代码。
//
// criteria 的写法按官方三原语各自定型, 序列化时由 lib/api.ts 的 parseEntry 决定
// 「以 { 或 [ 开头的值按结构发、其余按字符串发」, 所以字典形判定标准在表单里
// 就是一段 JSON 文本, 发出去才是真对象。
// noul 的是/否描述留空 —— 空的描述不会被带上请求体, 等价于「不给 criteria」的官方最简形。

import type { QuestionType } from './types'

export interface CritSeed {
  key: string
  value: string
}

export interface QuestionSeed {
  name: string
  type: QuestionType
  instructions: string
  criteria: CritSeed[]
}

export interface Preset {
  id: string
  labelKey: string
  state: string
  questions: QuestionSeed[]
}

/** noul 的两个固定标签; 描述留空即「不给判定标准」, 与官方最简请求等价。 */
const NOUL_BLANK: CritSeed[] = [
  { key: 'true', value: '' },
  { key: 'false', value: '' },
]

/** 有序档位: 档位没有名字, 只有描述, 所以键留空。 */
const levels = (values: string[]): CritSeed[] => values.map((value) => ({ key: '', value }))

/** 选项 + 描述成对给出。 */
const described = (pairs: [string, string][]): CritSeed[] => pairs.map(([key, value]) => ({ key, value }))

export const PRESETS: Preset[] = [
  {
    id: 'refund',
    labelKey: 'presets.refund',
    state: 'Customer: I ordered a hot dog at noon and it arrived stone cold at 1:30 PM. I want my money back.',
    questions: [
      {
        name: 'refund',
        type: 'noul',
        instructions: 'Does the customer explicitly request a refund?',
        criteria: NOUL_BLANK,
      },
    ],
  },
  {
    id: 'sky',
    labelKey: 'presets.sky',
    // 空 state 是合法请求: 判定完全由问题与候选自己承担。
    state: '',
    questions: [
      {
        name: 'sky_color',
        type: 'choice',
        instructions: 'What color does the sky appear?',
        criteria: described([
          ['blue', 'Rayleigh scattering of short wavelengths'],
          ['green', 'filtered by clouds'],
          ['red', 'only near sunset'],
        ]),
      },
    ],
  },
  {
    id: 'monkey',
    labelKey: 'presets.monkey',
    state: 'An unattended monkey smacked paint onto this canvas. Critics are debating its artistic merit.',
    questions: [
      {
        name: 'quality',
        type: 'score',
        instructions: 'Rate the artistic quality of the painting.',
        criteria: levels(['amateur', 'decent', 'good', 'great', 'masterpiece']),
      },
    ],
  },
  {
    id: 'ticket',
    labelKey: 'presets.ticket',
    state: 'Ticket #4021: My payments have been failing for 3 days, I was charged twice, and I have been on hold for an hour. Fix it today or I am leaving.',
    questions: [
      {
        name: 'is_urgent',
        type: 'noul',
        instructions: 'Does this message convey real urgency?',
        criteria: NOUL_BLANK,
      },
      {
        name: 'department',
        type: 'choice',
        instructions: 'Which team should own this ticket?',
        criteria: described([
          ['billing', 'payments and invoices'],
          ['tech', 'technical issues'],
          ['ops', 'logistics'],
        ]),
      },
      {
        name: 'frustration',
        type: 'score',
        instructions: 'How frustrated is the customer?',
        criteria: levels(['calm', 'mildly annoyed', 'frustrated', 'very frustrated', 'furious']),
      },
    ],
  },
  {
    id: 'structured',
    labelKey: 'presets.structured',
    // state 是数组 (对话记录), criteria 的值是对象 (EntryType) —— 官方契约两者都收。
    state: JSON.stringify(
      [
        { role: 'user', text: 'hey, did my refund arrive yet?' },
        { role: 'agent', text: 'let me check the payment system' },
      ],
      null,
      2,
    ),
    questions: [
      {
        name: 'intent',
        type: 'choice',
        instructions: 'Classify the latest user intent.',
        criteria: described([
          ['refund_status', '{ "topic": "money", "detail": "asking about a pending refund" }'],
          ['small_talk', '{ "topic": "other", "detail": "greeting or chatter" }'],
        ]),
      },
    ],
  },
  {
    id: 'zhTicket',
    labelKey: 'presets.zhTicket',
    state: '客户：我昨天在你们店买的蓝牙耳机只有一个耳朵有声音，客服让我等三天了还没回复，今天必须给我解决，不然我就投诉到消费者协会！',
    questions: [
      {
        name: 'is_urgent',
        type: 'noul',
        instructions: '这条消息是否表达了真实的紧急情绪？',
        criteria: NOUL_BLANK,
      },
      {
        name: 'department',
        type: 'choice',
        instructions: '这个问题应该由哪个部门处理？',
        criteria: described([
          ['售后', '产品质量与退换'],
          ['技术', '软件与网络故障'],
          ['物流', '发货与配送'],
        ]),
      },
      {
        name: 'anger',
        type: 'score',
        instructions: '客户的愤怒程度打几分？',
        criteria: levels(['平静', '有点不满', '生气', '很生气', '暴怒']),
      },
    ],
  },
]

/** 打开界面时铺在编辑器里的默认场景: 一次前向出三份契约, 信息量最大。 */
export const DEFAULT_PRESET = PRESETS.find((preset) => preset.id === 'ticket') ?? PRESETS[0]
