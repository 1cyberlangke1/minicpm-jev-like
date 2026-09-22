// 问题编辑器: 表单模式逐题可增删改排序, 原始 JSON 模式直接改整个数组。
// 输入: 无 (读写 questions / rawMode signals); 输出: 面板 DOM;
// 预期: 类型切换会把 criteria 换成该类型的默认形状; 原始 JSON 解析失败时不切回表单, 只提示。

import type { JSX } from 'preact'
import {
  addQuestion,
  addCriterion,
  changeType,
  duplicateQuestion,
  editor,
  moveQuestion,
  patchCriterion,
  patchQuestion,
  questions,
  rawMode,
  rawText,
  removeCriterion,
  removeQuestion,
  t,
  toggleRaw,
} from '../store'
import type { QuestionForm } from '../store'
import type { QuestionType } from '../lib/types'

const TYPES: QuestionType[] = ['noul', 'choice', 'score']

function Criteria({ question }: { question: QuestionForm }): JSX.Element | null {
  const translate = t.value
  if (question.type === 'noul') {
    return (
      <div class="critList">
        {question.criteria.map((row) => (
          <div class="critRow" key={row.id}>
            <span class="critKey locked">{row.key}</span>
            <input
              class="critValue"
              placeholder={translate(row.key === 'true' ? 'q.desc' : 'q.desc')}
              value={row.value}
              onInput={(event) => patchCriterion(row.id, { value: (event.currentTarget as HTMLInputElement).value })}
            />
          </div>
        ))}
      </div>
    )
  }
  if (question.type === 'choice') {
    return (
      <div class="critList">
        {question.criteria.map((row) => (
          <div class="critRow" key={row.id}>
            <input
              class="critKey"
              placeholder={translate('q.option')}
              value={row.key}
              onInput={(event) => patchCriterion(row.id, { key: (event.currentTarget as HTMLInputElement).value })}
            />
            <input
              class="critValue"
              placeholder={translate('q.desc')}
              value={row.value}
              onInput={(event) => patchCriterion(row.id, { value: (event.currentTarget as HTMLInputElement).value })}
            />
            <button class="iconBtn danger" title={translate('q.remove')} onClick={() => removeCriterion(question.id, row.id)}>
              ✕
            </button>
          </div>
        ))}
        <button class="addBtn" onClick={() => addCriterion(question.id)}>
          {translate('q.addOption')}
        </button>
      </div>
    )
  }
  return (
    <div class="critList levels">
      {question.criteria.map((row, index) => (
        <div class="critRow" key={row.id}>
          <span class="levelIndex">{index}</span>
          <input
            class="critValue"
            placeholder={`${translate('q.level')} ${index}`}
            value={row.value}
            onInput={(event) => patchCriterion(row.id, { value: (event.currentTarget as HTMLInputElement).value })}
          />
          <button class="iconBtn danger" title={translate('q.remove')} onClick={() => removeCriterion(question.id, row.id)}>
            ✕
          </button>
        </div>
      ))}
      <button class="addBtn" onClick={() => addCriterion(question.id)}>
        {translate('q.addLevel')}
      </button>
    </div>
  )
}

function Card(props: { question: QuestionForm; index: number; total: number }): JSX.Element {
  const translate = t.value
  const { question } = props
  return (
    <article class="qCard" style={{ '--i': props.index } as JSX.CSSProperties}>
      <header class="qHead">
        <span class="qIndex">{props.index + 1}</span>
        <input
          class="qName"
          value={question.name}
          spellcheck={false}
          placeholder={translate('q.name')}
          onInput={(event) => patchQuestion(question.id, { name: (event.currentTarget as HTMLInputElement).value })}
        />
        <div class="seg" role="group">
          {TYPES.map((type) => (
            <button
              class={question.type === type ? 'segItem on' : 'segItem'}
              onClick={() => changeType(question.id, type)}
              type="button"
            >
              {type}
            </button>
          ))}
        </div>
        <div class="qOps">
          <button class="iconBtn" title={translate('q.up')} disabled={props.index === 0} onClick={() => moveQuestion(question.id, -1)}>
            ↑
          </button>
          <button
            class="iconBtn"
            title={translate('q.down')}
            disabled={props.index === props.total - 1}
            onClick={() => moveQuestion(question.id, 1)}
          >
            ↓
          </button>
          <button class="iconBtn" title={translate('q.dup')} onClick={() => duplicateQuestion(question.id)}>
            ⧉
          </button>
          <button class="iconBtn danger" title={translate('q.remove')} onClick={() => removeQuestion(question.id)}>
            ✕
          </button>
        </div>
      </header>

      <label class="qField">
        <span class="qFieldLabel">instructions</span>
        <textarea
          class="code qInstructions"
          spellcheck={false}
          value={question.instructions}
          placeholder={translate('q.instructions')}
          onInput={(event) => patchQuestion(question.id, { instructions: (event.currentTarget as HTMLTextAreaElement).value })}
        />
      </label>

      <div class="qField">
        <span class="qFieldLabel">criteria</span>
        <Criteria question={question} />
      </div>
    </article>
  )
}

export function QuestionsPanel(): JSX.Element {
  const translate = t.value
  return (
    <section class="panel questionsPanel">
      <div class="panelHead">
        <span class="panelTitle">{translate('q.title')}</span>
        <div class="headActions">
          {/* 计数跟随当前生效的题目集: raw 模式下显示解析结果, 不再停留在表单旧状态 */}
          <span class="badge">{editor.value.questions.length}</span>
          <button class="toolBtn" onClick={toggleRaw}>
            {rawMode.value ? translate('q.form') : translate('q.raw')}
          </button>
          {!rawMode.value && (
            <button class="toolBtn primary" onClick={addQuestion}>
              {translate('q.add')}
            </button>
          )}
        </div>
      </div>

      {rawMode.value ? (
        <textarea
          class="code tall"
          spellcheck={false}
          value={rawText.value}
          onInput={(event) => {
            rawText.value = (event.currentTarget as HTMLTextAreaElement).value
          }}
        />
      ) : questions.value.length === 0 ? (
        <p class="emptyLine">{translate('q.empty')}</p>
      ) : (
        <div class="qList">
          {questions.value.map((question, index) => (
            <Card key={question.id} question={question} index={index} total={questions.value.length} />
          ))}
        </div>
      )}
    </section>
  )
}
