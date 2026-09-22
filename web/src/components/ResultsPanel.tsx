// 结果工作台: 三种原语各按官方语义渲染, 概率条按降序 (score 按档位序)。
// 输入: 无 (读 result / failure / running); 输出: 结果区 DOM;
// 预期: 加载中给骨架屏, 错误分「客户端拦截 / HTTP 业务错 / 422 字段数组 / 网络不通」四种形状。

import type { JSX } from 'preact'
import { signal } from '@preact/signals'
import { bodyPreview, failure, flash, lang, questions, result, running, t } from '../store'
import { humanize } from '../lib/i18n'
import type { Answer, ScoreAnswer } from '../lib/types'
import { Gauge, ProbBar } from './ProbBar'

const showRaw = signal<'off' | 'request' | 'response'>('off')

function sortProbabilities(entries: [string, number][]): [string, number][] {
  return [...entries].sort((left, right) => right[1] - left[1])
}

function Body(props: { name: string; answer: Answer; index: number }): JSX.Element {
  const translate = t.value
  const { answer } = props
  const head = (
    <header class="ansHead">
      <span class="qName">{props.name}</span>
      <span class={`typeTag ${answer.type}`}>{answer.type}</span>
    </header>
  )

  if (answer.type === 'noul') {
    return (
      <article class="ansCard" style={{ '--i': props.index } as JSX.CSSProperties}>
        {head}
        <div class="noulBody">
          <Gauge value={answer.noul} caption={translate('res.pYes')} />
          <div class="probList grow">
            <ProbBar label="yes" probability={answer.noul} winner={answer.noul >= 0.5} index={0} />
            <ProbBar label="no" probability={1 - answer.noul} winner={answer.noul < 0.5} index={1} />
            <p class="answerLine">P(yes) = {answer.noul.toFixed(6)}</p>
          </div>
        </div>
      </article>
    )
  }

  const entries =
    answer.type === 'choice'
      ? sortProbabilities(Object.entries(answer.probabilities))
      : Object.entries((answer as ScoreAnswer).probabilities).map(
          ([key, value]) => [`${key} · ${(answer as ScoreAnswer).legend[key] ?? ''}`, value] as [string, number],
        )
  const winner = answer.type === 'choice' ? answer.choice : entries.length ? sortProbabilities(entries)[0][0] : ''

  return (
    <article class="ansCard" style={{ '--i': props.index } as JSX.CSSProperties}>
      {head}
      <div class="ansHead2">
        {answer.type === 'choice' && (
          <span class="kv">
            <i>{translate('res.choice')}</i>
            <b>{answer.choice}</b>
          </span>
        )}
        {answer.type === 'score' && (
          <span class="kv">
            <i>{translate('res.score')}</i>
            <b>{answer.score.toFixed(3)}</b>
          </span>
        )}
        <span class="kv">
          <i>{translate('res.confidence')}</i>
          <b>{(answer.confidence * 100).toFixed(2)}%</b>
        </span>
      </div>
      <div class="probList">
        {entries.map(([label, probability], order) => (
          <ProbBar label={label} probability={probability} winner={label === winner} index={order} />
        ))}
      </div>
    </article>
  )
}

// 骨架条数与本次请求的问题数对齐, 但封顶 6 张: 几十条问题时铺满闪烁卡片反而吵。
// 封顶只是「不画那么多」, 下方留白即暗示还有内容, 不影响真实结果逐条渲染。
const SK_MAX = 6

function Skeleton(): JSX.Element {
  const cards = Math.min(Math.max(questions.value.length, 1), SK_MAX)
  return (
    <div class="skeleton">
      {Array.from({ length: cards }, (_, index) => (
        <div class="skCard" style={{ '--i': index } as JSX.CSSProperties} key={index}>
          <div class="skLine w40" />
          <div class="skBar" />
          <div class="skBar" />
          <div class="skBar short" />
        </div>
      ))}
    </div>
  )
}

export function ResultsPanel(): JSX.Element {
  const translate = t.value
  const payload = result.value
  const error = failure.value

  // 思考预算非零时耗时能翻十倍, 页脚不写出来就像界面卡住了; 零预算是默认路径, 不占版面。
  const usage = payload
    ? translate('res.usage', { in: payload.usage.input_tokens, out: payload.usage.output_tokens, ms: payload.elapsedMs }) +
      (payload.thinkTokens ? ` · ${translate('res.think', { n: payload.thinkTokens })}` : '')
    : ''

  return (
    <section class="panel resultsPanel">
      <div class="panelHead">
        <span class="panelTitle">{translate('res.title')}</span>
        <div class="headActions">
          <button
            class={showRaw.value === 'request' ? 'toolBtn on' : 'toolBtn'}
            onClick={() => (showRaw.value = showRaw.value === 'request' ? 'off' : 'request')}
          >
            {translate('res.raw')} req
          </button>
          <button
            class={showRaw.value === 'response' ? 'toolBtn on' : 'toolBtn'}
            onClick={() => (showRaw.value = showRaw.value === 'response' ? 'off' : 'response')}
          >
            {translate('res.raw')} res
          </button>
          {payload && (
            <button
              class="toolBtn"
              onClick={() => {
                void navigator.clipboard?.writeText(payload.raw).then(() => flash(translate('res.copied')))
              }}
            >
              {translate('res.copy')}
            </button>
          )}
        </div>
      </div>

      {showRaw.value !== 'off' && (
        <pre class="code rawBlock">{showRaw.value === 'request' ? bodyPreview.value : (payload?.raw ?? '')}</pre>
      )}

      {running.value ? (
        <Skeleton />
      ) : error ? (
        <div class="errBox">
          <div class="errHead">
            {/* 传输层 2xx 但内容不对时, 挂「HTTP 200」只会让人以为状态码才是问题 */}
            <span class="errBadge">
              {error.status >= 400 ? `HTTP ${error.status}` : translate('err.title')}
            </span>
            {/* 422 的 detail 本身就是逐字段数组, 标题再挑首条复述一遍只会重复; 改成「字段错误 (n)」摘要 */}
            <span class="errMsg">
              {error.fields.length
                ? `${translate('err.fields')} (${error.fields.length})`
                : humanize(lang.value, error.message)}
            </span>
          </div>
          {error.fields.length > 0 && (
            <ul class="fieldErrs">
              {error.fields.map((item) => (
                <li key={item.loc.join(' / ') + item.msg}>
                  <code>{item.loc.join(' / ')}</code>
                  <span>{item.msg}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : payload ? (
        <div class="ansList">
          {Object.entries(payload.answers).map(([name, answer], index) => (
            <Body key={name} name={name} answer={answer} index={index} />
          ))}
        </div>
      ) : (
        <div class="emptyState">
          <div class="emptyIcon">✦</div>
          <p>{translate('res.empty')}</p>
          <p class="muted">{translate('res.emptyHint')}</p>
        </div>
      )}

      {payload && !running.value && (
        <footer class="resFoot">
          <span class="muted">
            {translate('res.model')}: {payload.model}
          </span>
          <span class="muted">{usage}</span>
        </footer>
      )}
    </section>
  )
}
