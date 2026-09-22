// 上下文编辑器: 带行号的 state 输入区。
// 输入: 无 (读写 signals); 输出: 面板 DOM;
// 预期: 内容是字符串还是 JSON 由前端原样交给后端, 这里只给一个「看起来像 JSON」的提示, 不擅自改写。
//       行号栏靠 scrollTop 跟随 textarea 同步滚动; 该输入区禁软换行, 保证「逻辑行 = 视觉行」, 行号才不会错位。

import type { JSX } from 'preact'
import { useRef } from 'preact/hooks'
import { stateText, t } from '../store'

const countLines = (text: string): number => text.split('\n').length

export function StatePanel(): JSX.Element {
  const text = stateText.value
  const looksJson = /^\s*[[{]/.test(text)
  const gutter = useRef<HTMLDivElement>(null)
  return (
    <section class="panel statePanel">
      <div class="panelHead">
        <span class="panelTitle">{t.value('state.title')}</span>
        <div class="headActions">
          <span class="muted hint" title={t.value('state.hintFull')}>{t.value('state.hint')}</span>
          <span class="badge">{text.length} chars · {countLines(text)} ln</span>
        </div>
      </div>
      <div class="editorWrap">
        <div class="gutter" ref={gutter} aria-hidden="true">
          {Array.from({ length: countLines(text) }, (_, index) => (
            <span key={index}>{index + 1}</span>
          ))}
        </div>
        <textarea
          class="code"
          spellcheck={false}
          value={text}
          onInput={(event) => (stateText.value = (event.currentTarget as HTMLTextAreaElement).value)}
          onScroll={(event) => {
            if (gutter.current) gutter.current.scrollTop = (event.currentTarget as HTMLTextAreaElement).scrollTop
          }}
        />
        {looksJson && (
          <span class="jsonFlag" title="JSON">
            {'{ }'}
          </span>
        )}
      </div>
    </section>
  )
}
