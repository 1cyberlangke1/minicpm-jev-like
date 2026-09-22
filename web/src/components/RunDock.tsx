// 执行坞: 思考预算 + 发起请求 + 实时耗时。
// 输入: 无; 输出: 吸底控制条;
// 预期: think_tokens 默认 0 (纯 prefill 决策), Ctrl+↵ 与点按钮走同一条 run() 路径。

import type { JSX } from 'preact'
import { useEffect } from 'preact/hooks'
import { elapsed, problems, run, running, thinkTokens, t } from '../store'
import { useTween } from '../lib/useTween'

export function RunDock(): JSX.Element {
  const translate = t.value
  const seconds = useTween(elapsed.value / 1000, 240)
  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault()
        void run()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <div class="runDock">
      <label class="thinkField">
        <span class="thinkLabel">{t.value('run.thinking')}</span>
        <input type="range" min={0} max={1024} step={32} value={thinkTokens.value} onInput={(event) => {
          thinkTokens.value = Number((event.currentTarget as HTMLInputElement).value)
        }} />
        <span class="thinkValue">
          {thinkTokens.value} <i>{t.value('run.tokens')}</i>
        </span>
      </label>

      <div class="runActions">
        {problems.value.length > 0 && (
          <span class="inlineWarn">
            {problems.value.map((item) => (item.detail ? `${translate(item.key)}: ${item.detail}` : translate(item.key))).join(' · ')}
          </span>
        )}
        <button class="runBtn" disabled={running.value || problems.value.length > 0} onClick={() => void run()}>
          <span class="runLabel">{running.value ? translate('run.running') : translate('run.button')}</span>
          <kbd>{t.value('hint.shortcut')}</kbd>
        </button>
        {/* 亚 10ms 的请求 (客户端拦截、422) 停在 "0.00s" 看着像没计时, 显式写成 <0.01s */}
        <span class="elapsed">
          {running.value || elapsed.value ? `${seconds < 0.01 ? '<0.01' : seconds.toFixed(2)}s` : ''}
        </span>
      </div>
    </div>
  )
}
