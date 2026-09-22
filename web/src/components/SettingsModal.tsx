// 连接设置弹窗: 后端地址 + Bearer key。
// 输入: onClose; 输出: 遮罩 + 表单卡片;
// 预期: 地址留空即同源 /v1 (由独立 web 服务转发)，关闭或回车都立即重拉模型清单，
//       连接徽标因此不会停留在改地址之前的旧结论上。

import type { JSX } from 'preact'
import { apiKey, base, refreshModels, t } from '../store'

export function SettingsModal(props: { onClose: () => void }): JSX.Element {
  const translate = t.value
  const apply = (event: Event) => {
    event.preventDefault()
    event.stopPropagation()
    void refreshModels()
    props.onClose()
  }
  return (
    <div class="overlay" onClick={props.onClose}>
      <form class="modal" onClick={(event) => event.stopPropagation()} onSubmit={apply}>
        <h2>{translate('settings.title')}</h2>
        <label class="modalField">
          <span>{translate('settings.base')}</span>
          <input
            class="mono"
            value={base.value}
            spellcheck={false}
            onInput={(event) => (base.value = (event.currentTarget as HTMLInputElement).value)}
          />
          <small>{translate('settings.baseHint')}</small>
        </label>
        <label class="modalField">
          <span>API Key</span>
          <input
            class="mono"
            type="password"
            autocomplete="new-password"
            value={apiKey.value}
            spellcheck={false}
            placeholder="Bearer"
            onInput={(event) => (apiKey.value = (event.currentTarget as HTMLInputElement).value)}
          />
          <small>{translate('settings.keyHint')}</small>
        </label>
        <div class="modalFoot">
          <button
            type="button"
            class="toolBtn"
            title={translate('settings.resetHint')}
            onClick={() => {
              base.value = '/v1'
              // 地址回填 + 顶栏连接徽标已经说明结果, 再弹一条 toast 只是重复
              void refreshModels()
            }}
          >
            {translate('settings.reset')}
          </button>
          <button type="submit" class="runBtn compact">
            {translate('settings.close')}
          </button>
        </div>
      </form>
    </div>
  )
}
