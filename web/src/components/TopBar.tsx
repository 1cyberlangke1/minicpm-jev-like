// 顶栏: 品牌 + 模型选择 + 连接状态 + 三个开关 (设置 / 语言 / 主题)。
// 输入: onSettings -- 打开连接设置弹窗; 输出: 顶栏 DOM;
// 预期: 模型列表来自 GET /v1/models, 后端没起来时选择器显示占位符而不是空壳。

import type { JSX } from 'preact'
import { useEffect } from 'preact/hooks'
import { apiKey, base, lang, model$, models, refreshModels, running, setTheme, theme, t, toggleLang } from '../store'

export function TopBar(props: { onSettings: () => void }): JSX.Element {
  useEffect(() => {
    void refreshModels()
  }, [])

  const offline = models.value.length === 0
  const statusClass = running.value ? 'chip status busy' : offline ? 'chip status bad' : 'chip status good'

  return (
    <header class="topbar">
      <div class="brand">
        <span class="brandMark" aria-hidden="true">
          ◍
        </span>
        <span class="brandText">
          <strong>{t.value('brand.title')}</strong>
          <small>{t.value('brand.sub')}</small>
        </span>
      </div>

      <div class="topActions">
        <label class="field">
          <span class="fieldLabel">{t.value('nav.models')}</span>
          <span class="selectWrap">
            <select
              value={model$.value}
              // 窄屏会隐藏可见标签 (见 styles.css 手机档), 无障碍名靠这里兜住
              aria-label={t.value('nav.models')}
              onChange={(event) => {
                model$.value = (event.currentTarget as HTMLSelectElement).value
              }}
            >
              {/* 后端不可达时清单为空, 仍把上一次选中的模型显式列一项, 免得选择器看着像坏了 */}
              {offline && <option value={model$.value}>{model$.value || '—'}</option>}
              {models.value.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.name}
                </option>
              ))}
            </select>
            <span class="selectChevron" aria-hidden="true">
              ▾
            </span>
          </span>
        </label>

        <button
          class={statusClass}
          onClick={() => void refreshModels()}
          title={`${base.value} · ${apiKey.value ? 'key ✓' : 'key —'}`}
        >
          <span class="statusDot" aria-hidden="true" />
          {running.value ? t.value('run.running') : offline ? t.value('status.offline') : t.value('status.online')}
        </button>

        <button class="ghost" onClick={props.onSettings}>
          {t.value('nav.key')}
        </button>
        <button class="ghost" onClick={toggleLang} title={t.value('nav.lang')}>
          {lang.value === 'zh' ? 'EN' : '中'}
        </button>
        <button
          class="ghost iconOnly"
          onClick={() => setTheme(theme.value === 'dark' ? 'light' : 'dark')}
          title={t.value('nav.theme')}
          aria-label={t.value('nav.theme')}
        >
          {theme.value === 'dark' ? '☀' : '☾'}
        </button>
      </div>
    </header>
  )
}
