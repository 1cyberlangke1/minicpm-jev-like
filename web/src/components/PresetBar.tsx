// 示例快捷条: 一键把 state + 问题列表换成预置场景。
// 输入: 无; 输出: 横滑的 chip 条;
// 预期: 当前生效的 preset 高亮, 点击会覆盖编辑器内容 (结果区同时清空)。

import type { JSX } from 'preact'
import { signal } from '@preact/signals'
import { applyPreset, t } from '../store'
import { PRESETS } from '../lib/presets'

const active = signal('')

export function PresetBar(): JSX.Element {
  return (
    <div class="presetBar">
      <span class="presetLabel">{t.value('presets.title')}</span>
      <div class="presetTrack">
        {PRESETS.map((preset) => (
          <button
            class={active.value === preset.id ? 'presetChip on' : 'presetChip'}
            onClick={() => {
              active.value = preset.id
              applyPreset(preset.id)
            }}
          >
            {t.value(preset.labelKey)}
          </button>
        ))}
      </div>
    </div>
  )
}
