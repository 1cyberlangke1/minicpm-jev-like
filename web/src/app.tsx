// 应用外壳: 顶栏 / 示例条 / 左编辑右结果 / 设置弹窗 / 轻提示。
// 输入: 无; 输出: 整页 DOM;
// 预期: 两栏在宽屏并排、窄屏堆叠，滚动条各自独立，编辑区不会被结果挤出视口。

import type { JSX } from 'preact'
import { useState } from 'preact/hooks'
import { toast, running } from './store'
import { TopBar } from './components/TopBar'
import { PresetBar } from './components/PresetBar'
import { StatePanel } from './components/StatePanel'
import { QuestionsPanel } from './components/QuestionsPanel'
import { RunDock } from './components/RunDock'
import { ResultsPanel } from './components/ResultsPanel'
import { SettingsModal } from './components/SettingsModal'

export function App(): JSX.Element {
  const [settings, setSettings] = useState(false)
  return (
    <div class={`shell${running.value ? ' working' : ''}`}>
      <TopBar onSettings={() => setSettings(true)} />
      <PresetBar />
      <main class="workbench">
        <section class="colLeft">
          <StatePanel />
          <QuestionsPanel />
          <RunDock />
        </section>
        <section class="colRight">
          <ResultsPanel />
        </section>
      </main>
      {settings && <SettingsModal onClose={() => setSettings(false)} />}
      {toast.value && <div class="toast">{toast.value}</div>}
    </div>
  )
}
