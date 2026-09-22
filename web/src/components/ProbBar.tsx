// 概率条 + 环形表盘: 把 0~1 的概率画成可读的形。
// 输入: label / probability / winner / index; 输出: 一行条形或一个表盘;
// 预期: 条宽走 transform: scaleX (GPU 合成, 不触发回流), 极小概率也留一个可见的起点,
//       读数按量级给小数位, 入场逐条错开 45ms。

import type { JSX } from 'preact'
import { useTween } from '../lib/useTween'

/** 概率读数: 大数一位小数, 中数两位, 极小给四位, 免得全是 0.00%. */
export function formatProbability(probability: number): string {
  const percent = probability * 100
  if (percent === 0) return '0%'
  const digits = percent >= 10 ? 1 : percent >= 0.1 ? 2 : 4
  return `${percent.toFixed(digits)}%`
}

export function ProbBar(props: { label: string; probability: number; winner: boolean; index: number }): JSX.Element {
  const swept = useTween(props.probability, 720)
  const nub = props.probability > 0 ? 0.008 : 0
  const scale = Math.max(swept, nub)
  return (
    <div class={props.winner ? 'probRow win' : 'probRow'} style={{ '--i': props.index } as JSX.CSSProperties}>
      <span class="probLabel" title={props.label}>
        {props.label}
      </span>
      <span class="probTrack">
        <span class="probFill" style={{ transform: `scaleX(${scale.toFixed(4)})` }} />
      </span>
      <span class="probValue">{formatProbability(props.probability)}</span>
    </div>
  )
}

/** 环形表盘: 扫过角度写进 --sweep, 由 CSS 的 conic-gradient + mask 画出圆环. */
export function Gauge(props: { value: number; caption: string }): JSX.Element {
  const swept = useTween(props.value, 760)
  const angle = Math.max(0, Math.min(1, swept)) * 360
  return (
    <div class="gauge" title={props.caption}>
      <span class="gaugeFace" style={{ '--sweep': `${angle.toFixed(2)}deg` } as JSX.CSSProperties}>
        <span class="gaugeHole">
          <b>{(props.value * 100).toFixed(1)}</b>
          <i>%</i>
        </span>
      </span>
      <span class="gaugeCaption">{props.caption}</span>
    </div>
  )
}
