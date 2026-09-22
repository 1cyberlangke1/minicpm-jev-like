// 数字缓动: 把离散跳变 (耗时、token 数、概率) 变成一段短动画, 避免读数一闪一闪。
// 输入: 目标值; 输出: 当前动画值;
// 预期: 600ms ease-out 收敛, 期间再次设值会从当前值续接; prefers-reduced-motion 下直接落位。

import { useEffect, useRef, useState } from 'preact/hooks'

export function useTween(target: number, duration = 600): number {
  const [value, setValue] = useState(target)
  const from = useRef(target)
  const frame = useRef(0)

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      from.current = target
      setValue(target)
      return
    }
    cancelAnimationFrame(frame.current)
    const start = performance.now()
    const origin = from.current
    const step = (now: number): void => {
      const progress = Math.min(1, (now - start) / duration)
      const eased = 1 - (1 - progress) ** 3
      const next = origin + (target - origin) * eased
      from.current = next
      setValue(next)
      if (progress < 1) frame.current = requestAnimationFrame(step)
    }
    frame.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frame.current)
  }, [target, duration])

  return value
}
