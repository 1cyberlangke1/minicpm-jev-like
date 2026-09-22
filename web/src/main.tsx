// 入口: 挂 DOM + 引入样式。
// 输入: 无; 输出: 渲染完成的页面;
// 预期: 样式必须在 render 之前进文档，避免首帧闪一下裸 HTML。

import { render } from 'preact'
import { App } from './app'
import './styles.css'

render(<App />, document.getElementById('app')!)
