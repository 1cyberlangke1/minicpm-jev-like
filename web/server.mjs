// 独立 web 服务: 托管 dist/ 静态资源, 并把 /v1/* 原样转发给决策引擎后端。
// 输入: 环境变量 WEB_HOST (默认 127.0.0.1) / WEB_PORT (默认 5273) / JEV_BACKEND (默认 http://127.0.0.1:8000);
// 输出: 常驻 HTTP 进程, 启动时打印监听地址;
// 预期: 不依赖任何第三方包, 也不依赖后端框架 —— 后端只会说 HTTP, 换语言实现照样能连。
import { createServer } from 'node:http'
import { createReadStream, existsSync, statSync } from 'node:fs'
import { extname, join, normalize, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)), 'dist')
const HOST = process.env.WEB_HOST ?? '127.0.0.1'
const PORT = Number(process.env.WEB_PORT ?? 5273)
const BACKEND = new URL(process.env.JEV_BACKEND ?? 'http://127.0.0.1:8000')

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.map': 'application/json; charset=utf-8',
}

/** 把 /v1/* 转发到后端, 状态码 / 头 / 流式响应体都原样回传. */
async function proxy(req, res) {
  const target = new URL(req.url, BACKEND)
  const chunks = []
  for await (const part of req) chunks.push(part)
  let upstream
  try {
    upstream = await fetch(target, {
      method: req.method,
      headers: {
        'content-type': req.headers['content-type'] ?? 'application/json',
        ...(req.headers.authorization ? { authorization: req.headers.authorization } : {}),
      },
      body: chunks.length && req.method !== 'GET' && req.method !== 'HEAD' ? Buffer.concat(chunks) : undefined,
    })
  } catch {
    res.writeHead(502, { 'content-type': 'application/json; charset=utf-8' })
    res.end(JSON.stringify({ detail: { error_type: 'bad_gateway', message: `Backend ${target.origin} is unreachable.` } }))
    return
  }
  const headers = {}
  upstream.headers.forEach((value, key) => {
    if (['content-length', 'content-encoding', 'transfer-encoding'].includes(key)) return
    headers[key] = value
  })
  res.writeHead(upstream.status, headers)
  const reader = upstream.body?.getReader()
  if (!reader) return res.end()
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    res.write(Buffer.from(value))
  }
  res.end()
}

/** 静态文件: 只允许 dist/ 内的路径; 单页回退只对 GET/HEAD 生效, 其它方法按 404 契约回. */
function serveStatic(req, res) {
  const pathname = decodeURIComponent(new URL(req.url, 'http://x').pathname)
  let file = join(ROOT, normalize(pathname).replace(/^(\.\.[/\\])+/, ''))
  if (!file.startsWith(ROOT + sep) && file !== ROOT) {
    res.writeHead(403).end('forbidden')
    return
  }
  if (!existsSync(file) || statSync(file).isDirectory()) {
    // 把 index.html 发给一个 POST 只会让客户端拿 HTML 去 JSON.parse;
    // 未知方法/路径直接回后端同款 detail 结构, 前端能一行读出来。
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(404, { 'content-type': 'application/json; charset=utf-8' }).end(
        JSON.stringify({ detail: { error_type: 'not_found', message: `${req.method} ${pathname} is not served by the web static host.` } }),
      )
      return
    }
    file = join(ROOT, 'index.html')
  }
  if (!existsSync(file)) {
    res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' }).end('dist/ 缺失: 先跑 npm run build')
    return
  }
  res.writeHead(200, {
    'content-type': MIME[extname(file)] ?? 'application/octet-stream',
    'cache-control': extname(file) === '.html' ? 'no-cache' : 'public, max-age=31536000, immutable',
  })
  createReadStream(file).pipe(res)
}

createServer((req, res) => {
  if (new URL(req.url, 'http://x').pathname.startsWith('/v1/')) return void proxy(req, res)
  serveStatic(req, res)
}).listen(PORT, HOST, () => {
  console.log(`web  → http://${HOST}:${PORT}   (后端 ${BACKEND.origin})`)
})
