# 本地部署 Qwen3-0.6B 流式 Markdown 渲染 — 基础实现清单

本文为最小可行实现（MVP）清单，包含必要步骤、基础命令和关键注意点，供快速搭建本地网页化对话并支持流式 Markdown 渲染。

## 目标
- 在本地启动 Qwen3-0.6B 模型并通过 WebSocket 将生成的增量文本流式推送到浏览器。
- 浏览器端接收增量文本并用 Markdown 渲染（防 XSS）显示。

## 前提（最基本）
- 已在仓库 `./models/Qwen3-0.6B` 下放置 Qwen3-0.6B 模型文件。
- 操作系统：Windows（或其它），Python >= 3.11，建议 conda 环境。
- 基本依赖：PyTorch、transformers、FastAPI（或任意支持 WS 的框架）、uvicorn。

## 目录及文件（建议）
- `Qwen3_0_6B_Chat.py`：仓库已有 CLI 示例，可作为 prompt、tokenizer 与生成参数参考。
- `streaming_markdown.py`：仓库已有的终端级流式 Markdown 工具，提供 `normalize_markdown` 等函数参考。
- 新增 `web_server.py`：FastAPI/uvicorn + WebSocket 服务，包含 WebSocketStreamer。
- 新增 `web_static/index.html`：前端页面（markdown-it + DOMPurify），用于增量渲染。

## 环境准备（最少步骤）
1. 新建并激活虚拟环境：
```bash
conda create -n qwen3 python=3.11 -y
conda activate qwen3
pip install --upgrade pip
```
2. 安装依赖：
```bash
pip install torch transformers fastapi "uvicorn[standard]" websockets markdown-it-py
# 前端工具可使用 CDN，不必通过 npm
```

## 后端（简要实现要点）
- 使用 `FastAPI`，启用 WebSocket 路径 `/ws/chat`。
- 使用 `transformers.AutoTokenizer`、`AutoModelForCausalLM` 加载模型（可放到 GPU 或 CPU）。
- 自定义 `WebSocketStreamer`（可参照 `TextStreamer`），在 `put()` 中把 decode 后字符串回传给 websocket（通过 `asyncio.Queue` 做线程到协程的桥接）。
- Generate 在后台线程运行，主协程异步从队列读取并通过 WebSocket 发送增量 `delta` 消息，最后发送 `done` 消息。

### 关键逻辑（要点）
- `prefill_len`：跳过 prompt 部分的 token
- `skip_prompt`：确保前端只见到生成的文本
- 安全：对外只允许本地访问（或加鉴权），限制 `max_new_tokens` 和并发

## 前端（最少实现）
- 一个简单的 `index.html`，使用 WebSocket 与后端通信。
- 使用 `markdown-it` 或 `markdown-it-py` 渲染增量文本。
- 使用 `DOMPurify`（或类似）对生成 HTML 做 sanitize，以防 XSS。
- 保持缓存字符串（buffer），对每次收到的 `delta` 替换/更新并渲染。
- 对未闭合的 ``` fence 执行临时闭合，最终在 `done` 时进行完整渲染。

### 前端 UI 功能要求（MVP）
- 侧边栏：展示历史会话（最近的会话在上），支持选择历史会话。
- 新建对话按钮：在侧栏顶部，清除当前对话并创建新的会话条目。
- 主聊天区：显示 User（用户）与 Assistant（模型）的消息，按时间顺序排列，支持增量更新（流式显示）。
- 聊天输入区：多行文本输入框、发送按钮，支持 Ctrl/Cmd+Enter 快捷发送。
- 停止按钮：当模型正在生成时显示，允许客户端发送停止请求来中断生成。
- 深度思考（Thinking）开关：当打开时，发送的消息中带上 `thinking: true` 字段（后端可据此切换思维模式或策略）。
- 会话标题：支持修改当前会话标题（保存到本地存储），并在侧边栏展示。
- 本地会话持久化：使用 `localStorage` 保存会话与消息，页面刷新后仍能恢复历史记录。
- Markdown 渲染：增量渲染过程中，前端临时闭合未闭合代码 fence，以避免渲染错误；最终结束时进行完整渲染并执行语法高亮。


## 启动与测试（最少命令）
1. 本地启动 API Server：
```bash
uvicorn web_server:app --host 0.0.0.0 --port 8000 --workers 1
```
2. 在浏览器打开 `web_static/index.html`（或由服务器静态托管）。
3. 在页面输入消息并发送，观察增量 `delta` 和最终 `done`。

## 最低安全与稳定性建议
- 前端强制 sanitize 输出（DOMPurify）。
- 限制 `max_new_tokens` 和并发（Semaphore / 队列）。
- 对模型部署机器的内存/GPU 做监控，避免 OOM。

## 测试项（最基本）
- 单用户流式：能否收到增量消息、渲染 Markdown、代码块高亮。
- 未闭合 fence 的临时修复能否使渲染无报错。
- 并发连接限制：如果超过阈值，应返回 429 或拒绝新连接。

## 说明与扩展（仅点到为止）
- 若需更高吞吐，考虑 vLLM、TGI、或量化模型（4/8-bit）
- 若需多用户会话管理，建议引入 Redis/SQLite 做会话存储

---
文件基于仓库现有 `Qwen3_0_6B_Chat.py` 与 `streaming_markdown.py`，仅包含最基础要点和命令，后续如需我可以把 `web_server.py` 和前端样例文件添加到仓库并自动测试。