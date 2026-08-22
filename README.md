# AI Code Service

基于 Apple MLX / Qwen 编程大语言模型的高性能 AI Code 服务，提供完整的 OpenAI 兼容 API、FIM (Fill-In-The-Middle) 代码自动补全，以及面向 IDE 插件（如 VS Code, Cursor, Continue）的专有快捷编程助手接口。原生支持 **256K (262,144 Tokens)** 超长上下文与 Apple Silicon 统一显存硬件加速。

---

## 🌟 核心特性

1. **OpenAI 标准兼容**:
   - `POST /v1/chat/completions`: 对话生成（支持 Stream SSE 实时流式响应与 System Prompt 注入）。
   - `GET /v1/models` & `GET /v1/models/{model_id}`: 模型列表查询与 256K 元数据。
2. **FIM 代码自动补全**:
   - `POST /v1/completions`: 支持补全前缀（Prefix）与后缀（Suffix）填空（Fill-In-The-Middle），直接适配 IDE Autocomplete。
3. **256K 超长上下文支持 (262,144 Tokens)**:
   - 全面适配整库代码检索、跨多文件重构及超大项目分析，在 128GB 统一内存 Mac 下显存占用平稳且无 OOM 风险。
4. **编程助手专用接口**:
   - `POST /v1/code/explain`: 代码解释。
   - `POST /v1/code/refactor`: 代码重构。
   - `POST /v1/code/generate-tests`: 自动化单测生成（支持 pytest / unittest / jest 等）。
   - `POST /v1/code/fix-bugs`: 自动分析并修复代码 Bug。
5. **🚀 深度性能优化 (Apple Silicon & LLM Acceleration)**:
   - **异步非阻塞执行**: 推理任务完全从 FastAPI 事件循环中剥离到线程池执行，支持真正的并发请求与异步流式 SSE 管道。
   - **LRU + TTL 响应缓存**: 高频重复代码查询、IDE 补全直接命中缓存（响应延迟 < 1ms），并在 Header 输出 `X-Cache: HIT`。
   - **Metal 显存管理**: 针对 128G 内存配置 4GB Metal 缓存池，防止大模型长时间运行导致显存碎片膨胀。
   - **Tokenizer LRU 缓存**: 高频 Prompt 与 Token 编码哈希缓存，消除重复分词开销。
   - **全链路 TPS 与显存监控**: `/metrics` 接口输出实时显存使用情况（Active/Cache/Peak）、Prompt TPS、Generation TPS 及缓存命中率。
6. **安全与监控**:
   - 支持可选的 `API_KEY` Bearer Token 身份验证。
   - 跨域 CORS 支持与 `X-Request-ID` / `X-Process-Time` / `X-Prompt-Tokens` / `X-Completion-Tokens` 全链路追踪。
   - `GET /health` & `GET /metrics` 监控节点健康、显存与运行状态。

---

## 📁 架构目录结构

```
llmqwen/
├── app.py                     # FastAPI 服务入口与中间件
├── config.py                  # Pydantic 动态配置加载器（默认 256K 上下文）
├── config.yaml                # 服务与模型配置文件（256K 上下文 & 性能参数）
├── start.sh                   # 启动与测试自动化脚本（内置 Conda 自动激活）
├── download.py                # ModelScope/HuggingFace 模型下载脚本
├── client_demo.py             # 客户端 API 功能演示脚本
├── environment.yml            # Conda 环境定义文件 (Python 3.13+)
├── requirements.txt           # 项目依赖清单
├── pytest.ini                 # Pytest 测试配置文件
├── .vscode/                   # VS Code / Cursor IDE 解释器配置
├── engine/                    # 模型推理引擎模块
│   ├── base.py                # 推理引擎抽象基类（支持异步非阻塞及流式生成）
│   ├── cache.py               # 线程安全的 LRU + TTL 响应缓存器
│   ├── mlx_engine.py          # Apple MLX 推理引擎（显存调优与 TPS 统计）
│   ├── mock_engine.py         # Mock 仿真推理引擎
│   └── manager.py             # 模型统一加载、Prompt 构建器与性能指标聚合
├── schemas/                   # Pydantic 数据契约对象
│   └── openai.py              # OpenAI 兼容接口及代码工具 Schema
├── routers/                   # FastAPI 路由模块
│   ├── chat.py                # Chat Completions 路由（异步 + 缓存）
│   ├── completions.py         # FIM Autocomplete 路由（异步 + 缓存）
│   ├── code.py                # 编程助手专用工具路由
│   ├── models.py              # 模型管理路由
│   └── health.py              # 健康检查与显存/缓存监控路由
└── tests/                     # 完整单元测试与集成测试套件
    ├── conftest.py            # Pytest Fixtures
    ├── test_chat.py           # 对话与流式测试
    ├── test_completions.py    # FIM 补全测试
    ├── test_code_helpers.py   # 代码助手工具测试
    ├── test_models_and_health.py # 模型与健康检查测试
    ├── test_auth_and_config.py# 鉴权与配置测试
    ├── test_engine_and_prompt.py # 引擎与 Prompt 测试
    └── test_performance_and_cache.py # 性能优化与缓存专项测试
```

---

## 🐍 环境准备 (Conda & Python 3.13+)

本项目推荐使用 **Python 3.13+** 以及专用 Conda 环境 `llmqwen`。

### 1. 创建并激活 Conda 环境

#### 方式 A：通过 `environment.yml` 一键创建
```bash
conda env create -f environment.yml
conda activate llmqwen
```

#### 方式 B：手动创建与安装
```bash
# 1. 创建 Python 3.13 环境
conda create -y -n llmqwen python=3.13 -c conda-forge

# 2. 激活环境
conda activate llmqwen

# 3. 安装项目依赖
pip install -r requirements.txt
```

---

## 🚀 快速开始

### 1. 运行测试套件
`start.sh` 会自动检测并激活 `llmqwen` Conda 环境：
```bash
./start.sh --test
# 或手动在 conda 环境中运行
pytest -v
```

### 2. 启动 API 服务
```bash
./start.sh
# 或手动在 conda 环境中运行
python app.py
```
服务默认运行在 `http://localhost:8000`，交互式 API 文档可通过 `http://localhost:8000/docs` 访问。

### 3. 运行 API 验证客户端
在另一个终端窗口中：
```bash
python client_demo.py
```

---

## 🔌 在编程助手中配置与使用 (IDE & Copilot 配置指南)

本项目提供标准的 **OpenAI 兼容接口** 与 **FIM 代码补全接口**，可直接无缝接入 VS Code、Cursor、Continue、Cline、GitHub Copilot 生态等各类 AI 编程插件中。

### 📌 核心连接参数速查

| 参数项 | 推荐填入值 | 说明 |
| :--- | :--- | :--- |
| **API Provider / 类型** | `OpenAI` 或 `OpenAI Compatible` | 兼容标准 OpenAI 协议 |
| **API Base URL** | `http://localhost:8000/v1` | 服务 API 端点地址 |
| **API Key / Token** | `sk-aicodeservice-secret` 或 `dummy` | 若在 `config.yaml` 中配置了密钥则填密钥，未启用可填任意字符 |
| **Model Name (模型名)** | `qwen3.8-27b` | 或 `qwen3.8-27b-8bit` / `qwen3.8-27b-mlx` |
| **Context Window (上下文)** | `262144` (256K) | 模型最大上下文 Token 数 |
| **Max Output Tokens** | `2048` ~ `4096` | 单次输出最大 Token 数 |

---

### 🛠️ 常见编程插件配置示例

#### 1. Continue 插件配置 (VS Code / JetBrains / Cursor)
Continue 是目前最主流的开源代码助手插件，支持**对话聊天**与 **Tab 键行内 FIM 代码补全**。

打开 Continue 配置文件 `~/.continue/config.json`（或在插件中点击设置），添加如下配置：

```json
{
  "models": [
    {
      "title": "Qwen 27B (Local MLX 256K)",
      "provider": "openai",
      "model": "qwen3.8-27b",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "dummy",
      "contextLength": 262144
    }
  ],
  "tabAutocompleteModel": {
    "title": "Qwen FIM Autocomplete",
    "provider": "openai",
    "model": "qwen3.8-27b",
    "apiBase": "http://localhost:8000/v1",
    "apiKey": "dummy"
  }
}
```

---

#### 2. Cursor IDE 配置
Cursor 原生支持切换为自定义 OpenAI 端点：

1. 打开 Cursor 设置（`Cmd + ,` 或右上方齿轮图标）。
2. 进入 **Cursor Settings** -> **Models**。
3. 开启 **OpenAI API Key**，点击 **Override OpenAI Base URL** 并填入：
   - **Base URL**: `http://localhost:8000/v1`
   - **API Key**: `dummy`（或您的 `api_key`）
4. 在模型列表下方点击 **Add model**，输入 `qwen3.8-27b` 并保存选中。

---

#### 3. GitHub Copilot Chat (VS Code / 自定义代理或 Language Model API)
在 VS Code 中，GitHub Copilot Chat 支持通过语言模型扩展接口（VS Code Language Model API）或配合自定义端点转发插件（如 Copilot Custom Endpoint Proxy）接入私有模型：

1. **方式一：通过 Continue / Cline 替代并在界面中作为主要 Copilot 面板**
   - 直接使用 Continue 插件左侧面板，体验完全一致的 Copilot 对话与代码操作。
2. **方式二：在 VS Code `settings.json` 中配置语言模型端点（适用于支持自定义端点的 Copilot 扩展版本）**：
   ```json
   {
     "github.copilot.advanced": {
       "debug.overrideEngine": "qwen3.8-27b",
       "debug.overrideProxyUrl": "http://localhost:8000/v1"
     }
   }
   ```

---

#### 4. Cline / Roo Code 插件配置 (VS Code Autonomous Agent)
1. 在 VS Code 安装 **Cline** 或 **Roo Code** 插件。
2. 点击插件设置图标（齿轮），进入 API Provider 选择：
   - **API Provider**: 选择 `OpenAI Compatible`
   - **Base URL**: `http://localhost:8000/v1`
   - **API Key**: `dummy`
   - **Model ID**: `qwen3.8-27b`
   - **Context Window**: `262144`

---

#### 5. Aider / 命令行终端编程助手
使用终端 AI 结对编程工具 Aider 时，直接指定环境变量：

```bash
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"

# 启动 aider 并指定模型
aider --model openai/qwen3.8-27b
```

---

## ⚡ 性能与配置文件说明 (`config.yaml`)

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  reload: false
  cors_origins: ["*"]
  # api_key: "sk-aicodeservice-secret"  # 取消注释开启 API Key 鉴权

# 性能优化与显存调优 (针对 128G 统一内存 Mac 调优)
performance:
  enable_cache: true               # 启用响应与 Prompt LRU 缓存（高频重复请求返回 < 1ms）
  cache_max_size: 1024              # 缓存最大条目数
  cache_ttl_seconds: 3600           # 缓存过期时间（秒，默认 1 小时）
  max_concurrency: 3                # 并发生成信号量 (256K 上下文推荐 2~3 并发)
  metal_cache_limit_mb: 4096        # Apple Silicon MLX Metal 显存缓存上限 (4GB)
  clear_cache_after_generation: false # 单次大文本生成后是否主动清理显存碎片
  stream_chunk_size: 1              # 流式响应 Token 聚合块

default_model: "qwen3.8-27b"
use_mock: false  # 设置为 true 可在无 GPU 环境下全仿真运行

# 256K 超长上下文模型配置 (262,144 Tokens)
models:
  qwen3.8-27b-mlx:
    path: "inferencerlabs/Qwen3.8-27B-MTP-MLX"
    description: "Qwen3.8-27B MLX MTP 版本 (支持 256K 上下文)"
    engine_type: "mlx_vlm"
    context_length: 262144

  qwen3.8-27b-8bit:
    path: "lmstudio-community/Qwen3.8-27B-MLX-8bit"
    description: "Qwen3.8-27B MLX 8bit 量化版本 (支持 256K 上下文)"
    engine_type: "mlx_lm"
    context_length: 262144

  qwen3.8-27b:
    path: "Qwen/Qwen3.8-27B"
    description: "Qwen3.8-27B 标准版本 (支持 256K 上下文)"
    engine_type: "auto"
    context_length: 262144
```
