# AI Code Service

基于 Apple MLX / Qwen 编程大语言模型的高性能 AI Code 服务，提供完整的 OpenAI 兼容 API、FIM (Fill-In-The-Middle) 代码自动补全，以及面向 IDE 插件（如 VS Code, Cursor, Continue）的专有快捷编程助手接口。原生支持 **128K / 256K 超长上下文**、**8-bit KV Cache 显存量化** 与 **MTP 投机采样多 Token 预测加速**。默认模型为 **Qwen3.8-Flash-Next oQ4e MTP (Qwen4Exp 架构)**。

---

## 🌟 核心特性

1. **OpenAI 标准兼容**:
   - `POST /v1/chat/completions`: 对话生成（支持 Stream SSE 实时打字机流式响应与 System Prompt 注入）。
   - `GET /v1/models` & `GET /v1/models/{model_id}`: 模型列表查询与 256K 元数据。
2. **FIM 代码自动补全**:
   - `POST /v1/completions`: 支持补全前缀（Prefix）与后缀（Suffix）填空（Fill-In-The-Middle），直接适配 IDE Autocomplete。
3. **256K 超长上下文支持 (262,144 Tokens)**:
   - 全面适配整库代码检索、跨多文件重构及超大项目分析，在 128GB 统一内存 Mac 下显存占用平稳且无 OOM 风险。
4. **编程助手专用接口 (全量支持 Stream SSE)**:
   - `POST /v1/code/explain`: 代码解释。
   - `POST /v1/code/refactor`: 代码重构。
   - `POST /v1/code/generate-tests`: 自动化单测生成（支持 pytest / unittest / jest 等）。
   - `POST /v1/code/fix-bugs`: 自动分析并修复代码 Bug。
   - `POST /v1/code/edit`: 行内代码智能编辑。
   - `POST /v1/code/review`: 代码审查与性能隐患诊断。
   - `POST /v1/code/docstring`: 文档字符串与注释自动化生成。
5. **🚀 深度性能优化 (Apple Silicon & LLM Acceleration)**:
   - **Qwen4Exp 实验架构模型**: `qwen3.8-flash-next-oq4e-mtp-128k` 默认首选，内置 MoE MTP，128K 上下文，通过自定义引擎注入加载。
   - **8-bit + MTP 综合版模型**: 28GB 8bit 量化主权重 + 238MB MTP 投机采样头，实现 2x ~ 3x 吞吐跃升（45~65+ tok/s）。
   - **KV Cache 显存量化 (`kv_bits: 8`)**: 256K 上下文 KV 显存从 64GB 降至 32GB，长文本带宽搬运减半，吞吐提升 30%~50%。
   - **分块预填充 (`prefill_step_size: 2048`)**: 流式 Prefill 计算，平抑超大代码文件输入的瞬间峰值显存。
   - **前缀/系统提示词缓存 (`enable_prompt_cache: true`)**: System Prompt 与高频代码前缀常驻 Metal 显存，首字延迟（TTFT）`< 5ms`。
   - **LRU + TTL 响应缓存**: 高频重复代码查询命中缓存直接返回（延迟 `< 1ms`），并在 Header 输出 `X-Cache: HIT`。
   - **全链路 TPS 与显存监控**: `/metrics` 接口输出实时显存使用情况（Active/Cache/Peak）、Prompt TPS、Generation TPS 及缓存命中率。

---

## 📁 架构目录结构

```
llmqwen/
├── app.py                     # FastAPI 服务入口与中间件
├── config.py                  # Pydantic 动态配置加载器（支持 .env 环境变量覆盖）
├── config.yaml                # 服务与模型配置文件（性能调优参数）
├── .env                       # 环境变量配置 (本地生效，已 gitignore)
├── .env.example               # 环境变量配置模板 (提交到仓库)
├── start.sh                   # 启动与测试自动化脚本（兼容旧版）
├── scripts/                   # 脚本工具集
│   ├── common.sh              # 公共函数库 (Conda 激活/环境加载/端口清理)
│   ├── start-all.sh           # 一键启动后端 + 前端
│   ├── start-backend.sh       # 启动 Python LLM 后端
│   ├── start-frontend.sh      # 启动 Go Web 前端网关
│   ├── stop-all.sh            # 停止所有服务
│   ├── test-mock.sh           # Mock 单元测试
│   ├── test-real.sh           # 真实模型集成测试
│   └── test-all.sh            # 全部测试
├── download.py                # ModelScope 模型下载与校验工具
├── convert_to_mlx.py          # 模型 MLX 转换与 4bit/8bit 量化脚本
├── extract_mtp.py             # MTP 投机采样头提取工具
├── merge_mtp.py               # MTP 投机采样头与基础模型一键整合工具
├── link_to_lmstudio.py        # LM Studio 模型软链接配置工具
├── client_demo.py             # 客户端 API 功能演示与打字机流式验证脚本
├── benchmark.py               # 🚀 多模型性能基准压测与自动化对比套件
├── environment.yml            # Conda 环境定义文件 (Python 3.13+)
├── requirements.txt           # 项目依赖清单
├── pytest.ini                 # Pytest 测试配置文件
├── .vscode/                   # VS Code / Cursor IDE 解释器配置
├── engine/                    # 模型推理引擎模块
│   ├── base.py                # 推理引擎抽象基类（支持异步非阻塞及流式生成）
│   ├── cache.py               # 线程安全的 LRU + TTL 响应缓存器
│   ├── mlx_engine.py          # Apple MLX 推理引擎（KV 量化、分块 Prefill、TPS 统计）
│   ├── qwen4_exp_engine.py    # Qwen4Exp 实验架构专用引擎（自定义代码注入加载）
│   ├── mock_engine.py         # Mock 仿真推理引擎
│   ├── mtp_draft.py           # MTP 投机采样 Draft 逻辑
│   └── manager.py             # 模型统一加载、Prompt 构建器与性能指标聚合
├── docs/                      # 架构设计与运维操作文档
│   ├── PERFORMANCE_DESIGN.md  # 5大核心维度深度性能优化与系统设计文档
│   ├── BENCHMARK_REPORT.md    # 📊 多模型性能基准压测报告与对比指南
│   ├── OLLAMA_GUIDE.md        # Ollama 部署指南
│   └── OPS_MANUAL.md          # 运维部署、全链路监控与故障排查手册
├── schemas/                   # Pydantic 数据契约对象
│   └── openai.py              # OpenAI 兼容接口及代码工具 Schema (支持 Stream)
├── routers/                   # FastAPI 路由模块
│   ├── chat.py                # Chat Completions 路由（异步 + 缓存）
│   ├── completions.py         # FIM Autocomplete 路由（异步 + 缓存）
│   ├── code.py                # 编程助手专用工具路由 (全接口支持 Stream)
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
    ├── test_performance_and_cache.py # 性能优化与缓存专项测试
    └── test_real_model_integration.py # 真实模型全链路集成测试
└── webapp/                    # Web 聊天界面 (Vite + Go 网关)
    ├── main.go                # Go 网关服务（代理 Python API + 嵌入前端）
    ├── start.sh               # Web UI 启动脚本 (支持 --dev)
    ├── frontend/              # Vite 前端项目
    │   ├── index.html         # Vite 入口 HTML
    │   ├── package.json       # 前端依赖与脚本
    │   ├── vite.config.js     # Vite 配置 (API 代理到后端)
    │   └── src/
    │       ├── main.js        # Vite 入口 JS
    │       ├── style.css      # 样式表
    │       └── app.js         # 前端逻辑 (SSE 流式 + 模型切换 + 性能指标)
    └── dist/                  # Vite 构建输出 (Go embed 嵌入)
```

---

## 📚 详细设计与运维手册

- 📖 [深度性能优化与系统架构设计文档 (`docs/PERFORMANCE_DESIGN.md`)](file:///Users/charles/Documents/AI/Python/llmqwen/docs/PERFORMANCE_DESIGN.md)：详细推导 256K 下 8-bit KV Cache 显存量化、分块预填充、Metal 提示词缓存与 MTP 投机采样的数学原理与加速机制。
- 🛠️ [运维部署、监控与故障排查手册 (`docs/OPS_MANUAL.md`)](file:///Users/charles/Documents/AI/Python/llmqwen/docs/OPS_MANUAL.md)：涵盖 Conda 环境部署、端口自愈、全链路 `/metrics` 监控、launchd 后台常驻守护与常见故障排查。

---

## 🤖 模型生态与本地支持列表

本项目已在本地配置并支持以下模型规格：

| 模型代号 (Model ID) | 权重类型与架构 | 显存占用 | 上下文长度 | 推荐场景 | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`qwen3.8-flash-next-oq4e-mtp-128k`**<br>*(默认首选)* | Qwen4Exp 实验架构 + oQ4e 量化 + 内置 MoE MTP | **~16 GB** | **128K** | **日常编程 / IDE 实时补全** | 默认模型，内置 MoE MTP 加速，通过 Qwen4ExpEngine 自定义注入加载。 |
| **`qwen3.8-27b-oq4e-fp16-mtp`** | oQ 混合精度 4bit + MTP 推测解码 | **~15 GB** | **256K** | 本地高精度推理 | 本地 MLX oQ 混合量化，支持 256K 上下文。 |
| **`qwen3.8-27b-8bit-mtp`** | 8-bit 量化 + MTP 投机采样一体化 | **~28.2 GB** | **256K** | 编程 / IDE 补全 | 8bit 内存优势与 238MB MTP 前瞻加速，输出速率达 **45~65+ tok/s**。 |
| **`qwen3.8-27b-8bit`** | 8-bit 量化独立版 (`lmstudio-community`) | **~28 GB** | **256K** | 高速独立推理 | 精度无损，显存占用低，极度稳定。 |
| **`qwen3.8-27b-mlx`** | MTP 投机采样辅助层 (`inferencerlabs`) | **~238 MB** | **256K** | 辅助投机层 | 专用于与 27B 基座配合进行多 Token 前瞻预测。 |

---

## 🛠️ 模型管理工具集

### 1. 下载模型 (`download.py`)
通过国内 ModelScope 高速源下载官方或社区模型：
```bash
# 下载全部预设模型
python download.py

# 指定下载特定模型
python download.py --model qwen3.8-27b-8bit
```

### 2. 转换与量化模型 (`convert_to_mlx.py`)
将标准全精度权重转换为 MLX 原生 8-bit / 4-bit 量化格式：
```bash
# 转换为 8-bit 量化版 (推荐：精度无损，速度翻倍)
python convert_to_mlx.py --bits 8 --output ./models/qwen3.8-27b-mlx-8bit

# 转换为 4-bit 量化版 (极致速度 3.5x，体积仅 15GB)
python convert_to_mlx.py --bits 4 --output ./models/qwen3.8-27b-mlx-4bit
```

### 3. 整合 MTP 投机采样模型 (`merge_mtp.py`)
一键将 238MB MTP 辅助层与 28GB 8-bit 主模型整合为一体化综合版：
```bash
python merge_mtp.py --output ./models/qwen3.8-27b-8bit-mtp
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

### 1. 配置环境变量
```bash
cp .env.example .env
```

### 2. 一键启动全部服务
```bash
./scripts/start-all.sh
```
启动后访问：
- 🌐 **Web 聊天界面**: `http://localhost:8080`
- 📖 **API 文档**: `http://localhost:1235/docs`
- 💊 **健康检查**: `http://localhost:1235/health`

### 3. 运行测试
```bash
./scripts/test-mock.sh    # Mock 单元测试 (不需要模型)
./scripts/test-real.sh    # 真实模型集成测试
./scripts/test-all.sh     # 全部测试
```

---

## 🌐 Web 聊天界面

前端基于 **Vite** 构建，支持 SSE 流式对话、多模型切换、性能指标展示 (TTFT / TPS / Tokens)。

### 开发模式 (Vite Dev Server + HMR)
```bash
# 先启动 Python 后端
./scripts/start-backend.sh

# 再启动 Vite 前端 (支持热更新)
./scripts/start-frontend.sh --dev
# 访问 http://localhost:5173
```

### 生产模式 (Vite Build + Go 网关)
```bash
# 一键构建并启动
./scripts/start-frontend.sh
# 或
./scripts/start-all.sh
# 访问 http://localhost:8080
```

生产模式下 Vite 构建产物会被 Go 网关通过 `embed` 嵌入二进制，单文件部署无额外依赖。

---

## 📜 脚本速查表

所有脚本位于 `scripts/` 目录，均支持自动加载 `.env` 和 Conda 环境。

### 服务管理

| 脚本 | 说明 | 示例 |
| :--- | :--- | :--- |
| `scripts/start-all.sh` | 一键启动后端 + 前端 | `./scripts/start-all.sh` |
| `scripts/start-backend.sh` | 仅启动 Python LLM 后端 | `./scripts/start-backend.sh` |
| `scripts/start-frontend.sh` | 启动 Go Web 前端 (生产) | `./scripts/start-frontend.sh` |
| `scripts/start-frontend.sh --dev` | 启动 Vite Dev Server (HMR) | `./scripts/start-frontend.sh --dev` |
| `scripts/stop-all.sh` | 停止所有服务 | `./scripts/stop-all.sh` |

### 测试

| 脚本 | 说明 | 示例 |
| :--- | :--- | :--- |
| `scripts/test-mock.sh` | Mock 单元测试 (无需模型) | `./scripts/test-mock.sh` |
| `scripts/test-real.sh` | 真实模型集成测试 | `./scripts/test-real.sh` |
| `scripts/test-all.sh` | 全部测试 | `./scripts/test-all.sh` |

测试脚本支持传递 pytest 参数：
```bash
# 运行单个测试
./scripts/test-real.sh test_engine_chat

# 按关键字筛选
./scripts/test-real.sh -k "stream or thinking"

# 只收集不运行
./scripts/test-mock.sh --collect-only
```

### 兼容旧脚本

原有的 `./start.sh` 和 `./start.sh --test` 仍然可用，功能等价于：
```bash
./start.sh           # ≈ ./scripts/start-backend.sh
./start.sh --test    # ≈ ./scripts/test-mock.sh
```

---

## 🔌 在编程助手中配置与使用 (IDE & Copilot 配置指南)

本项目提供标准的 **OpenAI 兼容接口** 与 **FIM 代码补全接口**，可直接无缝接入 VS Code、Cursor、Continue、Cline、GitHub Copilot 生态等各类 AI 编程插件中。

### 📌 核心连接参数速查

| 参数项 | 推荐填入值 | 说明 |
| :--- | :--- | :--- |
| **API Provider / 类型** | `OpenAI` 或 `OpenAI Compatible` | 兼容标准 OpenAI 协议 |
| **API Base URL** | `http://localhost:1235/v1` | 服务 API 端点地址 |
| **API Key / Token** | `dummy` 或 `sk-aicodeservice-secret` | 未开启鉴权可填任意字符，开启后填对应密钥 |
| **Model Name (模型名)** | `qwen3.8-flash-next-oq4e-mtp-128k` | 或 `qwen3.8-27b-8bit-mtp` |
| **Context Window (上下文)** | `131072` (128K) 或 `262144` (256K) | 取决于所选模型 |
| **Max Output Tokens** | `2048` ~ `4096` | 单次输出最大 Token 数 |

---

### 🛠️ 常见编程插件配置示例

#### 1. Continue 插件配置 (VS Code / JetBrains / Cursor)
Continue 是目前最主流的开源代码助手插件，支持**对话聊天**与 **Tab 键行内 FIM 代码补全**。

打开 Continue 配置文件 `~/.continue/config.json`，添加如下配置：

```json
{
  "models": [
    {
      "title": "Qwen3.8 Flash Next oQ4e MTP 128K",
      "provider": "openai",
      "model": "qwen3.8-flash-next-oq4e-mtp-128k",
      "apiBase": "http://localhost:1235/v1",
      "apiKey": "dummy",
      "contextLength": 131072
    }
  ],
  "tabAutocompleteModel": {
    "title": "Qwen FIM Autocomplete",
    "provider": "openai",
    "model": "qwen3.8-flash-next-oq4e-mtp-128k",
    "apiBase": "http://localhost:1235/v1",
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
   - **Base URL**: `http://localhost:1235/v1`
   - **API Key**: `dummy`
4. 在模型列表下方点击 **Add model**，输入 `qwen3.8-flash-next-oq4e-mtp-128k` 并保存选中。

---

#### 3. GitHub Copilot Chat (VS Code `settings.json`)
在 VS Code 中若配合支持自定义端点的扩展或代理，可在 `settings.json` 中配置：
```json
{
  "github.copilot.advanced": {
    "debug.overrideEngine": "qwen3.8-flash-next-oq4e-mtp-128k",
    "debug.overrideProxyUrl": "http://localhost:1235/v1"
  }
}
```

---

#### 4. Cline / Roo Code 插件配置 (VS Code Autonomous Agent)
1. 在 VS Code 安装 **Cline** 或 **Roo Code** 插件。
2. 点击插件设置图标（齿轮），进入 API Provider 选择：
   - **API Provider**: 选择 `OpenAI Compatible`
   - **Base URL**: `http://localhost:1235/v1`
   - **API Key**: `dummy`
   - **Model ID**: `qwen3.8-flash-next-oq4e-mtp-128k`
   - **Context Window**: `131072`

---

#### 5. Aider / 命令行终端编程助手
使用终端 AI 结对编程工具 Aider 时，直接指定环境变量：

```bash
export OPENAI_API_BASE="http://localhost:1235/v1"
export OPENAI_API_KEY="dummy"

# 启动 aider 并指定综合加速模型
aider --model openai/qwen3.8-flash-next-oq4e-mtp-128k
```

---

## ⚡ 配置说明

本项目采用 **三层配置优先级**：系统环境变量 > `.env` 文件 > `config.yaml` > 代码默认值。

### 1. 环境变量配置 (`.env`)

所有可覆盖的配置项均可通过 `.env` 文件集中管理。首次使用请复制模板：

```bash
cp .env.example .env
```

`.env` 支持的配置项：

| 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | 服务绑定地址 |
| `PORT` | `1235` | 服务监听端口 |
| `API_KEY` | *(空)* | Bearer Token 鉴权密钥，设置后开启鉴权 |
| `DEFAULT_MODEL` | `qwen3.8-flash-next-oq4e-mtp-128k` | 默认使用的模型名称 |
| `USE_MOCK` | `false` | Mock 模式（不加载真实模型权重） |
| `ENABLE_CACHE` | `true` | 是否启用 LRU + TTL 响应缓存 |
| `RUN_REAL_MODEL_TESTS` | *(未设置)* | 设为 `1` 启用真实模型集成测试 |
| `WEB_PORT` | `8080` | Web 聊天界面网关端口 |
| `BACKEND_URL` | `http://localhost:1235` | Python 后端 API 地址 |

### 2. 服务与模型配置 (`config.yaml`)

```yaml
server:
  host: "0.0.0.0"
  port: 1235
  reload: false
  cors_origins: ["*"]
  # api_key: "sk-aicodeservice-secret"  # 取消注释开启 API Key 鉴权

# 深度性能优化与显存调优 (针对 Apple Silicon 统一内存调优)
performance:
  enable_cache: true               # 启用响应与 Prompt LRU 缓存
  cache_max_size: 1024              # 缓存最大条目数
  cache_ttl_seconds: 3600           # 缓存过期时间（秒）
  max_concurrency: 3                # 并发生成信号量
  metal_cache_limit_mb: 4096        # Apple Silicon MLX Metal 显存缓存上限 (4GB)
  clear_cache_after_generation: false
  stream_chunk_size: 1              # 流式响应 Token 聚合块大小
  kv_bits: 8                        # 🚀 KV Cache 量化位数 (8bit 节省 50% 显存)
  kv_group_size: 64                 # KV Cache 量化 Group 大小
  prefill_step_size: 2048           # 🚀 分块预填充 (平抑超大文本峰值显存)
  enable_prompt_cache: true         # 🚀 提示词/系统前缀 KV Cache 显存复用

# 默认模型：Qwen3.8 Flash Next oQ4e MTP 128K
default_model: "qwen3.8-flash-next-oq4e-mtp-128k"
use_mock: false

# 模型配置 (支持多架构: qwen4_exp / mlx_lm / mlx_vlm 等)
models:
  qwen3.8-flash-next-oq4e-mtp-128k:
    path: "jedisrt1/Qwen3.8-Flash-Next-oQ4e-MTP-128k"
    description: "Qwen3.8 Flash Next oQ4e MTP 128K (Qwen4Exp 架构，内置 MoE MTP，支持 128K 上下文)"
    engine_type: "qwen4_exp"
    context_length: 131072

  qwen3.8-27b-oq4e-fp16-mtp:
    path: "./models/Qwen3.8-27B-oQ4e-fp16-mtp"
    description: "Qwen3.8-27B 本地 MLX oQ 混合精度 4bit + MTP (支持 256K 上下文)"
    engine_type: "mlx_lm"
    context_length: 262144

  qwen3.8-27b-8bit-mtp:
    path: "./models/qwen3.8-27b-8bit-mtp"
    description: "Qwen3.8-27B MLX 8bit + MTP 完整综合加速版 (支持 256K 上下文)"
    engine_type: "mlx_lm"
    context_length: 262144

  qwen3.8-27b-8bit:
    path: "lmstudio-community/Qwen3.8-27B-MLX-8bit"
    description: "Qwen3.8-27B MLX 8bit 量化版本 (支持 256K 上下文)"
    engine_type: "mlx_lm"
    context_length: 262144

  qwen3.8-27b-mlx:
    path: "inferencerlabs/Qwen3.8-27B-MTP-MLX"
    description: "Qwen3.8-27B MLX MTP 辅助层 (238MB)"
    engine_type: "mlx_vlm"
    context_length: 262144
```
