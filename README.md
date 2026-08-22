# AI Code Service

基于 Apple MLX / Qwen 编程大语言模型的高性能 AI Code 服务，提供完整的 OpenAI 兼容 API、FIM (Fill-In-The-Middle) 代码自动补全，以及面向 IDE 插件（如 VS Code, Cursor, Continue）的专有快捷编程助手接口。原生支持 **256K (262,144 Tokens)** 超长上下文、**8-bit KV Cache 显存量化** 与 **MTP 投机采样多 Token 预测加速**。

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
├── config.py                  # Pydantic 动态配置加载器（默认 256K 上下文 & 性能配置）
├── config.yaml                # 服务与模型配置文件（256K 上下文 & 8bit+MTP 调优参数）
├── start.sh                   # 启动与测试自动化脚本（内置 Conda 环境锁定与端口自愈）
├── download.py                # ModelScope 模型下载与校验工具
├── convert_to_mlx.py          # 模型 MLX 转换与 4bit/8bit 量化脚本
├── merge_mtp.py               # MTP 投机采样头与 8bit 基础模型一键整合工具
├── client_demo.py             # 客户端 API 功能演示与打字机流式验证脚本
├── environment.yml            # Conda 环境定义文件 (Python 3.13+)
├── requirements.txt           # 项目依赖清单
├── pytest.ini                 # Pytest 测试配置文件
├── .vscode/                   # VS Code / Cursor IDE 解释器配置
├── engine/                    # 模型推理引擎模块
│   ├── base.py                # 推理引擎抽象基类（支持异步非阻塞及流式生成）
│   ├── cache.py               # 线程安全的 LRU + TTL 响应缓存器
│   ├── mlx_engine.py          # Apple MLX 推理引擎（KV 量化、分块 Prefill、TPS 统计）
│   ├── mock_engine.py         # Mock 仿真推理引擎
│   └── manager.py             # 模型统一加载、Prompt 构建器与性能指标聚合
├── benchmark.py               # 🚀 多模型性能基准压测与自动化对比套件
├── docs/                      # 架构设计与运维操作文档
│   ├── PERFORMANCE_DESIGN.md  # 5大核心维度深度性能优化与系统设计文档
│   ├── BENCHMARK_REPORT.md    # 📊 多模型性能基准压测报告与对比指南
│   └── OPS_MANUAL.md          # 运维部署、全链路监控与故障排查手册
├── schemas/                   # Pydantic 数据契约对象
│   └── openai.py              # OpenAI 兼容接口及代码工具 Schema (支持 Stream)
├── routers/                   # FastAPI 路由模块
│   ├── chat.py                # Chat Completions 路由（异步 + 缓存）
│   ├── completions.py         # FIM Autocomplete 路由（异步 + 缓存）
│   ├── code.py                # 编程助手专用工具路由 (全接口支持 Stream)
│   ├── models.py              # 模型管理路由 (输出 256K 元数据)
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

## 📚 详细设计与运维手册

- 📖 [深度性能优化与系统架构设计文档 (`docs/PERFORMANCE_DESIGN.md`)](file:///Users/charles/Documents/AI/Python/llmqwen/docs/PERFORMANCE_DESIGN.md)：详细推导 256K 下 8-bit KV Cache 显存量化、分块预填充、Metal 提示词缓存与 MTP 投机采样的数学原理与加速机制。
- 🛠️ [运维部署、监控与故障排查手册 (`docs/OPS_MANUAL.md`)](file:///Users/charles/Documents/AI/Python/llmqwen/docs/OPS_MANUAL.md)：涵盖 Conda 环境部署、端口自愈、全链路 `/metrics` 监控、launchd 后台常驻守护与常见故障排查。

---

## 🤖 模型生态与本地支持列表

本项目已在本地配置并完美支持以下模型规格（统一支持 **256K (262,144 Tokens)** 超长上下文）：

| 模型代号 (Model ID) | 权重类型与架构 | 显存占用 | 推荐场景 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **`qwen3.8-27b-8bit-mtp`**<br>*(默认首选)* | 8-bit 量化 + MTP 投机采样一体化 | **~28.2 GB** | **日常编程 / IDE 实时补全** | 结合 8bit 内存优势与 238MB MTP 前瞻加速，输出速率达 **45~65+ tok/s**。 |
| **`qwen3.8-27b-8bit`** | 8-bit 量化独立版 (`lmstudio-community`) | **~28 GB** | 高速独立推理 | 精度无损，显存占用低，极度稳定。 |
| **`qwen3.8-27b-mlx`** | MTP 投机采样辅助层 (`inferencerlabs`) | **~238 MB** | 辅助投机层 | 专用于与 27B 基座配合进行多 Token 前瞻预测。 |
| **`qwen3.8-27b`** | 原始 16-bit 全精度标准版 (`Qwen`) | **~54 GB** | 原始基准比对 | 官方未量化原始权重，可用于基准精度评估。 |

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

### 1. 运行测试套件
`start.sh` 会自动检测并锁定 `llmqwen` Conda 环境：
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
服务默认运行在 `http://localhost:1235`，交互式 API 文档可通过 `http://localhost:1235/docs` 访问。

### 3. 运行打字机流式验证客户端
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
| **API Base URL** | `http://localhost:1235/v1` | 服务 API 端点地址 |
| **API Key / Token** | `dummy` 或 `sk-aicodeservice-secret` | 未开启鉴权可填任意字符，开启后填对应密钥 |
| **Model Name (模型名)** | `qwen3.8-27b-8bit-mtp` | 或 `qwen3.8-27b-8bit` |
| **Context Window (上下文)** | `262144` (256K) | 模型最大上下文 Token 数 |
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
      "title": "Qwen 27B (MLX 8bit + MTP 256K)",
      "provider": "openai",
      "model": "qwen3.8-27b-8bit-mtp",
      "apiBase": "http://localhost:1235/v1",
      "apiKey": "dummy",
      "contextLength": 262144
    }
  ],
  "tabAutocompleteModel": {
    "title": "Qwen FIM Autocomplete",
    "provider": "openai",
    "model": "qwen3.8-27b-8bit-mtp",
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
4. 在模型列表下方点击 **Add model**，输入 `qwen3.8-27b-8bit-mtp` 并保存选中。

---

#### 3. GitHub Copilot Chat (VS Code `settings.json`)
在 VS Code 中若配合支持自定义端点的扩展或代理，可在 `settings.json` 中配置：
```json
{
  "github.copilot.advanced": {
    "debug.overrideEngine": "qwen3.8-27b-8bit-mtp",
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
   - **Model ID**: `qwen3.8-27b-8bit-mtp`
   - **Context Window**: `262144`

---

#### 5. Aider / 命令行终端编程助手
使用终端 AI 结对编程工具 Aider 时，直接指定环境变量：

```bash
export OPENAI_API_BASE="http://localhost:1235/v1"
export OPENAI_API_KEY="dummy"

# 启动 aider 并指定综合加速模型
aider --model openai/qwen3.8-27b-8bit-mtp
```

---

## ⚡ 性能与配置文件说明 (`config.yaml`)

```yaml
server:
  host: "0.0.0.0"
  port: 1235
  reload: false
  cors_origins: ["*"]
  # api_key: "sk-aicodeservice-secret"  # 取消注释开启 API Key 鉴权

# 深度性能优化与显存调优 (针对 128G 统一内存 Mac 深度调优)
performance:
  enable_cache: true               # 启用响应与 Prompt LRU 缓存（高频重复请求返回 < 1ms）
  cache_max_size: 1024              # 缓存最大条目数
  cache_ttl_seconds: 3600           # 缓存过期时间（秒，默认 1 小时）
  max_concurrency: 3                # 并发生成信号量 (256K 上下文推荐 2~3 并发)
  metal_cache_limit_mb: 4096        # Apple Silicon MLX Metal 显存缓存上限 (4GB)
  clear_cache_after_generation: false # 单次大文本生成后是否主动清理显存碎片
  stream_chunk_size: 1              # 流式响应 Token 聚合块大小
  kv_bits: 8                        # 🚀 KV Cache 量化位数 (8bit 节省 50% 显存，长文本提速 30%~50%)
  kv_group_size: 64                 # KV Cache 量化 Group 大小
  prefill_step_size: 2048           # 🚀 分块预填充 (平抑超大代码上下文峰值显存)
  enable_prompt_cache: true         # 🚀 提示词/系统前缀 KV Cache 显存复用 (<5ms 首字秒出)

# 默认模型：8bit + MTP 完整综合加速版
default_model: "qwen3.8-27b-8bit-mtp"
use_mock: false  # 设置为 true 可在无 GPU 环境下全仿真运行

# 256K 超长上下文模型配置 (262,144 Tokens)
models:
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

  qwen3.8-27b:
    path: "Qwen/Qwen3.8-27B"
    description: "Qwen3.8-27B 原始标准版本 (16bit 全精度，54GB 权重)"
    engine_type: "auto"
    context_length: 262144
```
