# 将本地 Qwen 模型导入 Ollama 使用指南与生态原理解析

---

## ❓ 核心问题：Ollama 支持 MLX 格式吗？

**不支持。** Ollama **无法直接加载或运行 MLX 格式**的模型。

### 为什么 Ollama 不支持 MLX？

| 维度 | Ollama 生态 | Apple MLX 生态 (本项目) |
| :--- | :--- | :--- |
| **底层推理引擎** | 基于 **`llama.cpp`** (C/C++ 跨平台框架) | 基于 Apple 官方 **`MLX`** (针对 Apple Silicon 原生定制) |
| **依赖的模型格式** | 强制使用 **GGUF 格式** (`.gguf` 单文件) | 使用 **Safetensors + JSON** (`model.safetensors`, `config.json`) |
| **硬件优化深度** | 跨平台 Metal/CUDA 缓冲区抽象映射 | **Apple 统一内存零拷贝 (Zero-Copy UMA)**，Metal 内核深度融合 |
| **投机加速支持** | 基础 Draft 模型机制 | **原生 MTP (Multi-Token Prediction) 投机前瞻加速** |

> **💡 关键结论**：  
> - 如果你想在 **Ollama** 中运行本地模型，**必须先将 Safetensors 模型转换为 GGUF 格式**。
> - 如果你的目的是在 **WebUI（如 Open WebUI、Chatbox）、VS Code (Continue)、Cursor** 等软件中使用该模型，**完全不需要经过 Ollama 转换**！只需直接将客户端的 API Base URL 指向本项目的 `http://localhost:8000/v1`，即可同时享受 **MLX 极致性能 (55+ tok/s)** 与 **OpenAI 兼容生态**。

---

## 🧭 将本地 Safetensors 转换为 GGUF 并导入 Ollama 的流程

```
本地已下载 Safetensors 权重 (~/.cache/modelscope/)
                │
                ▼ (使用 llama.cpp convert_hf_to_gguf.py)
   转换为 GGUF 格式 (例如 qwen3.8-27b-q8_0.gguf)
                │
                ▼ (编写 Modelfile 配置 256K 上下文与 ChatML 模板)
     ollama create qwen3.8-27b-local -f Modelfile
                │
                ▼ (在 Ollama 客户端或 WebUI 中使用)
        ollama run qwen3.8-27b-local
```

---

## 步骤 1：安装 `llama.cpp` 转换工具

在当前 `llmqwen` 环境中安装转换依赖：

```bash
# 激活环境
conda activate llmqwen

# 安装 GGUF 转换依赖
pip install gguf numpy sentencepiece protobuf

# 下载 llama.cpp 转换工具
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp
```

---

## 步骤 2：将本地模型转换为 GGUF 格式

将本地已有的 54GB 原始 Safetensors 快照转换为 8-bit 或 4-bit 量化的 GGUF 文件：

#### 方案 A：转换为 8-bit 量化 (推荐：精度无损)
```bash
python /tmp/llama.cpp/convert_hf_to_gguf.py \
  ~/.cache/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master \
  --outfile ./models/qwen3.8-27b-q8_0.gguf \
  --outtype q8_0
```

#### 方案 B：转换为 4-bit 量化 (极致轻量，体积约 16GB)
```bash
python /tmp/llama.cpp/convert_hf_to_gguf.py \
  ~/.cache/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master \
  --outfile ./models/qwen3.8-27b-q4_k_m.gguf \
  --outtype q4_k_m
```

---

## 步骤 3：配置 `Modelfile`

本项目根目录下已为您预置了标准的 [`Modelfile`](file:///Users/charles/Documents/AI/Python/llmqwen/Modelfile)：

```dockerfile
# 指向刚才生成的 GGUF 文件路径
FROM ./models/qwen3.8-27b-q8_0.gguf

# 256K 上下文配置 (262,144 Tokens)
PARAMETER num_ctx 262144
PARAMETER num_predict 4096
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

# Stop 标记词
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# Qwen ChatML 对话模板
TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}
{{- range .Messages }}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{- else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{- end }}
{{- end }}
<|im_start|>assistant
"""

SYSTEM """你是一个专业的 AI 编程助手。你的职责是帮助开发者编写、调试、重构和优化代码，解释技术概念并提供最佳实践。"""
```

---

## 步骤 4：在 Ollama 中构建与运行

在终端中执行导入：

```bash
# 1. 在 Ollama 中注册新模型
ollama create qwen3.8-27b-local -f Modelfile

# 2. 验证模型已成功注册
ollama list

# 3. 启动交互式终端对话
ollama run qwen3.8-27b-local
```

---

## 🔌 在前端工具（Chatbox / Open WebUI / Continue）中直连 MLX 的最佳方案

如果您使用 Ollama 只是为了对接第三方客户端或前端界面，**完全不需要进行 GGUF 格式转换**：

本项目提供的 AI Code Service 服务已经**原生提供了 100% 兼容 OpenAI 的 API 接口**：

- **API Base URL**: `http://localhost:8000/v1`
- **API Key**: 填 `dummy`（或任意字符串）
- **Model Name**: `qwen3.8-27b-8bit-mtp`

在此模式下，您可以直接在 **Chatbox**、**Open WebUI**、**NextChat**、**VS Code (Continue / Cline)**、**Cursor** 中获得比 Ollama 快 **30% ~ 50%** 的 MLX 原生硬件加速生成体验！
