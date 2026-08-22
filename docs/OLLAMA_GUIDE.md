# 将本地 Qwen 模型导入 Ollama 使用指南

Ollama 底层基于 **`llama.cpp`** 运行时，核心使用 **GGUF 格式**。如果您希望在 Ollama 软件（或配合 Open WebUI、Page Assist 等）中使用本地下载的 Qwen 权重，请按照以下标准步骤进行转换与导入。

---

## 🧭 整体流程概览

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
```

下载 `llama.cpp` 的轻量转换脚本：
```bash
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

## 🔌 在 IDE 或 WebUI 中使用 Ollama

- **Ollama API 地址**：`http://localhost:11434/v1`
- **模型名称**：`qwen3.8-27b-local`
- **兼容性**：支持 VS Code (Continue 插件选择 Provider 为 `ollama`)、Cursor、Open WebUI 等。

---

## ⚖️ MLX (当前 AI Code Service) 与 Ollama 的对比

| 对比维度 | Apple MLX 原生服务 (本项目当前架构) | Ollama (llama.cpp GGUF 架构) |
| :--- | :--- | :--- |
| **底层核心** | Apple 官方原生 **MLX 框架** | **llama.cpp (C++ / Metal)** |
| **显存机制** | **统一内存零拷贝** (Zero-Copy UMA) | 统一内存 Metal 缓冲区映射 |
| **MTP 投机采样**| **原生支持 (45~65+ tok/s 极速)** | 需要配置双模型 Draft 运行 |
| **KV Cache 显存量化**| **原生 8-bit KV Cache 量化 (节约 50% 显存)** | 支持 `--flash-attn` 与量化 KV |
| **代码专有接口** | **自带 /v1/code/* 专有快捷重构/审查工具** | 仅标准 Chat / Generate 接口 |
| **推荐适用场景** | **专业编程助手、IDE 深度结对编程、长代码库分析** | **跨平台部署、通用对话、WebUI 图形界面** |
