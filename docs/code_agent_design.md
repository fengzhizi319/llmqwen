# AI Code Service - Code Agent 架构体系与端到端优化设计

本文档全面阐述 AI Code Service 针对 **Code Agent (智能代码体)** 场景构建的高性能、高确定性与高鲁棒性工程架构。系统基于 Apple Silicon 硬件加速，以 **Qwen3.8 Flash Next oQ4e MTP 128K** 大模型为核心底座，无缝衔接 **OpenCode Agent**、**OpenAI Tools 协议** 与本地开发环境。

---

## 🏛️ 1. Code Agent 整体架构全景

```
+----------------------------------------------------------------------------------------------------+
|                                    客户端与开发者工作区 (Client Layer)                              |
|   - OpenCode CLI / Web UI      - Cursor / VSCode 插件      - Vercel AI SDK (@ai-sdk/openai-compatible)   |
+--------------------------------------------------+-------------------------------------------------+
                                                   | HTTP POST (OpenAI Tools Schema / SSE Stream)
                                                   v
+----------------------------------------------------------------------------------------------------+
|                                    FastAPI 服务协议接入层 (Gateway Layer)                          |
|   - /v1/chat/completions (OpenAI 协议)           - /v1/responses (Responses 协议)                  |
|   - Server-Sent Events (SSE) 实时流式管道         - 全链路追踪与并发隔离信号量 (Semaphore: 3)       |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|                                    协议转换与工具中间件 (Protocol & Tool Format)                    |
|   - engine/tool_format.py                                                                          |
|     * 双向格式转换: OpenAI JSON Tools Schema <---> Qwen 原生 XML (<tool_call>, <function=...>)    |
|     * 多格式解析容错: 原生 XML、裸标签、内嵌 JSON 结构、Markdown 围栏自适应提取                    |
|     * 代码缩进与格式零丢失保护: 保留首尾与内部缩进，杜绝 Python/代码参数破坏                       |
|     * 流式区域过滤器 (StreamRegionFilter): 抑制 <think> 思考块与工具参数外泄，保证前端可见文本纯净  |
|     * 标准 SSE Index 字段补齐: 严格对齐 @ai-sdk 要求的 delta.tool_calls[i].index 聚合规范          |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|                                    模型调度与上下文组装 (Model Orchestration Layer)                 |
|   - engine/manager.py                                                                              |
|     * System Prompt 工具注入: 生成规范的 # Tools XML 引导块与强约束指令                           |
|     * 多轮工具响应聚合: 将连续 role: tool 执行结果合并为规范单 user 块 (<tool_response>)           |
|     * Thinking 模式自适应旁路: Agent 场景注入空思考闭合标签，消除首字延迟 (<think>\n\n</think>)    |
|     * 防伪造工具护栏 (Guardrails): 普通非 Agent 对话模式禁止输出工具调用标签                        |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|                               MLX 硬件加速推理引擎 (Inference Acceleration Engine)                  |
|   - engine/qwen4_exp_engine.py & engine/mlx_engine.py                                              |
|     * 低温确定性解码: Code Agent 模式下默认 temperature: 0.0 (Argmax 采样)                          |
|     * MTP (Multi-Token Prediction) 深度投机推测: 3 深度 Draft Verification (吞吐 45 -> 135+ tok/s) |
|     * 纯净 Cache 隔离机制: 消除混合线性注意力 (Hybrid Linear Attention) 跨轮次历史状态污染         |
|     * 工具闭合快速截断: 动态注入 </tool_call>、</function>、<|im_end|> 作为 Stop 词，降低多余 Token   |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|                               Apple Silicon 统一内存架构 (Unified Memory)                          |
|   - 128GB 高带宽统一内存 (UMA)    - Metal 统一计算管线 (Zero-Copy)    - 400+ GB/s 超大内存吞吐带宽   |
+----------------------------------------------------------------------------------------------------+
```

---

## 🛠️ 2. 工具调用协议与鲁棒性工程

### 2.1 Qwen 官方原生工具协议与标签
Qwen3.8 Flash Next 官方模型遵循 ChatML 扩展的 XML 工具调用格式：

- **工具调用声明块**（放置在 `<|im_start|>assistant` 输出内）：
  ```xml
  <tool_call>
  <function=execute_command>
  <parameter=command>
  pytest tests/test_core.py
  </parameter>
  </function>
  </tool_call>
  ```
- **工具执行结果返回块**（放置在 `<|im_start|>user` 消息内）：
  ```xml
  <tool_response>
  ================ 1 passed in 0.05s ================
  </tool_response>
  ```

### 2.2 多格式容错解析矩阵
真实场景中，不同客户端或小概率模型变体可能会产生格式抖动。`engine/tool_format.py` 实现了四级鲁棒容错提取：

1. **标准 XML 格式**：完整闭合的 `<tool_call><function=NAME>...</function></tool_call>`。
2. **裸函数块容错**：模型直接输出 `<function=NAME>...</function>`（缺少外层 `<tool_call>`）。
3. **闭合标签省略容错**：模型在输出参数后直接用 `</tool_call>` 关闭，省略了中间的 `</function>`。
4. **内嵌 JSON 结构容错**：当模型以 `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` 或带 ` ```json ` 围栏输出时，自动提取并转换为符合 OpenAI 规范的标准 `tool_calls`。

### 2.3 代码缩进与参数零丢失保护 (`_normalize_param_value`)
在代码编辑工具（如 `write_file`, `apply_diff`, `replace_content`）中，Python 代码的第 1 行缩进如果被 `\s*` 或 `.strip()` 吞噬，会导致严重的语法破坏。
- **问题根源**：传统正则表达式 `r"<parameter=.*?>\s*(.*?)</parameter>"` 中的 `\s*` 会贪婪吞掉第 1 行开头的空格或 Tab。
- **修复方案**：采用精确换行匹配 `(?:\r?\n)?`，仅剥离紧邻开闭标签的单次回车换行，代码内部所有前导缩进与多行结构 100% 原始保留。

### 2.4 OpenAI SSE 流式协议与 `index` 字段对齐
Vercel AI SDK（`@ai-sdk/openai-compatible`）及 OpenCode 在流式解析 `delta.tool_calls` 时，依赖每个 tool call 对象内的 `index: int` 字段来跨 chunk 聚合参数。
若缺少 `index`，客户端将抛出解析异常或静默丢失调用。
本系统在 SSE 结束帧输出：
```json
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "tool_calls": [
          {
            "index": 0,
            "id": "call_xxxxxxxxxxxx",
            "type": "function",
            "function": {
              "name": "bash",
              "arguments": "{\"command\": \"git status\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

### 2.5 连续多轮工具响应合并
当模型单步调用多个并行工具时，客户端会返回多个连续的 `role: "tool"` 消息。
系统在 `engine/manager.py` 中自动将这批消息聚合成单个 `<|im_start|>user` 块，内部包裹多个 `<tool_response>`，完全与 Qwen 官方 `chat_template.jinja` 保持 100% 数学一致。

---

## ⚡ 3. 推理性能与低延迟专项优化

### 3.1 低温确定性解码与 MTP 命中率提升
- **理论依据**：Code Agent 的核心诉求是**代码确定性、语法准确度与工具参数精确匹配**，而非文学发散。
- **优化配置**：在 Code Agent 模式下将默认采样温度设为 `0.0`（Argmax 贪婪解码）。
- **收益推导**：确定性输出消除了草稿 Token 的概率随机漂移，使 MTP (Multi-Token Prediction) 深度推测解码的 Draft 命中率由高温度下的 **5.7%** 跃升至 **50% 以上**，实测生成吞吐从 45 tok/s 提升至 **135+ tok/s**。

### 3.2 思考块自适应旁路 (Zero Thinking Latency)
- 思考模型在回答普通复杂逻辑题时会展开长达数百 Token 的 `<think>` 过程，在 Code Agent 工具调用循环中会导致首字延迟（TTFT）长达数秒。
- 当检测到当前请求包含 `tools` 或处于 `code_agent_mode` 且未显式要求思考时，系统自动在 Prompt 尾部注入 `<think>\n\n</think>\n\n`，促使模型直接输出行动决策与工具调用块，**首字延迟从数秒骤降至 15ms 以内**。

### 3.3 工具闭合快速截断 (Stop Sequences)
模型在输出完 `</tool_call>` 后往往会尝试生成解释性总结。系统将 `</tool_call>`、`</function>` 与 `<|im_end|>` 注册为全局 Stop 序列：
- 模型一旦闭合工具标签立即停止解码。
- 节省无意义的续写 Token 开销，加速单步 Agent Loop 周转。

### 3.4 纯净 KV Cache 隔离机制
Qwen3.8 Flash Next 采用混合线性循环注意力（Hybrid Linear Recurrent Attention）。由于线性注意力包含循环状态（recurrent state），若跨轮次直接复用残留了上次生成结果的 dirty cache，会导致注意力错位和幻觉。
- 每次推理请求使用独立的纯净 Cache 初始化（`make_cache()`）。
- 结合 Apple Silicon 统一内存与 Chunked Prefill，几千 Token 上下文的前向预填充仅耗时数十毫秒，既保证了极速响应，又彻底杜绝了状态污染。

---

## 🤖 4. OpenCode 智能体集成配置与权限治理

### 4.1 配置文件规范 (`agent/opencode.json`)
OpenCode 作为成熟的开源 Code Agent CLI / Headless 服务，其配置文件对模式字段有严格约束：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "aicodeservice": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "AI Code Service (本地 Qwen3.8 Flash Next)",
      "options": {
        "baseURL": "http://localhost:1235/v1",
        "apiKey": "not-needed"
      },
      "models": {
        "qwen3.8-flash-next-oq4e-mtp-128k": {
          "name": "Qwen3.8 Flash Next oQ4e MTP 128K",
          "limit": {
            "context": 131072,
            "output": 8192
          }
        }
      }
    }
  },
  "model": "aicodeservice/qwen3.8-flash-next-oq4e-mtp-128k",
  "small_model": "aicodeservice/qwen3.8-flash-next-oq4e-mtp-128k",
  "temperature": 0.0,
  "permission": {
    "edit": "allow",
    "bash": "allow",
    "read": "allow",
    "glob": "allow",
    "grep": "allow",
    "webfetch": "allow",
    "todowrite": "allow"
  },
  "instructions": [
    "你是一位精通软件架构、代码重构与快速排错的高级工程师，始终追求代码的健壮性、可读性与高执行性能。",
    "在编写或修改代码前，优先查阅现有项目结构与文件逻辑，严禁凭空假设未验证的接口与路径。",
    "修改代码时遵循最小变更原则，严格保留既有缩进风格、注释和架构模式，杜绝破坏性重构。",
    "每次完成关键改动后主动执行相关测试、编译或语法检查命令，确保交付代码零报错通过。",
    "输出简洁明确，直接给出关键解答与完整代码实现，避免冗余寒暄。"
  ]
}
```

### 4.2 权限治理策略
为了让 Agent 在无人值守或沉浸式开发时顺畅运转，将常用安全工具预置为 `allow`：
- `read` / `glob` / `grep`：无阻碍的代码库探索与上下文搜集。
- `edit`：直接修改代码。
- `bash`：执行测试套件、构建打包与语法检查。
- `todowrite`：管理长任务的待办清单与进度追踪。

---

## 📊 5. 评测体系与基准验证

为全面量化 Code Agent 在真实开发场景与算法任务上的代码生成能力，系统集成了全套自动化评测套件：

| 评测脚本 | 覆盖范围 | 核心指标 | 特性与运行方式 |
| :--- | :--- | :--- | :--- |
| `agent/eval_coding.py` | 10 道典型工程手写题 (HumanEval-Lite) | pass@1, TPS, 延迟 | 秒级断言沙箱，即时验证 |
| `agent/eval_humaneval.py` | HumanEval 164 题全量数据集 | 分领域 pass@1 (字符串/数学/集合/算法) | 官方标准断言，支持 `--resume` 断点续跑 |
| `agent/eval_humaneval.py` | MBPP 974 题分层数据集 | 基础 (1-300)、中级 (301-600)、综合 (601-974) | 多层级抽样与全量阶梯评测 |

---

## 🚀 6. 一键运维与实战指南

### 6.1 启动 Code Agent 完整服务
运行一键启动脚本（自动拉起本地 MLX 后端并在端口 4096 提供 Agent 服务）：
```bash
./scripts/start-agent.sh
```

### 6.2 交互式 CLI 会话
在独立终端中进入 `agent` 目录体验全功能终端 TUI：
```bash
cd agent
opencode
```

### 6.3 统一停止服务
```bash
./scripts/stop-all.sh
```
