# Qwen4Exp 跨请求前缀缓存：实现与真实模型验证

更新日期：2026-10-13。本文对应 `engine/qwen4_exp_engine.py`、`engine/prefix_cache.py` 和 `scripts/benchmark-prefix-cache.py`。

## 本次修复的问题

旧实现虽然设置了 `enable_prompt_cache: true`，但 MTP 请求每次仍然调用 `make_cache()`，`common_len` 固定为零；生成结束时又把持久化缓存设成 `None`。因此没有跨请求复用，也没有证据支持“首字 <5ms”的描述。

真实权重对比还发现旧 MTP 生成错误：对于“Write a Python function add(a, b). Only output code.”，标准解码输出完整函数，而旧 MTP 输出为 `` ```\n add,):    ``。原因包括验证输入缺少当前 token、预测与草稿位置错位，以及拒绝草稿后只尝试局部回滚，未完整恢复 PLE 状态。本次同时修复这些问题，避免将错误结果当作性能提升。

## 缓存保存什么

Qwen4Exp 是混合线性注意力、QSA 稀疏注意力、PLE 历史嵌入的模型，缓存并非只有 K/V 张量。

| 状态 | 本地模型实现 | 为什么必须保留 |
| --- | --- | --- |
| 线性注意力卷积与循环状态 | `ArraysCache[0]`、`ArraysCache[1]` | 保存累计历史，不能通过裁剪 K/V 回退 |
| PLE 卷积历史与 n-gram token 历史 | `ArraysCache[2]`、`ArraysCache[3]` | 决定下一段输入的局部历史嵌入 |
| QSA K/V、offset、索引 keys | `QSAKVCache` 完整对象 | 保证位置编码与稀疏检索位置一致 |

缓存条目是完整缓存对象的 `deepcopy`，包含这些状态与元数据。存储时复制一次，命中时再复制一次；生成请求不持有已保存快照的可变对象。真实 MLX 数组复制与取消后重复生成已经验证，生成或草稿验证不会修改持久化快照。

## 请求执行流程

1. 在引擎锁内，按 MLX 原生规则完整分词，包括 BOS / 特殊 token 处理。
2. 在 LRU 中查找**最长完全匹配的 token 前缀**。匹配到的快照长度必须小于当前请求长度，至少留一个 token 用来计算下一 token 的 logits。
3. 命中则复制完整状态；没有匹配则新建空缓存。不会将任意公共前缀长度套用到一个无法回退的循环状态上。
4. 使用 `prefill_step_size` 分块计算未缓存的前缀。在可保存的边界上写入完整状态快照。
5. 留下最后一个 prompt token，传给原生解码器；MTP 路径也从相同的前缀状态计算该 token 的 logits 和 hidden state。
6. 生成与草稿验证只修改请求副本。工具停止、客户端取消、异常或生成结束，都不会将生成状态写回前缀缓存。

默认保存的边界：

- **系统消息结束处**：包含工具声明与固定指令，支持工具轮次、用户消息分叉和同系统提示词的请求。
- **完整 prompt 的倒数一个 token**：支持相同请求重复执行。
- **缓存 token 上限处**：prompt 超过上限时保留一个有界前缀，剩余内容照常预填充。

系统边界必须经完整请求 token 前缀校验，不能只比较字符串。多轮 Prompt 可能会重新组织 assistant 思考标签，因而不保证整个上轮 prompt 都可复用；本次多轮实测复用了系统前缀的 764 tokens。

缓存按**每个模型引擎**共享，不是以 session ID 分组。共享的是相同 token 前缀的内部计算状态，模型输出仍由当前请求的完整输入与采样参数决定。服务重启或模型卸载后缓存消失。

## 容量与配置

```yaml
performance:
  enable_prompt_cache: true
  prompt_cache_max_entries: 4
  prompt_cache_max_tokens: 8192
  prefill_step_size: 2048
  enable_mtp: true
  mtp_depth: 3
  code_agent_mode: true
  default_temperature: 0.0
```

`prompt_cache_max_entries` 是每个 Qwen4Exp 引擎的 LRU 条目上限；`prompt_cache_max_tokens` 是**每条快照**允许覆盖的 token 上限。两者任一为零都不保存快照。它们不修改模型 128K 上下文限制，也不限制请求总输入长度。

单次短请求可能保存系统边界和 prompt 边界两个条目，所以“4 条”并不等于“4 个会话”。快照的状态大小依赖模型、token 数、K/V 扩容以及固定循环状态，不能用 token 数直接换算为固定显存。默认最多覆盖 4 × 8192 个快照 token，实际内存以 `prefix_cache.state_bytes` 和 Metal 内存指标为准；请求工作副本及 MTP 验证前快照还会占用额外内存。

Qwen4Exp 自定义缓存没有应用标准 `kv_bits` / `kv_group_size` 量化。本次分块预填充实际生效，但不能套用普通 27B 模型的“256K KV 显存减半”结论。

## MTP 正确性修复

仅在有效温度为 `0.0`、无 logits processor 时采用贪婪 MTP；随机采样和 repetition penalty 路径使用原生解码器。现有 Code Agent 规则仍会把显式低温（≤0.2）映射为贪婪采样。

设当前已输出但尚未进入主干缓存的 token 为 `c`，草稿为 `d0, d1, d2`。主干验证输入为：

```text
[c, d0, d1, d2]
 logits[0] -> 验证 d0
 logits[1] -> 验证 d1
 logits[2] -> 验证 d2
 logits[3] -> 全部接受时的额外主干 token
```

如果接受了 `A` 个草稿后出现分歧，就恢复验证前的**完整缓存副本**，重放 `[c] + drafts[:A]`，再继续输出主干预测。这个操作同时恢复线性注意力、PLE 和 QSA，避免依赖不完整的 `mtp_partial_rollback()`。

MTP 辅助头缓存按每个草稿周期重建，不跨请求或周期保存。接受率较低时，草稿计算、状态复制和重放会增加成本；因此深度 3 并非所有任务都更快。不会在已经输出部分文本后静默重新执行原生生成，以免重复输出和隐藏推理错误。

本次真实测试中深度 1、2、3 均与标准贪婪解码输出一致；benchmark 深度 3 的两个测量共接受 46 / 52 个草稿（88.5%）。这只描述测试输入，不保证所有输入都达到此接受率。浮点计算的不同分块/批量顺序仍可能使临界 logits 的选择不同，验证结果不能推广为所有输入的数学逐位等价证明。

## 真实模型测量

原始记录：[qwen4-prefix-cache-2026-10-13.json](benchmarks/qwen4-prefix-cache-2026-10-13.json)。

环境：128 GiB Apple Silicon 统一内存，Python 3.13.15，MLX 0.32.1，mlx-lm 0.31.3，transformers 5.14.1。模型为 `jedisct1/Qwen3.8-Flash-Next-oQ4e-MTP-128k`，从现有 ModelScope 本地缓存加载，不使用 Mock，不经过响应缓存。

加载耗时 **14.575 秒**。首次 kernel warmup 的首字耗时 **12.862 秒**，不纳入缓存加速比。基准在同一加载进程中运行，生成上限 32 tokens；代码输出可能被此上限截断，完整代码正确性由独立真实集成测试验证。

| 场景 | 输入 tokens | 复用 tokens | 无缓存 TTFT | 缓存命中 TTFT | 无缓存总耗时 | 缓存总耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 相同请求，标准解码 | 790 | 789 | 3.148 s | 0.267 s | 4.132 s | 1.252 s |
| 同系统提示词，用户消息分支 | 790 | 764 | 3.200 s | 0.394 s | 4.198 s | 1.386 s |
| 多轮 assistant 历史追加 | 843 | 764 | 3.262 s | 0.567 s | 4.281 s | 1.652 s |
| 缩短输入并更换系统消息 | 73 | 0 | 0.512 s | 0.547 s（未命中） | 1.048 s | 1.087 s |

标准解码相同请求 TTFT 约减少 **91.5%（11.8×）**；这是单次样本，非 p50/p95。首次构造快照的 TTFT 为 **4.631 秒**，高于无缓存的 **3.148 秒**，反映了分边界预填充、状态复制与首次计算的成本；缓存不是让所有请求都更快。

在已有前缀快照上，修复后的 MTP 两次测量分别为：

| MTP 测量 | TTFT | 总耗时 | 复用 tokens | 输出与标准解码 |
| --- | ---: | ---: | ---: | --- |
| 首次使用 MTP、已有前缀 | 0.127 s | 0.810 s | 789 | 相同 |
| 再次使用 MTP、已有前缀 | 0.124 s | 0.717 s | 789 | 相同 |

取消一次生成后再次请求，输出仍与无缓存基线相同，TTFT **0.256 秒**。LRU 始终没有超过 4 条，模型卸载后缓存条目和计数清零。

benchmark 峰值 Metal 活跃内存 **92155.72 MiB（约 90.0 GiB）**，末尾活跃内存 **90582.91 MiB**，最终快照状态约 **501.8 MiB**。这里只测了 73–843 token 请求，不能宣称 128K/256K 或多模型并发不 OOM；本模型权重本身已占用约 88 GiB。

## 如何复现

从项目根目录使用兼容的专用环境：

```bash
conda run -n llmqwen python -u scripts/benchmark-prefix-cache.py \
  --output docs/benchmarks/qwen4-prefix-cache-local.json

RUN_REAL_MODEL_TESTS=1 conda run -n llmqwen python -m pytest \
  tests/test_real_model_integration.py -q --capture=fd

conda run -n llmqwen python -m pytest tests \
  --ignore=tests/test_real_model_integration.py -q --capture=fd
```

脚本会显式关闭 Mock，要求真实 Qwen4Exp 引擎加载成功；任一输出对比或前缀复用校验失败时退出失败，并保留已经完成的测量与错误记录。用 `--filler-lines` 和 `--max-tokens` 调整输入与输出规模，注意 kernel warmup 不应算进命中效果。

本机默认 `python` 为 base Python 3.10，其 MLX 扩展与动态库符号不兼容，直接导入 `mlx.core` 会失败。专用 `llmqwen` 环境测试通过；本次没有修改全局环境。

## 回归验证范围

- 74 项不加载权重的测试：覆盖最长精确前缀、分支/缩短不误用、存储/恢复双向隔离、LRU/容量/清空，以及队列背压、取消、停止生成。
- 原有 38 项真实集成测试通过：覆盖真实加载、对话、流式 SSE、FIM、代码工具、响应缓存、长输入、可执行代码等。
- 新增 4 项真实集成测试通过：MTP 深度 1/2/3 与标准贪婪输出一致；真实路由生成 `add_numbers(a=19, b=23)`，消费模拟工具返回值 `42` 后正确回答并复用系统前缀。没有执行模型生成的任意工具命令。
- 实际 benchmark：重复、分支、多轮、改系统/缩短、取消恢复、MTP、卸载清理的输出与状态检查全部通过。

## 监控与排查

`GET /metrics` 的 `performance.engines[模型名称]` 包含：

| 字段 | 含义 |
| --- | --- |
| `prefix_cache_hits` | 与 `prefix_cache.hits` 相同，兼容已有监控 |
| `prefix_cache.entries` / `max_entries` | 当前快照数和条目上限 |
| `prefix_cache.max_tokens_per_entry` | 单快照 token 上限 |
| `prefix_cache.hits` / `misses` | 启用前缀缓存时的查找计数 |
| `prefix_cache.reused_tokens` | 累计复用 token 数 |
| `prefix_cache.retained_tokens` | 所有快照 token 数之和，重叠前缀会重复计数 |
| `prefix_cache.state_bytes` | 保存快照缓存对象报告的状态字节数，不含请求工作副本 |
| `last_reused_tokens` | 上次请求跳过的 token 数 |
| `last_prefill_tokens` | 上次手动预填充的新增 token 数，不含最后一个解码输入 token |
| `last_prefill_seconds` | 前缀恢复、分块预填充与保存快照耗时，不含最终 token 和生成 |
| `mtp_drafted_tokens` / `mtp_accepted_tokens` | 累计提出和接受的草稿 token 数 |

响应头 `X-Cache` 表示**响应缓存**命中，与上述前缀缓存不同；工具请求绕过响应缓存，仍可以复用前缀状态。

命中率低时先检查系统提示词与工具声明是否稳定、是否更换模型、LRU 是否淘汰了相关边界、prompt token 是否完全相同。若首字仍慢，应分别测量分词、剩余 token 预填充、最终模型前向和 HTTP 排队；不要把 `last_prefill_seconds` 当作端到端 TTFT。
