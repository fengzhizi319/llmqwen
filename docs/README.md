# 文档导航

默认 Qwen4Exp 模型的最新实现与实测以 2026-10-13 专项记录为准；普通 27B 模型的历史内存/吞吐表不适用于它。

| 文档 | 内容 |
| --- | --- |
| [Qwen4Exp 前缀缓存与真实验证](QWEN4_PREFIX_CACHE.md) | 完整状态快照、MTP 验证修复、容量配置、延迟与内存实测、复现命令、限制 |
| [推理引擎与缓存分层](llm_design.md) | 路由、调度、三类缓存、引擎锁、加载与释放 |
| [Code Agent 设计](code_agent_design.md) | 工具协议、多轮 Prompt、思考旁路、停止与取消 |
| [性能设计](PERFORMANCE_DESIGN.md) | KV 量化适用范围、分块预填充、前缀缓存、异步生成生命周期 |
| [Benchmark 报告](BENCHMARK_REPORT.md) | 当前 Qwen4Exp 测量与历史 27B 数据分开记录 |
| [运维手册](OPS_MANUAL.md) | 环境、模型资产、监控、前缀容量调整与故障定位 |
| [Ollama 指南](OLLAMA_GUIDE.md) | 独立的 Ollama 部署路径 |

原始测量数据：[qwen4-prefix-cache-2026-10-13.json](benchmarks/qwen4-prefix-cache-2026-10-13.json)。

- [编程能力评测报告](CODING_EVAL_REPORT.md)：HumanEval 全量与用户停止后的 MBPP 阶段结果、失败分析和实际能力边界。
