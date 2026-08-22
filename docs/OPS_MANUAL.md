# AI Code Service - 运维部署、监控与故障排查手册 (Ops Manual)

本文档面向系统运维人员与开发者，详细介绍 AI Code Service 在生产与本地开发环境中的部署标准、日常运维操作、全链路监控指标体系以及常见故障排查方法。

---

## 📋 目录
1. [系统运行环境与依赖配置](#1-系统运行环境与依赖配置)
2. [服务生命周期管理与启停](#2-服务生命周期管理与启停)
3. [模型资产与生命周期管理](#3-模型资产与生命周期管理)
4. [全链路监控指标与健康探针](#4-全链路监控指标与健康探针)
5. [常见故障排查指南 (Troubleshooting)](#5-常见故障排查指南-troubleshooting)
6. [macOS 后台守护进程配置 (launchd)](#6-macos-后台守护进程配置-launchd)

---

## 1. 系统运行环境与依赖配置

### 1.1 推荐硬件配置
- **操作系统**：macOS Sonoma 14.0+ 或 macOS Sequoia 15.0+
- **硬件芯片**：Apple Silicon（M1/M2/M3/M4 系列，推荐 Pro/Max/Ultra 芯片）
- **统一内存**：推荐 **128 GB 统一内存**（运行 27B 8-bit 模型占用约 28GB 权重 + 32GB 256K KV Cache，预留 68GB 给操作系统与 IDE）
- **磁盘空间**：至少预留 60GB 高速 NVMe 固态存储

### 1.2 Python 与 Conda 环境隔离
本项目强制要求在独立的 Conda 环境 **`llmqwen`** 下运行（Python **3.13+**）：

```bash
# 1. 创建专用 Conda 环境
conda create -y -n llmqwen python=3.13 -c conda-forge

# 2. 激活环境并安装依赖
conda activate llmqwen
pip install -r requirements.txt
```

---

## 2. 服务生命周期管理与启停

### 2.1 推荐启动方式 (`start.sh`)
项目根目录下的 [`start.sh`](file:///Users/charles/Documents/AI/Python/llmqwen/start.sh) 已经内置了环境自愈、端口占用清理与版本锁定机制：

```bash
# 启动 API 服务（默认监听 0.0.0.0:8000）
./start.sh

# 运行自动化测试套件
./start.sh --test
```

### 2.2 启动流程自愈机制
1. **自动锁定 Python 3.13**：无论当前终端处于 `base` 还是其他环境，脚本会自动解析并使用 `/opt/homebrew/Caskroom/miniforge/base/envs/llmqwen/bin/python3` 解释器。
2. **自动释放残留端口**：在启动服务前，脚本会自动检测 `8000` 端口是否被旧进程占有并执行优雅释放，杜绝 `[Errno 48] address already in use` 报错。

### 2.3 停止服务
- 前台运行：按下 `Ctrl + C`，服务将捕获信号并执行平滑退出（释放 Metal 显存）。
- 后台强杀：
  ```bash
  lsof -ti :8000 | xargs kill -9
  ```

---

## 3. 模型资产与生命周期管理

### 3.1 模型下载 (`download.py`)
利用国内 ModelScope 高速源下载与校验模型资产：
```bash
# 下载/校验默认 8-bit 模型
python download.py --model qwen3.8-27b-8bit

# 下载全部预置模型资产
python download.py --model all
```

### 3.2 8-bit + MTP 投机综合模型整合 (`merge_mtp.py`)
一键将 238MB 的 MTP 投机采样辅助层与 28GB 的 8-bit 主模型整合为一个综合版模型：
```bash
python merge_mtp.py --output ./models/qwen3.8-27b-8bit-mtp
```

### 3.3 原始模型 MLX 量化转换 (`convert_to_mlx.py`)
若需要从原始 16-bit 权重重新量化：
```bash
# 转换为 8-bit 量化版 (推荐：精度无损，速度 2x)
python convert_to_mlx.py --bits 8 --output ./models/qwen3.8-27b-mlx-8bit

# 转换为 4-bit 量化版 (极致速度 3.5x，仅占 15GB 显存)
python convert_to_mlx.py --bits 4 --output ./models/qwen3.8-27b-mlx-4bit
```

---

## 4. 全链路监控指标与健康探针

服务提供了符合云原生规范的 `/health` 健康检查与 `/metrics` 细粒度指标端点。

### 4.1 健康检查探针 (`GET /health`)
- **请求示例**：`curl http://localhost:8000/health`
- **响应示例**：
  ```json
  {
    "status": "ok",
    "service": "AI Code Service",
    "version": "1.1.0",
    "default_model": "qwen3.8-27b-8bit-mtp",
    "available_models": [
      "qwen3.8-27b-8bit-mtp",
      "qwen3.8-27b-8bit",
      "qwen3.8-27b"
    ],
    "active_engines": {
      "qwen3.8-27b-8bit-mtp": true
    },
    "use_mock": false
  }
  ```

### 4.2 性能与显存监控指标 (`GET /metrics`)
- **请求示例**：`curl http://localhost:8000/metrics`
- **指标说明表**：

| 指标字段 | 类型 | 单位 | 运维说明与健康阈值 |
| :--- | :--- | :--- | :--- |
| `active_memory_mb` | float | MB | Apple Metal 驱动当前活跃物理显存（正常范围：28,000MB ~ 70,000MB） |
| `cache_memory_mb` | float | MB | Metal 驱动显存缓存池（由 `metal_cache_limit_mb` 限制，默认 4096MB） |
| `peak_memory_mb` | float | MB | 历史峰值显存占用（128G 机器建议告警阈值 > 105,000MB） |
| `last_prompt_tps` | float | Tokens/s | 最近一次请求的提示词预填充速度（Prefill TPS） |
| `last_generation_tps` | float | Tokens/s | 最近一次请求的代码生成速度（8bit+MTP 正常值：45 ~ 65+ tok/s） |
| `cache.hit_rate` | float | % | 响应/前缀缓存命中率（高频编程场景预期 > 35%） |
| `cache.size` | int | 条目 | 当前内存中缓存的高频响应条数（最大 1024） |

---

## 5. 常见故障排查指南 (Troubleshooting)

### 5.1 报错 `[Errno 48] address already in use`
- **现象**：启动服务时提示 8000 端口已被绑定。
- **原因**：之前的旧版本服务进程仍在后台运行。
- **解决方法**：
  ```bash
  # 运行 start.sh 会自动清理，或手动执行：
  lsof -ti :8000 | xargs kill -9
  ```

### 5.2 报错 `Symbol not found: __ZN3mlx4core...`
- **现象**：加载 MLX 引擎时提示 `.dylib` 符号未找到。
- **原因**：当前终端使用了非 Python 3.13 的 Conda `base` 环境，其下的 MLX 动态库版本损坏。
- **解决方法**：
  ```bash
  conda activate llmqwen
  # 或直接运行封装好的 start.sh
  ./start.sh
  ```

### 5.3 启动时提示从 Hugging Face 重新下载 25.7GB 模型
- **现象**：服务启动时尝试连接外网 Hugging Face 下载权重。
- **原因**：MLX 默认仅识别 HuggingFace 缓存路径，而模型存放在 ModelScope 目录下。
- **状态**：**已修复**。底层引擎已集成 `resolve_local_model_path` 智能解析器，会自动秒级重定向到本地 `~/.cache/modelscope/` 快照目录。

### 5.4 客户端请求报 500 `unexpected keyword argument 'temp'`
- **现象**：非流式请求报错 500。
- **原因**：新版 `mlx_lm >= 0.31` 要求使用 `make_sampler` 构造采样器。
- **状态**：**已修复**。底层已重构采样器构造逻辑，支持标准 `temp`、`top_p` 与 `repetition_penalty`。

---

## 6. macOS 后台守护进程配置 (launchd)

如需让 AI Code Service 在 macOS 开机自启并在后台作为系统服务静默运行，可配置 launchd：

### 6.1 创建服务配置文件
创建文件 `~/Library/LaunchAgents/com.aicodeservice.server.plist`：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.aicodeservice.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/Caskroom/miniforge/base/envs/llmqwen/bin/python3</string>
        <string>app.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/charles/Documents/AI/Python/llmqwen</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/aicodeservice.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/aicodeservice.stderr.log</string>
</dict>
</plist>
```

### 6.2 注册与管理服务
```bash
# 启动并注册常驻守护进程
launchctl load ~/Library/LaunchAgents/com.aicodeservice.server.plist

# 查看实时日志
tail -f /tmp/aicodeservice.stdout.log

# 停止并卸载守护进程
launchctl unload ~/Library/LaunchAgents/com.aicodeservice.server.plist
```
