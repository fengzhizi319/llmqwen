#!/bin/bash

# AI Code Service 启动与测试脚本

set -e

echo "=========================================="
echo "  🚀 AI Code Service"
echo "=========================================="

ENV_NAME="llmqwen"

# 1. 尝试自动激活 Conda 环境
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    if command -v conda &> /dev/null; then
        CONDA_BASE=$(conda info --base 2>/dev/null || true)
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            if conda env list | grep -q -E "(^|[ /])${ENV_NAME}($|[ /])"; then
                echo "⚡ 自动激活 Conda 环境: ${ENV_NAME}..."
                conda activate "$ENV_NAME"
            fi
        fi
    fi
fi

# 2. 检查 Python 命令与版本
if command -v python3 &> /dev/null; then
    PY_BIN="python3"
elif command -v python &> /dev/null; then
    PY_BIN="python"
else
    echo "❌ 错误: 未找到 Python，请先配置 Conda 环境: conda activate ${ENV_NAME}"
    exit 1
fi

PY_VERSION=$($PY_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_FULL_VERSION=$($PY_BIN --version 2>&1)
echo "🐍 当前 Python 环境: $PY_FULL_VERSION ($(which $PY_BIN))"

# 检查 Python 版本是否满足推荐的 3.13+
$PY_BIN -c "import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)" || {
    echo "⚠️  警告: 当前 Python 版本为 $PY_VERSION，推荐使用 Python 3.13+ 运行本项目。"
}

# 3. 检查配置文件
if [ ! -f config.yaml ]; then
    echo "❌ 错误: 未找到 config.yaml 配置文件"
    exit 1
fi

# 4. 支持命令行参数
if [ "$1" == "--test" ]; then
    echo "🧪 运行单元测试与集成测试套件..."
    $PY_BIN -m pytest -v
    exit 0
fi

# 5. 启动服务
echo "🚀 启动服务..."
echo "API 地址: http://localhost:8000"
echo "API 文档: http://localhost:8000/docs"
echo "=========================================="

$PY_BIN app.py
