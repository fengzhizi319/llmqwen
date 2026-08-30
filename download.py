#!/usr/bin/env python3
"""
AI Code Service - 模型下载脚本
通过 ModelScope 国内高速源下载 Qwen / MLX 模型权重至本地缓存
"""

import argparse
import sys
from modelscope import snapshot_download


MODEL_REGISTRY = {
    "qwen3.8-flash-next-oQ4e-mtp-128k": "jedisct1/Qwen3.8-Flash-Next-oQ4e-MTP-128k",
    # "qwen3.8-27b-mlx": "inferencerlabs/Qwen3.8-27B-MTP-MLX",
    # "qwen3.8-27b-8bit": "lmstudio-community/Qwen3.8-27B-MLX-8bit",
    # "qwen3.8-27b": "Qwen/Qwen3.8-27B",
}


def download_model(model_key_or_repo: str) -> str:
    """下载指定模型并返回本地路径"""
    repo_id = MODEL_REGISTRY.get(model_key_or_repo, model_key_or_repo)
    print(f"\n📥 正在通过 ModelScope 下载/校验模型: {repo_id}...")
    local_path = snapshot_download(repo_id)
    print(f"✅ 模型下载就绪: {repo_id}")
    print(f"📁 本地存储路径: {local_path}")
    return local_path


def main():
    parser = argparse.ArgumentParser(description="AI Code Service - ModelScope 模型下载器")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="指定要下载的模型名称 (默认下载全部)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  🚀 AI Code Service - 模型下载工具")
    print("=" * 60)

    if args.model == "all":
        for key in MODEL_REGISTRY:
            download_model(key)
    else:
        download_model(args.model)

    print("\n🎉 模型下载/校验全部完成！可直接运行 ./start.sh 启动服务。")


if __name__ == "__main__":
    main()