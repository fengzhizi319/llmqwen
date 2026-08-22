#!/usr/bin/env python3
"""
AI Code Service - 模型 MLX 格式转换与量化优化脚本
将标准 HuggingFace / ModelScope 模型转换为 Apple Silicon 原生 MLX 格式并进行 4-bit / 8-bit 量化加速
"""

import argparse
import os
import shutil
import sys
import time
from typing import Optional
import yaml

# 导入本地路径解析器
from engine.mlx_engine import resolve_local_model_path


def get_dir_size_mb(path: str) -> float:
    """计算文件夹总大小 (MB)"""
    total_size = 0
    if not os.path.exists(path):
        return 0.0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return round(total_size / (1024 * 1024), 2)


def convert_model(
    input_path: str,
    output_dir: str,
    quantize: bool = True,
    q_bits: int = 8,
    q_group_size: int = 64,
    auto_update_config: bool = False,
):
    print("=" * 65)
    print("  🚀 AI Code Service - MLX 模型转换与量化加速工具")
    print("=" * 65)

    resolved_src = resolve_local_model_path(input_path)
    print(f"📌 源模型输入路径: {input_path}")
    print(f"📁 本地定位快照路径: {resolved_src}")
    
    if not os.path.exists(resolved_src):
        print(f"❌ 错误: 未在本地找到源模型 '{resolved_src}'，请先运行 python download.py 下载。")
        sys.exit(1)

    src_size_mb = get_dir_size_mb(resolved_src)
    print(f"📊 源模型原始体积: {round(src_size_mb / 1024, 2)} GB")

    target_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(target_dir, exist_ok=True)
    print(f"🎯 MLX 目标输出目录: {target_dir}")
    print(f"⚙️ 量化配置: {'启用' if quantize else '禁用'} (位数: {q_bits}-bit, Group Size: {q_group_size})")
    print("-" * 65)
    print("⏳ 正在转换与量化模型权重（在 Apple Silicon 统一内存上高速执行，请稍候...）")

    t0 = time.time()

    # 判断是否为 VLM 模型
    is_vlm = False
    config_json_path = os.path.join(resolved_src, "config.json")
    if os.path.exists(config_json_path):
        with open(config_json_path, "r", encoding="utf-8") as f:
            content = f.read()
            if "vision_config" in content or "Qwen3_5ForConditionalGeneration" in content:
                is_vlm = True

    try:
        if is_vlm:
            print("🔍 检测到视觉/多模态架构，使用 mlx_vlm.convert 进行转换...")
            import mlx_vlm
            mlx_vlm.convert(
                hf_path=resolved_src,
                mlx_path=target_dir,
                quantize=quantize,
                q_bits=q_bits,
                q_group_size=q_group_size,
            )
        else:
            print("🔍 检测到纯文本/代码架构，使用 mlx_lm.convert 进行转换...")
            import mlx_lm
            mlx_lm.convert(
                hf_path=resolved_src,
                mlx_path=target_dir,
                quantize=quantize,
                q_bits=q_bits,
                q_group_size=q_group_size,
            )

        duration = round(time.time() - t0, 2)
        dst_size_mb = get_dir_size_mb(target_dir)

        print("\n" + "=" * 65)
        print("🎉 MLX 优化模型转换成功！")
        print(f"⏱️ 总耗时: {duration} 秒")
        print(f"💾 转换前大小: {round(src_size_mb / 1024, 2)} GB")
        print(f"⚡ 转换后大小: {round(dst_size_mb / 1024, 2)} GB (内存节省 {round((1 - dst_size_mb/src_size_mb)*100, 1)}%)")
        print(f"📁 输出目录: {target_dir}")
        print("=" * 65)

        # 自动更新 config.yaml
        if auto_update_config:
            update_config_yaml(target_dir, q_bits)

    except Exception as e:
        print(f"\n❌ 转换过程发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def update_config_yaml(mlx_model_path: str, q_bits: int):
    """将新转换的 MLX 模型注册到 config.yaml"""
    config_file = "config.yaml"
    if not os.path.exists(config_file):
        return

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model_key = f"qwen3.8-27b-{q_bits}bit-custom"
    if "models" not in cfg:
        cfg["models"] = {}

    cfg["models"][model_key] = {
        "path": mlx_model_path,
        "description": f"Qwen3.8-27B 本地定制 MLX {q_bits}bit 量化加速版 (支持 256K 上下文)",
        "engine_type": "auto",
        "context_length": 262144,
    }
    cfg["default_model"] = model_key

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"\n📝 已自动将新模型 '{model_key}' 写入 config.yaml 并设为默认模型！")


def main():
    parser = argparse.ArgumentParser(description="AI Code Service - MLX 模型转换与量化工具")
    parser.add_argument(
        "--input",
        type=str,
        default="Qwen/Qwen3.8-27B",
        help="源模型路径或 ModelScope 仓库名称 (默认: Qwen/Qwen3.8-27B)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/qwen3.8-27b-mlx-8bit",
        help="MLX 目标输出目录 (默认: ./models/qwen3.8-27b-mlx-8bit)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=8,
        help="量化比特数: 8 (推荐代码生成，精度无损且提速 2x) 或 4 (提速 3.5x，体积仅 15GB)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="量化分组大小 (默认: 64)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="仅转换 MLX 格式，不进行量化压缩",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        default=True,
        help="自动将转换后的模型配置写入 config.yaml 并设为默认模型",
    )

    args = parser.parse_args()

    # 动态调整输出目录后缀
    if args.output == "./models/qwen3.8-27b-mlx-8bit" and args.bits == 4:
        args.output = "./models/qwen3.8-27b-mlx-4bit"

    convert_model(
        input_path=args.input,
        output_dir=args.output,
        quantize=not args.no_quantize,
        q_bits=args.bits,
        q_group_size=args.group_size,
        auto_update_config=args.update_config,
    )


if __name__ == "__main__":
    main()
