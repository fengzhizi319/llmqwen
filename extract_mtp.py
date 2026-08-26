#!/usr/bin/env python3
"""
AI Code Service - 从 Qwen3.5 模型权重中提取内嵌 MTP 层到独立文件

Qwen3.5 架构在训练时将 MTP (Multi-Token Prediction) 层嵌入主权重
（路径: language_model.mtp.*）。mlx_lm 加载时会在 sanitize() 中丢弃这些权重。
本脚本将 MTP 权重提取到独立 safetensors 文件，供推测解码使用。
"""

import argparse
import json
import os
import sys

import mlx.core as mx
from safetensors import safe_open


def extract_mtp_weights(model_path: str, output_file: str = "mtp.safetensors") -> int:
    """从模型 safetensors 分片中提取所有 MTP 权重"""
    model_dir = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.isdir(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        sys.exit(1)

    # 收集所有 MTP 权重
    mtp_weights = {}
    shard_files = sorted(
        f for f in os.listdir(model_dir) if f.endswith(".safetensors")
    )
    if not shard_files:
        print(f"❌ 未找到 safetensors 文件: {model_dir}")
        sys.exit(1)

    print(f"📂 模型目录: {model_dir}")
    print(f"🔍 扫描 {len(shard_files)} 个权重分片...")

    for fname in shard_files:
        fpath = os.path.join(model_dir, fname)
        with safe_open(fpath, framework="numpy") as f:
            mtp_keys = [k for k in f.keys() if "mtp." in k]
            if mtp_keys:
                print(f"  📦 {fname}: 发现 {len(mtp_keys)} 个 MTP 权重")
                for key in mtp_keys:
                    tensor = f.get_tensor(key)
                    # 去掉 language_model. 前缀，保存为 mtp.* 格式
                    short_key = key.replace("language_model.", "")
                    mtp_weights[short_key] = mx.array(tensor)

    if not mtp_weights:
        print("❌ 未找到任何 MTP 权重（模型可能不包含内嵌 MTP 层）")
        sys.exit(1)

    # 保存提取的权重
    output_path = os.path.join(model_dir, output_file)
    # 转换为可序列化格式
    save_dict = {}
    for k, v in mtp_weights.items():
        save_dict[k] = v

    mx.save_safetensors(output_path, save_dict)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ 提取完成！保存至: {output_path} ({file_size_mb:.1f} MB)")
    print(f"   共 {len(mtp_weights)} 个权重张量")

    # 更新 config.json
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        cfg["has_mtp"] = True
        cfg["mtp_weights_path"] = output_file
        if "mtp_block_size" not in cfg:
            cfg["mtp_block_size"] = 1  # Qwen3.5 内嵌 MTP 为 depth-1
        if "text_config" in cfg:
            cfg["text_config"]["mtp_num_hidden_layers"] = 1

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
        print(f"📝 已更新 config.json (has_mtp=true, mtp_weights_path={output_file})")

    return len(mtp_weights)


def main():
    parser = argparse.ArgumentParser(
        description="从 Qwen3.5 模型中提取内嵌 MTP 权重到独立文件"
    )
    parser.add_argument(
        "model_path",
        help="模型本地路径 (如 ./models/Qwen3.8-27B-oQ4e-fp16-mtp)",
    )
    parser.add_argument(
        "--output",
        default="mtp.safetensors",
        help="输出文件名 (默认: mtp.safetensors)",
    )
    args = parser.parse_args()
    extract_mtp_weights(args.model_path, args.output)


if __name__ == "__main__":
    main()
