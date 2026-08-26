"""
AI Code Service - MTP (Multi-Token Prediction) 推测解码 Draft 模型

从 Qwen3.5 模型中提取的内嵌 MTP 层构建轻量级 Draft 头，
用于推测解码加速：MTP 层基于主模型 hidden state 预测下一个 token，
主模型验证后一次推理产出多个 token，实现生成加速。
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs


class MTPHead(nn.Module):
    """Qwen3.5 内嵌 MTP 推测解码头

    架构:
        1. pre_fc_norm_hidden / pre_fc_norm_embedding: 归一化输入
        2. fc: 拼接 hidden_state + embedding → 投影到 hidden_size
        3. norm: 后投影归一化
        4. layer: 单层 Transformer Decoder (softmax attention + MLP)
        → 输出预测的下一个 token hidden state
    """

    def __init__(self, text_args: TextModelArgs, quant_config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = text_args.hidden_size

        # 双输入归一化
        self.pre_fc_norm_hidden = nn.RMSNorm(text_args.hidden_size, eps=text_args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(text_args.hidden_size, eps=text_args.rms_norm_eps)

        # 拼接投影层: concat(hidden, emb) → hidden  (10240 → 5120)
        self.fc = nn.Linear(text_args.hidden_size * 2, text_args.hidden_size, bias=False)

        # 后投影归一化
        self.norm = nn.RMSNorm(text_args.hidden_size, eps=text_args.rms_norm_eps)

        # 单层 Transformer Decoder (full attention, layer_idx=3 确保 is_linear=False)
        self.layers = [DecoderLayer(args=text_args, layer_idx=text_args.full_attention_interval - 1)]

        # 对 DecoderLayer 内的 Linear 层进行混合精度量化 (匹配原始 oQ 配置)
        # attention=5bit, MLP=4bit
        group_size = quant_config.get("group_size", 64)
        attn_bits = quant_config.get("attn_bits", 5)
        mlp_bits = quant_config.get("mlp_bits", 4)

        layer = self.layers[0]
        if hasattr(layer, 'self_attn'):
            nn.quantize(layer.self_attn, bits=attn_bits, group_size=group_size)
        if hasattr(layer, 'linear_attn'):
            nn.quantize(layer.linear_attn, bits=attn_bits, group_size=group_size)
        nn.quantize(layer.mlp, bits=mlp_bits, group_size=group_size)

    def __call__(
        self,
        hidden_state: mx.array,
        embedding: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """基于主模型 hidden state 预测下一个 token 的 hidden state

        Args:
            hidden_state: 主模型最后一层的输出 [batch, hidden_size]
            embedding: 当前 token 的嵌入向量 [batch, hidden_size]
            cache: MTP 层的 KV cache

        Returns:
            预测的下一个 token hidden state [batch, hidden_size]
        """
        # 归一化双输入
        h_norm = self.pre_fc_norm_hidden(hidden_state)
        e_norm = self.pre_fc_norm_embedding(embedding)

        # 拼接 → 投影 → 归一化
        combined = mx.concatenate([h_norm, e_norm], axis=-1)
        projected = self.fc(combined)
        x = self.norm(projected)

        # 添加序列维度 [batch, 1, hidden_size] 以适配 DecoderLayer
        x = mx.expand_dims(x, axis=1)

        # 通过 Transformer 层 (使用 1x1 causal mask)
        mask = None
        if cache is not None and cache[0] is not None:
            from mlx_lm.models.base import create_attention_mask
            mask = create_attention_mask(x, cache[0])

        layer_cache = cache[0] if cache else None
        next_hidden = self.layers[0](x, mask=mask, cache=layer_cache)

        # 去掉序列维度 → [batch, hidden_size]
        return next_hidden.squeeze(1)


def load_mtp_head(
    model_path: str,
    mtp_weights_path: str = "mtp.safetensors",
) -> Optional[MTPHead]:
    """从模型目录加载 MTP 推测解码头

    Args:
        model_path: 模型本地路径 (包含 config.json 和 mtp.safetensors)
        mtp_weights_path: MTP 权重文件名

    Returns:
        加载成功的 MTPHead 实例，若权重文件不存在则返回 None
    """
    config_file = os.path.join(model_path, "config.json")
    weights_file = os.path.join(model_path, mtp_weights_path)

    if not os.path.exists(weights_file):
        return None
    if not os.path.exists(config_file):
        return None

    # 读取模型配置
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    text_cfg = cfg.get("text_config", cfg)
    quant_cfg = cfg.get("quantization", {})

    # 构建 TextModelArgs
    text_args = TextModelArgs(
        model_type=text_cfg.get("model_type", "qwen3_5_text"),
        hidden_size=text_cfg.get("hidden_size", 5120),
        intermediate_size=text_cfg.get("intermediate_size", 17408),
        num_hidden_layers=1,  # MTP 只有 1 层
        num_attention_heads=text_cfg.get("num_attention_heads", 24),
        num_key_value_heads=text_cfg.get("num_key_value_heads", 4),
        head_dim=text_cfg.get("head_dim", 256),
        rms_norm_eps=text_cfg.get("rms_norm_eps", 1e-6),
        vocab_size=text_cfg.get("vocab_size", 248320),
        full_attention_interval=text_cfg.get("full_attention_interval", 4),
        rope_parameters=text_cfg.get("rope_parameters", {}),
    )

    # 构建 MTP 头
    quant_config = {
        "mlp_bits": quant_cfg.get("bits", 4),
        "attn_bits": quant_cfg.get("bits", 4),
        "group_size": quant_cfg.get("group_size", 64),
    }
    mtp_head = MTPHead(text_args, quant_config)

    # 加载权重
    weights = mx.load(weights_file) if weights_file.endswith(".npz") else None
    if weights is None:
        # safetensors 格式
        from safetensors import safe_open
        weight_dict = {}
        with safe_open(weights_file, framework="numpy") as f:
            for key in f.keys():
                weight_dict[key] = mx.array(f.get_tensor(key))
        weights = weight_dict

    # 权重 key 映射: 去掉 mtp. 前缀以匹配 MTPHead 内部结构
    remapped = {}
    for k, v in weights.items():
        new_key = k
        if new_key.startswith("mtp."):
            new_key = new_key[4:]  # 去掉 "mtp." 前缀
        remapped[new_key] = v

    # 使用 strict=False 允许部分权重不匹配 (如量化后结构差异)
    mtp_head.load_weights(list(remapped.items()), strict=False)
    print(f"[MTPHead] ✅ 成功加载 MTP 推测解码头 ({len(weights)} 个权重)")
    return mtp_head
