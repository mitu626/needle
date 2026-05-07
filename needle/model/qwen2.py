"""Qwen2 model — reuses LLaMA components with minor differences.

Key differences from LLaMA:
  - QKV bias = True
  - Uses the same SwiGLU MLP
  - rope_theta typically 1_000_000 (set in config)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .llama import LlamaMLP, LlamaModel
from ..layers.attention import Attention
from ..layers.linear import ColumnParallelLinear, RowParallelLinear
from ..layers.norm import RMSNorm
from ..layers.rope import RotaryEmbedding


class Qwen2Attention(Attention):
    """Qwen2 adds bias to QKV projections."""

    def __init__(self, config: ModelConfig, layer_idx: int, tp_size: int = 1, tp_rank: int = 0):
        super().__init__(config, layer_idx, tp_size=tp_size, tp_rank=tp_rank)
        # Override projections to add bias
        self.q_proj = ColumnParallelLinear(
            config.hidden_size, config.num_attention_heads * config.head_dim,
            bias=True, tp_size=tp_size, tp_rank=tp_rank,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size, config.num_key_value_heads * config.head_dim,
            bias=True, tp_size=tp_size, tp_rank=tp_rank,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size, config.num_key_value_heads * config.head_dim,
            bias=True, tp_size=tp_size, tp_rank=tp_rank,
        )


class Qwen2DecoderLayer(nn.Module):
    """Qwen2 decoder layer — same structure as LLaMA, uses Qwen2Attention."""

    def __init__(self, config: ModelConfig, layer_idx: int, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        self.self_attn = Qwen2Attention(config, layer_idx, tp_size=tp_size, tp_rank=tp_rank)
        self.mlp = LlamaMLP(config, tp_size=tp_size, tp_rank=tp_rank)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, positions, kv_cache, block_table,
                cu_seqlens, context_lens, max_seqlen, is_prefill, cos, sin):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, positions, kv_cache, block_table,
            cu_seqlens, context_lens, max_seqlen, is_prefill, cos, sin,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen2Model(nn.Module):
    def __init__(self, config: ModelConfig, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config, i, tp_size=tp_size, tp_rank=tp_rank)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(self, input_ids, positions, kv_caches, block_table,
                cu_seqlens, context_lens, max_seqlen, is_prefill):
        cos, sin = self.rotary_emb.get_cos_sin(positions)
        hidden_states = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, positions, kv_caches[i],
                block_table, cu_seqlens, context_lens, max_seqlen, is_prefill, cos, sin,
            )
        return self.norm(hidden_states)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config, tp_size=tp_size, tp_rank=tp_rank)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, positions, kv_caches, block_table,
                cu_seqlens, context_lens, max_seqlen, is_prefill):
        hidden = self.model(input_ids, positions, kv_caches, block_table,
                            cu_seqlens, context_lens, max_seqlen, is_prefill)
        if is_prefill:
            last_indices = cu_seqlens[1:] - 1
            last_hidden = hidden[last_indices]
        else:
            last_hidden = hidden
        return self.lm_head(last_hidden)
