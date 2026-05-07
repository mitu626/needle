"""LLaMA model implementation (LLaMA-2 / LLaMA-3 compatible).

Supports GQA (num_key_value_heads != num_attention_heads).
RMSNorm and SiLU activation are used throughout.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from ..layers.attention import Attention
from ..layers.linear import ColumnParallelLinear, RowParallelLinear
from ..layers.norm import RMSNorm
from ..layers.rope import RotaryEmbedding

class LlamaMLP(nn.Module):
    """SwiGLU MLP: gate_proj and up_proj fused, then silu * up, then down."""

    def __init__(self, config: ModelConfig, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size,
            tp_size=tp_size, tp_rank=tp_rank,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size,
            tp_size=tp_size, tp_rank=tp_rank,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size,
            tp_size=tp_size, tp_rank=tp_rank,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        self.self_attn = Attention(config, layer_idx, tp_size=tp_size, tp_rank=tp_rank)
        self.mlp = LlamaMLP(config, tp_size=tp_size, tp_rank=tp_rank)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cu_seqlens: torch.Tensor,
        context_lens: torch.Tensor,
        max_seqlen: int,
        is_prefill: bool,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
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


class LlamaModel(nn.Module):
    def __init__(self, config: ModelConfig, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, i, tp_size=tp_size, tp_rank=tp_rank)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(
        self,
        input_ids: torch.Tensor,         # [total_tokens]
        positions: torch.Tensor,          # [total_tokens]
        kv_caches: list[torch.Tensor],    # per-layer KV caches
        block_table: torch.Tensor,
        cu_seqlens: torch.Tensor,
        context_lens: torch.Tensor,
        max_seqlen: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        cos, sin = self.rotary_emb.get_cos_sin(positions)
        hidden_states = self.embed_tokens(input_ids)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, positions, kv_caches[i],
                block_table, cu_seqlens, context_lens, max_seqlen, is_prefill,
                cos, sin,
            )
        return self.norm(hidden_states)


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config, tp_size=tp_size, tp_rank=tp_rank)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        block_table: torch.Tensor,
        cu_seqlens: torch.Tensor,
        context_lens: torch.Tensor,
        max_seqlen: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        """Returns logits for the last token of each sequence: [num_seqs, vocab]."""
        hidden = self.model(
            input_ids, positions, kv_caches, block_table,
            cu_seqlens, context_lens, max_seqlen, is_prefill,
        )
        if is_prefill:
            last_indices = (cu_seqlens[1:] - 1).long()  # [batch] — must be int64 for indexing
            last_hidden = hidden[last_indices]
        else:
            last_hidden = hidden  # decode: one token per seq

        return self.lm_head(last_hidden)  # [num_seqs, vocab_size]
