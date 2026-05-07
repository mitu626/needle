"""Multi-head / Grouped-query Attention with PagedAttention support.

Prefill:  flashinfer.prefill.single_prefill_with_kv_cache (varlen, causal)
Decode:   flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper (paged KV)

Falls back to torch.scaled_dot_product_attention + pure-PyTorch gather when
flashinfer is not available.

KV cache layout: [2, num_blocks, num_kv_heads, block_size, head_dim]
  dim 0: 0=K, 1=V
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import ColumnParallelLinear, RowParallelLinear
from ..model.config import ModelConfig

try:
    from flashinfer.prefill import single_prefill_with_kv_cache as _fi_prefill
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper as _FiDecodeWrapper
    _HAS_FLASHINFER = True
except ImportError:
    _HAS_FLASHINFER = False


class Attention(nn.Module):
    """MHA / GQA Attention layer."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = ColumnParallelLinear(
            config.hidden_size, config.num_attention_heads * config.head_dim,
            tp_size=tp_size, tp_rank=tp_rank,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size, config.num_key_value_heads * config.head_dim,
            tp_size=tp_size, tp_rank=tp_rank,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size, config.num_key_value_heads * config.head_dim,
            tp_size=tp_size, tp_rank=tp_rank,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * config.head_dim, config.hidden_size,
            tp_size=tp_size, tp_rank=tp_rank,
        )
        # Lazily allocated per-layer decode workspace (flashinfer requirement)
        self._decode_workspace: Optional[torch.Tensor] = None
        self._decode_wrapper: Optional[object] = None

    def forward(
        self,
        hidden_states: torch.Tensor,   # [total_tokens, hidden_size]
        positions: torch.Tensor,        # [total_tokens]  (used upstream for RoPE lookup)
        kv_cache: torch.Tensor,         # [2, num_blocks, num_kv_heads, block_size, head_dim]
        block_table: torch.Tensor,      # [num_seqs, max_blocks_per_seq]
        cu_seqlens: torch.Tensor,       # [batch+1]  int32
        context_lens: torch.Tensor,     # [num_seqs]  int32
        max_seqlen: int,               # noqa: ARG002 — kept for API symmetry with prefill
        is_prefill: bool,
        cos: torch.Tensor,              # [total_tokens, head_dim]
        sin: torch.Tensor,              # [total_tokens, head_dim]
    ) -> torch.Tensor:
        total_tokens = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(total_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(total_tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(total_tokens, self.num_kv_heads, self.head_dim)

        q, k = _apply_rope(q, k, cos, sin)

        if is_prefill:
            attn_out = self._prefill(q, k, v, kv_cache, block_table,
                                     cu_seqlens, context_lens, max_seqlen)
        else:
            attn_out = self._decode(q, k, v, kv_cache, block_table, context_lens)

        return self.o_proj(attn_out.view(total_tokens, self.num_heads * self.head_dim))

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _prefill(self, q, k, v, kv_cache, block_table, cu_seqlens, context_lens, max_seqlen):
        _write_kv(k, v, kv_cache, block_table, cu_seqlens, context_lens)

        if _HAS_FLASHINFER and q.is_cuda:
            return self._prefill_flashinfer(q, k, v, cu_seqlens)
        return self._prefill_pytorch(q, k, v, cu_seqlens)

    def _prefill_flashinfer(self, q, k, v, cu_seqlens):
        outputs = []
        for i in range(cu_seqlens.shape[0] - 1):
            s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            outputs.append(_fi_prefill(q[s:e], k[s:e], v[s:e],
                                       causal=True, sm_scale=self.scale))
        return torch.cat(outputs, dim=0)

    def _prefill_pytorch(self, q, k, v, cu_seqlens):
        outputs = []
        for i in range(cu_seqlens.shape[0] - 1):
            s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            qi = q[s:e].unsqueeze(0).transpose(1, 2)   # [1, H, S, D]
            ki = k[s:e].unsqueeze(0).transpose(1, 2)
            vi = v[s:e].unsqueeze(0).transpose(1, 2)
            if self.num_kv_heads != self.num_heads:
                factor = self.num_heads // self.num_kv_heads
                ki = ki.repeat_interleave(factor, dim=1)
                vi = vi.repeat_interleave(factor, dim=1)
            out_i = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True)
            outputs.append(out_i.transpose(1, 2).squeeze(0))  # [S, H, D]
        return torch.cat(outputs, dim=0)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def _decode(self, q, k, v, kv_cache, block_table, context_lens):
        num_seqs   = q.shape[0]
        block_size = kv_cache.shape[3]
        ctx        = context_lens.long()
        phys       = block_table[torch.arange(num_seqs, device=q.device), ctx // block_size]
        block_off  = ctx % block_size

        # Write new token K/V into cache — vectorized scatter
        kv_cache[0, phys, :, block_off, :] = k
        kv_cache[1, phys, :, block_off, :] = v

        if _HAS_FLASHINFER and q.is_cuda and _fi_group_size_ok(self.num_heads, self.num_kv_heads):
            return self._decode_flashinfer(q, kv_cache, block_table, context_lens)
        return _decode_pytorch(q, kv_cache, block_table, context_lens, self.scale,
                               self.num_heads, self.num_kv_heads, self.head_dim)

    def _decode_flashinfer(self, q, kv_cache, block_table, context_lens):
        num_seqs   = q.shape[0]
        block_size = kv_cache.shape[3]
        seq_lens   = (context_lens + 1).to(torch.int32)

        num_blocks_per_seq = (seq_lens + block_size - 1) // block_size
        indptr = torch.zeros(num_seqs + 1, dtype=torch.int32, device=q.device)
        indptr[1:] = num_blocks_per_seq.cumsum(0)

        indices = torch.cat([
            block_table[i, :int(num_blocks_per_seq[i])]
            for i in range(num_seqs)
        ]).to(torch.int32)

        last_page_lens = (seq_lens - 1) % block_size + 1

        # Allocate workspace buffer on first use (32 MB is sufficient for most configs)
        if self._decode_workspace is None:
            self._decode_workspace = torch.empty(
                32 * 1024 * 1024, dtype=torch.uint8, device=q.device
            )
        if self._decode_wrapper is None:
            self._decode_wrapper = _FiDecodeWrapper(self._decode_workspace)

        wrapper = self._decode_wrapper
        wrapper.begin_forward(
            indptr, indices, last_page_lens,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=block_size,
            q_data_type=q.dtype,
        )
        out = wrapper.forward(q, kv_cache[0], kv_cache[1], sm_scale=self.scale)
        wrapper.end_forward()
        return out  # [num_seqs, num_heads, head_dim]


# ---------------------------------------------------------------------------
# KV cache write
# ---------------------------------------------------------------------------

def _write_kv(
    k: torch.Tensor,            # [total_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,
    kv_cache: torch.Tensor,     # [2, num_blocks, num_kv_heads, block_size, head_dim]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens: torch.Tensor,   # [batch+1]
    context_lens: torch.Tensor, # [batch]
) -> None:
    """Scatter K/V tokens into the paged KV cache.

    Uses PyTorch advanced indexing — no custom CUDA kernel needed.
    kv_cache[0, phys, :, block_off, :] = k  (and same for V)
    """
    block_size  = kv_cache.shape[3]
    batch_size  = cu_seqlens.shape[0] - 1
    seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()

    # Absolute KV position for every token in the batch
    abs_pos = torch.cat([
        torch.arange(int(context_lens[i]),
                     int(context_lens[i]) + int(seq_lengths[i]),
                     device=k.device, dtype=torch.long)
        for i in range(batch_size)
    ])  # [total_tokens]

    seq_idx = torch.repeat_interleave(
        torch.arange(batch_size, device=k.device, dtype=torch.long), seq_lengths
    )  # [total_tokens]

    phys      = block_table[seq_idx, abs_pos // block_size]  # [total_tokens]
    block_off = abs_pos % block_size                          # [total_tokens]

    kv_cache[0, phys, :, block_off, :] = k
    kv_cache[1, phys, :, block_off, :] = v


# ---------------------------------------------------------------------------
# swap_kv_blocks: CPU <-> GPU block swap for preemption
# ---------------------------------------------------------------------------

def swap_kv_blocks(
    src_cache: torch.Tensor,    # [2, num_blocks, num_kv_heads, block_size, head_dim]
    dst_cache: torch.Tensor,
    block_mapping: torch.Tensor,  # [num_pairs, 2]: [[src_id, dst_id], ...]
) -> None:
    """Copy physical KV blocks between two caches (GPU<->CPU or GPU<->GPU).

    PyTorch handles cross-device transfers automatically — no cudaMemcpyAsync needed.
    """
    src_ids = block_mapping[:, 0].long()
    dst_ids = block_mapping[:, 1].long()
    dst_cache[:, dst_ids] = src_cache[:, src_ids]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fi_group_size_ok(num_heads: int, num_kv_heads: int) -> bool:
    """FlashInfer decode only supports GQA group sizes that are powers of 2."""
    g = num_heads // num_kv_heads
    return g > 0 and (g & (g - 1)) == 0


def _apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # [T, 1, D]
    sin = sin.unsqueeze(1)
    return (
        q * cos + _rotate_half(q) * sin,
        k * cos + _rotate_half(k) * sin,
    )


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _decode_pytorch(q, kv_cache, block_table, context_lens, scale,
                    num_heads, num_kv_heads, head_dim):
    """Pure-PyTorch paged decode fallback."""
    num_seqs   = q.shape[0]
    block_size = kv_cache.shape[3]
    outputs    = []
    for i in range(num_seqs):
        ctx_len = int(context_lens[i]) + 1
        nb      = (ctx_len + block_size - 1) // block_size
        phys    = block_table[i, :nb]
        k_full  = kv_cache[0][phys].transpose(0, 1).reshape(num_kv_heads, -1, head_dim)[:, :ctx_len]
        v_full  = kv_cache[1][phys].transpose(0, 1).reshape(num_kv_heads, -1, head_dim)[:, :ctx_len]
        expand  = num_heads // num_kv_heads
        k_exp   = k_full.repeat_interleave(expand, dim=0)
        v_exp   = v_full.repeat_interleave(expand, dim=0)
        attn_w  = torch.einsum("hd,hsd->hs", q[i], k_exp) * scale
        out_i   = torch.einsum("hs,hsd->hd", attn_w.softmax(dim=-1), v_exp)
        outputs.append(out_i)
    return torch.stack(outputs, dim=0)
