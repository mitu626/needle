"""Rotary Position Embedding (RoPE) — used by LLaMA, Qwen2, and most modern models."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for RoPE."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        t = torch.arange(max_seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def get_cos_sin(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[positions], self.sin_cached[positions]
