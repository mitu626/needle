"""Tensor-parallel Linear layers.

ColumnParallelLinear: splits output features across TP ranks.
RowParallelLinear:    splits input features across TP ranks, all-reduces output.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _get_tp_group():
    """Return the process group used for tensor parallelism."""
    if dist.is_available() and dist.is_initialized():
        return dist.group.WORLD
    return None


class ColumnParallelLinear(nn.Module):
    """Linear layer that shards output dimension across TP ranks.

    Each rank holds weight[:, local_out_start:local_out_end].
    No communication needed after forward — used in QKV projections and
    the first MLP linear where the result is consumed locally.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
        gather_output: bool = False,
    ):
        super().__init__()
        assert out_features % tp_size == 0, (
            f"out_features {out_features} must be divisible by tp_size {tp_size}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.gather_output = gather_output

        self.local_out = out_features // tp_size
        self.weight = nn.Parameter(torch.empty(self.local_out, in_features))
        self.bias = nn.Parameter(torch.zeros(self.local_out)) if bias else None
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.gather_output and self.tp_size > 1:
            gathered = [torch.zeros_like(out) for _ in range(self.tp_size)]
            dist.all_gather(gathered, out, group=_get_tp_group())
            out = torch.cat(gathered, dim=-1)
        return out


class RowParallelLinear(nn.Module):
    """Linear layer that shards input dimension across TP ranks.

    Each rank holds weight[local_in_start:local_in_end, :].
    All-reduces output across ranks — used after GQA attention output
    projection and the second MLP linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
        reduce_output: bool = True,
    ):
        super().__init__()
        assert in_features % tp_size == 0
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.reduce_output = reduce_output

        self.local_in = in_features // tp_size
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight)
        if self.reduce_output and self.tp_size > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=_get_tp_group())
        if self.bias is not None:
            out = out + self.bias
        return out
