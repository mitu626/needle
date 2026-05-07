"""Core data structures for LeanLLM."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_new_tokens: int = 512
    stop_token_ids: List[int] = field(default_factory=list)
    stream: bool = False

    def __post_init__(self):
        assert self.temperature >= 0.0, "temperature must be >= 0"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
        assert self.max_new_tokens > 0, "max_new_tokens must be > 0"


@dataclass
class Request:
    request_id: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    arrival_time: float = field(default_factory=time.time)


class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED = "finished"


@dataclass
class Sequence:
    seq_id: int
    request: Request
    token_ids: List[int]        # prompt + generated tokens
    logical_blocks: List[int]   # logical block id list
    block_size: int
    status: SequenceStatus = SequenceStatus.WAITING
    num_generated_tokens: int = 0

    @property
    def length(self) -> int:
        return len(self.token_ids)

    @property
    def prompt_len(self) -> int:
        return len(self.request.prompt_token_ids)

    @property
    def num_logical_blocks(self) -> int:
        return (self.length + self.block_size - 1) // self.block_size

    @property
    def last_token_id(self) -> int:
        return self.token_ids[-1]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.num_generated_tokens += 1
        # check if we need a new logical block
        if self.length % self.block_size == 1:
            self.logical_blocks.append(len(self.logical_blocks))

    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def finish(self) -> None:
        self.status = SequenceStatus.FINISHED


@dataclass
class PhysicalBlock:
    block_id: int
    ref_count: int = 0
    device: str = "gpu"     # "gpu" | "cpu"


@dataclass
class SchedulerOutput:
    prefill_seqs: List[Sequence]
    decode_seqs: List[Sequence]
    block_tables: Dict[int, List[int]]  # seq_id -> physical block ids
    swap_in_map: Dict[int, int]         # cpu_block_id -> gpu_block_id
    swap_out_map: Dict[int, int]        # gpu_block_id -> cpu_block_id
    blocks_to_free: List[int]

    @property
    def num_seqs(self) -> int:
        return len(self.prefill_seqs) + len(self.decode_seqs)

    @property
    def is_empty(self) -> bool:
        return self.num_seqs == 0


@dataclass
class ExecuteInput:
    input_ids: torch.Tensor         # [total_tokens]
    position_ids: torch.Tensor      # [total_tokens]
    cu_seqlens: torch.Tensor        # [batch+1] cumulative sequence lengths
    max_seqlen: int
    block_table: torch.Tensor       # [num_seqs, max_blocks_per_seq]
    context_lens: torch.Tensor      # [num_seqs] number of tokens in KV cache
    is_prefill: bool


@dataclass
class ExecuteOutput:
    seq_ids: List[int]
    next_token_ids: List[int]
    logprobs: Optional[torch.Tensor] = None
