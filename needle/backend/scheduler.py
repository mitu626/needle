"""Continuous Batching scheduler.

Scheduling policy:
  1. Swap-in sequences that were previously swapped out (SWAPPED -> RUNNING).
  2. Promote WAITING sequences to RUNNING if GPU blocks are available.
  3. If GPU memory is tight, swap out lower-priority RUNNING sequences.
  4. Build SchedulerOutput with prefill / decode split.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from .block_allocator import BlockAllocator
from .sequence import (
    SchedulerOutput,
    Sequence,
    SequenceStatus,
)


class Scheduler:
    def __init__(
        self,
        block_allocator: BlockAllocator,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
    ):
        self.allocator = block_allocator
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens

        self.waiting: deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self.swapped: List[Sequence] = []

        # seq_id -> list of physical block ids on current device
        self._block_tables: Dict[int, List[int]] = {}
        # seq_id -> list of physical block ids on CPU (for swapped seqs)
        self._cpu_block_tables: Dict[int, List[int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_sequence(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def schedule(self) -> SchedulerOutput:
        swap_in_map: Dict[int, int] = {}
        swap_out_map: Dict[int, int] = {}
        blocks_to_free: List[int] = []

        # 1. Try to swap in SWAPPED sequences
        still_swapped: List[Sequence] = []
        for seq in self.swapped:
            cpu_blocks = self._cpu_block_tables[seq.seq_id]
            if self.allocator.can_allocate_gpu(len(cpu_blocks)):
                mapping = self.allocator.swap_in(cpu_blocks)
                swap_in_map.update(mapping)
                self._block_tables[seq.seq_id] = [mapping[c] for c in cpu_blocks]
                del self._cpu_block_tables[seq.seq_id]
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                still_swapped.append(seq)
        self.swapped = still_swapped

        # 2. Promote WAITING sequences
        num_batched_tokens = sum(s.length for s in self.running)
        while self.waiting:
            seq = self.waiting[0]
            num_blocks_needed = seq.num_logical_blocks
            tokens_needed = seq.length

            if (
                len(self.running) >= self.max_num_seqs
                or num_batched_tokens + tokens_needed > self.max_num_batched_tokens
                or not self.allocator.can_allocate_gpu(num_blocks_needed)
            ):
                break

            self.waiting.popleft()
            gpu_blocks = self.allocator.allocate_gpu_blocks(num_blocks_needed)
            self._block_tables[seq.seq_id] = gpu_blocks
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            num_batched_tokens += tokens_needed

        # 3. Swap out if still over budget (preemption)
        while self.running and not self.allocator.can_allocate_gpu(1):
            victim = self.running.pop()  # evict last (lowest priority FIFO)
            gpu_blocks = self._block_tables[victim.seq_id]
            if self.allocator.num_free_cpu_blocks >= len(gpu_blocks):
                mapping = self.allocator.swap_out(gpu_blocks)
                swap_out_map.update(mapping)
                self._cpu_block_tables[victim.seq_id] = [
                    mapping[g] for g in gpu_blocks
                ]
                del self._block_tables[victim.seq_id]
                victim.status = SequenceStatus.SWAPPED
                self.swapped.append(victim)
            else:
                # No CPU space either — free and recompute later
                self.allocator.free_gpu_blocks(gpu_blocks)
                blocks_to_free.extend(gpu_blocks)
                del self._block_tables[victim.seq_id]
                victim.status = SequenceStatus.WAITING
                self.waiting.appendleft(victim)

        # 4. Allocate one extra block for each decode sequence that needs it
        for seq in self.running:
            if seq.num_generated_tokens > 0:  # decode phase
                blocks_needed = seq.num_logical_blocks
                current_blocks = len(self._block_tables[seq.seq_id])
                if blocks_needed > current_blocks:
                    new_block = self.allocator.allocate_gpu()
                    self._block_tables[seq.seq_id].append(new_block)

        # 5. Split into prefill vs decode
        prefill_seqs = [s for s in self.running if s.num_generated_tokens == 0]
        decode_seqs = [s for s in self.running if s.num_generated_tokens > 0]

        return SchedulerOutput(
            prefill_seqs=prefill_seqs,
            decode_seqs=decode_seqs,
            block_tables={
                sid: list(blocks)
                for sid, blocks in self._block_tables.items()
            },
            swap_in_map=swap_in_map,
            swap_out_map=swap_out_map,
            blocks_to_free=blocks_to_free,
        )

    def free_sequence(self, seq: Sequence) -> None:
        """Release all blocks for a finished sequence.

        Bug #6 fix: use discard-style removal so calling free_sequence
        twice (double-free) does not raise an exception.
        """
        if seq.seq_id in self._block_tables:
            self.allocator.free_gpu_blocks(self._block_tables.pop(seq.seq_id))
        if seq.seq_id in self._cpu_block_tables:
            self.allocator.free_cpu_blocks(self._cpu_block_tables.pop(seq.seq_id))
        seq.finish()
        # Use discard-style removal to avoid ValueError on double-free
        try:
            self.running.remove(seq)
        except ValueError:
            pass

    def has_unfinished_seqs(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)
