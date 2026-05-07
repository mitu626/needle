"""PagedAttention-style block allocator for KV cache management."""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

from .sequence import PhysicalBlock


class BlockAllocator:
    """Manages physical KV cache blocks on GPU and CPU.

    Blocks are fixed-size chunks of KV cache memory. Each sequence maps
    logical block indices to physical block ids via a block table.
    Reference counting enables copy-on-write for prefix sharing (future).
    """

    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int, block_size: int):
        self.block_size = block_size

        # GPU block pool
        self._gpu_free: deque[int] = deque(range(num_gpu_blocks))
        self._gpu_blocks: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i, device="gpu") for i in range(num_gpu_blocks)
        }

        # CPU block pool (for swap)
        self._cpu_free: deque[int] = deque(range(num_cpu_blocks))
        self._cpu_blocks: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i, device="cpu") for i in range(num_cpu_blocks)
        }

    # ------------------------------------------------------------------
    # GPU allocation
    # ------------------------------------------------------------------

    def allocate_gpu(self) -> int:
        if not self._gpu_free:
            raise MemoryError("No free GPU blocks available")
        block_id = self._gpu_free.popleft()
        self._gpu_blocks[block_id].ref_count = 1
        return block_id

    def allocate_gpu_blocks(self, num: int) -> List[int]:
        if len(self._gpu_free) < num:
            raise MemoryError(
                f"Requested {num} GPU blocks but only {len(self._gpu_free)} available"
            )
        return [self.allocate_gpu() for _ in range(num)]

    def free_gpu(self, block_id: int) -> None:
        block = self._gpu_blocks[block_id]
        assert block.ref_count > 0, f"Block {block_id} already free"
        block.ref_count -= 1
        if block.ref_count == 0:
            self._gpu_free.append(block_id)

    def free_gpu_blocks(self, block_ids: List[int]) -> None:
        for bid in block_ids:
            self.free_gpu(bid)

    # ------------------------------------------------------------------
    # CPU allocation (swap target)
    # ------------------------------------------------------------------

    def allocate_cpu(self) -> int:
        if not self._cpu_free:
            raise MemoryError("No free CPU blocks available")
        block_id = self._cpu_free.popleft()
        self._cpu_blocks[block_id].ref_count = 1
        return block_id

    def free_cpu(self, block_id: int) -> None:
        block = self._cpu_blocks[block_id]
        assert block.ref_count > 0
        block.ref_count -= 1
        if block.ref_count == 0:
            self._cpu_free.append(block_id)

    def free_cpu_blocks(self, block_ids: List[int]) -> None:
        for bid in block_ids:
            self.free_cpu(bid)

    # ------------------------------------------------------------------
    # Swap helpers
    # ------------------------------------------------------------------

    def swap_out(self, gpu_block_ids: List[int]) -> Dict[int, int]:
        """Allocate CPU blocks and return gpu->cpu mapping."""
        mapping: Dict[int, int] = {}
        for gpu_bid in gpu_block_ids:
            cpu_bid = self.allocate_cpu()
            mapping[gpu_bid] = cpu_bid
            self.free_gpu(gpu_bid)
        return mapping

    def swap_in(self, cpu_block_ids: List[int]) -> Dict[int, int]:
        """Allocate GPU blocks and return cpu->gpu mapping."""
        mapping: Dict[int, int] = {}
        for cpu_bid in cpu_block_ids:
            gpu_bid = self.allocate_gpu()
            mapping[cpu_bid] = gpu_bid
            self.free_cpu(cpu_bid)
        return mapping

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def num_free_gpu_blocks(self) -> int:
        return len(self._gpu_free)

    @property
    def num_free_cpu_blocks(self) -> int:
        return len(self._cpu_free)

    def can_allocate_gpu(self, num: int = 1) -> bool:
        return len(self._gpu_free) >= num

    def get_block(self, block_id: int, device: str = "gpu") -> PhysicalBlock:
        if device == "gpu":
            return self._gpu_blocks[block_id]
        return self._cpu_blocks[block_id]
