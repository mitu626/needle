"""Unit tests for BlockAllocator."""
import pytest
from needle.core.block_allocator import BlockAllocator


@pytest.fixture
def allocator():
    return BlockAllocator(num_gpu_blocks=8, num_cpu_blocks=4, block_size=16)


class TestAllocateFree:
    def test_allocate_single(self, allocator):
        bid = allocator.allocate_gpu()
        assert 0 <= bid < 8
        assert allocator.num_free_gpu_blocks == 7

    def test_allocate_multiple(self, allocator):
        blocks = allocator.allocate_gpu_blocks(4)
        assert len(blocks) == 4
        assert len(set(blocks)) == 4  # no duplicates
        assert allocator.num_free_gpu_blocks == 4

    def test_free_returns_to_pool(self, allocator):
        bid = allocator.allocate_gpu()
        allocator.free_gpu(bid)
        assert allocator.num_free_gpu_blocks == 8

    def test_free_double_raises(self, allocator):
        bid = allocator.allocate_gpu()
        allocator.free_gpu(bid)
        with pytest.raises(AssertionError):
            allocator.free_gpu(bid)

    def test_oom_raises(self, allocator):
        allocator.allocate_gpu_blocks(8)
        with pytest.raises(MemoryError):
            allocator.allocate_gpu()

    def test_can_allocate_check(self, allocator):
        assert allocator.can_allocate_gpu(8)
        assert not allocator.can_allocate_gpu(9)


class TestSwap:
    def test_swap_out_frees_gpu_allocates_cpu(self, allocator):
        gpu_blocks = allocator.allocate_gpu_blocks(3)
        mapping = allocator.swap_out(gpu_blocks)

        assert allocator.num_free_gpu_blocks == 8  # blocks freed
        assert allocator.num_free_cpu_blocks == 1  # 4 - 3 = 1
        assert set(mapping.keys()) == set(gpu_blocks)
        assert all(0 <= v < 4 for v in mapping.values())

    def test_swap_in_frees_cpu_allocates_gpu(self, allocator):
        gpu_blocks = allocator.allocate_gpu_blocks(3)
        out_map = allocator.swap_out(gpu_blocks)
        cpu_blocks = list(out_map.values())

        in_map = allocator.swap_in(cpu_blocks)
        assert allocator.num_free_cpu_blocks == 4  # all cpu blocks freed
        assert allocator.num_free_gpu_blocks == 5  # 8 - 3 = 5

    def test_round_trip_mapping(self, allocator):
        original = allocator.allocate_gpu_blocks(2)
        out_map = allocator.swap_out(original)
        cpu = list(out_map.values())
        in_map = allocator.swap_in(cpu)
        new_gpu = [in_map[c] for c in cpu]
        assert len(new_gpu) == 2
        assert all(0 <= g < 8 for g in new_gpu)


class TestRefCount:
    def test_ref_count_increments(self, allocator):
        bid = allocator.allocate_gpu()
        block = allocator.get_block(bid, "gpu")
        assert block.ref_count == 1

    def test_partial_free(self, allocator):
        bid = allocator.allocate_gpu()
        block = allocator.get_block(bid)
        block.ref_count += 1  # simulate shared reference
        allocator.free_gpu(bid)
        assert block.ref_count == 1  # still held
        assert allocator.num_free_gpu_blocks == 7  # not returned yet
        allocator.free_gpu(bid)
        assert allocator.num_free_gpu_blocks == 8
