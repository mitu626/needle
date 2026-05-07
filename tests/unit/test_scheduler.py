"""Unit tests for the Continuous Batching Scheduler."""
import pytest
from needle.core.block_allocator import BlockAllocator
from needle.core.scheduler import Scheduler
from needle.core.sequence import Request, SamplingParams, Sequence, SequenceStatus


def make_seq(seq_id: int, prompt_len: int = 8, block_size: int = 16) -> Sequence:
    prompt = list(range(prompt_len))
    req = Request(
        request_id=f"req-{seq_id}",
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(max_new_tokens=32),
    )
    num_blocks = (prompt_len + block_size - 1) // block_size
    return Sequence(
        seq_id=seq_id,
        request=req,
        token_ids=list(prompt),
        logical_blocks=list(range(num_blocks)),
        block_size=block_size,
    )


@pytest.fixture
def scheduler():
    allocator = BlockAllocator(num_gpu_blocks=16, num_cpu_blocks=8, block_size=16)
    return Scheduler(allocator, max_num_seqs=4, max_num_batched_tokens=512)


class TestBasicScheduling:
    def test_waiting_to_running(self, scheduler):
        seq = make_seq(1)
        scheduler.add_sequence(seq)
        out = scheduler.schedule()
        assert seq in out.prefill_seqs
        assert seq.status == SequenceStatus.RUNNING

    def test_prefill_then_decode(self, scheduler):
        seq = make_seq(1)
        scheduler.add_sequence(seq)

        # First step: prefill
        out = scheduler.schedule()
        assert seq in out.prefill_seqs

        # Simulate token generation
        seq.num_generated_tokens = 1
        seq.token_ids.append(100)

        # Second step: decode
        out2 = scheduler.schedule()
        assert seq in out2.decode_seqs

    def test_max_seqs_limit(self, scheduler):
        seqs = [make_seq(i) for i in range(6)]
        for s in seqs:
            scheduler.add_sequence(s)

        out = scheduler.schedule()
        running = out.prefill_seqs + out.decode_seqs
        assert len(running) <= 4  # max_num_seqs

    def test_block_table_populated(self, scheduler):
        seq = make_seq(1)
        scheduler.add_sequence(seq)
        out = scheduler.schedule()
        assert seq.seq_id in out.block_tables
        assert len(out.block_tables[seq.seq_id]) >= 1

    def test_free_sequence(self, scheduler):
        seq = make_seq(1)
        scheduler.add_sequence(seq)
        scheduler.schedule()
        scheduler.free_sequence(seq)
        assert seq.status == SequenceStatus.FINISHED
        assert not scheduler.has_unfinished_seqs()


class TestPreemption:
    def test_swap_out_when_oom(self):
        # Very tight GPU budget: only 2 blocks total
        allocator = BlockAllocator(num_gpu_blocks=2, num_cpu_blocks=4, block_size=16)
        sched = Scheduler(allocator, max_num_seqs=4, max_num_batched_tokens=512)

        seq1 = make_seq(1, prompt_len=8)  # needs 1 block
        seq2 = make_seq(2, prompt_len=8)  # needs 1 block — fills GPU

        sched.add_sequence(seq1)
        sched.add_sequence(seq2)
        out = sched.schedule()

        # Both should be scheduled (exactly 2 blocks needed)
        running = out.prefill_seqs + out.decode_seqs
        assert len(running) == 2

        # Adding a third forces a swap
        seq3 = make_seq(3, prompt_len=8)
        sched.add_sequence(seq3)
        out2 = sched.schedule()
        # One was swapped out
        total_active = len(out2.prefill_seqs) + len(out2.decode_seqs)
        assert total_active <= 2

    def test_empty_schedule_when_no_seqs(self, scheduler):
        out = scheduler.schedule()
        assert out.is_empty

    def test_has_unfinished_false_when_empty(self, scheduler):
        assert not scheduler.has_unfinished_seqs()
