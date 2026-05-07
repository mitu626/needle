"""Integration test: tensor parallel output matches single-GPU output.

Requires 2+ GPUs and TEST_MODEL_PATH set.

Run with:
  TEST_MODEL_PATH=/path/to/model pytest tests/integration/test_tensor_parallel.py -v
"""
import os
import pytest
import torch

TEST_MODEL_PATH = os.environ.get("TEST_MODEL_PATH", "")

pytestmark = pytest.mark.skipif(
    not TEST_MODEL_PATH
    or not os.path.isdir(TEST_MODEL_PATH)
    or torch.cuda.device_count() < 2,
    reason="TEST_MODEL_PATH not set or < 2 GPUs available",
)


def run_single_gpu(prompt_ids, max_new_tokens=16):
    from needle.core.engine import LLMEngine
    from needle.core.sequence import SamplingParams

    engine = LLMEngine(
        model_path=TEST_MODEL_PATH,
        block_size=16,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
        dtype="bfloat16",
    )
    params = SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens)
    return engine.generate(prompt_ids, params)


def run_tensor_parallel(prompt_ids, tp_size=2, max_new_tokens=16):
    """Launch TP workers and get output from rank 0."""
    from needle.worker.worker import launch_workers
    from needle.core.block_allocator import BlockAllocator
    from needle.core.scheduler import Scheduler
    from needle.core.sequence import SamplingParams, Request, Sequence
    import time

    procs, cmd_queues, result_queue = launch_workers(
        model_path=TEST_MODEL_PATH,
        world_size=tp_size,
        block_size=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.4,
    )

    # Wait for worker init
    init_info = result_queue.get(timeout=120)

    from needle.core.engine import LLMEngine
    from needle.model.config import ModelConfig
    model_config = ModelConfig.from_pretrained(TEST_MODEL_PATH)

    allocator = BlockAllocator(
        num_gpu_blocks=init_info["num_gpu_blocks"],
        num_cpu_blocks=init_info["num_cpu_blocks"],
        block_size=16,
    )
    scheduler = Scheduler(allocator)

    req = Request(
        request_id="tp-test",
        prompt_token_ids=prompt_ids,
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens),
        arrival_time=time.time(),
    )
    num_blocks = (len(prompt_ids) + 15) // 16
    seq = Sequence(
        seq_id=1, request=req, token_ids=list(prompt_ids),
        logical_blocks=list(range(num_blocks)), block_size=16,
    )
    scheduler.add_sequence(seq)

    generated = []
    while scheduler.has_unfinished_seqs() and len(generated) < max_new_tokens:
        sched_out = scheduler.schedule()
        for q in cmd_queues:
            q.put(sched_out)
        out = result_queue.get(timeout=30)
        token = out.next_token_ids[0]
        generated.append(token)
        seq.append_token(token)
        params = seq.request.sampling_params
        if token in params.stop_token_ids or seq.num_generated_tokens >= params.max_new_tokens:
            scheduler.free_sequence(seq)
            break

    for q in cmd_queues:
        q.put(None)  # shutdown
    for p in procs:
        p.join(timeout=10)

    return generated


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TEST_MODEL_PATH)


def test_tp2_matches_single_gpu(tokenizer):
    prompt = "The meaning of life is"
    prompt_ids = tokenizer.encode(prompt)

    single_output = run_single_gpu(prompt_ids, max_new_tokens=16)
    tp2_output = run_tensor_parallel(prompt_ids, tp_size=2, max_new_tokens=16)

    assert single_output == tp2_output, (
        f"TP2 output mismatch!\nSingle: {single_output}\nTP2:    {tp2_output}"
    )
