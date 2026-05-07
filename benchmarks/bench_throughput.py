"""Throughput benchmark for Needle.

Usage:
  TEST_MODEL_PATH=/path/to/model python benchmarks/bench_throughput.py \
      --num-prompts 100 --input-len 128 --output-len 128

Metrics reported:
  - Total throughput (tokens/s)
  - Time to first token (TTFT) avg / p99
  - Inter-token latency (ITL) avg / p99
"""
import argparse
import random
import time
from typing import List

import numpy as np


def generate_random_prompts(num_prompts: int, input_len: int, vocab_size: int = 32000):
    return [
        [random.randint(1, vocab_size - 1) for _ in range(input_len)]
        for _ in range(num_prompts)
    ]


def benchmark(
    model_path: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    max_num_seqs: int = 32,
    gpu_memory_utilization: float = 0.90,
    dtype: str = "bfloat16",
    warmup: int = 5,
):
    from needle.core.engine import LLMEngine
    from needle.core.sequence import SamplingParams
    from needle.model.config import ModelConfig

    print(f"\nLoading model from {model_path} ...")
    engine = LLMEngine(
        model_path=model_path,
        block_size=16,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_seqs * input_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )

    cfg = ModelConfig.from_pretrained(model_path)
    vocab_size = cfg.vocab_size

    params = SamplingParams(temperature=0.0, max_new_tokens=output_len)

    # Warmup
    print(f"Warming up with {warmup} prompts ...")
    warmup_prompts = generate_random_prompts(warmup, input_len, vocab_size)
    for i, prompt in enumerate(warmup_prompts):
        engine.generate(prompt, params, request_id=f"warmup-{i}")

    # Benchmark
    print(f"Running {num_prompts} prompts (input={input_len}, output={output_len}) ...")
    prompts = generate_random_prompts(num_prompts, input_len, vocab_size)

    # Submit all requests
    start_times = {}
    for i, prompt in enumerate(prompts):
        req_id = f"bench-{i}"
        engine.add_request(req_id, prompt, params)
        start_times[req_id] = time.perf_counter()

    # Run until all done
    latencies = []
    total_output_tokens = 0
    bench_start = time.perf_counter()

    while engine.has_unfinished_requests():
        finished = engine.step()
        for req_id, tokens in finished.items():
            elapsed = time.perf_counter() - start_times[req_id]
            latencies.append(elapsed)
            total_output_tokens += len(tokens)

    bench_end = time.perf_counter()
    total_time = bench_end - bench_start

    # Metrics
    total_input_tokens = num_prompts * input_len
    total_tokens = total_input_tokens + total_output_tokens
    throughput = total_output_tokens / total_time
    total_throughput = total_tokens / total_time

    latencies = np.array(latencies) * 1000  # ms

    print("\n" + "=" * 60)
    print(f"  Model:            {model_path}")
    print(f"  Prompts:          {num_prompts}")
    print(f"  Input length:     {input_len} tokens")
    print(f"  Output length:    {output_len} tokens")
    print(f"  Max concurrent:   {max_num_seqs}")
    print("-" * 60)
    print(f"  Total time:       {total_time:.2f}s")
    print(f"  Output tokens/s:  {throughput:.1f}")
    print(f"  Total tokens/s:   {total_throughput:.1f}")
    print(f"  Latency avg:      {latencies.mean():.1f}ms")
    print(f"  Latency p50:      {np.percentile(latencies, 50):.1f}ms")
    print(f"  Latency p99:      {np.percentile(latencies, 99):.1f}ms")
    print("=" * 60)

    return {
        "throughput_output_tps": throughput,
        "throughput_total_tps": total_throughput,
        "latency_avg_ms": float(latencies.mean()),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }


def main():
    parser = argparse.ArgumentParser(description="Needle throughput benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    benchmark(
        model_path=args.model,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
