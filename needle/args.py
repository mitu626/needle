"""Command-line argument parsing for Needle.

Usage:
    python -m needle --model /path/to/model [options]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ServerArgs:
    model_path: str

    # HTTP server
    host: str = "0.0.0.0"
    port: int = 8000

    # Tensor parallelism
    tp_size: int = 1

    # Model
    dtype: str = "bfloat16"

    # KV cache
    gpu_memory_utilization: float = 0.90
    page_size: int = 16

    # Scheduler
    max_running_requests: int = 256
    max_prefill_tokens: int = 8192

    # API
    model_name: str = "needle"


def parse_args(args: List[str]) -> ServerArgs:
    parser = argparse.ArgumentParser(
        prog="needle",
        description="Needle — sharp, lightweight LLM inference engine.",
    )

    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        type=str,
        required=True,
        help="Path to model weights (local directory or HuggingFace repo ID).",
    )

    parser.add_argument(
        "--host",
        type=str,
        default=ServerArgs.host,
        help=f"HTTP server host (default: {ServerArgs.host}).",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=ServerArgs.port,
        help=f"HTTP server port (default: {ServerArgs.port}).",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        dest="tp_size",
        type=int,
        default=ServerArgs.tp_size,
        help=f"Tensor-parallel size / number of GPUs (default: {ServerArgs.tp_size}).",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default=ServerArgs.dtype,
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model weight dtype. 'auto' reads from model config (default: bfloat16).",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        "--memory-ratio",
        dest="gpu_memory_utilization",
        type=float,
        default=ServerArgs.gpu_memory_utilization,
        help=f"Fraction of GPU memory to use for KV cache (default: {ServerArgs.gpu_memory_utilization}).",
    )

    parser.add_argument(
        "--page-size",
        dest="page_size",
        type=int,
        default=ServerArgs.page_size,
        help=f"KV cache page size in tokens (default: {ServerArgs.page_size}).",
    )

    parser.add_argument(
        "--max-running-requests",
        dest="max_running_requests",
        type=int,
        default=ServerArgs.max_running_requests,
        help=f"Maximum concurrent running sequences (default: {ServerArgs.max_running_requests}).",
    )

    parser.add_argument(
        "--max-prefill-tokens",
        "--max-extend-length",
        dest="max_prefill_tokens",
        type=int,
        default=ServerArgs.max_prefill_tokens,
        help=f"Max prefill tokens per step (default: {ServerArgs.max_prefill_tokens}).",
    )

    parser.add_argument(
        "--model-name",
        dest="model_name",
        type=str,
        default=ServerArgs.model_name,
        help=f"Model name reported in API responses (default: {ServerArgs.model_name}).",
    )

    kwargs = parser.parse_args(args).__dict__

    # Resolve ~ in path
    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    # Validate model path
    if not os.path.exists(kwargs["model_path"]):
        print(
            f"[needle] error: model path does not exist: {kwargs['model_path']}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve dtype=auto
    if kwargs["dtype"] == "auto":
        try:
            import json
            config_file = os.path.join(kwargs["model_path"], "config.json")
            with open(config_file) as f:
                cfg = json.load(f)
            torch_dtype = cfg.get("torch_dtype", "bfloat16")
            kwargs["dtype"] = torch_dtype if torch_dtype in {"float16", "bfloat16", "float32"} else "bfloat16"
        except Exception:
            kwargs["dtype"] = "bfloat16"

    return ServerArgs(**kwargs)
