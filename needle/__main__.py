"""Entry point for `python -m needle`."""
from __future__ import annotations

import os
import sys

from .args import parse_args
from .launch import launch
from .utils.logging import get_logger

logger = get_logger(__name__)


def launch_server() -> None:
    server_args = parse_args(sys.argv[1:])
    logger.info("Starting Needle server with args:\n%s", server_args)

    launch(
        model_path=server_args.model_path,
        tp_size=server_args.tp_size,
        dtype=server_args.dtype,
        gpu_memory_utilization=server_args.gpu_memory_utilization,
        max_prefill_tokens=server_args.max_prefill_tokens,
        page_size=server_args.page_size,
        max_running_reqs=server_args.max_running_requests,
        model_name=server_args.model_name,
        host=server_args.host,
        port=server_args.port,
        # Pass CUDA_VISIBLE_DEVICES through to subprocess (spawn doesn't inherit shell env)
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )


if __name__ == "__main__":
    launch_server()
