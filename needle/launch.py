"""Multi-process launcher for LeanLLM.

Start order:
  1. Spawn TP rank=0..N-1 (Scheduler+Engine processes)
  2. Wait for all ranks to report "ready" via pipe
  3. Start API Server in main process (or spawn separately)

All TP ranks share the same SchedulerConfig except tp_rank.
rank=0 binds ZMQ sockets; rank=1..N-1 connect to rank=0's broadcast addr.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from typing import Optional

from .backend.backend import SchedulerConfig, run_backend_process
from .serving.server import serve
from .utils.logging import get_logger

logger = get_logger(__name__)


def launch(
    model_path: str,
    tp_size: int = 1,
    # ZMQ
    zmq_backend_addr: str = "tcp://127.0.0.1:5555",
    zmq_result_addr: str  = "tcp://127.0.0.1:5556",
    zmq_broadcast_addr: str = "tcp://127.0.0.1:5557",
    # Distributed rendezvous (for NCCL/gloo)
    distributed_addr: str = "tcp://127.0.0.1:29500",
    # Model / cache
    page_size: int = 16,
    max_running_reqs: int = 256,
    max_prefill_tokens: int = 8192,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.90,
    # HTTP
    host: str = "0.0.0.0",
    port: int = 8000,
    model_name: str = "leanllm",
) -> None:
    ctx = mp.get_context("spawn")
    procs = []
    readers = []

    for tp_rank in range(tp_size):
        config = SchedulerConfig(
            model_path=model_path,
            tp_rank=tp_rank,
            tp_size=tp_size,
            distributed_addr=distributed_addr,
            zmq_backend_addr=zmq_backend_addr,
            zmq_result_addr=zmq_result_addr,
            zmq_broadcast_addr=zmq_broadcast_addr,
            page_size=page_size,
            max_running_reqs=max_running_reqs,
            max_prefill_tokens=max_prefill_tokens,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        reader, writer = ctx.Pipe(duplex=False)
        p = ctx.Process(
            target=run_backend_process,
            args=(config, writer),
            daemon=True,
        )
        p.start()
        writer.close()   # parent doesn't write
        procs.append(p)
        readers.append(reader)

    # Wait for all ranks to be ready
    logger.info("Waiting for %d TP rank(s) to initialise...", tp_size)
    for rank, reader in enumerate(readers):
        msg = reader.recv()          # blocks until rank sends "ready"
        assert msg == "ready", f"rank {rank} sent unexpected: {msg}"
        reader.close()
        logger.info("rank %d ready", rank)

    logger.info("All ranks ready. Starting API Server on %s:%d", host, port)

    try:
        # API Server runs in the main process (uvicorn blocks here)
        serve(
            model_path=model_path,
            zmq_push_addr=zmq_backend_addr,
            zmq_pull_addr=zmq_result_addr,
            model_name=model_name,
            host=host,
            port=port,
        )
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=5)
