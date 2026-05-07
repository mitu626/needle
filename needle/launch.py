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
import socket
import sys
import time
from typing import Optional

from .backend.backend import SchedulerConfig, run_backend_process
from .serving.server import serve
from .utils.logging import get_logger

logger = get_logger(__name__)


def _find_free_port() -> int:
    """Return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _find_free_ports(n: int) -> list:
    """Return n distinct free TCP ports, keeping sockets open until all are selected."""
    socks = []
    ports = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ports.append(s.getsockname()[1])
        socks.append(s)
    for s in socks:
        s.close()
    return ports


def launch(
    model_path: str,
    tp_size: int = 1,
    # ZMQ — auto-select free ports if None
    zmq_backend_addr: Optional[str] = None,
    zmq_result_addr: Optional[str]  = None,
    zmq_broadcast_addr: Optional[str] = None,
    # Distributed rendezvous (for NCCL/gloo) — auto-select free port if None
    distributed_addr: Optional[str] = None,
    # GPU visibility (e.g. "0,1") — empty string means inherit CUDA_VISIBLE_DEVICES
    cuda_visible_devices: str = "",
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
    if distributed_addr is None or zmq_backend_addr is None or \
            zmq_result_addr is None or zmq_broadcast_addr is None:
        ports = _find_free_ports(4)
        if distributed_addr is None:
            distributed_addr = f"tcp://127.0.0.1:{ports[0]}"
        if zmq_backend_addr is None:
            zmq_backend_addr = f"tcp://127.0.0.1:{ports[1]}"
        if zmq_result_addr is None:
            zmq_result_addr = f"tcp://127.0.0.1:{ports[2]}"
        if zmq_broadcast_addr is None:
            zmq_broadcast_addr = f"tcp://127.0.0.1:{ports[3]}"
    logger.info(
        "Ports — dist:%s zmq_backend:%s zmq_result:%s zmq_bcast:%s",
        distributed_addr, zmq_backend_addr, zmq_result_addr, zmq_broadcast_addr,
    )
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
            cuda_visible_devices=cuda_visible_devices,
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
