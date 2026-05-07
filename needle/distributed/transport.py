"""ZMQ transport layer for LeanLLM inter-process communication.

Topology:
  API Server  ──PUSH──▶  Scheduler rank=0  ──PUSH──▶  API Server
                              │
                           PUB/SUB
                              │
                         rank=1..N-1

Socket ownership:
  addr_backend   : rank=0 binds PULL  /  API binds PUSH (connect)
  addr_result    : API binds PULL     /  rank=0 connects PUSH
  addr_broadcast : rank=0 binds PUB  /  rank=1..N-1 connect SUB
"""
from __future__ import annotations

import pickle
from typing import Callable, Dict, Generic, List, Optional, TypeVar

import msgpack
import numpy as np
import torch
import torch.distributed as dist
import zmq
import zmq.asyncio

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Low-level typed queues  (mirrors mini-sglang utils/mp.py)
# ---------------------------------------------------------------------------

class ZmqPushQueue(Generic[T]):
    def __init__(self, addr: str, *, bind: bool, encoder: Callable):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUSH)
        self._sock.bind(addr) if bind else self._sock.connect(addr)
        self._encoder = encoder

    def put(self, obj: T) -> None:
        self._sock.send(msgpack.packb(self._encoder(obj), use_bin_type=True), copy=False)

    def put_raw(self, raw: bytes) -> None:
        self._sock.send(raw, copy=False)

    def close(self) -> None:
        self._sock.close()
        self._ctx.term()


class ZmqPullQueue(Generic[T]):
    def __init__(self, addr: str, *, bind: bool, decoder: Callable):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PULL)
        self._sock.bind(addr) if bind else self._sock.connect(addr)
        self._decoder = decoder

    def get(self) -> T:
        return self._decoder(msgpack.unpackb(self._sock.recv(), raw=False))

    def get_raw(self) -> bytes:
        return self._sock.recv()

    def decode(self, raw: bytes) -> T:
        return self._decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        return self._sock.poll(timeout=0) == 0

    def close(self) -> None:
        self._sock.close()
        self._ctx.term()


class ZmqPubQueue(Generic[T]):
    def __init__(self, addr: str, *, bind: bool, encoder: Callable):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.bind(addr) if bind else self._sock.connect(addr)
        self._encoder = encoder

    def put_raw(self, raw: bytes) -> None:
        self._sock.send(raw, copy=False)

    def close(self) -> None:
        self._sock.close()
        self._ctx.term()


class ZmqSubQueue(Generic[T]):
    def __init__(self, addr: str, *, bind: bool, decoder: Callable):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.bind(addr) if bind else self._sock.connect(addr)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._decoder = decoder

    def get(self) -> T:
        return self._decoder(msgpack.unpackb(self._sock.recv(), raw=False))

    def empty(self) -> bool:
        return self._sock.poll(timeout=0) == 0

    def close(self) -> None:
        self._sock.close()
        self._ctx.term()


# ---------------------------------------------------------------------------
# Async variants for the API Server (uvicorn asyncio event loop)
# ---------------------------------------------------------------------------

class AsyncZmqPushQueue(Generic[T]):
    def __init__(self, addr: str, *, bind: bool, encoder: Callable):
        self._ctx = zmq.asyncio.Context()
        self._sock = self._ctx.socket(zmq.PUSH)
        self._sock.bind(addr) if bind else self._sock.connect(addr)
        self._encoder = encoder

    async def put(self, obj: T) -> None:
        await self._sock.send(
            msgpack.packb(self._encoder(obj), use_bin_type=True), copy=False
        )

    def close(self) -> None:
        self._sock.close()
        self._ctx.term()


class AsyncZmqPullQueue(Generic[T]):
    def __init__(self, addr: str, *, bind: bool, decoder: Callable):
        self._ctx = zmq.asyncio.Context()
        self._sock = self._ctx.socket(zmq.PULL)
        self._sock.bind(addr) if bind else self._sock.connect(addr)
        self._decoder = decoder

    async def get(self) -> T:
        return self._decoder(msgpack.unpackb(await self._sock.recv(), raw=False))

    def close(self) -> None:
        self._sock.close()
        self._ctx.term()


# ---------------------------------------------------------------------------
# broadcast_pyobj  (mirrors SGLang srt/utils/common.py)
#
# Broadcasts a list of arbitrary Python objects from src rank to all ranks
# in a torch.distributed process group, using CPU tensors (gloo backend).
# ---------------------------------------------------------------------------

def broadcast_pyobj(
    data: Optional[List],
    rank: int,
    cpu_group: dist.ProcessGroup,
    src: int = 0,
) -> List:
    """Broadcast Python objects via gloo (CPU).

    rank=src serialises *data* with pickle, broadcasts size then payload.
    Other ranks receive and deserialise.  Returns the list on all ranks.
    """
    device = torch.device("cpu")

    if rank == src:
        if not data:
            size_t = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(size_t, src=src, group=cpu_group)
            return data or []

        payload = pickle.dumps(data)
        size_t = torch.tensor([len(payload)], dtype=torch.long, device=device)
        data_t = torch.frombuffer(bytearray(payload), dtype=torch.uint8)

        dist.broadcast(size_t, src=src, group=cpu_group)
        dist.broadcast(data_t, src=src, group=cpu_group)
        return data
    else:
        size_t = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(size_t, src=src, group=cpu_group)
        n = int(size_t.item())
        if n == 0:
            return []
        data_t = torch.empty(n, dtype=torch.uint8, device=device)
        dist.broadcast(data_t, src=src, group=cpu_group)
        return pickle.loads(data_t.numpy().tobytes())
