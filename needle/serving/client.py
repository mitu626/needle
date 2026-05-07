"""BackendClient — API Server's interface to the Scheduler backend.

Transport is an implementation detail (currently ZMQ); this class exposes
only generation semantics:

  generate / generate_stream
      │
      ├─ send UserMsg(uid, token_ids, sampling_params)  ──▶ Scheduler
      └─ wait on asyncio.Queue keyed by uid
              ◀── recv DetokenizeMsg(uid, next_token, finished)

A single background task (_result_loop) receives all DetokenizeMsgs and
routes each token to the right per-request Queue.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import AsyncIterator, Dict, List, Optional, Tuple

import torch

from ..backend.message import (
    AbortMsg,
    BaseApiMsg,
    BatchDetokenizeMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from ..distributed.transport import AsyncZmqPullQueue, AsyncZmqPushQueue


class BackendClient:
    """Async ZMQ client for the API Server process.

    Args:
        push_addr: address to CONNECT and PUSH UserMsg/AbortMsg to Scheduler.
        pull_addr: address to CONNECT and PULL DetokenizeMsg from Scheduler.
    """

    def __init__(self, push_addr: str, pull_addr: str) -> None:
        self._push: AsyncZmqPushQueue = AsyncZmqPushQueue(
            push_addr, bind=False, encoder=lambda m: m.encode()
        )
        self._pull: AsyncZmqPullQueue = AsyncZmqPullQueue(
            pull_addr, bind=True, decoder=BaseApiMsg.decode
        )
        # uid → (token_queue, request_id)
        self._queues: Dict[int, asyncio.Queue] = {}
        self._uid_to_rid: Dict[int, str] = {}
        self._uid_counter: int = 0
        self._loop_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._loop_task = asyncio.create_task(self._result_loop())

    async def stop(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._push.close()
        self._pull.close()

    # ------------------------------------------------------------------
    # uid helpers
    # ------------------------------------------------------------------

    def alloc_uid(self, request_id: Optional[str] = None) -> Tuple[int, str]:
        """Return (uid: int, request_id: str). Allocates a new uid."""
        self._uid_counter += 1
        uid = self._uid_counter
        rid = request_id or f"chatcmpl-{uuid.uuid4().hex}"
        self._uid_to_rid[uid] = rid
        q: asyncio.Queue = asyncio.Queue()
        self._queues[uid] = q
        return uid, rid

    def _release_uid(self, uid: int) -> None:
        self._queues.pop(uid, None)
        self._uid_to_rid.pop(uid, None)

    # ------------------------------------------------------------------
    # Background result dispatch loop
    # ------------------------------------------------------------------

    async def _result_loop(self) -> None:
        while True:
            msg = await self._pull.get()
            msgs: List[DetokenizeMsg] = (
                msg.data if isinstance(msg, BatchDetokenizeMsg) else [msg]
            )
            for m in msgs:
                q = self._queues.get(m.uid)
                if q is not None:
                    await q.put((m.next_token, m.finished))
                    if m.finished:
                        self._release_uid(m.uid)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        token_ids: List[int],
        sampling_params: dict,
        request_id: Optional[str] = None,
    ) -> Tuple[str, List[int]]:
        """Non-streaming generation. Returns (request_id, output_token_ids)."""
        uid, rid = self.alloc_uid(request_id)
        await self._push.put(
            UserMsg(
                uid=uid,
                input_ids=torch.tensor(token_ids, dtype=torch.int32),
                sampling_params=sampling_params,
            )
        )
        q = self._queues[uid]
        output: List[int] = []
        while True:
            token, finished = await q.get()
            output.append(token)
            if finished:
                return rid, output

    async def generate_stream(
        self,
        token_ids: List[int],
        sampling_params: dict,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[Tuple[str, int, bool]]:
        """Streaming generation. Yields (request_id, token_id, finished)."""
        uid, rid = self.alloc_uid(request_id)
        await self._push.put(
            UserMsg(
                uid=uid,
                input_ids=torch.tensor(token_ids, dtype=torch.int32),
                sampling_params=sampling_params,
            )
        )
        q = self._queues[uid]
        while True:
            token, finished = await q.get()
            yield rid, token, finished
            if finished:
                return

    async def abort(self, uid: int) -> None:
        """Cancel an in-flight request."""
        await self._push.put(AbortMsg(uid=uid))
        self._release_uid(uid)

    async def shutdown(self) -> None:
        """Ask Scheduler to exit."""
        await self._push.put(ExitMsg())
