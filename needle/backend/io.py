"""Scheduler I/O mixin — message receive/send for all TP ranks.

Design mirrors mini-sglang scheduler/io.py + SGLang recv_requests():

  rank=0:
    - PULL from API Server (ZMQ)
    - PUB broadcast raw bytes to rank=1..N-1 (ZMQ PUB/SUB)
    - sync message count to all ranks via gloo broadcast
    - PUSH DetokenizeMsg back to API Server

  rank=1..N-1:
    - SUB from rank=0 broadcast (ZMQ)
    - receive message count via gloo broadcast
    - send_result is a no-op

Single-rank (TP=1) path skips all distributed logic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

from .message import BaseBackendMsg, BatchDetokenizeMsg, DetokenizeMsg
from ..distributed.transport import (
    ZmqPubQueue,
    ZmqPullQueue,
    ZmqPushQueue,
    ZmqSubQueue,
)

if TYPE_CHECKING:
    from .backend import SchedulerConfig


class SchedulerIOMixin:
    """Mixin that provides receive_msgs / send_result to the Scheduler.

    Must be initialised after the subclass __init__ via:
        SchedulerIOMixin.__init__(self, config, tp_rank, tp_size, cpu_group)
    """

    def __init__(
        self,
        config: SchedulerConfig,
        tp_rank: int,
        tp_size: int,
        cpu_group: dist.ProcessGroup,
    ) -> None:
        self._tp_rank = tp_rank
        self._tp_size = tp_size
        self._cpu_group = cpu_group

        if tp_size == 1:
            self._pull = ZmqPullQueue(
                config.zmq_backend_addr, bind=True, decoder=BaseBackendMsg.decode
            )
            self._push = ZmqPushQueue(
                config.zmq_result_addr, bind=False, encoder=lambda m: m.encode()
            )
            self.receive_msgs = self._recv_single
            self.send_result = self._send_rank0

        elif tp_rank == 0:
            self._pull = ZmqPullQueue(
                config.zmq_backend_addr, bind=True, decoder=BaseBackendMsg.decode
            )
            self._push = ZmqPushQueue(
                config.zmq_result_addr, bind=False, encoder=lambda m: m.encode()
            )
            self._pub = ZmqPubQueue(
                config.zmq_broadcast_addr, bind=True, encoder=lambda m: m.encode()
            )
            self.receive_msgs = self._recv_rank0
            self.send_result = self._send_rank0

        else:
            self._sub = ZmqSubQueue(
                config.zmq_broadcast_addr, bind=False, decoder=BaseBackendMsg.decode
            )
            self.receive_msgs = self._recv_rankN
            self.send_result = self._send_rankN

    # ------------------------------------------------------------------
    # TP=1 path
    # ------------------------------------------------------------------

    def _recv_single(self, blocking: bool = False) -> List[BaseBackendMsg]:
        msgs: List[BaseBackendMsg] = []
        if blocking:
            msgs.append(self._pull.get())
        while not self._pull.empty():
            msgs.append(self._pull.get())
        return msgs

    # ------------------------------------------------------------------
    # rank=0 path
    # ------------------------------------------------------------------

    def _recv_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        msgs: List[BaseBackendMsg] = []
        pending_raw: List[bytes] = []

        if blocking:
            raw = self._pull.get_raw()
            self._pub.put_raw(raw)          # forward immediately to SUB peers
            msgs.append(self._pull.decode(raw))

        while not self._pull.empty():
            pending_raw.append(self._pull.get_raw())

        # sync count with rank=1..N-1
        count_t = torch.tensor([len(pending_raw)], dtype=torch.long)
        dist.broadcast(count_t, src=0, group=self._cpu_group)

        for raw in pending_raw:
            self._pub.put_raw(raw)
            msgs.append(self._pull.decode(raw))

        return msgs

    # ------------------------------------------------------------------
    # rank=1..N-1 path
    # ------------------------------------------------------------------

    def _recv_rankN(self, blocking: bool = False) -> List[BaseBackendMsg]:
        msgs: List[BaseBackendMsg] = []

        if blocking:
            msgs.append(self._sub.get())

        count_t = torch.tensor([-1], dtype=torch.long)
        dist.broadcast(count_t, src=0, group=self._cpu_group)
        n = int(count_t.item())

        for _ in range(n):
            msgs.append(self._sub.get())

        return msgs

    # ------------------------------------------------------------------
    # send_result
    # ------------------------------------------------------------------

    def _send_rank0(self, replies: List[DetokenizeMsg]) -> None:
        if not replies:
            return
        msg: BaseBackendMsg = (
            replies[0] if len(replies) == 1 else BatchDetokenizeMsg(data=replies)
        )
        self._push.put(msg)

    def _send_rankN(self, replies: List[DetokenizeMsg]) -> None:
        pass  # no-op for non-primary ranks

    # ------------------------------------------------------------------
    # Barrier
    # ------------------------------------------------------------------

    def sync_all_ranks(self) -> None:
        if self._tp_size > 1:
            dist.barrier(group=self._cpu_group)
