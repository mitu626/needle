"""Backend process — scheduling + inference in one process.

Each TP rank runs run_backend_process() in its own subprocess.
All ranks run identical logic (symmetric design):
  - receive_msgs() synchronises messages across ranks via gloo broadcast
  - model forward uses NCCL AllReduce internally
  - rank=0 sends DetokenizeMsg results back to API Server via ZMQ
  - rank=1..N-1 send_result() is a no-op

Option-A: wraps the existing LLMEngine / Scheduler / BlockAllocator.
Overlap scheduling is a TODO pending data-structure refactoring (Option-B).
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Optional

import torch
import torch.distributed as dist

from .message import (
    AbortMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from .io import SchedulerIOMixin
from .sequence import SamplingParams


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchedulerConfig:
    model_path: str

    # TP distributed
    tp_rank: int = 0
    tp_size: int = 1
    distributed_addr: str = "tcp://127.0.0.1:29500"
    distributed_timeout: int = 300          # seconds

    # ZMQ addresses (this process binds all of them)
    zmq_backend_addr: str = "tcp://127.0.0.1:5555"    # PULL from API
    zmq_result_addr: str  = "tcp://127.0.0.1:5556"    # PUSH to API
    zmq_broadcast_addr: str = "tcp://127.0.0.1:5557"  # PUB to rank=1..N-1

    # Model / cache
    page_size: int = 16
    max_running_reqs: int = 256
    max_prefill_tokens: int = 8192
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.90

    # Scheduling
    disable_overlap: bool = False   # overlap TODO: enable after Option-B refactor


# ---------------------------------------------------------------------------
# Scheduler process
# ---------------------------------------------------------------------------

class BackendProc(SchedulerIOMixin):
    """Wraps LLMEngine with ZMQ I/O and TP coordination.

    Responsibilities:
      - Scheduling: continuous batching, KV cache allocation
      - Inference: model forward pass, sampling
      - I/O: ZMQ message recv/send, gloo broadcast for TP sync
    """

    def __init__(self, config: SchedulerConfig, cpu_group: dist.ProcessGroup) -> None:
        from .engine import LLMEngine

        self.config = config
        self.engine = LLMEngine(
            model_path=config.model_path,
            block_size=config.page_size,
            max_num_seqs=config.max_running_reqs,
            max_num_batched_tokens=config.max_prefill_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tp_size,
            dtype=config.dtype,
        )

        # SchedulerIOMixin sets up ZMQ sockets based on tp_rank
        SchedulerIOMixin.__init__(
            self, config, config.tp_rank, config.tp_size, cpu_group
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        try:
            # TODO: enable overlap after Option-B data-structure refactor
            self._normal_loop()
        except KeyboardInterrupt:
            pass

    def _normal_loop(self) -> None:
        while True:
            blocking = not self.engine.has_unfinished_requests()
            for msg in self.receive_msgs(blocking=blocking):
                self._process_msg(msg)

            replies = self._step()
            if replies:
                self.send_result(replies)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _process_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for m in msg.data:
                self._process_msg(m)
        elif isinstance(msg, UserMsg):
            self._handle_user_msg(msg)
        elif isinstance(msg, AbortMsg):
            self._handle_abort(msg.uid)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt

    def _handle_user_msg(self, msg: UserMsg) -> None:
        params_dict = msg.sampling_params
        sampling_params = SamplingParams(
            temperature=params_dict.get("temperature", 1.0),
            top_p=params_dict.get("top_p", 1.0),
            top_k=params_dict.get("top_k", -1),
            max_new_tokens=params_dict.get("max_new_tokens", 512),
        )
        self.engine.add_request(
            request_id=str(msg.uid),
            prompt_token_ids=msg.input_ids.tolist(),
            sampling_params=sampling_params,
        )

    def _handle_abort(self, uid: int) -> None:
        request_id = str(uid)
        scheduler = self.engine.scheduler
        for queue in (scheduler.waiting, scheduler.running, scheduler.swapped):
            seqs = list(queue)
            for seq in seqs:
                if seq.request.request_id == request_id:
                    scheduler.free_sequence(seq)
                    return

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _step(self) -> List[DetokenizeMsg]:
        sched_out = self.engine.scheduler.schedule()
        if sched_out.is_empty:
            return []

        execute_out = self.engine.model_runner.execute(sched_out)

        replies: List[DetokenizeMsg] = []
        all_seqs = sched_out.prefill_seqs + sched_out.decode_seqs

        for seq, token_id in zip(all_seqs, execute_out.next_token_ids):
            seq.append_token(token_id)
            params = seq.request.sampling_params
            finished = (
                int(token_id) in params.stop_token_ids
                or seq.num_generated_tokens >= params.max_new_tokens
            )
            if finished:
                self.engine.scheduler.free_sequence(seq)
            replies.append(DetokenizeMsg(
                uid=int(seq.request.request_id),
                next_token=int(token_id),
                finished=finished,
            ))

        return replies


# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------

def _init_distributed(config: SchedulerConfig) -> dist.ProcessGroup:
    """Initialise gloo process group for CPU-side message sync.

    Returns the CPU (gloo) group used by SchedulerIOMixin.
    NCCL AllReduce for GPU is handled inside ModelRunner via torch.distributed
    (init_process_group is called once with nccl when tp_size > 1, and a
    separate gloo group is created for CPU coordination).
    """
    if config.tp_size == 1:
        dist.init_process_group(
            backend="gloo",
            init_method=config.distributed_addr,
            world_size=1,
            rank=0,
            timeout=timedelta(seconds=config.distributed_timeout),
        )
        return dist.group.WORLD

    # Multi-rank: init nccl for GPU AllReduce (used by ModelRunner)
    dist.init_process_group(
        backend="nccl",
        init_method=config.distributed_addr,
        world_size=config.tp_size,
        rank=config.tp_rank,
        timeout=timedelta(seconds=config.distributed_timeout),
    )
    # Separate gloo group for CPU-side message count broadcast
    cpu_group = dist.new_group(
        ranks=list(range(config.tp_size)),
        backend="gloo",
        timeout=timedelta(seconds=config.distributed_timeout),
    )
    return cpu_group


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------

def run_backend_process(
    config: SchedulerConfig,
    pipe_writer: Optional[mp.connection.Connection] = None,
) -> None:
    """Entry point for each TP rank subprocess."""
    torch.cuda.set_device(config.tp_rank)

    cpu_group = _init_distributed(config)

    proc = BackendProc(config, cpu_group)

    if pipe_writer is not None:
        pipe_writer.send("ready")
        pipe_writer.close()

    proc.run_forever()

    dist.destroy_process_group()
