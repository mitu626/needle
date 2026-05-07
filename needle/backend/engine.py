"""LLMEngine — main loop that ties scheduler + worker together."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from .block_allocator import BlockAllocator
from .scheduler import Scheduler
from .sequence import Request, SamplingParams, Sequence, SequenceStatus
from ..utils.logging import get_logger

logger = get_logger(__name__)

_SEQ_COUNTER = 0


def _next_seq_id() -> int:
    global _SEQ_COUNTER
    _SEQ_COUNTER += 1
    return _SEQ_COUNTER


class LLMEngine:
    """Single-process engine (TP=1). For multi-GPU wrap with AsyncEngine."""

    def __init__(
        self,
        model_path: str,
        block_size: int = 16,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
    ):
        self.block_size = block_size
        self.tensor_parallel_size = tensor_parallel_size

        # Lazy imports to avoid circular deps at module load
        from ..model.config import ModelConfig
        from ..model.runner import ModelRunner
        from ..model.loader import load_model

        self.model_config = ModelConfig.from_pretrained(model_path)

        # Build worker / model runner
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            block_size=block_size,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
        )
        load_model(self.model_runner, model_path)

        # Profile GPU memory and determine block counts
        num_gpu_blocks, num_cpu_blocks = self.model_runner.profile_memory(
            gpu_memory_utilization
        )
        logger.info(
            "KV cache: %d GPU blocks, %d CPU blocks (block_size=%d)",
            num_gpu_blocks,
            num_cpu_blocks,
            block_size,
        )

        self.allocator = BlockAllocator(
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )
        self.scheduler = Scheduler(
            block_allocator=self.allocator,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )

        # Allocate KV cache tensors on the model runner
        self.model_runner.init_kv_cache(num_gpu_blocks, num_cpu_blocks)

        self._request_outputs: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
    ) -> None:
        req = Request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            arrival_time=time.time(),
        )
        seq_id = _next_seq_id()
        # Initial logical block allocation
        num_blocks = (len(prompt_token_ids) + self.block_size - 1) // self.block_size
        seq = Sequence(
            seq_id=seq_id,
            request=req,
            token_ids=list(prompt_token_ids),
            logical_blocks=list(range(num_blocks)),
            block_size=self.block_size,
        )
        self.scheduler.add_sequence(seq)

    def step(self) -> Dict[str, List[int]]:
        """Execute one scheduling + forward step. Returns finished outputs."""
        sched_out = self.scheduler.schedule()
        if sched_out.is_empty:
            return {}

        # Execute forward pass
        execute_out = self.model_runner.execute(sched_out)

        # Update sequences with new tokens
        finished: Dict[str, List[int]] = {}
        all_seqs = sched_out.prefill_seqs + sched_out.decode_seqs

        for seq, token_id in zip(all_seqs, execute_out.next_token_ids):
            seq.append_token(token_id)
            params = seq.request.sampling_params
            done = (
                token_id in params.stop_token_ids
                or seq.num_generated_tokens >= params.max_new_tokens
            )
            if done:
                self.scheduler.free_sequence(seq)
                finished[seq.request.request_id] = seq.token_ids[seq.prompt_len:]

        return finished

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_seqs()

    def generate(
        self,
        prompt_token_ids: List[int],
        sampling_params: Optional[SamplingParams] = None,
        request_id: str = "sync-0",
    ) -> List[int]:
        """Synchronous generation helper (single request)."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        self.add_request(request_id, prompt_token_ids, sampling_params)
        while self.has_unfinished_requests():
            outputs = self.step()
            if request_id in outputs:
                return outputs[request_id]
        return []
