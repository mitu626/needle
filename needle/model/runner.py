"""ModelRunner: assembles ExecuteInput from SchedulerOutput and runs forward."""
from __future__ import annotations

from typing import List, Optional

import torch

from ..backend.sequence import ExecuteInput, ExecuteOutput, SchedulerOutput
from . import build_model
from .config import ModelConfig
from .sampler import Sampler


class ModelRunner:
    """Owns the model, KV caches, and drives one forward pass per step."""

    def __init__(
        self,
        model_config: ModelConfig,
        block_size: int = 16,
        dtype: str = "bfloat16",
        device: str = "cuda",
        tensor_parallel_size: int = 1,
        tp_rank: int = 0,
    ):
        self.model_config = model_config
        self.block_size = block_size
        self.device = device
        self.tp_size = tensor_parallel_size
        self.tp_rank = tp_rank
        self.dtype = getattr(torch, dtype)

        self.model = build_model(model_config, tp_size=tensor_parallel_size, tp_rank=tp_rank)
        self.model.to(device=device, dtype=self.dtype)
        self.model.eval()

        self.sampler = Sampler()

        # KV caches: list of [2, num_blocks, num_kv_heads, block_size, head_dim]
        self.kv_caches: List[torch.Tensor] = []

    # ------------------------------------------------------------------

    def profile_memory(self, gpu_memory_utilization: float = 0.90):
        """Estimate how many KV blocks fit in available GPU memory."""
        if self.device == "cpu":
            return 512, 512

        total_mem = torch.cuda.get_device_properties(0).total_memory
        # Conservative: subtract model weight footprint
        allocated = torch.cuda.memory_allocated()
        available = int((total_mem - allocated) * gpu_memory_utilization)

        cfg = self.model_config
        # bytes per block per layer: 2 (K+V) * num_kv_heads * block_size * head_dim * dtype_bytes
        dtype_bytes = 2 if self.dtype == torch.bfloat16 else 4
        bytes_per_block = (
            2 * cfg.num_kv_heads * self.block_size * cfg.head_dim * dtype_bytes
            * cfg.num_hidden_layers
        )
        num_gpu_blocks = max(1, available // bytes_per_block)
        num_cpu_blocks = num_gpu_blocks // 2  # half as swap buffer
        return num_gpu_blocks, num_cpu_blocks

    def init_kv_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """Allocate KV cache tensors for all layers."""
        cfg = self.model_config
        kv_shape = (
            2,                   # K, V
            num_gpu_blocks,
            cfg.num_kv_heads,
            self.block_size,
            cfg.head_dim,
        )
        self.kv_caches = [
            torch.zeros(kv_shape, dtype=self.dtype, device=self.device)
            for _ in range(cfg.num_hidden_layers)
        ]

    # ------------------------------------------------------------------

    def execute(self, sched_out: SchedulerOutput) -> ExecuteOutput:
        """Build ExecuteInput and run one model forward pass."""
        execute_input = self._prepare_inputs(sched_out)
        with torch.no_grad():
            logits = self.model(
                input_ids=execute_input.input_ids,
                positions=execute_input.position_ids,
                kv_caches=self.kv_caches,
                block_table=execute_input.block_table,
                cu_seqlens=execute_input.cu_seqlens,
                context_lens=execute_input.context_lens,
                max_seqlen=execute_input.max_seqlen,
                is_prefill=execute_input.is_prefill,
            )

        all_seqs = sched_out.prefill_seqs + sched_out.decode_seqs
        sampling_params = [seq.request.sampling_params for seq in all_seqs]
        next_token_ids = self.sampler.sample(logits, sampling_params)

        return ExecuteOutput(
            seq_ids=[s.seq_id for s in all_seqs],
            next_token_ids=next_token_ids,
        )

    def _prepare_inputs(self, sched_out: SchedulerOutput) -> ExecuteInput:
        """Convert SchedulerOutput into batched tensors."""
        all_seqs = sched_out.prefill_seqs + sched_out.decode_seqs
        is_prefill = len(sched_out.decode_seqs) == 0 or len(sched_out.prefill_seqs) > 0

        input_ids_list: List[int] = []
        position_ids_list: List[int] = []
        cu_seqlens_list: List[int] = [0]
        context_lens_list: List[int] = []

        for seq in all_seqs:
            if seq.num_generated_tokens == 0:
                # Prefill: feed all prompt tokens
                tokens = seq.token_ids
                ctx_len = 0
            else:
                # Decode: feed only the last generated token
                tokens = [seq.last_token_id]
                ctx_len = seq.length - 1

            input_ids_list.extend(tokens)
            start_pos = ctx_len
            position_ids_list.extend(range(start_pos, start_pos + len(tokens)))
            cu_seqlens_list.append(cu_seqlens_list[-1] + len(tokens))
            context_lens_list.append(ctx_len)

        # Build block table: [num_seqs, max_blocks_per_seq]
        max_blocks = max(
            len(sched_out.block_tables.get(s.seq_id, []))
            for s in all_seqs
        ) if all_seqs else 1

        block_table_data = []
        for seq in all_seqs:
            blocks = sched_out.block_tables.get(seq.seq_id, [])
            padded = blocks + [0] * (max_blocks - len(blocks))
            block_table_data.append(padded)

        dev = self.device
        return ExecuteInput(
            input_ids=torch.tensor(input_ids_list, dtype=torch.long, device=dev),
            position_ids=torch.tensor(position_ids_list, dtype=torch.long, device=dev),
            cu_seqlens=torch.tensor(cu_seqlens_list, dtype=torch.int32, device=dev),
            max_seqlen=max(
                (cu_seqlens_list[i + 1] - cu_seqlens_list[i])
                for i in range(len(all_seqs))
            ) if all_seqs else 1,
            block_table=torch.tensor(block_table_data, dtype=torch.int32, device=dev),
            context_lens=torch.tensor(context_lens_list, dtype=torch.int32, device=dev),
            is_prefill=is_prefill,
        )
