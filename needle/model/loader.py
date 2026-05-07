"""Load weights from SafeTensors / HuggingFace checkpoint into model."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import torch

from .config import ModelConfig


# Weight name remapping: HF name -> LeanLLM name
_LLAMA_KEY_MAP = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    "model.norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}

# Layer-level pattern replacements
_LAYER_REPLACEMENTS = [
    ("model.layers.{i}.self_attn.q_proj.weight", "model.layers.{i}.self_attn.q_proj.weight"),
    ("model.layers.{i}.self_attn.k_proj.weight", "model.layers.{i}.self_attn.k_proj.weight"),
    ("model.layers.{i}.self_attn.v_proj.weight", "model.layers.{i}.self_attn.v_proj.weight"),
    ("model.layers.{i}.self_attn.o_proj.weight", "model.layers.{i}.self_attn.o_proj.weight"),
    ("model.layers.{i}.self_attn.q_proj.bias",   "model.layers.{i}.self_attn.q_proj.bias"),
    ("model.layers.{i}.self_attn.k_proj.bias",   "model.layers.{i}.self_attn.k_proj.bias"),
    ("model.layers.{i}.self_attn.v_proj.bias",   "model.layers.{i}.self_attn.v_proj.bias"),
    ("model.layers.{i}.mlp.gate_proj.weight",    "model.layers.{i}.mlp.gate_proj.weight"),
    ("model.layers.{i}.mlp.up_proj.weight",      "model.layers.{i}.mlp.up_proj.weight"),
    ("model.layers.{i}.mlp.down_proj.weight",    "model.layers.{i}.mlp.down_proj.weight"),
    ("model.layers.{i}.input_layernorm.weight",  "model.layers.{i}.input_layernorm.weight"),
    ("model.layers.{i}.post_attention_layernorm.weight", "model.layers.{i}.post_attention_layernorm.weight"),
]


def _shard_tensor(tensor: torch.Tensor, dim: int, tp_size: int, tp_rank: int) -> torch.Tensor:
    """Shard tensor along dim for tensor parallelism."""
    if tp_size == 1:
        return tensor
    chunk_size = tensor.shape[dim] // tp_size
    start = tp_rank * chunk_size
    return tensor.narrow(dim, start, chunk_size).contiguous()


def load_model(model_runner, model_path: str) -> None:
    """Load weights into model_runner.model from SafeTensors or PyTorch bin files."""
    path = Path(model_path)
    state_dict = _load_state_dict(path)
    _load_weights_into_model(
        model_runner.model,
        state_dict,
        model_runner.model_config,
        tp_size=model_runner.tp_size,
        tp_rank=model_runner.tp_rank,
    )


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """Auto-detect and load weights from safetensors or pytorch_model.bin."""
    # Try safetensors first (preferred)
    index_file = path / "model.safetensors.index.json"
    single_file = path / "model.safetensors"

    if single_file.exists():
        from safetensors.torch import load_file
        return load_file(str(single_file))

    if index_file.exists():
        import json
        from safetensors.torch import load_file

        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        # Collect unique shard files
        shard_files = sorted(set(weight_map.values()))
        state_dict: Dict[str, torch.Tensor] = {}
        for shard in shard_files:
            state_dict.update(load_file(str(path / shard)))
        return state_dict

    # Fall back to pytorch_model.bin
    bin_file = path / "pytorch_model.bin"
    if bin_file.exists():
        return torch.load(str(bin_file), map_location="cpu")

    # Sharded pytorch bins
    import glob
    bins = sorted(glob.glob(str(path / "pytorch_model-*.bin")))
    if bins:
        state_dict = {}
        for b in bins:
            state_dict.update(torch.load(b, map_location="cpu"))
        return state_dict

    raise FileNotFoundError(f"No weight files found in {path}")


def _load_weights_into_model(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    config: ModelConfig,
    tp_size: int = 1,
    tp_rank: int = 0,
) -> None:
    """Copy state_dict weights into model, sharding TP dimensions."""
    model_sd = model.state_dict()
    loaded = set()

    for name, param in model_sd.items():
        if name not in state_dict:
            continue
        tensor = state_dict[name].to(param.dtype)

        # Column parallel: shard output dim (dim 0)
        if any(
            name.endswith(sfx)
            for sfx in ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                        "gate_proj.weight", "up_proj.weight",
                        "q_proj.bias", "k_proj.bias", "v_proj.bias")
        ):
            tensor = _shard_tensor(tensor, dim=0, tp_size=tp_size, tp_rank=tp_rank)

        # Row parallel: shard input dim (dim 1)
        elif any(name.endswith(sfx) for sfx in ("o_proj.weight", "down_proj.weight")):
            tensor = _shard_tensor(tensor, dim=1, tp_size=tp_size, tp_rank=tp_rank)

        if tensor.shape != param.shape:
            raise ValueError(
                f"Shape mismatch for {name}: checkpoint {tensor.shape} vs model {param.shape}"
            )
        param.data.copy_(tensor)
        loaded.add(name)

    missing = set(model_sd.keys()) - loaded
    if missing:
        # Weight tying: lm_head.weight shares embed_tokens.weight in Qwen2/LLaMA
        if "lm_head.weight" in missing and "model.embed_tokens.weight" in state_dict:
            model.lm_head.weight.data.copy_(
                state_dict["model.embed_tokens.weight"].to(model.lm_head.weight.dtype)
            )
            missing.discard("lm_head.weight")

        # Warn about any remaining missing params (excluding non-persistent buffers)
        missing_params = [k for k in missing if "cached" not in k and "inv_freq" not in k]
        if missing_params:
            import warnings
            warnings.warn(f"Missing weights for: {missing_params[:10]}...")
