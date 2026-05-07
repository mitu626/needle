"""Parse model configuration from HuggingFace config.json."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    # Architecture
    model_type: str = "llama"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32      # GQA: may differ from num_attention_heads
    intermediate_size: int = 11008
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    hidden_act: str = "silu"

    # Derived
    head_dim: int = field(init=False)

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def is_gqa(self) -> bool:
        return self.num_key_value_heads != self.num_attention_heads

    @classmethod
    def from_pretrained(cls, model_path: str) -> "ModelConfig":
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")
        with open(config_path) as f:
            cfg = json.load(f)
        return cls._from_dict(cfg)

    @classmethod
    def _from_dict(cls, cfg: dict) -> "ModelConfig":
        model_type = cfg.get("model_type", "llama")
        return cls(
            model_type=model_type,
            hidden_size=cfg.get("hidden_size", 4096),
            num_hidden_layers=cfg.get("num_hidden_layers", 32),
            num_attention_heads=cfg.get("num_attention_heads", 32),
            num_key_value_heads=cfg.get(
                "num_key_value_heads", cfg.get("num_attention_heads", 32)
            ),
            intermediate_size=cfg.get("intermediate_size", 11008),
            vocab_size=cfg.get("vocab_size", 32000),
            max_position_embeddings=cfg.get("max_position_embeddings", 4096),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
            rope_theta=cfg.get("rope_theta", 10000.0),
            hidden_act=cfg.get("hidden_act", "silu"),
        )
