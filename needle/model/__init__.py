from .config import ModelConfig
from .llama import LlamaForCausalLM
from .qwen2 import Qwen2ForCausalLM

__all__ = ["ModelConfig", "LlamaForCausalLM", "Qwen2ForCausalLM", "build_model"]


def build_model(config: ModelConfig, tp_size: int = 1, tp_rank: int = 0):
    """Factory: instantiate the right model class from config.model_type."""
    model_type = config.model_type.lower()
    if model_type in ("llama",):
        return LlamaForCausalLM(config, tp_size=tp_size, tp_rank=tp_rank)
    elif model_type in ("qwen2",):
        return Qwen2ForCausalLM(config, tp_size=tp_size, tp_rank=tp_rank)
    else:
        raise ValueError(f"Unsupported model_type: {model_type!r}")
