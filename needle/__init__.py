"""Needle — sharp, lightweight LLM inference engine."""
from .backend import LLMEngine, SamplingParams
from .llm import LLM
from .serving import serve

__version__ = "0.1.0"
__all__ = ["LLM", "LLMEngine", "SamplingParams", "serve"]
