"""Inter-process messages for LeanLLM.

Two message flows:
  API Server → Scheduler (rank=0):   UserMsg | AbortMsg | ExitMsg
  Scheduler  → API Server:           DetokenizeMsg | BatchDetokenizeMsg

Serialization: msgpack via serialize_type / deserialize_type,
matching mini-sglang's wire format convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_any(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    if isinstance(value, (int, float, str, bool, bytes, type(None))):
        return value
    return _serialize_obj(value)


def _serialize_obj(obj: Any) -> Dict:
    if isinstance(obj, torch.Tensor):
        assert obj.dim() == 1, "only 1-D tensors are supported"
        return {
            "__type__": "Tensor",
            "buffer": obj.numpy().tobytes(),
            "dtype": str(obj.dtype),
        }
    out: Dict = {"__type__": obj.__class__.__name__}
    for k, v in obj.__dict__.items():
        out[k] = _serialize_any(v)
    return out


def _deserialize_any(cls_map: Dict, data: Any) -> Any:
    if isinstance(data, dict):
        if "__type__" in data:
            return _deserialize_obj(cls_map, data)
        return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    return data


def _deserialize_obj(cls_map: Dict, data: Dict) -> Any:
    type_name = data["__type__"]
    if type_name == "Tensor":
        dtype_str = data["dtype"].replace("torch.", "")
        np_arr = np.frombuffer(data["buffer"], dtype=getattr(np, dtype_str))
        return torch.from_numpy(np_arr.copy())
    cls = cls_map[type_name]
    kwargs = {k: _deserialize_any(cls_map, v) for k, v in data.items() if k != "__type__"}
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# API → Scheduler messages
# ---------------------------------------------------------------------------

@dataclass
class BaseBackendMsg:
    def encode(self) -> Dict:
        return _serialize_obj(self)

    @staticmethod
    def decode(data: Dict) -> BaseBackendMsg:
        return _deserialize_obj(_BACKEND_CLS, data)


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor   # CPU 1-D int32
    sampling_params: dict     # flat dict, avoids circular import with SamplingParams


@dataclass
class AbortMsg(BaseBackendMsg):
    uid: int


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    """Wraps multiple messages into one ZMQ send for batching efficiency."""
    data: List[BaseBackendMsg]


_BACKEND_CLS = {
    cls.__name__: cls
    for cls in [UserMsg, AbortMsg, ExitMsg, BatchBackendMsg]
}


# ---------------------------------------------------------------------------
# Scheduler → API messages
# ---------------------------------------------------------------------------

@dataclass
class BaseApiMsg:
    def encode(self) -> Dict:
        return _serialize_obj(self)

    @staticmethod
    def decode(data: Dict) -> BaseApiMsg:
        return _deserialize_obj(_API_CLS, data)


@dataclass
class DetokenizeMsg(BaseApiMsg):
    uid: int
    next_token: int
    finished: bool


@dataclass
class BatchDetokenizeMsg(BaseApiMsg):
    data: List[DetokenizeMsg]


_API_CLS = {
    cls.__name__: cls
    for cls in [DetokenizeMsg, BatchDetokenizeMsg]
}
