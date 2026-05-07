"""OpenAI-compatible API schema (pydantic v2)."""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Request schemas
# ------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = Field(default=-1, description="Top-k sampling; -1 = disabled")
    max_tokens: int = Field(default=512, alias="max_new_tokens")
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    n: int = 1  # only n=1 supported

    model_config = {"populate_by_name": True}


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 512
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


# ------------------------------------------------------------------
# Response schemas
# ------------------------------------------------------------------

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# Streaming delta
class ChatCompletionDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]


# Error response
class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    code: Optional[int] = None
