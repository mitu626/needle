"""FastAPI HTTP server — OpenAI-compatible /v1/chat/completions.

API Server is now a pure ZMQ client. It does not own the engine.
All generation requests are forwarded to the Scheduler process via ZMQ.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .protocol import (
    ChatCompletionChoice,
    ChatCompletionDelta,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    UsageInfo,
)
from .client import BackendClient
from ..utils.logging import get_logger
from ..utils.metrics import METRICS

logger = get_logger(__name__)

_client: Optional[BackendClient] = None
_tokenizer = None
_model_name: str = "leanllm"


def create_app(
    zmq_push_addr: str,
    zmq_pull_addr: str,
    model_path: str,
    model_name: str = "leanllm",
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        zmq_push_addr: address to PUSH requests to Scheduler (Scheduler binds).
        zmq_pull_addr: address to PULL results from Scheduler (Scheduler binds).
        model_path: HuggingFace model path (used only for tokenizer).
        model_name: model name reported in API responses.
    """
    global _client, _tokenizer, _model_name

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _client, _tokenizer, _model_name
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _model_name = model_name

        _client = BackendClient(
            push_addr=zmq_push_addr,
            pull_addr=zmq_pull_addr,
        )
        await _client.start()
        logger.info("LeanLLM API Server ready. Model: %s", model_name)

        yield

        await _client.stop()

    app = FastAPI(title="LeanLLM", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [{"id": _model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, http_req: Request):
        assert _client is not None and _tokenizer is not None

        # Tokenize
        prompt = _tokenizer.apply_chat_template(
            [m.model_dump() for m in body.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = _tokenizer.encode(prompt)
        sampling_params = {
            "temperature": body.temperature,
            "top_p": body.top_p,
            "top_k": body.top_k,
            "max_new_tokens": body.max_tokens,
        }

        try:
            if body.stream:
                return StreamingResponse(
                    _stream_chat(http_req, token_ids, sampling_params, body.model),
                    media_type="text/event-stream",
                )

            # Non-streaming
            start = time.perf_counter()
            rid, output_ids = await _client.generate(token_ids, sampling_params)
            output_text = _tokenizer.decode(output_ids, skip_special_tokens=True)
            elapsed = time.perf_counter() - start

            METRICS.record_request(
                prompt_tokens=len(token_ids),
                completion_tokens=len(output_ids),
                latency=elapsed,
            )
            return ChatCompletionResponse(
                id=rid,
                model=body.model,
                choices=[
                    ChatCompletionChoice(
                        message=ChatCompletionMessage(content=output_text),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=len(token_ids),
                    completion_tokens=len(output_ids),
                    total_tokens=len(token_ids) + len(output_ids),
                ),
            )

        except Exception:
            logger.exception("Error in /v1/chat/completions")
            raise HTTPException(status_code=500, detail="Internal server error")

    return app


async def _stream_chat(
    http_req: Request,
    token_ids: list,
    sampling_params: dict,
    model_name: str,
) -> AsyncIterator[str]:
    assert _client is not None and _tokenizer is not None

    # Pre-allocate a uid so we can abort on disconnect
    uid, rid = _client.alloc_uid()
    import torch
    from ..backend.message import UserMsg

    await _client._push.put(
        UserMsg(
            uid=uid,
            input_ids=torch.tensor(token_ids, dtype=torch.int32),
            sampling_params=sampling_params,
        )
    )
    q = _client._queues[uid]

    # Bug #11 fix: accumulate tokens and use incremental decode to avoid
    # UTF-8 split artefacts for multi-byte characters (e.g. Chinese).
    generated_ids: list = []
    prev_text = ""

    try:
        while True:
            # Check for client disconnect before each token
            if await http_req.is_disconnected():
                await _client.abort(uid)
                return

            token, finished = await q.get()
            generated_ids.append(token)

            # Decode the full sequence so far; only emit the new suffix.
            new_text = _tokenizer.decode(generated_ids, skip_special_tokens=True)
            delta = new_text[len(prev_text):]
            prev_text = new_text

            chunk = ChatCompletionStreamResponse(
                id=rid,
                model=model_name,
                choices=[
                    ChatCompletionStreamChoice(
                        delta=ChatCompletionDelta(content=delta),
                        finish_reason="stop" if finished else None,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            if finished:
                break
    finally:
        yield "data: [DONE]\n\n"


def serve(
    model_path: str,
    zmq_push_addr: str = "tcp://127.0.0.1:5555",
    zmq_pull_addr: str = "tcp://127.0.0.1:5556",
    model_name: str = "leanllm",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    app = create_app(zmq_push_addr, zmq_pull_addr, model_path, model_name)
    uvicorn.run(app, host=host, port=port, log_level="info")
