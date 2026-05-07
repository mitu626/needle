"""Top-level LLM class for library-mode usage (no ZMQ / FastAPI).

Example:
    from needle import LLM

    llm = LLM("/path/to/model")
    print(llm.generate("你好"))

    for chunk in llm.stream("你好"):
        print(chunk, end="", flush=True)
"""
from __future__ import annotations

from typing import Generator, Iterator, List, Optional, Union

from .backend.engine import LLMEngine
from .backend.sequence import SamplingParams
from .utils.logging import get_logger

logger = get_logger(__name__)


class LLM:
    """High-level synchronous inference interface (TP=1, in-process).

    Wraps LLMEngine and a HuggingFace tokenizer. No subprocesses are
    spawned — all inference runs in the calling thread.
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.90,
        page_size: int = 16,
        max_running_requests: int = 256,
        max_prefill_tokens: int = 8192,
    ) -> None:
        from transformers import AutoTokenizer

        logger.info("Loading tokenizer from %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        logger.info("Initialising LLMEngine from %s", model_path)
        self._engine = LLMEngine(
            model_path=model_path,
            block_size=page_size,
            max_num_seqs=max_running_requests,
            max_num_batched_tokens=max_prefill_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            dtype=dtype,
        )

        # Collect eos / pad token ids for stop condition
        eos_id = self._tokenizer.eos_token_id
        self._default_stop_ids: List[int] = [eos_id] if eos_id is not None else []

        self._req_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: Union[str, List[int]],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop_token_ids: Optional[List[int]] = None,
    ) -> str:
        """Generate a completion for *prompt* and return the full text."""
        token_ids = self._encode(prompt)
        params = self._make_params(
            max_new_tokens, temperature, top_p, top_k, stop_token_ids
        )
        output_ids = self._engine.generate(
            token_ids, params, request_id=self._next_rid()
        )
        return self._tokenizer.decode(output_ids, skip_special_tokens=True)

    def batch_generate(
        self,
        prompts: List[Union[str, List[int]]],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop_token_ids: Optional[List[int]] = None,
    ) -> List[str]:
        """Generate completions for a batch of prompts."""
        params = self._make_params(
            max_new_tokens, temperature, top_p, top_k, stop_token_ids
        )
        rids: List[str] = []
        for prompt in prompts:
            rid = self._next_rid()
            rids.append(rid)
            self._engine.add_request(
                request_id=rid,
                prompt_token_ids=self._encode(prompt),
                sampling_params=params,
            )

        results: dict = {}
        while self._engine.has_unfinished_requests():
            finished = self._engine.step()
            results.update(finished)

        return [
            self._tokenizer.decode(results.get(rid, []), skip_special_tokens=True)
            for rid in rids
        ]

    def stream(
        self,
        prompt: Union[str, List[int]],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop_token_ids: Optional[List[int]] = None,
    ) -> Iterator[str]:
        """Yield decoded text chunks token-by-token.

        Uses an incremental decoding buffer to avoid UTF-8 split artefacts
        for multi-byte characters (e.g. Chinese).
        """
        token_ids = self._encode(prompt)
        params = self._make_params(
            max_new_tokens, temperature, top_p, top_k, stop_token_ids
        )
        rid = self._next_rid()
        self._engine.add_request(
            request_id=rid,
            prompt_token_ids=token_ids,
            sampling_params=params,
        )

        generated: List[int] = []
        prev_text = ""
        while self._engine.has_unfinished_requests():
            finished = self._engine.step()
            if rid in finished:
                # Flush remaining text on finish
                full_text = self._tokenizer.decode(
                    finished[rid], skip_special_tokens=True
                )
                tail = full_text[len(prev_text):]
                if tail:
                    yield tail
                return

            # Peek at the latest token via scheduler state (best-effort)
            # We reconstruct from the running sequence's token buffer.
            seq = self._find_running_seq(rid)
            if seq is None:
                continue
            new_ids = seq.token_ids[seq.prompt_len:]
            if len(new_ids) <= len(generated):
                continue
            generated = new_ids
            new_text = self._tokenizer.decode(generated, skip_special_tokens=True)
            chunk = new_text[len(prev_text):]
            if chunk:
                yield chunk
                prev_text = new_text

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _encode(self, prompt: Union[str, List[int]]) -> List[int]:
        if isinstance(prompt, list):
            return prompt
        return self._tokenizer.encode(prompt)

    def _make_params(
        self,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_token_ids: Optional[List[int]],
    ) -> SamplingParams:
        stops = list(self._default_stop_ids)
        if stop_token_ids:
            for s in stop_token_ids:
                if s not in stops:
                    stops.append(s)
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stops,
        )

    def _next_rid(self) -> str:
        self._req_counter += 1
        return f"llm-{self._req_counter}"

    def _find_running_seq(self, request_id: str):
        scheduler = self._engine.scheduler
        for queue in (scheduler.running, scheduler.waiting, scheduler.swapped):
            for seq in queue:
                if seq.request.request_id == request_id:
                    return seq
        return None
