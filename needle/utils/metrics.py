"""Prometheus metrics for LeanLLM serving."""
from __future__ import annotations

import time
from typing import Optional


class _Metrics:
    """Wraps prometheus_client counters/histograms with a no-op fallback."""

    def __init__(self):
        self._enabled = False
        try:
            from prometheus_client import Counter, Histogram, Gauge, start_http_server
            self._enabled = True

            self.request_counter = Counter(
                "leanllm_requests_total", "Total number of requests"
            )
            self.token_counter = Counter(
                "leanllm_tokens_total", "Total tokens generated",
                ["type"]  # "prompt" | "completion"
            )
            self.latency_histogram = Histogram(
                "leanllm_request_latency_seconds",
                "End-to-end request latency",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            )
            self.throughput_gauge = Gauge(
                "leanllm_throughput_tokens_per_sec",
                "Rolling tokens/sec throughput",
            )
            self._start_http_server = start_http_server
        except ImportError:
            pass

        self._window_tokens = 0
        self._window_start = time.time()

    def record_request(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
    ) -> None:
        if not self._enabled:
            return
        self.request_counter.inc()
        self.token_counter.labels("prompt").inc(prompt_tokens)
        self.token_counter.labels("completion").inc(completion_tokens)
        self.latency_histogram.observe(latency)

        # Update rolling throughput
        self._window_tokens += completion_tokens
        elapsed = time.time() - self._window_start
        if elapsed >= 10.0:
            tps = self._window_tokens / elapsed
            self.throughput_gauge.set(tps)
            self._window_tokens = 0
            self._window_start = time.time()

    def start_server(self, port: int = 9090) -> None:
        if self._enabled:
            self._start_http_server(port)


METRICS = _Metrics()
