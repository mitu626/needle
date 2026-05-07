"""Top-k / top-p / temperature sampler."""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from ..backend.sequence import SamplingParams


class Sampler:
    """Stateless sampler: given logits [num_seqs, vocab], returns next token ids."""

    @torch.no_grad()
    def sample(
        self,
        logits: torch.Tensor,
        sampling_params_list: List[SamplingParams],
    ) -> List[int]:
        assert logits.shape[0] == len(sampling_params_list)
        results = []

        for i, params in enumerate(sampling_params_list):
            logit = logits[i]

            if params.temperature == 0.0:
                results.append(int(logit.argmax()))
                continue

            logit = logit / params.temperature

            if params.top_k > 0:
                top_k = min(params.top_k, logit.size(-1))
                kth_val = logit.topk(top_k).values[-1]
                logit = logit.masked_fill(logit < kth_val, float("-inf"))

            if params.top_p < 1.0:
                sorted_logits, sorted_idx = logit.sort(descending=True)
                cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                remove = cumprobs - sorted_logits.softmax(dim=-1) > params.top_p
                sorted_logits[remove] = float("-inf")
                logit = logit.scatter(0, sorted_idx, sorted_logits)

            probs = F.softmax(logit, dim=-1)
            results.append(int(torch.multinomial(probs, num_samples=1)))

        return results
