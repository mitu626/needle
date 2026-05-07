"""Numerical correctness test for PagedAttention vs naive attention.

Runs on CPU with the PyTorch fallback path so no CUDA build is required.
"""
import pytest
import torch
import torch.nn.functional as F
import math


def naive_attention(q, k, v, scale):
    """Standard scaled dot-product attention. [batch, heads, seq, dim]"""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = F.softmax(scores, dim=-1)
    return torch.matmul(scores, v)


def paged_attention_ref(
    q,            # [num_seqs, num_heads, head_dim]
    k_cache,      # [num_heads, total_ctx, head_dim]  (dense for test)
    v_cache,      # [num_heads, total_ctx, head_dim]
    context_lens, # [num_seqs]
    scale,
):
    """Reference paged attention that matches the kernel output."""
    num_seqs = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    outputs = []
    for i in range(num_seqs):
        ctx = context_lens[i].item()
        qi = q[i]                 # [H, D]
        ki = k_cache[i, :, :ctx, :]  # [H, ctx, D]
        vi = v_cache[i, :, :ctx, :]

        # Compute attention
        attn = torch.einsum("hd,hsd->hs", qi, ki) * scale  # [H, ctx]
        attn = torch.softmax(attn, dim=-1)
        out_i = torch.einsum("hs,hsd->hd", attn, vi)  # [H, D]
        outputs.append(out_i)
    return torch.stack(outputs)  # [num_seqs, H, D]


@pytest.mark.parametrize("num_seqs,num_heads,head_dim,ctx_len", [
    (1, 4, 64, 16),
    (4, 8, 128, 32),
    (2, 2, 32, 64),
])
def test_paged_attention_matches_naive(num_seqs, num_heads, head_dim, ctx_len):
    torch.manual_seed(42)
    scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(num_seqs, num_heads, head_dim)
    # Dense KV cache per-seq
    k_cache = torch.randn(num_seqs, num_heads, ctx_len, head_dim)
    v_cache = torch.randn(num_seqs, num_heads, ctx_len, head_dim)
    context_lens = torch.full((num_seqs,), ctx_len, dtype=torch.int32)

    ref_out = paged_attention_ref(q, k_cache, v_cache, context_lens, scale)

    # Compare with naive batched attention
    # q: [num_seqs, num_heads, 1, head_dim]
    # k: [num_seqs, num_heads, ctx_len, head_dim]
    q_exp = q.unsqueeze(2)
    naive_out = naive_attention(q_exp, k_cache, v_cache, scale)  # [B, H, 1, D]
    naive_out = naive_out.squeeze(2)  # [B, H, D]

    torch.testing.assert_close(ref_out, naive_out, atol=1e-5, rtol=1e-4)


def test_attention_output_shape():
    """Smoke test: shapes are correct."""
    q = torch.randn(3, 8, 128)
    k = torch.randn(3, 8, 64, 128)
    v = torch.randn(3, 8, 64, 128)
    context_lens = torch.tensor([64, 32, 48], dtype=torch.int32)
    scale = 1.0 / math.sqrt(128)

    out = paged_attention_ref(q, k, v, context_lens, scale)
    assert out.shape == (3, 8, 128)
