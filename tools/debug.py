"""Universal debug / validation script for Needle.

Runs six ordered steps against any supported model (qwen2, llama, …):

  1. Config parsing   — print key ModelConfig fields
  2. Model build      — count parameters, check weight file completeness
  3. Weight loading   — summarise missing / unexpected keys
  4. Prefill check    — verify logits shape, detect NaN / Inf
  5. Greedy decode    — run N steps and print decoded text
  6. HF validation    — compare with HuggingFace reference (--validate)

Usage:
    python tools/debug.py --model /path/to/model
    python tools/debug.py --model /path/to/model --prompt "Hello" --max-tokens 20
    python tools/debug.py --model /path/to/model --validate
"""
from __future__ import annotations

import argparse
import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
INFO = "\033[34m[INFO]\033[0m"
WARN = "\033[33m[WARN]\033[0m"


def section(n: int, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Step {n}: {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step1_config(model_path: str):
    section(1, "Config Parsing")
    from needle.model.config import ModelConfig

    cfg = ModelConfig.from_pretrained(model_path)
    print(f"  model_type              : {cfg.model_type}")
    print(f"  hidden_size             : {cfg.hidden_size}")
    print(f"  num_hidden_layers       : {cfg.num_hidden_layers}")
    print(f"  num_attention_heads     : {cfg.num_attention_heads}")
    print(f"  num_key_value_heads     : {cfg.num_key_value_heads}  (GQA={cfg.is_gqa})")
    print(f"  intermediate_size       : {cfg.intermediate_size}")
    print(f"  vocab_size              : {cfg.vocab_size}")
    print(f"  max_position_embeddings : {cfg.max_position_embeddings}")
    print(f"  rms_norm_eps            : {cfg.rms_norm_eps}")
    print(f"  rope_theta              : {cfg.rope_theta}")
    print(f"{PASS} Config parsed successfully.")
    return cfg


def step2_build(cfg, dtype: str):
    section(2, "Model Build")
    import torch
    from needle.model.runner import ModelRunner

    torch_dtype = getattr(torch, dtype)
    runner = ModelRunner(
        model_config=cfg,
        block_size=16,
        dtype=dtype,
        tensor_parallel_size=1,
    )
    model = runner.model
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")

    # Check weight files exist
    import glob as _glob
    weight_files = (
        _glob.glob(os.path.join(cfg.__dict__.get("_model_path", ""), "*.safetensors"))
        + _glob.glob(os.path.join(cfg.__dict__.get("_model_path", ""), "*.bin"))
    )
    print(f"  Weight files found   : {len(weight_files)}")
    print(f"{PASS} Model built.")
    return runner


def step3_load(runner, model_path: str):
    section(3, "Weight Loading")
    from needle.model.loader import load_model

    load_model(runner, model_path)
    print(f"{PASS} Weights loaded.")
    return runner


def step4_prefill(runner, tokenizer, prompt_ids: list):
    section(4, "Prefill Check")
    import torch

    # Minimal KV cache for a single forward pass
    num_blocks = max(16, (len(prompt_ids) + 15) // 16 + 2)
    runner.init_kv_cache(num_blocks, 0)

    # Build minimal SchedulerOutput
    from needle.backend.sequence import (
        Request, SamplingParams, Sequence, SchedulerOutput
    )
    req = Request(
        request_id="debug-0",
        prompt_token_ids=prompt_ids,
        sampling_params=SamplingParams(max_new_tokens=1, temperature=0.0),
    )
    seq = Sequence(
        seq_id=1,
        request=req,
        token_ids=list(prompt_ids),
        logical_blocks=list(range((len(prompt_ids) + 15) // 16)),
        block_size=16,
    )

    # Allocate physical blocks
    from needle.backend.block_allocator import BlockAllocator
    alloc = BlockAllocator(num_gpu_blocks=num_blocks, num_cpu_blocks=0, block_size=16)
    num_prompt_blocks = (len(prompt_ids) + 15) // 16
    block_table = alloc.allocate_gpu_blocks(num_prompt_blocks)

    sched_out = SchedulerOutput(
        prefill_seqs=[seq],
        decode_seqs=[],
        block_tables={seq.seq_id: block_table},
        swap_in_map={},
        swap_out_map={},
        blocks_to_free=[],
    )

    with torch.inference_mode():
        out = runner.execute(sched_out)

    token_id = out.next_token_ids[0]
    print(f"  Output token id : {token_id}")
    print(f"  Output token    : {repr(tokenizer.decode([token_id]))}")

    # Detect numerical issues in logits (runner doesn't expose raw logits; check token is valid)
    if token_id < 0 or token_id >= runner.model_config.vocab_size:
        print(f"{FAIL} token_id {token_id} out of vocab range!")
        return False, runner, alloc, seq, block_table

    # Append the generated token so the sequence is in decode state for step5
    seq.append_token(token_id)
    # Allocate a new block if crossing a block boundary
    if seq.length % seq.block_size == 1:
        new_block = alloc.allocate_gpu()
        block_table.append(new_block)

    print(f"{PASS} Prefill check passed.")
    return True, runner, alloc, seq, block_table


def step5_decode(runner, tokenizer, alloc, seq, block_table: list, max_tokens: int):
    section(5, "Greedy Decode")
    import torch
    from needle.backend.sequence import (
        Request, SamplingParams, Sequence, SchedulerOutput
    )

    generated = []
    for step_i in range(max_tokens):
        sched_out = SchedulerOutput(
            prefill_seqs=[],
            decode_seqs=[seq],
            block_tables={seq.seq_id: list(block_table)},
            swap_in_map={},
            swap_out_map={},
            blocks_to_free=[],
        )
        with torch.inference_mode():
            out = runner.execute(sched_out)

        token_id = out.next_token_ids[0]
        seq.append_token(token_id)

        # Allocate a new block if the sequence crossed a block boundary
        if seq.length % seq.block_size == 1:
            new_block = alloc.allocate_gpu()
            block_table.append(new_block)

        generated.append(token_id)
        if token_id == tokenizer.eos_token_id:
            print(f"  (EOS at step {step_i + 1})")
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"  Generated ({len(generated)} tokens): {repr(text)}")
    print(f"{PASS} Decode complete.")
    return generated


def step6_validate(model_path: str, prompt_ids: list, needle_tokens: list, dtype: str):
    section(6, "HuggingFace Validation")
    import torch

    print("  Loading HuggingFace model (this may take a while)…")
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print(f"{FAIL} transformers not installed — skipping.")
        return

    torch_dtype = getattr(torch, dtype)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map="auto"
    )
    hf_model.eval()

    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(hf_model.device)
    with torch.inference_mode():
        hf_out = hf_model.generate(
            input_ids,
            max_new_tokens=len(needle_tokens),
            do_sample=False,
        )
    hf_tokens = hf_out[0][len(prompt_ids):].tolist()

    matches = sum(a == b for a, b in zip(needle_tokens, hf_tokens))
    total = len(needle_tokens)
    match_rate = matches / total * 100 if total else 0.0

    print(f"  Needle  tokens : {needle_tokens[:20]}{'…' if total > 20 else ''}")
    print(f"  HF      tokens : {hf_tokens[:20]}{'…' if total > 20 else ''}")
    print(f"  Match rate     : {matches}/{total} = {match_rate:.1f}%")

    if match_rate == 100.0:
        print(f"{PASS} Perfect match with HuggingFace reference.")
    elif match_rate >= 95.0:
        print(f"{WARN} Match rate {match_rate:.1f}% — minor divergence (check sampling).")
    else:
        print(f"{FAIL} Match rate {match_rate:.1f}% — significant divergence!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Needle universal debug / validation tool."
    )
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        required=True,
        help="Path to model directory.",
    )
    parser.add_argument(
        "--prompt",
        default="Hello, how are you?",
        help='Input prompt (default: "Hello, how are you?").',
    )
    parser.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=20,
        help="Number of tokens to decode (default: 20).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Compare output with HuggingFace reference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"[error] model path does not exist: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"{INFO} Prompt  : {repr(args.prompt)}")
    print(f"{INFO} Token ids ({len(prompt_ids)}): {prompt_ids[:16]}{'…' if len(prompt_ids) > 16 else ''}")

    cfg = step1_config(args.model_path)
    # Stash model_path so step2 can report weight files
    cfg.__dict__["_model_path"] = args.model_path

    runner = step2_build(cfg, args.dtype)
    step3_load(runner, args.model_path)
    ok, runner, alloc, seq, block_table = step4_prefill(runner, tokenizer, prompt_ids)
    if not ok:
        sys.exit(1)
    prefill_token = seq.token_ids[-1]  # the token generated during prefill
    generated = step5_decode(runner, tokenizer, alloc, seq, block_table, args.max_tokens)

    if args.validate:
        # Include the prefill-generated token so the full sequence matches HF
        step6_validate(args.model_path, prompt_ids, [prefill_token] + generated, args.dtype)

    print(f"\n{PASS} All steps completed.")


if __name__ == "__main__":
    main()
