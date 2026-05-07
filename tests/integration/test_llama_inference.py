"""Integration test: end-to-end LLaMA inference vs HuggingFace output.

Requires:
  - A local LLaMA / Qwen2 checkpoint at TEST_MODEL_PATH
  - GPU with sufficient memory
  - pip install transformers

Run with:
  TEST_MODEL_PATH=/path/to/model pytest tests/integration/test_llama_inference.py -v
"""
import os
import pytest
import torch

TEST_MODEL_PATH = os.environ.get("TEST_MODEL_PATH", "")

pytestmark = pytest.mark.skipif(
    not TEST_MODEL_PATH or not os.path.isdir(TEST_MODEL_PATH),
    reason="TEST_MODEL_PATH not set or not a directory",
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TEST_MODEL_PATH)


@pytest.fixture(scope="module")
def lean_engine():
    from needle.core.engine import LLMEngine
    engine = LLMEngine(
        model_path=TEST_MODEL_PATH,
        block_size=16,
        max_num_seqs=4,
        gpu_memory_utilization=0.5,
        dtype="bfloat16",
    )
    return engine


@pytest.fixture(scope="module")
def hf_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    return model


def test_greedy_output_matches_hf(tokenizer, lean_engine, hf_model):
    """Greedy decode: Needle output should match HF output exactly."""
    from needle.core.sequence import SamplingParams

    prompt = "The quick brown fox"
    prompt_ids = tokenizer.encode(prompt)

    # Needle output
    params = SamplingParams(temperature=0.0, max_new_tokens=20)
    lean_ids = lean_engine.generate(prompt_ids, params)

    # HuggingFace output
    input_tensor = torch.tensor([prompt_ids], device="cuda")
    with torch.no_grad():
        hf_output = hf_model.generate(
            input_tensor,
            max_new_tokens=20,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    hf_ids = hf_output[0, len(prompt_ids):].tolist()

    assert lean_ids == hf_ids, (
        f"Mismatch!\nNeedle: {lean_ids}\nHF:      {hf_ids}"
    )


def test_output_is_non_empty(lean_engine, tokenizer):
    from needle.core.sequence import SamplingParams

    prompt_ids = tokenizer.encode("Hello, my name is")
    params = SamplingParams(temperature=0.0, max_new_tokens=10)
    output = lean_engine.generate(prompt_ids, params)
    assert len(output) > 0


def test_multiple_requests_sequential(lean_engine, tokenizer):
    from needle.core.sequence import SamplingParams

    prompts = ["Hello", "The capital of France is", "2 + 2 ="]
    params = SamplingParams(temperature=0.0, max_new_tokens=8)

    for i, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt)
        output = lean_engine.generate(ids, params, request_id=f"test-{i}")
        assert len(output) > 0
