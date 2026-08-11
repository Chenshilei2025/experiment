"""Shared record loading, deterministic HF generation, and JSONL persistence."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Awaitable, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_records(path: Path, mechanism: str) -> list[dict[str, Any]]:
    """Read and validate a non-empty test JSONL for one mechanism."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records or any(record.get("mechanism") != mechanism for record in records):
        raise ValueError(f"{path} is not a non-empty {mechanism} JSONL file")
    return records


def load_model(checkpoint: Path, device_name: str) -> tuple[Any, Any, torch.device]:
    """Load a decoder-only model with safe left-padded batch generation."""
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("test-set evaluation requires a CUDA device")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = torch.device(device_name)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device).eval()
    return model, tokenizer, device


def generate_batch(
    model: Any, tokenizer: Any, device: torch.device, records: list[dict[str, Any]],
    prompt_builder: Callable[[dict[str, Any]], list[dict[str, str]]], max_new_tokens: int,
    *, enable_thinking: bool | None,
) -> list[str]:
    """Generate deterministic responses from rendered policy messages."""
    rendered = []
    for record in records:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        rendered.append(tokenizer.apply_chat_template(prompt_builder(record), **kwargs))
    encoded = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
    encoded = {name: value.to(device) for name, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **encoded, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)


def run_async(coroutine: Awaitable[Any]) -> Any:
    """Run a batch-local async scorer without sharing its semaphores across loops."""
    return asyncio.run(coroutine)


def append_rows(handle: Any, rows: list[dict[str, Any]]) -> None:
    """Persist a completed batch atomically enough for recovery."""
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
