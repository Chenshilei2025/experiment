"""Bootstrap MATH, U-MATH, and GPQA-Diamond reasoning eval files.

The script avoids the optional ``datasets`` package so it can run in the
minimal training runtime.  It reuses local MATH500 when present and converts
downloaded Hugging Face csv/parquet/jsonl files into the schema consumed by
``scripts.evaluation.run_reasoning_benchmarks.sh``.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


DEFAULT_UMATH_REPO = "toloka/u-math"
DEFAULT_GPQA_REPO = "Idavidrein/gpqa"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    return []


def _scalar(row: dict[str, Any], *names: str) -> str:
    lower = {key.lower().replace(" ", "_"): key for key in row}
    for name in names:
        key = lower.get(name.lower().replace(" ", "_"))
        if key is None:
            continue
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _candidate_files(root: Path) -> Iterable[Path]:
    suffixes = {".parquet", ".jsonl", ".json", ".csv"}
    yield from sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes)


def _download(repo_id: str, cache_root: Path) -> Path:
    return Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            allow_patterns=["*.parquet", "*.jsonl", "*.json", "*.csv", "data/*", "**/data/*"],
            local_dir=cache_root / repo_id.replace("/", "__"),
            local_dir_use_symlinks=False,
        )
    )


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def copy_math(source: Path, output: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"MATH source does not exist: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != output.resolve():
        shutil.copyfile(source, output)
    rows = [line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"path": str(output), "n_total": len(rows)}


def build_umath(repo_root: Path, output: Path, *, limit: int | None) -> dict[str, Any]:
    converted: list[dict[str, Any]] = []
    for path in _candidate_files(repo_root):
        for index, row in enumerate(_read_rows(path), 1):
            question = _scalar(row, "problem", "question", "prompt", "body")
            answer = _scalar(row, "answer", "final_answer", "gold_answer", "solution")
            if not question or not answer:
                continue
            converted.append({
                "id": _scalar(row, "id", "problem_id", "uid") or f"{path.stem}-{index}",
                "question": question,
                "answer": answer,
            })
            if limit is not None and len(converted) >= limit:
                break
        if limit is not None and len(converted) >= limit:
            break
    _write_jsonl(converted, output)
    return {"path": str(output), "n_total": len(converted)}


def build_gpqa(repo_root: Path, output: Path, *, limit: int | None, seed: int) -> dict[str, Any]:
    converted: list[dict[str, Any]] = []
    rng = random.Random(seed)
    files = list(_candidate_files(repo_root))
    diamond_files = [path for path in files if "diamond" in str(path).lower()]
    for path in diamond_files or files:
        for index, row in enumerate(_read_rows(path), 1):
            question = _scalar(row, "question", "problem")
            correct = _scalar(row, "correct_answer", "correct answer", "answer", "label")
            incorrect = [
                _scalar(row, f"incorrect_answer_{item}", f"incorrect answer {item}", f"wrong_answer_{item}")
                for item in range(1, 4)
            ]
            choices = [item for item in [correct, *incorrect] if item]
            if not question or not correct or len(choices) < 2:
                continue
            rng.shuffle(choices)
            answer = chr(ord("A") + choices.index(correct))
            converted.append({
                "id": _scalar(row, "id", "record_id", "question_id") or f"{path.stem}-{index}",
                "question": question,
                "choices": choices,
                "answer": answer,
            })
            if limit is not None and len(converted) >= limit:
                break
        if limit is not None and len(converted) >= limit:
            break
    _write_jsonl(converted, output)
    return {"path": str(output), "n_total": len(converted)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/cache/reasoning_sources"))
    parser.add_argument("--math-source", type=Path, required=True)
    parser.add_argument("--umath-repo", default=DEFAULT_UMATH_REPO)
    parser.add_argument("--gpqa-repo", default=DEFAULT_GPQA_REPO)
    parser.add_argument("--umath-limit", type=int)
    parser.add_argument("--gpqa-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.cache_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "math": copy_math(args.math_source, args.output_root / "math500" / "test.jsonl"),
        "ugmath": build_umath(_download(args.umath_repo, args.cache_root), args.output_root / "ugmath" / "test.jsonl", limit=args.umath_limit),
        "gpqa": build_gpqa(_download(args.gpqa_repo, args.cache_root), args.output_root / "gpqa" / "diamond.jsonl", limit=args.gpqa_limit, seed=args.seed),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
