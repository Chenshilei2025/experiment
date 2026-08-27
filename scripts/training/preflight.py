"""Validate local training configuration before allocating GPUs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def require(value: str | None, name: str, errors: list[str]) -> None:
    if not value:
        errors.append(f"missing {name}")


def validate_records(mechanism: str, errors: list[str]) -> None:
    if mechanism == "MIXED":
        train_path = os.getenv("LOYAL_MIXED_TRAIN_RECORDS")
        require(train_path, "LOYAL_MIXED_TRAIN_RECORDS", errors)
        if train_path:
            source = Path(train_path)
            if not source.is_file():
                errors.append(f"record file does not exist: {source}")
            else:
                count = 0
                with source.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            errors.append(f"invalid JSONL in {source}:{line_number}: {exc}")
                            break
                        if (
                            record.get("mechanism") not in {"MIU", "EIL"}
                            or not isinstance(record.get("record_id"), str)
                            or not isinstance(record.get("messages"), list)
                        ):
                            errors.append(f"{source}:{line_number} is not a valid mixed training row")
                            break
                        count += 1
                        if count >= 100:
                            break
                if count == 0:
                    errors.append(f"record file is empty: {source}")
        for name in ("LOYAL_MIU_RECORDS", "LOYAL_EIL_RECORDS"):
            value = os.getenv(name)
            require(value, name, errors)
            for raw_path in (value or "").split(":"):
                if raw_path and not Path(raw_path).is_file():
                    errors.append(f"{name} path does not exist: {raw_path}")
        return
    if mechanism == "CREATIVE":
        path = os.getenv("LOYAL_CREATIVE_TRAIN_RECORDS")
        require(path, "LOYAL_CREATIVE_TRAIN_RECORDS", errors)
        if not path:
            return
        source = Path(path)
        if not source.is_file():
            errors.append(f"record file does not exist: {source}")
            return
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(source, columns=["messages"])
        except Exception as exc:  # pragma: no cover - reports external parquet/read errors.
            errors.append(f"cannot read creative SFT parquet messages column: {source} ({exc})")
            return
        rows = table.to_pylist()
        if not rows:
            errors.append(f"record file is empty: {source}")
            return
        for index, row in enumerate(rows[:100], 1):
            messages = row.get("messages")
            if (
                not isinstance(messages, list)
                or len(messages) < 2
                or messages[-1].get("role") != "assistant"
                or not isinstance(messages[-1].get("content"), str)
                or not messages[-1]["content"].strip()
            ):
                errors.append(f"{source}:row{index} is not a valid creative SFT messages row")
                break
        return
    for split in ("TRAIN", "VAL"):
        path = os.getenv(f"LOYAL_{mechanism}_{split}_RECORDS")
        if mechanism == "CREATIVE" and split == "VAL":
            continue
        require(path, f"LOYAL_{mechanism}_{split}_RECORDS", errors)
        if not path:
            continue
        source = Path(path)
        if not source.is_file():
            errors.append(f"record file does not exist: {source}")
            continue
        count = 0
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"invalid JSONL in {source}:{line_number}: {exc}")
                    break
                if mechanism in {"MIU", "EIL"} and (record.get("mechanism") != mechanism or not isinstance(record.get("id"), str)):
                    errors.append(f"{source}:{line_number} is not a valid {mechanism} record")
                    break
                if mechanism == "CREATIVE" and not isinstance(record.get("messages"), list):
                    errors.append(f"{source}:{line_number} is not a valid creative SFT row")
                    break
                count += 1
        if count == 0:
            errors.append(f"record file is empty: {source}")


def positive_int_env(name: str, errors: list[str]) -> int | None:
    try:
        value = int(os.environ[name])
    except KeyError:
        errors.append(f"missing {name}")
        return None
    except ValueError:
        errors.append(f"{name} must be an integer")
        return None
    if value <= 0:
        errors.append(f"{name} must be positive")
        return None
    return value


def gpu_device_set(name: str, errors: list[str]) -> set[int] | None:
    value = os.getenv(name, "")
    if not value:
        errors.append(f"missing {name}")
        return None
    try:
        devices = {int(item) for item in value.split(",") if item.strip()}
    except ValueError:
        errors.append(f"{name} must be a comma-separated GPU index list")
        return None
    if not devices or any(device < 0 for device in devices):
        errors.append(f"{name} must contain non-negative GPU indices")
        return None
    return devices


def validate_gpu_layout(mechanism: str, errors: list[str]) -> int | None:
    prefix = f"LOYAL_{mechanism}_"
    train = positive_int_env(f"{prefix}TRAIN_GPU_COUNT", errors)
    if mechanism == "CREATIVE":
        try:
            rollout = int(os.environ[f"{prefix}ROLLOUT_GPU_COUNT"])
        except KeyError:
            errors.append(f"missing {prefix}ROLLOUT_GPU_COUNT")
            rollout = None
        except ValueError:
            errors.append(f"{prefix}ROLLOUT_GPU_COUNT must be an integer")
            rollout = None
    else:
        rollout = positive_int_env(f"{prefix}ROLLOUT_GPU_COUNT", errors)
    ray_gpus = positive_int_env(f"{prefix}RAY_NUM_GPUS", errors)
    if train is not None and rollout is not None and ray_gpus is not None and train + rollout != ray_gpus:
        errors.append(f"{prefix}RAY_NUM_GPUS must equal training plus rollout GPU counts")
    if mechanism == "EIL":
        train_devices = gpu_device_set("LOYAL_EIL_TRAIN_GPU_DEVICES", errors)
        if train_devices is not None and ray_gpus is not None and len(train_devices) != ray_gpus:
            errors.append("LOYAL_EIL_TRAIN_GPU_DEVICES must contain LOYAL_EIL_RAY_NUM_GPUS devices")
    if mechanism == "CREATIVE" and rollout is not None and rollout != 0:
        errors.append("LOYAL_CREATIVE_ROLLOUT_GPU_COUNT must be 0 for SFT")
    return ray_gpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mechanism", choices=("miu", "eil", "mixed", "creative"))
    parser.add_argument("--runtime", action="store_true", help="also require installed SLIME GPU runtime packages")
    args = parser.parse_args()
    mechanism = args.mechanism.upper()
    errors: list[str] = []
    validate_records(mechanism, errors)
    required_gpu_count = validate_gpu_layout(mechanism, errors)
    if mechanism in {"MIU", "MIXED"}:
        require(os.getenv("LOYAL_MIU_JUDGE_BASE_URL"), "LOYAL_MIU_JUDGE_BASE_URL", errors)
        require(os.getenv("LOYAL_MIU_JUDGE_MODEL"), "LOYAL_MIU_JUDGE_MODEL", errors)
    if mechanism in {"EIL", "MIXED"}:
        require(os.getenv("LOYAL_EIL_JUDGE_BASE_URL"), "LOYAL_EIL_JUDGE_BASE_URL", errors)
        require(os.getenv("LOYAL_EIL_JUDGE_MODEL"), "LOYAL_EIL_JUDGE_MODEL", errors)
    if mechanism in {"EIL", "MIXED"}:
        require(os.getenv("LOYAL_EIL_ADVERSARY_BASE_URL"), "LOYAL_EIL_ADVERSARY_BASE_URL", errors)
        require(os.getenv("LOYAL_EIL_ADVERSARY_MODEL"), "LOYAL_EIL_ADVERSARY_MODEL", errors)
    if mechanism == "CREATIVE":
        require(os.getenv("LOYAL_CREATIVE_TRAIN_RECORDS"), "LOYAL_CREATIVE_TRAIN_RECORDS", errors)
    if args.runtime:
        modules = ("torch", "ray", "megatron", "slime") if mechanism == "CREATIVE" else ("torch", "ray", "sglang", "megatron", "slime")
        for module in modules:
            if importlib.util.find_spec(module) is None:
                errors.append(f"missing runtime package: {module}")
        checkpoint_paths = {
            "LOYAL_MODEL_HF_CHECKPOINT": os.getenv("LOYAL_MODEL_HF_CHECKPOINT"),
            "LOYAL_MODEL_REF_LOAD": os.getenv("LOYAL_MODEL_REF_LOAD"),
        }
        for path_name, path in checkpoint_paths.items():
            if not path:
                errors.append(f"missing {path_name}; source scripts/launch/model_profiles.sh before preflight")
                continue
            try:
                accessible = Path(path).is_dir()
            except OSError as exc:
                errors.append(f"checkpoint directory is inaccessible: {path} ({exc.strerror or type(exc).__name__})")
            else:
                if not accessible:
                    errors.append(f"checkpoint directory does not exist: {path}")
        try:
            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
            visible_gpu_count = len([line for line in result.stdout.splitlines() if line.strip()])
            if result.returncode or not visible_gpu_count:
                errors.append("NVIDIA GPU is unavailable to the training runtime")
            elif required_gpu_count is not None and visible_gpu_count < required_gpu_count:
                errors.append(
                    f"training runtime exposes {visible_gpu_count} GPUs but {mechanism} requires {required_gpu_count}"
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            errors.append("nvidia-smi is unavailable")
    if errors:
        print("Training preflight failed:", *[f"- {error}" for error in errors], sep="\n", file=sys.stderr)
        return 1
    print(f"{mechanism} training preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
