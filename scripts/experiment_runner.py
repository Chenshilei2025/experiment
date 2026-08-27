"""Run reproducible multi-stage training experiments from a JSON config.

The runner owns the mechanics shared by all training experiments: resolving a
checkpoint name, preserving the exact config, recording dataset metadata,
running stages, and writing a recoverable manifest.  Experiment directories
should therefore contain conditions (JSON) and analysis, not another copy of
the Docker-launch logic.

Example:
    python -m scripts.experiment_runner \
      --config experiments/mixed_training/configs/shuffle.json \
      --run-name pilot_01 \
      --output-dir artifacts/experiments/mixed_training/pilot_01
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Mapping

from scripts.common.experiment_logging import write_run_provenance
from scripts.data.prepare_mixed_slime import build_mixed_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MECHANISMS = {"miu", "eil", "mixed", "creative"}
TRAINING_MECHANISMS = {"miu", "eil"}
_SIMPLE_NAME = re.compile(r"[A-Za-z0-9._-]+\Z")


def _json_value(value: str) -> Any:
    """Parse a CLI override as JSON when possible, otherwise retain its text."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _set_path(config: dict[str, Any], expression: str) -> None:
    """Apply ``path=value`` to a JSON object without introducing shell syntax."""
    path, separator, raw_value = expression.partition("=")
    keys = path.split(".")
    if not separator or not keys or any(not key for key in keys):
        raise ValueError("--set must use a dotted JSON path, for example seed=42 or stages.0.rollouts=800")
    target: Any = config
    for key in keys[:-1]:
        if isinstance(target, dict) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) and isinstance(target.get(key), (dict, list)):
            target = target[key]
        elif isinstance(target, list) and key.isdigit() and int(key) < len(target):
            target = target[int(key)]
        else:
            raise ValueError(f"cannot set {path}: {key!r} does not select an existing object or list")
    last = keys[-1]
    if isinstance(target, dict) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", last):
        target[last] = _json_value(raw_value)
    elif isinstance(target, list) and last.isdigit() and int(last) < len(target):
        target[int(last)] = _json_value(raw_value)
    else:
        raise ValueError(f"cannot set {path}: final key is invalid")


def load_config(path: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a versioned JSON experiment condition and apply explicit overrides."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a JSON object")
    config = copy.deepcopy(config)
    for expression in overrides or []:
        _set_path(config, expression)
    return config


def _string_environment(values: Mapping[str, Any], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name.startswith("LOYAL_") or not re.fullmatch(r"LOYAL_[A-Z0-9_]+", name):
            raise ValueError(f"{label} may only contain LOYAL_* environment names; got {name!r}")
        if name.endswith("_API_KEY") or name.endswith("_API_KEYS") or name.endswith("_BASE_URL") or name in {
            "LOYAL_MIU_JUDGE_MODEL", "LOYAL_EIL_JUDGE_MODEL", "LOYAL_EIL_ADVERSARY_MODEL",
        }:
            raise ValueError(f"{label}.{name} is API/evaluator configuration and must stay in .env")
        if name == "LOYAL_CREATIVE_TRAIN_RECORDS":
            result[name] = str(value)
            continue
        if name.endswith("_RECORDS") or name.endswith("_TRAIN_RECORDS") or name.endswith("_VAL_RECORDS"):
            raise ValueError(f"{label}.{name} is a dataset path and must stay in the mechanism recipe")
        if isinstance(value, bool):
            result[name] = "1" if value else "0"
        elif isinstance(value, (str, int, float)) and not isinstance(value, complex):
            result[name] = str(value)
        else:
            raise ValueError(f"{label}.{name} must be a string, number, or boolean")
    return result


def _file_signature(path: Path) -> dict[str, Any]:
    """Record lightweight provenance without hashing user datasets."""
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_unix": stat.st_mtime,
    }
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            result["line_count"] = sum(1 for line in handle if line.strip())
    return result


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON file must contain an object: {path}")
    return value


def _record_paths(stages: list[dict[str, Any]], environment: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Describe the canonical train split for each mechanism used by a run."""
    paths: dict[str, dict[str, Any]] = {}
    for stage in stages:
        mechanism = stage["mechanism"]
        if mechanism == "creative":
            raw_path = environment.get("LOYAL_CREATIVE_TRAIN_RECORDS")
            if not raw_path:
                raise ValueError("creative stages require environment.LOYAL_CREATIVE_TRAIN_RECORDS")
            path = Path(raw_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if not path.is_file():
                raise ValueError(f"creative training records do not exist: {path}")
            paths[mechanism] = _file_signature(path)
            continue
        if mechanism == "mixed":
            # The derived mixed file is created immediately before training, so
            # Record the two immutable source splits instead.
            for source_mechanism in sorted(TRAINING_MECHANISMS):
                path = PROJECT_ROOT / source_mechanism / "data" / "dataset" / f"{source_mechanism.upper()}-v2" / "train.jsonl"
                if not path.is_file():
                    raise ValueError(f"training records for {source_mechanism} do not exist: {path}")
                paths[source_mechanism] = _file_signature(path)
            continue
        variable = f"LOYAL_{mechanism.upper()}_TRAIN_RECORDS"
        raw_path = environment.get(variable)
        path = Path(raw_path) if raw_path else PROJECT_ROOT / mechanism / "data" / "dataset" / f"{mechanism.upper()}-v2" / "train.jsonl"
        if not path.is_file():
            raise ValueError(f"training records for {mechanism} do not exist: {path}")
        paths[mechanism] = _file_signature(path)
    return paths


def _evaluation_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the checkpoint evaluation schedule for an experiment condition.

    A stage checkpoint is immutable only until the next stage resumes the
    shared SLIME directory.  The default schedule consequently scores the
    baseline and every completed stage, on both benchmark families.
    """
    raw = config.get("evaluation", {})
    if not isinstance(raw, dict):
        raise ValueError("evaluation must be an object")
    allowed = {"baseline", "after_each_stage", "mechanisms", "checkpoint_iterations"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"evaluation contains unsupported fields: {', '.join(sorted(unknown))}")
    baseline = raw.get("baseline", True)
    after_each_stage = raw.get("after_each_stage", True)
    mechanisms = raw.get("mechanisms", ["miu", "eil"])
    if not isinstance(baseline, bool) or not isinstance(after_each_stage, bool):
        raise ValueError("evaluation.baseline and evaluation.after_each_stage must be booleans")
    if not isinstance(mechanisms, list) or not mechanisms or any(item not in TRAINING_MECHANISMS for item in mechanisms):
        raise ValueError("evaluation.mechanisms must be a non-empty list containing miu and/or eil")
    if len(set(mechanisms)) != len(mechanisms):
        raise ValueError("evaluation.mechanisms must not contain duplicates")
    checkpoint_iterations = raw.get("checkpoint_iterations", [])
    if not isinstance(checkpoint_iterations, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in checkpoint_iterations
    ):
        raise ValueError("evaluation.checkpoint_iterations must be a list of non-negative integers")
    if len(set(checkpoint_iterations)) != len(checkpoint_iterations):
        raise ValueError("evaluation.checkpoint_iterations must not contain duplicates")
    return {
        "baseline": baseline,
        "after_each_stage": after_each_stage,
        "mechanisms": mechanisms,
        "checkpoint_iterations": sorted(checkpoint_iterations),
    }


def _mixed_ablation_plan(config: Mapping[str, Any], stages: list[dict[str, Any]], environment: Mapping[str, str]) -> dict[str, Any] | None:
    """Validate and resolve the strict prompt-quota mixed-ablation contract."""
    raw = config.get("mixed_ablation")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("mixed_ablation must be an object")
    allowed = {"phase", "ratio", "fixed_parameters", "depends_on"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"mixed_ablation contains unsupported fields: {', '.join(sorted(unknown))}")
    phase = raw.get("phase")
    if phase not in {"reward_coefficient", "batch_ratio"}:
        raise ValueError("mixed_ablation.phase must be reward_coefficient or batch_ratio")
    ratio = raw.get("ratio")
    ratios = {"E1M2": 1 / 3, "E1M1": 1 / 2, "E2M1": 2 / 3}
    if ratio not in ratios:
        raise ValueError("mixed_ablation.ratio must be E1M2, E1M1, or E2M1")
    if not stages or any(stage["mechanism"] != "mixed" for stage in stages):
        raise ValueError("mixed_ablation requires only mixed stages")
    try:
        configured_fraction = float(environment["LOYAL_MIXED_EIL_BATCH_FRACTION"])
        rollout_batch = int(environment["LOYAL_MIXED_ROLLOUT_BATCH_SIZE"])
        samples_per_prompt = int(environment["LOYAL_MIXED_SAMPLES_PER_PROMPT"])
        global_batch = int(environment["LOYAL_MIXED_GLOBAL_BATCH_SIZE"])
    except KeyError as exc:
        raise ValueError(f"mixed_ablation requires environment.{exc.args[0]}") from exc
    except ValueError as exc:
        raise ValueError("mixed_ablation batch parameters must be numeric") from exc
    if abs(configured_fraction - ratios[ratio]) > 1e-9:
        raise ValueError("LOYAL_MIXED_EIL_BATCH_FRACTION does not match mixed_ablation.ratio")
    if rollout_batch < 1 or samples_per_prompt < 1 or global_batch != rollout_batch * samples_per_prompt:
        raise ValueError("mixed_ablation requires global_batch_size = rollout_batch_size * samples_per_prompt")
    fixed = raw.get("fixed_parameters")
    if not isinstance(fixed, dict) or not fixed:
        raise ValueError("mixed_ablation.fixed_parameters must document the controlled training parameters")
    dependency = raw.get("depends_on")
    if dependency is not None and (not isinstance(dependency, str) or not dependency):
        raise ValueError("mixed_ablation.depends_on must be a non-empty string when supplied")
    return {
        "phase": phase,
        "ratio": ratio,
        "eil_batch_fraction": configured_fraction,
        "prompt_groups_per_single_task_batch": rollout_batch,
        "candidates_per_prompt": samples_per_prompt,
        "candidates_per_update": global_batch,
        "fixed_parameters": fixed,
        "depends_on": dependency,
    }


def _validate(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    if config.get("version") != 1:
        raise ValueError("experiment config version must be 1")
    experiment = config.get("experiment")
    if not isinstance(experiment, str) or not _SIMPLE_NAME.fullmatch(experiment):
        raise ValueError("experiment must be a simple identifier")
    if config.get("base_model", "qwen3-4b") not in {"qwen3-4b", "glm-z1-9b", "llama3.1-8b-instruct"}:
        raise ValueError("base_model must be one of qwen3-4b, glm-z1-9b, llama3.1-8b-instruct")
    seed = config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    shared_environment = config.get("environment", {})
    if not isinstance(shared_environment, dict):
        raise ValueError("environment must be an object")
    environment = _string_environment(shared_environment, "environment")
    context = config.get("context", {})
    if not isinstance(context, dict) or any(not isinstance(key, str) or not isinstance(value, (str, int, float)) for key, value in context.items()):
        raise ValueError("context must map string keys to scalar values")
    stages = config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages must be a non-empty list")
    validated: list[dict[str, Any]] = []
    for index, stage in enumerate(stages, 1):
        if not isinstance(stage, dict):
            raise ValueError(f"stage {index} must be an object")
        mechanism = stage.get("mechanism")
        rollouts = stage.get("rollouts")
        if mechanism not in MECHANISMS:
            raise ValueError(f"stage {index}.mechanism must be miu, eil, mixed, or creative")
        if not isinstance(rollouts, int) or isinstance(rollouts, bool) or rollouts < 1:
            raise ValueError(f"stage {index}.rollouts must be a positive integer")
        stage_environment = stage.get("environment", {})
        if not isinstance(stage_environment, dict):
            raise ValueError(f"stage {index}.environment must be an object")
        validated.append({
            "mechanism": mechanism,
            "rollouts": rollouts,
            "environment": _string_environment(stage_environment, f"stages[{index}].environment"),
        })
    evaluation_plan = _evaluation_plan(config)
    mixed_plan = _mixed_ablation_plan(config, validated, environment)
    if mixed_plan and evaluation_plan["checkpoint_iterations"]:
        save_interval = int(environment.get("LOYAL_MIXED_SAVE_INTERVAL", "0"))
        final_iteration = sum(stage["rollouts"] for stage in validated) - 1
        if save_interval < 1 or any((item + 1) % save_interval or item > final_iteration for item in evaluation_plan["checkpoint_iterations"]):
            raise ValueError("checkpoint evaluations must align with saved checkpoints and stay within the rollout budget")
    return validated, environment, context, evaluation_plan, mixed_plan


def _checkpoint_name(config: Mapping[str, Any], run_name: str, context: Mapping[str, Any]) -> str:
    template = config.get("checkpoint_template", "{experiment}_{run_name}_seed{seed}")
    if not isinstance(template, str):
        raise ValueError("checkpoint_template must be a string")
    try:
        value = template.format(experiment=config["experiment"], run_name=run_name, seed=config["seed"], **context)
    except KeyError as exc:
        raise ValueError(f"checkpoint_template refers to missing field {exc.args[0]!r}") from exc
    if not _SIMPLE_NAME.fullmatch(value):
        raise ValueError(f"resolved checkpoint name must be a simple directory name: {value!r}")
    return value


def _latest_checkpoint_iteration(checkpoint: str, environment: Mapping[str, str]) -> int:
    """Read the stage checkpoint before the next stage can overwrite it."""
    checkpoint_root = Path(environment.get("LOYAL_CHECKPOINT_HOST_DIR", PROJECT_ROOT / "artifacts" / "checkpoints" / checkpoint))
    path = checkpoint_root / "latest_checkpointed_iteration.txt"
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"completed stage did not produce a readable checkpoint iteration: {path}") from exc
    if value < 0:
        raise RuntimeError(f"checkpoint iteration must be non-negative: {path}")
    return value


def _evaluation_output(mechanism: str, model_kind: str, label: str) -> str:
    return str((PROJECT_ROOT / "artifacts" / "evaluations" / f"{mechanism}_{model_kind}_{label}").resolve())


def _evaluation_summary_exists(mechanism: str, model_kind: str, label: str) -> bool:
    return (Path(_evaluation_output(mechanism, model_kind, label)) / "summary.json").is_file()


def _export_output(checkpoint: str, iteration: int, environment: Mapping[str, str]) -> Path:
    export_root = Path(environment.get("LOYAL_EXPORT_ROOT", PROJECT_ROOT / "artifacts" / "exported_models"))
    return export_root / checkpoint / f"iter_{iteration:07d}"


def _launcher(env_name: str, container_script: str, host_script: str) -> str:
    """Prefer host scripts automatically on machines without Docker."""
    configured = os.environ.get(env_name)
    if configured:
        return configured
    if shutil.which("docker") is None:
        return host_script
    return container_script


def _evaluate_baseline(*, label: str, mechanisms: list[str], environment: Mapping[str, str]) -> dict[str, str]:
    """Score the unmodified base model before the first stage begins."""
    results: dict[str, str] = {}
    launcher = _launcher("LOYAL_TEST_LAUNCHER", "scripts/run_test_container.sh", "scripts/run_test_host.sh")
    for mechanism in mechanisms:
        if os.environ.get("LOYAL_EXPERIMENT_RESUME") == "1" and _evaluation_summary_exists(mechanism, "baseline", label):
            results[mechanism] = _evaluation_output(mechanism, "baseline", label)
            continue
        subprocess.run(
            ["bash", launcher, mechanism, "baseline", label],
            cwd=PROJECT_ROOT, env=dict(environment), check=True,
        )
        results[mechanism] = _evaluation_output(mechanism, "baseline", label)
    return results


def _evaluate_stage_checkpoint(
    *, checkpoint: str, iteration: int, label: str, mechanisms: list[str], environment: Mapping[str, str],
) -> dict[str, str]:
    """Export one immutable stage checkpoint and test it on selected benchmarks."""
    export_launcher = _launcher("LOYAL_EXPORT_LAUNCHER", "scripts/export_final_checkpoint.sh", "scripts/export_final_checkpoint_host.sh")
    test_launcher = _launcher("LOYAL_TEST_LAUNCHER", "scripts/run_test_container.sh", "scripts/run_test_host.sh")
    exported_model = _export_output(checkpoint, iteration, environment)
    if not exported_model.joinpath("config.json").is_file():
        exported = subprocess.run(
            ["bash", export_launcher, checkpoint, str(iteration)],
            cwd=PROJECT_ROOT, env=dict(environment), check=True,
        )
        del exported  # subprocess success is the only result consumed here.
    results: dict[str, str] = {}
    for mechanism in mechanisms:
        if os.environ.get("LOYAL_EXPERIMENT_RESUME") == "1" and _evaluation_summary_exists(mechanism, "final", label):
            results[mechanism] = _evaluation_output(mechanism, "final", label)
            continue
        subprocess.run(
            ["bash", test_launcher, mechanism, "final", label, str(iteration)],
            cwd=PROJECT_ROOT,
            env=dict(environment),
            check=True,
        )
        results[mechanism] = _evaluation_output(mechanism, "final", label)
    return results


def _numeric_deltas(before: object, after: object) -> dict[str, float]:
    """Return final-minus-baseline for numeric metrics shared by two summaries."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {}
    deltas: dict[str, float] = {}
    for key in sorted(set(before) & set(after)):
        previous, current = before[key], after[key]
        if isinstance(previous, bool) or isinstance(current, bool):
            continue
        if isinstance(previous, (int, float)) and isinstance(current, (int, float)):
            deltas[key] = float(current) - float(previous)
    return deltas


def _write_before_after_comparison(
    *, output_dir: Path, baseline: Mapping[str, str], final: Mapping[str, str],
) -> dict[str, Any]:
    """Persist a compact, machine-readable comparison of standard test metrics."""
    comparison: dict[str, Any] = {"baseline": dict(baseline), "final": dict(final), "benchmarks": {}}
    for mechanism in sorted(set(baseline) & set(final)):
        before_path = Path(baseline[mechanism]) / "summary.json"
        after_path = Path(final[mechanism]) / "summary.json"
        try:
            before = json.loads(before_path.read_text(encoding="utf-8"))
            after = json.loads(after_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot compare {mechanism} baseline and final summaries: {exc}") from exc
        baseline_metrics = {
            key: value for key, value in before.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        final_metrics = {
            key: value for key, value in after.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        comparison["benchmarks"][mechanism] = {
            "baseline_metrics": baseline_metrics,
            "final_metrics": final_metrics,
            "delta": _numeric_deltas(before, after),
        }
    _write_json(output_dir / "before_after_comparison.json", comparison)
    return comparison


def _prepare_mixed_training_data(*, output_dir: Path, seed: int) -> tuple[Path, dict[str, Any]]:
    """Materialize the shuffled union inside the immutable experiment record."""
    path = output_dir / "mixed_train.jsonl"
    summary = build_mixed_file(
        miu_source=PROJECT_ROOT / "miu" / "data" / "dataset" / "MIU-v2" / "train.jsonl",
        eil_source=PROJECT_ROOT / "eil" / "data" / "dataset" / "EIL-v2" / "train.jsonl",
        destination=path,
        seed=seed,
    )
    summary["path"] = str(path.resolve())
    summary["file"] = _file_signature(path)
    _write_json(output_dir / "mixed_training_data.json", summary)
    return path, summary


def run_config(config: dict[str, Any], *, output_dir: Path, run_name: str, config_path: Path | None = None) -> dict[str, Any]:
    """Run a validated condition.  ``rollouts`` are additive stage budgets.

    SLIME persists its global rollout ID in the shared checkpoint.  A stage's
    requested target is therefore the sum of its own budget and preceding
    budgets, which makes both `miu-eil` and `eil-miu` work without a bespoke
    coordinator.
    """
    if not _SIMPLE_NAME.fullmatch(run_name):
        raise ValueError("run name must be a simple identifier")
    stages, shared_environment, context, evaluation_plan, mixed_plan = _validate(config)
    resume = os.environ.get("LOYAL_EXPERIMENT_RESUME") == "1"
    if output_dir.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite experiment output {output_dir}")
    checkpoint = _checkpoint_name(config, run_name, context)
    resolved = copy.deepcopy(config)
    resolved["checkpoint_name"] = checkpoint
    resolved["run_name"] = run_name
    if mixed_plan is not None:
        resolved["mixed_ablation_resolved"] = mixed_plan
    if resume:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True)
    if not resume or not (output_dir / "config.resolved.json").exists():
        _write_json(output_dir / "config.resolved.json", resolved)
    if config_path is not None and (not resume or not (output_dir / "config.source.json").exists()):
        (output_dir / "config.source.json").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    base_environment = os.environ.copy()
    base_environment.update(shared_environment)
    base_environment.update({
        "LOYAL_BASE_MODEL": str(config.get("base_model", "qwen3-4b")),
        "LOYAL_SHARED_CHECKPOINT_NAME": checkpoint,
        "LOYAL_TRAINING_SEED": str(config["seed"]),
        "LOYAL_ROLLOUT_SEED": str(config.get("rollout_seed", config["seed"])),
    })
    mixed_stages = [stage for stage in stages if stage["mechanism"] == "mixed"]
    if mixed_stages:
        mixed_path = output_dir / "mixed_train.jsonl"
        mixed_summary_path = output_dir / "mixed_training_data.json"
        if resume and mixed_path.is_file() and mixed_summary_path.is_file():
            mixed_summary = _read_json(mixed_summary_path)
        else:
            mixed_path, mixed_summary = _prepare_mixed_training_data(output_dir=output_dir, seed=config["seed"])
        base_environment.update({
            "LOYAL_MIXED_TRAIN_RECORDS": str(mixed_path.resolve()),
            "LOYAL_MIU_RECORDS": (
                f"{PROJECT_ROOT / 'miu' / 'data' / 'dataset' / 'MIU-v2' / 'train.jsonl'}:"
                f"{PROJECT_ROOT / 'miu' / 'data' / 'dataset' / 'MIU-v2' / 'val.jsonl'}"
            ),
            "LOYAL_EIL_RECORDS": (
                f"{PROJECT_ROOT / 'eil' / 'data' / 'dataset' / 'EIL-v2' / 'train.jsonl'}:"
                f"{PROJECT_ROOT / 'eil' / 'data' / 'dataset' / 'EIL-v2' / 'val.jsonl'}"
            ),
        })
    else:
        mixed_summary = None
    if not resume or not (output_dir / "command.json").exists() or not (output_dir / "environment.json").exists():
        write_run_provenance(output_dir, environment=base_environment)
    manifest_path = output_dir / "manifest.json"
    if resume and manifest_path.exists():
        manifest = _read_json(manifest_path)
        manifest.setdefault("evaluations", {})
        manifest.setdefault("stages", [{"mechanism": item["mechanism"], "rollouts": item["rollouts"], "status": "pending"} for item in stages])
        manifest["evaluation_plan"] = evaluation_plan
        manifest.pop("status", None)
        manifest.pop("finished_at_unix", None)
        manifest["resumed_at_unix"] = time.time()
    else:
        manifest = {
            "experiment": config["experiment"],
            "run_name": run_name,
            "base_model": base_environment["LOYAL_BASE_MODEL"],
            "seed": config["seed"],
            "checkpoint_name": checkpoint,
            "context": context,
            "evaluation_plan": evaluation_plan,
            "mixed_ablation": mixed_plan,
            "evaluations": {},
            "stages": [{"mechanism": item["mechanism"], "rollouts": item["rollouts"], "status": "pending"} for item in stages],
            "training_records": _record_paths(stages, base_environment),
            "mixed_training_data": mixed_summary,
            "config_file": _file_signature(config_path) if config_path else None,
            "started_at_unix": time.time(),
        }
    _write_json(output_dir / "manifest.json", manifest)

    if evaluation_plan["baseline"]:
        label = f"{checkpoint}-baseline"
        baseline_existing = {
            mechanism: _evaluation_output(mechanism, "baseline", label)
            for mechanism in evaluation_plan["mechanisms"]
            if _evaluation_summary_exists(mechanism, "baseline", label)
        }
        manifest["evaluations"]["baseline"] = {"status": "running", "label": label, "benchmarks": baseline_existing}
        _write_json(output_dir / "manifest.json", manifest)
        evaluations = _evaluate_baseline(
            label=label, mechanisms=evaluation_plan["mechanisms"], environment=base_environment,
        )
        manifest["evaluations"]["baseline"] = {
            "status": "completed", "label": label, "benchmarks": evaluations,
            "finished_at_unix": time.time(),
        }
        _write_json(output_dir / "manifest.json", manifest)

    completed_rollouts = 0
    for index, stage in enumerate(stages):
        completed_rollouts += stage["rollouts"]
        manifest_stage = manifest["stages"][index]
        if (
            resume
            and manifest_stage.get("status") == "completed"
            and manifest_stage.get("target_num_rollout") == completed_rollouts
            and isinstance(manifest_stage.get("checkpoint_iteration"), int)
            and manifest_stage["checkpoint_iteration"] >= completed_rollouts - 1
        ):
            if not evaluation_plan["after_each_stage"]:
                evaluation = manifest_stage.get("evaluation")
                if isinstance(evaluation, dict) and evaluation.get("status") == "running":
                    evaluation["status"] = "skipped_on_resume"
                    evaluation["skipped_reason"] = "after_each_stage disabled on resume"
                    evaluation_key = f"after_stage_{index + 1}"
                    if isinstance(manifest.get("evaluations", {}).get(evaluation_key), dict):
                        manifest["evaluations"][evaluation_key]["status"] = "skipped_on_resume"
                        manifest["evaluations"][evaluation_key]["skipped_reason"] = "after_each_stage disabled on resume"
                _write_json(output_dir / "manifest.json", manifest)
                continue
            if manifest_stage.get("evaluation", {}).get("status") == "completed":
                continue
        environment = base_environment.copy()
        environment.update(stage["environment"])
        environment[f"LOYAL_{stage['mechanism'].upper()}_NUM_ROLLOUT"] = str(completed_rollouts)
        manifest_stage.update({"target_num_rollout": completed_rollouts, "started_at_unix": time.time(), "status": "running"})
        _write_json(output_dir / "manifest.json", manifest)
        # Docker/Ray output is the primary diagnostic for a failed stage. Keep
        # it adjacent to the immutable config and stage manifest rather than
        # relying on a shared, manually named terminal log.
        with (output_dir / "run.log").open("a", encoding="utf-8", buffering=1) as log:
            print(f"[stage {index + 1}] launching {stage['mechanism']}", file=log, flush=True)
            launcher = _launcher(
                "LOYAL_TRAINING_LAUNCHER",
                "scripts/launch/run_training_container.sh",
                "scripts/launch/run_training_host.sh",
            )
            subprocess.run(
                ["bash", launcher, stage["mechanism"]],
                cwd=PROJECT_ROOT, env=environment, check=True, stdout=log, stderr=subprocess.STDOUT, text=True,
            )
        (output_dir / f"stage_{index + 1}_{stage['mechanism']}.complete").touch()
        if environment.get("LOYAL_SUBMIT_DRY_RUN") == "1":
            manifest["stages"][index].update({
                "finished_at_unix": time.time(), "status": "dry_run_completed",
            })
            _write_json(output_dir / "manifest.json", manifest)
            continue
        checkpoint_iteration = _latest_checkpoint_iteration(checkpoint, environment)
        manifest["stages"][index].update({
            "checkpoint_iteration": checkpoint_iteration,
            "finished_at_unix": time.time(), "status": "completed",
        })
        _write_json(output_dir / "manifest.json", manifest)

        # Export and score before a later stage resumes this shared checkpoint:
        # the recorded iteration then remains a stable, independently
        # testable model for every point on an order-training trajectory.
        if evaluation_plan["after_each_stage"]:
            label = f"{checkpoint}-stage{index + 1}-{stage['mechanism']}"
            evaluation_key = f"after_stage_{index + 1}"
            manifest["stages"][index].update({"evaluation": {"status": "running", "label": label}})
            manifest["evaluations"][evaluation_key] = {"status": "running", "label": label, "checkpoint_iteration": checkpoint_iteration}
            _write_json(output_dir / "manifest.json", manifest)
            evaluations = _evaluate_stage_checkpoint(
                checkpoint=checkpoint, iteration=checkpoint_iteration, label=label,
                mechanisms=evaluation_plan["mechanisms"], environment=environment,
            )
            completed_evaluation = {
                "status": "completed", "label": label, "checkpoint_iteration": checkpoint_iteration,
                "benchmarks": evaluations, "finished_at_unix": time.time(),
            }
            manifest["stages"][index]["evaluation"] = completed_evaluation
            manifest["evaluations"][evaluation_key] = completed_evaluation
            # A single-stage condition is the standard mixed-training design:
            # provide its requested before/after comparison alongside the
            # immutable evaluation directories.
            if len(stages) == 1 and "baseline" in manifest["evaluations"]:
                manifest["before_after_comparison"] = _write_before_after_comparison(
                    output_dir=output_dir,
                    baseline=manifest["evaluations"]["baseline"]["benchmarks"],
                    final=evaluations,
                )
            _write_json(output_dir / "manifest.json", manifest)

    manifest["finished_at_unix"] = time.time()
    manifest["status"] = "completed"
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, help="required unless --validate-only is supplied")
    parser.add_argument("--validate-only", action="store_true", help="validate and print the resolved condition without creating output or starting training")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE", help="override one JSON field; VALUE may be JSON")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if args.validate_only:
        stages, _, context, evaluation_plan, mixed_plan = _validate(config)
        print(json.dumps({
            "experiment": config["experiment"],
            "checkpoint_name": _checkpoint_name(config, args.run_name, context),
            "stages": [{"mechanism": stage["mechanism"], "rollouts": stage["rollouts"]} for stage in stages],
            "evaluation": evaluation_plan,
            "mixed_ablation": mixed_plan,
        }, ensure_ascii=False, indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only is supplied")
    run_config(config, output_dir=args.output_dir, run_name=args.run_name, config_path=args.config)


if __name__ == "__main__":
    main()
