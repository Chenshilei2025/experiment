"""Run reproducible multi-stage training experiments from a JSON config.

The runner owns the mechanics shared by all training experiments: resolving a
checkpoint name, preserving the exact config, recording dataset fingerprints,
running stages, and writing a recoverable manifest.  Experiment directories
should therefore contain conditions (JSON) and analysis, not another copy of
the Docker-launch logic.

Example:
    python -m scripts.experiment_runner \
      --config experiments/training_order/configs/miu_eil.json \
      --run-name pilot_01 \
      --output-dir artifacts/experiments/training_order/pilot_01
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping

from scripts.common.experiment_logging import write_run_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MECHANISMS = {"miu", "eil"}
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
        if name.endswith("_RECORDS") or name.endswith("_TRAIN_RECORDS") or name.endswith("_VAL_RECORDS"):
            raise ValueError(f"{label}.{name} is a dataset path and must stay in the mechanism recipe")
        if isinstance(value, bool):
            result[name] = "1" if value else "0"
        elif isinstance(value, (str, int, float)) and not isinstance(value, complex):
            result[name] = str(value)
        else:
            raise ValueError(f"{label}.{name} must be a string, number, or boolean")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record_paths(stages: list[dict[str, Any]], environment: Mapping[str, str]) -> dict[str, str]:
    """Fingerprint the canonical train split for each mechanism used by a run."""
    paths: dict[str, str] = {}
    for stage in stages:
        mechanism = stage["mechanism"]
        variable = f"LOYAL_{mechanism.upper()}_TRAIN_RECORDS"
        raw_path = environment.get(variable)
        path = Path(raw_path) if raw_path else PROJECT_ROOT / mechanism / "data" / "dataset" / mechanism.upper() / "train.jsonl"
        if not path.is_file():
            raise ValueError(f"training records for {mechanism} do not exist: {path}")
        paths[mechanism] = _sha256(path)
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
    allowed = {"baseline", "after_each_stage", "mechanisms"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"evaluation contains unsupported fields: {', '.join(sorted(unknown))}")
    baseline = raw.get("baseline", True)
    after_each_stage = raw.get("after_each_stage", True)
    mechanisms = raw.get("mechanisms", ["miu", "eil"])
    if not isinstance(baseline, bool) or not isinstance(after_each_stage, bool):
        raise ValueError("evaluation.baseline and evaluation.after_each_stage must be booleans")
    if not isinstance(mechanisms, list) or not mechanisms or any(item not in MECHANISMS for item in mechanisms):
        raise ValueError("evaluation.mechanisms must be a non-empty list containing miu and/or eil")
    if len(set(mechanisms)) != len(mechanisms):
        raise ValueError("evaluation.mechanisms must not contain duplicates")
    return {"baseline": baseline, "after_each_stage": after_each_stage, "mechanisms": mechanisms}


def _validate(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any], dict[str, Any]]:
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
            raise ValueError(f"stage {index}.mechanism must be miu or eil")
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
    return validated, environment, context, _evaluation_plan(config)


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


def _latest_checkpoint_iteration(checkpoint: str) -> int:
    """Read the stage checkpoint before the next stage can overwrite it."""
    path = PROJECT_ROOT / "artifacts" / "checkpoints" / checkpoint / "latest_checkpointed_iteration.txt"
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"completed stage did not produce a readable checkpoint iteration: {path}") from exc
    if value < 0:
        raise RuntimeError(f"checkpoint iteration must be non-negative: {path}")
    return value


def _evaluation_output(mechanism: str, model_kind: str, label: str) -> str:
    return str((PROJECT_ROOT / "artifacts" / "evaluations" / f"{mechanism}_{model_kind}_{label}").resolve())


def _evaluate_baseline(*, label: str, mechanisms: list[str], environment: Mapping[str, str]) -> dict[str, str]:
    """Score the unmodified base model before the first stage begins."""
    results: dict[str, str] = {}
    for mechanism in mechanisms:
        subprocess.run(
            ["bash", "scripts/run_test_container.sh", mechanism, "baseline", label],
            cwd=PROJECT_ROOT, env=dict(environment), check=True,
        )
        results[mechanism] = _evaluation_output(mechanism, "baseline", label)
    return results


def _evaluate_stage_checkpoint(
    *, checkpoint: str, iteration: int, label: str, mechanisms: list[str], environment: Mapping[str, str],
) -> dict[str, str]:
    """Export one immutable stage checkpoint and test it on selected benchmarks."""
    exported = subprocess.run(
        ["bash", "scripts/export_final_checkpoint.sh", checkpoint, str(iteration)],
        cwd=PROJECT_ROOT, env=dict(environment), check=True,
    )
    del exported  # subprocess success is the only result consumed here.
    results: dict[str, str] = {}
    for mechanism in mechanisms:
        subprocess.run(
            ["bash", "scripts/run_test_container.sh", mechanism, "final", label, str(iteration)],
            cwd=PROJECT_ROOT,
            env=dict(environment),
            check=True,
        )
        results[mechanism] = _evaluation_output(mechanism, "final", label)
    return results


def run_config(config: dict[str, Any], *, output_dir: Path, run_name: str, config_path: Path | None = None) -> dict[str, Any]:
    """Run a validated condition.  ``rollouts`` are additive stage budgets.

    SLIME persists its global rollout ID in the shared checkpoint.  A stage's
    requested target is therefore the sum of its own budget and preceding
    budgets, which makes both `miu-eil` and `eil-miu` work without a bespoke
    coordinator.
    """
    if not _SIMPLE_NAME.fullmatch(run_name):
        raise ValueError("run name must be a simple identifier")
    stages, shared_environment, context, evaluation_plan = _validate(config)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite experiment output {output_dir}")
    checkpoint = _checkpoint_name(config, run_name, context)
    resolved = copy.deepcopy(config)
    resolved["checkpoint_name"] = checkpoint
    resolved["run_name"] = run_name
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "config.resolved.json", resolved)
    if config_path is not None:
        (output_dir / "config.source.json").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    base_environment = os.environ.copy()
    base_environment.update(shared_environment)
    base_environment.update({
        "LOYAL_BASE_MODEL": str(config.get("base_model", "qwen3-4b")),
        "LOYAL_SHARED_CHECKPOINT_NAME": checkpoint,
        "LOYAL_TRAINING_SEED": str(config["seed"]),
        "LOYAL_ROLLOUT_SEED": str(config.get("rollout_seed", config["seed"])),
    })
    write_run_provenance(output_dir, environment=base_environment)
    manifest: dict[str, Any] = {
        "experiment": config["experiment"],
        "run_name": run_name,
        "base_model": base_environment["LOYAL_BASE_MODEL"],
        "seed": config["seed"],
        "checkpoint_name": checkpoint,
        "context": context,
        "evaluation_plan": evaluation_plan,
        "evaluations": {},
        "stages": [{"mechanism": item["mechanism"], "rollouts": item["rollouts"], "status": "pending"} for item in stages],
        "training_records_sha256": _record_paths(stages, base_environment),
        "config_sha256": _sha256(config_path) if config_path else None,
        "started_at_unix": time.time(),
    }
    _write_json(output_dir / "manifest.json", manifest)

    if evaluation_plan["baseline"]:
        label = f"{checkpoint}-baseline"
        manifest["evaluations"]["baseline"] = {"status": "running", "label": label}
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
        environment = base_environment.copy()
        environment.update(stage["environment"])
        environment[f"LOYAL_{stage['mechanism'].upper()}_NUM_ROLLOUT"] = str(completed_rollouts)
        manifest["stages"][index].update({"target_num_rollout": completed_rollouts, "started_at_unix": time.time(), "status": "running"})
        _write_json(output_dir / "manifest.json", manifest)
        # Docker/Ray output is the primary diagnostic for a failed stage. Keep
        # it adjacent to the immutable config and stage manifest rather than
        # relying on a shared, manually named terminal log.
        with (output_dir / "run.log").open("a", encoding="utf-8", buffering=1) as log:
            print(f"[stage {index + 1}] launching {stage['mechanism']}", file=log, flush=True)
            subprocess.run(
                ["bash", "scripts/launch/run_training_container.sh", stage["mechanism"]],
                cwd=PROJECT_ROOT, env=environment, check=True, stdout=log, stderr=subprocess.STDOUT, text=True,
            )
        (output_dir / f"stage_{index + 1}_{stage['mechanism']}.complete").touch()
        checkpoint_iteration = _latest_checkpoint_iteration(checkpoint)
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
            _write_json(output_dir / "manifest.json", manifest)

    manifest["finished_at_unix"] = time.time()
    manifest["status"] = "completed"
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true", help="validate and print the resolved condition without creating output or starting training")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE", help="override one JSON field; VALUE may be JSON")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if args.validate_only:
        stages, _, context, evaluation_plan = _validate(config)
        print(json.dumps({
            "experiment": config["experiment"],
            "checkpoint_name": _checkpoint_name(config, args.run_name, context),
            "stages": [{"mechanism": stage["mechanism"], "rollouts": stage["rollouts"]} for stage in stages],
            "evaluation": evaluation_plan,
        }, ensure_ascii=False, indent=2))
        return
    run_config(config, output_dir=args.output_dir, run_name=args.run_name, config_path=args.config)


if __name__ == "__main__":
    main()
