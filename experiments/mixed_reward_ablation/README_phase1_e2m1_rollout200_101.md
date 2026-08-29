# Phase 1 E2M1 Rollout200 Long Task on 10.220.5.101

This package runs the qwen3-4B EIL/MIU mixed reward ablation and the downstream
post-training pipeline on the 4xA100 host.

## Experiment

- Model: `qwen3-4B`, with Qwen thinking mode disabled by `model_profiles.sh`.
- Reward mix: `E2M1`, so EIL batch fraction is `0.6666666666666666`.
- Reward coefficients: `lambda=0.5`, `eta=0.5`.
- Training horizon: `200` rollouts, split as `10 x 20`.
- Checkpoints: `19 39 59 79 99 119 139 159 179 199`.
- Training-time eval: disabled.
- Post-training pipeline: direct EIL/MIU checkpoint tests after every saved
  checkpoint, best checkpoint selection, MATH/UGMATH/GPQA, creative SFT on
  WritingPrompts/ROCStories, then EIL/MIU re-evaluation.
- Best checkpoint score: average of MIU decision exact match, MIU reasoning
  faithfulness, EIL task utility, and EIL low leakage (`1 - leakage_mean`),
  with hard gates for MIU valid rate and EIL failure rate.

## Run

Clone or pull this branch on `10.220.5.101`, then run:

```bash
cd /tmp/loyal_agent_docker
LOYAL_FORCE_RESTART=1 bash scripts/launch/run_phase1_e2m1_rollout200_101.sh
```

The script copies `.env` from an existing local/shared checkout when needed.
If the env file lives somewhere else, pass:

```bash
LOYAL_ENV_SOURCE=/path/to/.env LOYAL_FORCE_RESTART=1 bash scripts/launch/run_phase1_e2m1_rollout200_101.sh
```

## Active Paths

By default the active paths are on the host overlay disk, not CephFS:

- Run root: `/tmp/experiment_g_longtask_101/experiments/mixed_reward_ablation_phase1_parallel`
- Checkpoints: `/tmp/experiment_g_longtask_101/checkpoints`
- Ray temp: `/tmp/r101` (kept short to avoid Ray AF_UNIX socket path limits)
- Post-train outputs: `/tmp/experiment_g_longtask_101/evaluations/phase1-lambda050-e2m1-rollout200_posttrain`

## Speed And Stability Choices

- API concurrency is already raised from the conservative launcher defaults:
  `LOYAL_EIL_RM_MAX_CONCURRENT=32` and
  `LOYAL_EIL_GROUP_RM_MAX_CONCURRENT=4`.
- The script does not raise API concurrency further by default.  The previous
  run showed long-tail reward/API latency near checkpoint time, so pushing
  higher risks retries and unstable wall time.
- The safe speedups are operational: active checkpoint/output/Ray/data paths
  are local, training-time eval is disabled, and each 20-rollout segment is
  evaluated immediately before the next segment starts.
- If we need to change learning rate after the next completed checkpoint, set
  `/tmp/experiment_g_longtask_101/evaluations/phase1-lambda050-e2m1-rollout200_posttrain/phase1_next_lr.txt`
  to the new value before relaunching the next segment.  The launcher reads
  that file on each restart, so the current in-flight segment stays untouched.
- The GPU layout is fixed to `2 train + 2 rollout` for 4xA100.  `1+3` creates
  optimizer pressure; `3+1` bottlenecks rollout generation.
- The EIL/MIU rollout schedule is computed against the full 200-rollout
  horizon, even though training is restarted in 20-rollout segments for
  immediate checkpoint testing.
- The dynamic sampling filter is still active, but zero-variance eligible
  groups are retained by default through `LOYAL_RETAIN_ZERO_STD_GROUPS=1`.
  They keep batch shape stable and contribute zero GRPO advantage rather than
  being replacement-sampled indefinitely.
- Old checkpoints are removed only after their EIL/MIU test summaries exist
  with per-sample outputs, and they are not the current best or one of the two
  most recent checkpoints.
- The creative-generation stage is a strict checkpoint continuation from the
  selected best phase-1 step.  It passes `--ckpt-step <best_step>`, refuses
  `--no-load-optim/--no-load-rng/--finetune`, and keeps
  `--use-checkpoint-opt_param-scheduler` so optimizer moments, RNG, scheduler,
  and global step are carried across the stage boundary.
- A separate metrics watcher writes
  `/tmp/experiment_g_longtask_101/evaluations/phase1-lambda050-e2m1-rollout200_posttrain/metrics_trend.json`
  and `.csv` so the active best checkpoint, EIL/MIU reward deltas, and
  per-step quality metric deltas are visible while training continues.

## Acceptance

The first health gate is checkpoint `iter_0000019`:

```bash
test -s /tmp/experiment_g_longtask_101/checkpoints/mixed-v2-phase1-lambda050-e2m1-rollout200-phase1-seed1234/iter_0000019/common.pt
test -f /tmp/experiment_g_longtask_101/checkpoints/mixed-v2-phase1-lambda050-e2m1-rollout200-phase1-seed1234/iter_0000019/.metadata
```

The full acceptance gate is:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("/tmp/experiment_g_longtask_101/evaluations/phase1-lambda050-e2m1-rollout200_posttrain/acceptance.json")
payload = json.loads(path.read_text(encoding="utf-8"))
assert payload["status"] == "passed", payload
print("ACCEPTANCE PASSED", path)
PY
```
