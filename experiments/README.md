# Experiments

This directory contains reproducible experiments that compare policy models
without creating a separate evaluator. Each run reuses the production EIL or
MIU prompt builders, judges, scorers, and metric aggregation.

`loyalty_cot_baseline/run.py` implements the first matrix:

`model × {EIL, MIU} × {base plain/no-thinking, base loyalty-CoT/thinking}`

The first experiment also evaluates the three trained open models under the
plain/no-thinking condition. See its experiment matrix for the complete table.

See that directory's README for commands and output conventions.

`adversary_diversity/run.py` is a fixed-response EIL rescore experiment for
testing temperature and adversary-model diversity without confounding it with
new policy generations.

`training_order/` runs Qwen3-4B MIU/EIL order conditions; `reward_ablation/`
runs independent single-task reward-coefficient conditions. Their conditions
are JSON files under `configs/`, run by `python -m scripts.experiment_runner`.
The runner records the exact config, data hashes, checkpoint name, and stage
status, so a new condition normally means copying one JSON file rather than
writing another launch program. Both use the fixed training and standard-test
protocol: Qwen adversary plus DeepSeek judges. Their configured standard
timeline is base-model MIU/EIL evaluation before training, followed by MIU/EIL
evaluation after every training stage at that stage's exported iteration.
`judge_sensitivity/` is the fixed-response judge counterpart to
`adversary_diversity/`; these two test-only experiments intentionally vary
judges or adversaries to measure evaluator sensitivity.

## Run records

Every newly started experiment creates these files directly in its
`--output-dir`:

- `command.json`: the exact command, working directory, and Python executable;
- `environment.json`: all `LOYAL_*` settings after redacting credential-like
  variables;
- `run.log`: stdout and stderr for test-only experiments, or Docker/Ray stage
  output for training experiments.

These are in addition to the experiment-specific `manifest.json`, per-response
JSONL, summaries, and (for training) resolved config and stage-completion
markers. Keep a run's log in its own output directory; `artifacts/logs/` is
only historical shared logging and should not be used as the canonical record
of a new condition.

## Execution paths

| Experiment | Entry | Shared production path |
| --- | --- | --- |
| Loyalty-CoT baseline | `experiments.loyalty_cot_baseline.run` | prompt condition → `scripts.evaluation.{miu,eil}` → production reward/scorers |
| Training order | `scripts.experiment_runner` + `training_order/configs/*.json` | baseline MIU/EIL → stages → `scripts/launch/run_training_container.sh` → recipes → export exact iteration → MIU/EIL |
| Reward ablation | `scripts.experiment_runner` + `reward_ablation/configs/*.json` | baseline MIU/EIL → one stage → same recipe path → export exact iteration → MIU/EIL |
| Adversary diversity | `experiments.adversary_diversity.run` | fixed responses → production EIL adversary, leakage, and utility scorers |
| Judge sensitivity | `experiments.judge_sensitivity.run` | fixed responses → production MIU/EIL scoring with the judge client substituted |
