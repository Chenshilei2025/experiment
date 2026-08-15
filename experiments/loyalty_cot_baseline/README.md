# Loyalty-aware CoT baseline

This experiment implements [Experiment 1](experiment_matrix.md). It compares
each base model under two policy-prompt conditions:

- `plain`: the existing EIL or MIU policy prompt unchanged, with template/API
  thinking explicitly disabled;
- `loyalty-cot`: the fixed Loyalty CoT instruction in `prompts.py` is appended
  to the existing system contract and thinking enabled. It requests internal
  reasoning but requires only the final response to be emitted.

The unit of execution is one `model × mechanism × condition` cell. The full
matrix has five models, two base conditions, and a post-training plain run for
the three open models. Each cell writes `manifest.json`, `per_sample.jsonl`,
and `summary.json`. Responses are scored by the production EIL/MIU judges and
aggregation code; this experiment does not introduce another metric.

## Local Hugging Face model

```bash
python3 -m experiments.loyalty_cot_baseline.run \
  --backend hf --model-name qwen3-4b-base --checkpoint /ssd/shilei/models/Qwen3-4B \
  --mechanism miu --condition plain --device cuda:0 \
  --output-dir artifacts/experiments/experiment_1/qwen3-4b/miu/base-plain
```

Repeat with `--condition loyalty-cot`, then with `--mechanism eil`. The script
sets Qwen template thinking from the condition: disabled for `plain`, enabled
for `loyalty-cot`. Do not pass a manual thinking flag, since that would make
the comparison ambiguous.

The command above is for direct execution on this host. `/models/Qwen3-4B` is
the corresponding path *inside* the training Docker container, where the host
model directory is mounted at `/models`; do not use that container-only path
when launching this Python command from the host.

For MIU, the runner uses 384 new tokens for `plain` and 2048 for
`loyalty-cot`: the latter budget includes Qwen's private thinking trace and
its final structured answer. Citation parsing accepts explicit `[E#]` tags
adjacent to their claim, including trailing punctuation such as `[E2].`; it
never invents a citation that the model omitted.

## Hosted / closed model

Configure a policy endpoint separately from the judge endpoints:

```bash
export LOYAL_EXPERIMENT_MODEL_BASE_URL='https://.../v1'
export LOYAL_EXPERIMENT_MODEL_MODEL='provider-model-id'
export LOYAL_EXPERIMENT_MODEL_API_KEY='...'
export LOYAL_EXPERIMENT_MODEL_JSON_MODE=0
```

For the plain control, the runner overrides any configured API thinking mode
and sends `thinking={"type":"disabled"}`. For the CoT condition it leaves
thinking enabled at the provider default while adding the baseline instruction.

```bash
python3 -m experiments.loyalty_cot_baseline.run \
  --backend api --model-name provider-model-id \
  --mechanism eil --condition loyalty-cot \
  --output-dir artifacts/experiments/experiment_1/provider-model/eil/base-loyalty-cot
```

Existing `LOYAL_EIL_*` and `LOYAL_MIU_*` environment variables continue to
configure the adversary and judge services. For EIL, start with
`--score-concurrency 2`: each sample uses four adversary temperatures.

## Matrix and parallelism

For the full planned table, see [experiment_matrix.md](experiment_matrix.md).
Each base model has these four cells:

```text
EIL/base-plain        EIL/base-loyalty-cot
MIU/base-plain        MIU/base-loyalty-cot
```

Each trained open model adds two post-training plain cells:

```text
EIL/posttrain-plain   MIU/posttrain-plain
```

For two GPUs, split the EIL JSONL into two disjoint files and run the same
model/condition cell in two processes with distinct output directories. Merge
the results with `scripts.evaluation.merge_eil_shards`. All shards must share
the model, prompt condition, decoding, and judge configuration.
