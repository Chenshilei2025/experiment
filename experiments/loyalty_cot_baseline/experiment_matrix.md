# Experiment 1: model, CoT, and loyalty-training comparison

The primary table compares five policy models under the same EIL and MIU
test-time scorer. Every cell below is run once on EIL and once on MIU.

| Model | Base plain / thinking off | Base Loyalty-CoT / thinking on | Post-training plain / thinking off |
|---|:---:|:---:|:---:|
| Qwen3-4B | yes | yes | yes |
| Llama-3.1-8B-Instruct | yes | yes | yes |
| GLM-Z1-9B-0414 | yes | yes | yes |
| Claude Opus 4.8 | yes | yes | not applicable |
| GPT-5.5 | yes | yes | not applicable |

This creates 13 cells per benchmark and 26 test runs in total. `plain` is the
existing policy prompt with thinking disabled. `loyalty-cot` appends the fixed
Loyalty CoT system instruction and enables thinking. The post-training column
must use the same plain/no-thinking condition as the base plain column.

## Run naming

Use this stable output layout:

```text
artifacts/experiments/experiment_1/
  <model-key>/
    eil/{base-plain,base-loyalty-cot,posttrain-plain}/
    miu/{base-plain,base-loyalty-cot,posttrain-plain}/
```

Closed models omit `posttrain-plain`. Each `manifest.json` records the exact
provider model identifier and endpoint-independent generation configuration.
For Claude Opus 4.8 and GPT-5.5, configure the provider's exact currently
available API model ID via `LOYAL_EXPERIMENT_MODEL_MODEL`; do not infer an ID
from this comparison label.

## Open-model preparation

The supported training profiles are:

| Model key | HF model directory | Megatron reference directory |
|---|---|---|
| `qwen3-4b` | `/ssd/shilei/models/Qwen3-4B` (container: `/models/Qwen3-4B`) | `/models/Qwen3-4B_torch_dist` |
| `llama3.1-8b-instruct` | `/models/Llama-3.1-8B-Instruct` (host: `/ssd/models/Llama-3.1-8B-Instruct`) | `/models/Llama-3.1-8B-Instruct_torch_dist` |
| `glm-z1-9b` | `/models/GLM-Z1-9B-0414` | `/models/GLM-Z1-9B-0414_torch_dist` |

After a model download, prepare its reference checkpoint once:

```bash
scripts/launch/prepare_model_checkpoint.sh glm-z1-9b
```

Then set `LOYAL_BASE_MODEL` to the same key before launching either EIL or MIU
training. The post-training test uses the exported HF checkpoint, not the
Megatron checkpoint directory.
