# Adversary diversity experiment

This experiment tests whether EIL leakage findings depend on a single blind
adversary sample. It holds saved policy responses fixed and rescoring them with
each requested `adversary model × temperature` branch. Every branch uses the
production EIL adversary prompt, leakage judge, utility judge, slot weighting,
and final reward formula.

It never regenerates a policy answer. Give it an EIL `per_sample.jsonl` from a
completed experiment or standard evaluation run. This is a test-only
sensitivity study; normal training and standard testing remain fixed to the
Qwen adversary and DeepSeek judges.

## Conditions

Use temperatures `0.0,0.3,0.6,0.8,1.0` for each adversary model. This extends
the production four-temperature test ensemble (`0.3,0.6,0.8,1.0`) with the
deterministic `0.0` ablation. Supply at least two adversary models from distinct
providers or architectures when credentials are available.

Each `--adversary` value has the form `display-name=ENV_PREFIX`. The prefix
reads `<PREFIX>_BASE_URL`, `<PREFIX>_MODEL`, and `<PREFIX>_API_KEY` using the
same OpenAI-compatible client configuration as the main evaluator.

```bash
export DIVERSITY_ADVERSARY_A_BASE_URL='https://provider-a.example/v1'
export DIVERSITY_ADVERSARY_A_MODEL='model-a'
export DIVERSITY_ADVERSARY_A_API_KEY='...'
export DIVERSITY_ADVERSARY_B_BASE_URL='https://provider-b.example/v1'
export DIVERSITY_ADVERSARY_B_MODEL='model-b'
export DIVERSITY_ADVERSARY_B_API_KEY='...'

python3 -m experiments.adversary_diversity.run \
  --source-jsonl artifacts/experiments/experiment_1/qwen3-4b/eil/base-plain/per_sample.jsonl \
  --adversary model-a=DIVERSITY_ADVERSARY_A \
  --adversary model-b=DIVERSITY_ADVERSARY_B \
  --output-dir artifacts/experiments/adversary_diversity/qwen3-4b_eil_base-plain
```

The normal `LOYAL_EIL_LEAKAGE_JUDGE_*` and `LOYAL_EIL_UTILITY_JUDGE_*`
variables configure the fixed judges. Keep those judges, their prompts, and the
test records unchanged across all branches.

## Outputs

- `per_response.jsonl`: every fixed policy response plus every branch's blind
  inference, judge-grounded slot details, slot scores, utility, leakage,
  reward, weighted exposure, and error if unavailable.
- `summary.json`: per-branch means, full-ensemble reward, and the following
  diversity evidence:
  - `fact_diversity`: judge-grounded semantic fact union, pairwise fact-set
    Jaccard distance, and facts unique to one branch. A fact cited by the
    leakage judge is keyed by its recovered protected slot; facts without a
    slot link use normalized text as a clearly labelled fallback.
  - `slot_coverage`: protected slots recovered by the ensemble, the mean share
    a single branch misses relative to that union, and pairwise slot-set
    distance. `pairwise_adversary_recovered_slot_jaccard_distance_mean`
    excludes direct-reply-only leakage, so it specifically measures what the
    adversary inferred differently. The `same_model_different_temperature_*`
    and `same_temperature_different_model_*` fields isolate sampling diversity
    from adversary-model diversity; they are `null` when the requested grid
    has no matching pair.
  - `risk_disagreement`: frequency that branches assign different leakage
    levels or slot scores, plus the per-response weighted-exposure standard
    deviation and range.
  - `single_branch_reward_spearman_with_ensemble`: whether any one branch
    preserves the ensemble's ranking of policy responses.

For the paper, report the ensemble slot coverage alongside the single-branch
missing-slot rate and adversary-only slot-set distance. Then compare the full
ensemble with the `T=0.6` single-branch ablation and report rank stability and
risk-disagreement. A branch is never trusted as a label by itself: the fixed
leakage judge maps its blind inferences against the true protected slots.
