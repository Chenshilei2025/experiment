# Scripts layout

- `common/`: shared Python helpers used by training and evaluation.
- `data/`: conversion of source records into SLIME prompt datasets.
- `training/`: preflight validation, scheduling, rollouts, rewards, and filters.
- `evaluation/`: model evaluation and saved-response rescoring.
- `launch/`: shell entry points, container wrapper, environment loading, and judge relay.

Model lifecycle helpers:

- `export_final_checkpoint.sh`: converts the latest complete training checkpoint to a Hugging Face model in `artifacts/exported_models/` using the pinned Docker image.
- `run_test_container.sh`: runs MIU or EIL baseline/final test generation and scoring inside the pinned Docker image. Test JSONL is mounted read-only; results are written to a new directory under `artifacts/evaluations/`.

Use the launchers from `scripts/launch/`:

```bash
bash scripts/launch/run_training_container.sh miu
bash scripts/launch/run_training_container.sh eil
bash scripts/launch/run_loyal_smoke.sh
bash scripts/export_final_checkpoint.sh Qwen3-4B_loyal_api_baseline
bash scripts/run_test_container.sh miu baseline baseline_miu_v1
bash scripts/run_test_container.sh eil final final_eil_v1
bash scripts/launch/run_full_api_training_and_tests.sh run_001
```
