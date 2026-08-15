# Experiment runtime

Only shared code used by the experiments is kept here.

- `experiment_runner.py`: executes JSON-defined multi-stage training conditions.
- `common/`: API client and prompt-safe shared helpers.
- `data/prepare_slime.py`: canonical MIU/EIL record → SLIME prompt conversion.
- `training/`: preflight checks, MIU rollout repair, and SLIME reward hooks.
- `evaluation/`: shared generation, scoring, rescoring, and EIL shard merging.
- `launch/run-miu.sh`, `launch/run-eil.sh`: visible mechanism recipes (data, reward paths, hyperparameters).
- `launch/submit_training.sh`, `launch/run_training_container.sh`: shared Ray and Docker mechanics.
- `launch/model_profiles.sh`: Qwen/GLM/Llama model setup.
- `export_final_checkpoint.sh`, `run_test_container.sh`: trained-model export and standard testing.

Experiment conditions belong in `experiments/*/configs/`. They may override
training hyperparameters, seed, and rollout budgets, but evaluator endpoints,
evaluator models, and credentials remain in `.env`.

The default JSON `evaluation` plan evaluates the base model on MIU and EIL
before training, then exports the exact checkpoint iteration and evaluates both
benchmarks after every stage.  `manifest.json` records the labels, iterations,
and result directories, so order-training stages cannot be overwritten by a
later stage before they are tested.

`run-miu.sh` and `run-eil.sh` are the source of truth for their default
training, rollout, reward-worker, checkpoint, evaluation, and W&B settings.
`.env` contains only host resource selection, model storage, and remote API
service configuration.

All supported open base models (`qwen3-4b`, `glm-z1-9b`, and
`llama3.1-8b-instruct`) use those same MIU/EIL recipes. `launch/model_profiles.sh`
is the only model-specific layer: it selects the HF/reference checkpoints,
SLIME architecture arguments, and chat-template options.
