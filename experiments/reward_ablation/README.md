# Reward 系数消融

固定基模 Qwen3-4B、Qwen adversary 与 DeepSeek judge。EIL 使用 `R = U - λL`，取 `λ ∈ {0, 0.5, 1.0, 2.0}`；MIU 使用 `R = -1 + (2-η)D + ηF`，取 `η ∈ {0, 0.5, 1.0}`。

EIL 与 MIU 是两个独立的消融矩阵：每个格子都从同一个 Qwen3-4B 基模重新开始，只训练一个机制，绝不先跑另一个机制，也不讨论训练顺序。执行器会先保存基模在 MIU/EIL 两个测试集上的 baseline，再在阶段结束后导出对应 iteration 并重测两个测试集；测试 evaluator 与训练保持同一套 Qwen adversary / DeepSeek judge 配置。

```bash
python3 -m scripts.experiment_runner \
  --config experiments/reward_ablation/configs/eil_lambda_1.json \
  --run-name v1 --output-dir artifacts/experiments/reward_ablation/v1/eil_lambda1/seed1234 \
  --set seed=1234 --set stages.0.rollouts=840
```

MIU 示例配置是 `configs/miu_eta_0_5.json`。复制一个小 JSON 并只改 reward 系数、seed 或 rollout 预算即可新增条件；所有条件走同一训练、checkpoint 和 provenance 路径。旧 CLI 仍可用作兼容包装。每个条件使用至少三个 seed；主表报告 EIL 的 utility/leakage 或 MIU 的 decision/faithfulness，而不是仅报告合成 reward。
