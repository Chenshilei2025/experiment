# MIU/EIL 训练顺序实验

比较 `miu-eil` 与 `eil-miu`。两种条件均使用 Qwen3-4B、同一份 train split、相同 MIU/EIL rollout 预算、相同 seed，以及固定的 Qwen adversary / DeepSeek judge。每个条件必须使用独立 checkpoint 名称。

每个条件使用同一条固定评测时间线，且不需要额外训练：训练开始前先在 **MIU 与 EIL 两个测试集**上测基模；每一阶段完成后，执行器立刻冻结该 iteration、导出 HF checkpoint，并在两个测试集上评测，然后才允许下一阶段继续共享 checkpoint。因此：

- `miu-eil` 在 MIU 后得到 **MIU-only**，在 EIL 后得到 **MIU→EIL**；
- `eil-miu` 在 EIL 后得到 **EIL-only**，在 MIU 后得到 **EIL→MIU**。

评测计划、每个 checkpoint iteration 与所有评测目录都会记录在运行目录的 `manifest.json` 中。默认输出为 `artifacts/evaluations/{miu,eil}_{baseline,final}_<checkpoint>-{baseline,stageN-<mechanism>}`；每个目录都有独立的 `summary.json` 与逐样本结果。

每个条件是版本控制的 JSON；同一执行器用于所有训练实验：

```bash
python3 -m scripts.experiment_runner \
  --config experiments/training_order/configs/miu_eil.json \
  --run-name v1 --output-dir artifacts/experiments/training_order/v1/miu-eil/seed1234 \
  --set seed=1234 --set stages.0.rollouts=660 --set stages.1.rollouts=840
```

把 config 换为 `eil_miu.json` 即可运行反向顺序。每一条件建议至少三个 seed。输出目录保存原始配置、解析后的配置、数据 SHA-256 和逐阶段 manifest，因而不需要每个实验各自实现这些逻辑。原 `python -m experiments.training_order.run` 命令仍可用，但只是兼容包装。
