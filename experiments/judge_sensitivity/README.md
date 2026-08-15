# Judge sensitivity（测试集）

这是与 `experiments/adversary_diversity/` 配套的**固定回答重评分**实验：固定一个已完成 EIL 或 MIU 测试运行的 `per_sample.jsonl`，逐个切换 judge，而不重新生成回答也不训练 policy。因此测到的是 judge 分数差异。

每个 `--judge` 的格式是 `显示名=环境变量前缀`，该前缀读取 `<PREFIX>_BASE_URL`、`<PREFIX>_MODEL`、`<PREFIX>_API_KEY`。EIL 的 adversary 不在这里切换，而是使用 `LOYAL_EIL_ADVERSARY_*`；每题四温度 adversary 推断只生成一次，再原样交给全部 judge。若要比较 adversary，请使用配套的 `adversary_diversity` 实验。

```bash
export JUDGE_A_BASE_URL='https://provider-a.example/v1'
export JUDGE_A_MODEL='judge-a'
export JUDGE_A_API_KEY='...'
export JUDGE_B_BASE_URL='https://provider-b.example/v1'
export JUDGE_B_MODEL='judge-b'
export JUDGE_B_API_KEY='...'

python3 -m experiments.judge_sensitivity.run \
  --mechanism eil \
  --source-jsonl artifacts/evaluations/eil_final_run/per_sample.jsonl \
  --judge judge-a=JUDGE_A --judge judge-b=JUDGE_B \
  --output-dir artifacts/experiments/judge_sensitivity/eil_final_run
```

输出的 `per_response.jsonl` 保存每个 judge 的逐题分数，`summary.json` 保存每个 judge 的生产评测汇总及指标范围。标准训练与标准测试固定为 Qwen adversary、DeepSeek judge；本目录只用于额外检验换 judge 后结论是否稳定。
