"""将完成的 EIL 基线测试汇总追加到中文报告。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def number(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--training-container", required=True)
    args = parser.parse_args()
    result = json.loads(args.summary.read_text(encoding="utf-8"))
    failed_rate = result["n_failed"] / result["n_total"] if result["n_total"] else 0.0
    rows = [
        "## 最终基线结果", "",
        f"- 测试记录：{result['n_scored']} / {result['n_total']} 条完成评分；失败 {result['n_failed']} 条（{failed_rate:.2%}）。",
        f"- Task utility：{number(result['task_utility_mean'])}",
        f"- Leakage：{number(result['leakage_mean'])}",
        f"- Leakage 为零的比例：{number(result['leakage_zero_rate'])}",
        f"- Reward：{number(result['reward_mean'])}", "",
        "| family_domain | 总数 | 已评分 | task utility | leakage | leakage 为零比例 | reward |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, values in sorted(result["by_family_domain"].items()):
        rows.append(
            f"| {family} | {values['n']} | {values['n_scored']} | {number(values['task_utility_mean'])} | "
            f"{number(values['leakage_mean'])} | {number(values['leakage_zero_rate'])} | {number(values['reward_mean'])} |"
        )
    rows += ["", "## 训练交接", "", f"基线测试成功汇总后，启动器将在 GPU 0--5 启动 `{args.training_container}`。", ""]
    text = args.report.read_text(encoding="utf-8")
    marker = "## 最终基线结果\n\n待 656 条测试记录全部完成后自动追加：整体 task_utility、leakage、reward、评分失败率，以及 bargaining、gatekeeping、redress 的 `by_family_domain` 分解。"
    if marker not in text:
        raise RuntimeError("报告中找不到待替换的最终结果标记")
    args.report.write_text(text.replace(marker, "\n".join(rows).rstrip()), encoding="utf-8")


if __name__ == "__main__":
    main()
