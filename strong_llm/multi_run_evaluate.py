"""
多轮实验结果汇总与可视化

用法:
  # 由 multi_run.py --evaluate 自动调用
  # 也可独立运行:
  python multi_run_evaluate.py --variants 1,2,3 --runs 3

功能:
  1. 对每个 v{V}_r{R}，调用 evaluate.main() 生成完整评估文件（已存在则跳过）
  2. 对每个 v{V}_r{R}，调用 plot.main() 生成完整图表集（已存在则跳过）
  3. 读取所有轮次的 summary_metrics.csv，按策略计算 mean ± std，
     写入 aggregate/ 并生成汇总图
"""

from __future__ import annotations
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import config

MULTI_RESULTS_ROOT = config.OUTPUT_DIR / "multi_run"

# 需要汇总的关键指标（列名与新版 evaluate.py summary_metrics.csv 保持一致）
KEY_METRICS = [
    "F1",          # 字段级 micro F1（主指标）
    "precision",   # 字段级 micro Precision
    "recall",      # 字段级 micro Recall
    "null_accuracy",
    "field_macro_f1",
    "field_supported_macro_f1",
    "record_f1",        # 记录级 F1
    "record_precision", # 记录级 Precision
    "record_recall",    # 记录级 Recall
    "record_pred_total",
    "record_gold_total",
    "record_tp",
]


# ============================================================
# 逐轮评估与画图
# ============================================================

def evaluate_one_run(variant: int, run: int, multi_root: Path) -> None:
    """对单轮 results 运行完整评估，覆盖已有文件。"""
    results_dir = multi_root / f"v{variant}_r{run}" / "results"
    eval_dir    = multi_root / f"v{variant}_r{run}" / "evaluation"

    if not results_dir.exists():
        print(f"  [评估] 跳过 v{variant}_r{run}: results 目录不存在")
        return

    print(f"  [评估] 运行 v{variant}_r{run} ...")
    import evaluate as eval_mod
    eval_mod.main(results_dir=results_dir, output_dir=eval_dir)
    print(f"  [评估] 完成: {eval_dir}")


def plot_one_run(variant: int, run: int, multi_root: Path) -> None:
    """对单轮生成完整图表集，覆盖已有文件。"""
    run_dir     = multi_root / f"v{variant}_r{run}"
    figures_dir = run_dir / "figures"

    if not (run_dir / "evaluation" / "summary_metrics.csv").exists():
        print(f"  [画图] 跳过 v{variant}_r{run}: 评估文件不存在")
        return

    print(f"  [画图] 运行 v{variant}_r{run} ...")
    _copy_bo_data(run_dir)
    import plot as plot_mod
    plot_mod.main(output_dir=figures_dir, results_dir=run_dir)
    print(f"  [画图] 完成: {figures_dir}")


# ============================================================
# 汇总求均值
# ============================================================

def compute_stats(values: List[float]) -> Dict[str, float]:
    """计算 mean / std / min / max。"""
    valid = [v for v in values if v is not None]
    if not valid:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    n = len(valid)
    mean = sum(valid) / n
    variance = sum((x - mean) ** 2 for x in valid) / n if n > 1 else 0.0
    return {
        "mean": round(mean, 4),
        "std": round(variance ** 0.5, 4),
        "min": round(min(valid), 4),
        "max": round(max(valid), 4),
        "n": n,
    }


def collect_all_results(
    multi_root: Path,
    variants: List[int],
    n_runs: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """读取所有轮次 summary_metrics.csv，返回 {strategy: [entry, ...]}。"""
    results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for v in variants:
        for r in range(1, n_runs + 1):
            csv_path = multi_root / f"v{v}_r{r}" / "evaluation" / "summary_metrics.csv"
            if not csv_path.exists():
                print(f"  [汇总] 未找到: {csv_path}")
                continue

            with csv_path.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    strategy = row.get("strategy", "")
                    entry: Dict[str, Any] = {"variant": v, "run": r}
                    for m in KEY_METRICS:
                        try:
                            entry[m] = float(row.get(m, ""))
                        except (ValueError, TypeError):
                            entry[m] = None
                    results[strategy].append(entry)

    return dict(results)


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# CSV 均值聚合（供汇总画图使用）
# ============================================================

def _avg_csv(all_rows: List[Dict], key_cols: List[str]) -> List[Dict]:
    """按 key_cols 分组，对其余数值列取均值。保持首次出现的列顺序。"""
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in all_rows:
        key = tuple(row.get(k, "") for k in key_cols)
        groups[key].append(row)

    if not all_rows:
        return []
    all_cols = list(all_rows[0].keys())
    num_cols = [c for c in all_cols if c not in key_cols]

    result: List[Dict] = []
    for key, rows in groups.items():
        out: Dict = dict(zip(key_cols, key))
        for col in num_cols:
            vals = []
            for r in rows:
                try:
                    v = r.get(col, "")
                    if v != "" and v is not None:
                        vals.append(float(v))
                except (ValueError, TypeError):
                    pass
            out[col] = round(sum(vals) / len(vals), 6) if vals else 0.0
        result.append(out)
    return result


def average_eval_csvs(multi_root: Path, variants: List[int], n_runs: int) -> Path:
    """
    从所有轮次的 evaluation/ 目录读取标准评估 CSV，
    按关键列分组取均值，写入 aggregate/evaluation/，
    返回该目录供 plot.main() 使用。
    """
    agg_eval_dir = multi_root / "aggregate" / "evaluation"
    agg_eval_dir.mkdir(parents=True, exist_ok=True)

    csv_specs = [
        ("summary_metrics.csv",         ["strategy"]),
        ("per_field_metrics.csv",        ["strategy", "field"]),
        ("per_field_pr_split.csv",       ["strategy", "field"]),
        ("per_doc_length_metrics.csv",   ["strategy", "bin"]),
        ("per_record_count_metrics.csv", ["strategy", "bin"]),
    ]

    for csv_name, key_cols in csv_specs:
        all_rows: List[Dict] = []
        for v in variants:
            for r in range(1, n_runs + 1):
                path = multi_root / f"v{v}_r{r}" / "evaluation" / csv_name
                if path.exists():
                    with path.open("r", encoding="utf-8") as f:
                        all_rows.extend(list(csv.DictReader(f)))

        if not all_rows:
            print(f"  [汇总图] 跳过 {csv_name}: 各轮未找到数据")
            continue

        averaged = _avg_csv(all_rows, key_cols)
        fieldnames = list(all_rows[0].keys())
        _write_csv(agg_eval_dir / csv_name, averaged, fieldnames)
        print(f"  [汇总图] 均值 CSV 已生成: {csv_name} ({len(averaged)} 行, 来自 {len(all_rows)} 条原始记录)")

    # 聚合 evaluation_details.json（fig1c 需要 per_paper 数据）
    _average_eval_json(multi_root, variants, n_runs, agg_eval_dir)

    return agg_eval_dir


def _average_eval_json(
    multi_root: Path,
    variants: List[int],
    n_runs: int,
    agg_eval_dir: Path,
) -> None:
    """
    读取所有轮次的 evaluation_details.json，
    按 (strategy, paper_id) 对 per_paper 数值字段取均值，
    写入 aggregate/evaluation/evaluation_details.json 供 fig1c 使用。
    """
    # { strategy: { paper_id: [entry, ...] } }
    per_paper_groups: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    found = 0
    for v in variants:
        for r in range(1, n_runs + 1):
            json_path = multi_root / f"v{v}_r{r}" / "evaluation" / "evaluation_details.json"
            if not json_path.exists():
                continue
            found += 1
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for strategy, strat_data in data.items():
                for entry in strat_data.get("per_paper", []):
                    paper_id = entry.get("paper_id", "")
                    per_paper_groups[strategy][paper_id].append(entry)

    if not per_paper_groups:
        print("  [汇总图] 跳过 evaluation_details.json: 各轮未找到数据")
        return

    # 数值字段取均值，布尔字段取均值（float），字符串字段取第一个值
    NUM_FIELDS = {"gold_record_count", "pred_record_count", "record_tp"}
    BOOL_FIELDS = {"has_prediction", "is_failed"}

    output: Dict[str, Any] = {}
    for strategy, paper_map in per_paper_groups.items():
        averaged_papers: List[Dict] = []
        for paper_id, entries in paper_map.items():
            avg_entry: Dict[str, Any] = {"strategy": strategy, "paper_id": paper_id}
            for field in NUM_FIELDS | BOOL_FIELDS:
                vals = []
                for e in entries:
                    v = e.get(field)
                    if v is not None:
                        vals.append(float(v))
                avg_entry[field] = round(sum(vals) / len(vals), 4) if vals else 0.0
            averaged_papers.append(avg_entry)
        output[strategy] = {"per_paper": averaged_papers}

    out_path = agg_eval_dir / "evaluation_details.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    total_papers = sum(len(v["per_paper"]) for v in output.values())
    print(f"  [汇总图] 均值 JSON 已生成: evaluation_details.json "
          f"({len(output)} 策略, {total_papers} 篇论文, 来自 {found} 个轮次)")


def _copy_bo_data(target_dir: Path, multi_root: Path = None) -> None:
    """
    将 study.db 复制到 target_dir/bo/，让 plot.main() 能生成 fig3/fig4。
    查找顺序：multi_run 各轮 → output/results-0 → output/results。
    已存在则跳过。
    """
    import shutil

    dest = target_dir / "bo" / "study.db"

    # 候选路径
    candidates: List[Path] = []
    if multi_root:
        candidates += [
            multi_root / f"v{v}_r{r}" / "bo" / "study.db"
            for v in range(1, 4) for r in range(1, 4)
        ]
    candidates += [
        config.OUTPUT_DIR / "results-0" / "bo" / "study.db",
        config.OUTPUT_DIR / "results"   / "bo" / "study.db",
    ]

    for src in candidates:
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            print(f"  [画图] BO study.db 已复制: {src.name} → {dest.parent}")
            return

    print("  [画图] 未找到 study.db，跳过 fig3/fig4")


def _plot_aggregate(summary_rows: List[Dict], output_dir: Path) -> None:
    """生成汇总图：P/R/F1 均值 ± std 柱状图（PDF + PNG）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[汇总] matplotlib 未安装, 跳过画图")
        return

    if not summary_rows:
        return

    strategies = [r["strategy"] for r in summary_rows]
    pp   = [r.get("precision_mean", 0) for r in summary_rows]
    pr   = [r.get("recall_mean",    0) for r in summary_rows]
    pf1  = [r.get("F1_mean",        0) for r in summary_rows]
    pp_e = [r.get("precision_std",  0) for r in summary_rows]
    pr_e = [r.get("recall_std",     0) for r in summary_rows]
    pf1_e= [r.get("F1_std",         0) for r in summary_rows]

    x = np.arange(len(strategies))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(strategies) * 1.8), 5))
    ax.bar(x - w, pp,  w, yerr=pp_e,  label="Precision", color="#4C72B0", capsize=3, alpha=0.85)
    ax.bar(x,     pr,  w, yerr=pr_e,  label="Recall",    color="#55A868", capsize=3, alpha=0.85)
    b3 = ax.bar(x + w, pf1, w, yerr=pf1_e, label="F1",  color="#DD8452", capsize=3, alpha=0.85)

    for bar, val in zip(b3, pf1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    n_runs = summary_rows[0].get("n_runs", "?")
    ax.set_ylabel("Score")
    ax.set_title(f"Multi-Run Aggregated P / R / F1  (mean ± std,  n = {n_runs} runs)")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    for fmt in ("pdf", "png"):
        fig.savefig(output_dir / f"aggregate_summary.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[汇总] 汇总图已保存: aggregate_summary.pdf / .png")


# ============================================================
# 主流程
# ============================================================

def aggregate_all(
    multi_root: Path = None,
    variants: List[int] = None,
    n_runs: int = 1,
    strategies: List[str] = None,
    run_evaluate: bool = False,
    run_plot: bool = False,
) -> None:
    """
    主流程：
      1. (可选) 逐轮调用 evaluate.main()
      2. (可选) 逐轮调用 plot.main()
      3. 读取所有 summary_metrics.csv → 计算 mean ± std → 写入 aggregate/
    """
    if multi_root is None:
        multi_root = MULTI_RESULTS_ROOT
    if variants is None:
        variants = [1, 2, 3]

    print(f"\n{'=' * 60}")
    print(f"[汇总] variants={variants}, n_runs={n_runs}")
    print(f"{'=' * 60}")

    # --- Step 1: 逐轮评估 ---
    if run_evaluate:
        print("\n[Step 1] 逐轮评估 ...")
        for v in variants:
            for r in range(1, n_runs + 1):
                evaluate_one_run(v, r, multi_root)

    # --- Step 2: 逐轮画图 ---
    if run_plot:
        print("\n[Step 2] 逐轮画图 ...")
        for v in variants:
            for r in range(1, n_runs + 1):
                plot_one_run(v, r, multi_root)

    # --- Step 3: 汇总求均值 ---
    print("\n[Step 3] 汇总所有轮次结果 ...")
    all_results = collect_all_results(multi_root, variants, n_runs)

    if not all_results:
        print("[汇总] 未找到任何评估结果!")
        return

    used_strategies = strategies or sorted(all_results.keys())

    # 按策略计算 mean ± std
    summary_rows: List[Dict] = []
    detail_rows:  List[Dict] = []

    for strategy in used_strategies:
        entries = all_results.get(strategy, [])
        if not entries:
            continue

        row: Dict[str, Any] = {"strategy": strategy, "n_runs": len(entries)}
        for m in KEY_METRICS:
            vals = [e[m] for e in entries if e.get(m) is not None]
            stats = compute_stats(vals)
            row[f"{m}_mean"] = stats["mean"]
            row[f"{m}_std"]  = stats["std"]
            row[f"{m}_min"]  = stats["min"]
            row[f"{m}_max"]  = stats["max"]
        summary_rows.append(row)

        for e in entries:
            dr: Dict[str, Any] = {"strategy": strategy, "variant": e["variant"], "run": e["run"]}
            for m in KEY_METRICS:
                dr[m] = e.get(m)
            detail_rows.append(dr)

    # 写汇总 CSV
    agg_dir = multi_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    if summary_rows:
        _write_csv(agg_dir / "aggregate_summary.csv", summary_rows, list(summary_rows[0].keys()))
        print(f"[汇总] 汇总表已保存: {agg_dir / 'aggregate_summary.csv'}")

    if detail_rows:
        _write_csv(agg_dir / "all_runs_detail.csv", detail_rows, list(detail_rows[0].keys()))
        print(f"[汇总] 明细表已保存: {agg_dir / 'all_runs_detail.csv'}")

    # 写 JSON
    json_path = agg_dir / "aggregate_summary.json"
    json_path.write_text(
        json.dumps({"summary": summary_rows, "detail": detail_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[汇总] JSON 已保存: {json_path}")

    # 打印表格
    print(f"\n{'=' * 90}")
    print(f"{'策略':<12} {'n':>3} {'F1 mean':>10} {'F1 std':>9} "
          f"{'Prec mean':>10} {'Recall mean':>12} {'Field F1':>9}")
    print("-" * 90)
    for row in summary_rows:
        print(
            f"{row['strategy']:<12} "
            f"{row.get('n_runs', 0):>3} "
            f"{row.get('F1_mean', 0):>10.4f} "
            f"{row.get('F1_std',  0):>9.4f} "
            f"{row.get('precision_mean', 0):>10.4f} "
            f"{row.get('recall_mean',    0):>12.4f} "
            f"{row.get('field_macro_f1_mean', 0):>9.4f}"
        )
    print("=" * 90)

    # 生成汇总图（均值±std 概览柱状图）
    _plot_aggregate(summary_rows, agg_dir)

    # --- Step 4: 生成与单轮等价的完整图表集（均值版） ---
    print("\n[Step 4] 生成完整汇总图表集 (均值版) ...")
    average_eval_csvs(multi_root, variants, n_runs)
    _copy_bo_data(agg_dir, multi_root)
    agg_figures_dir = agg_dir / "figures"
    agg_figures_dir.mkdir(parents=True, exist_ok=True)
    import plot as plot_mod
    plot_mod.main(output_dir=agg_figures_dir, results_dir=agg_dir)
    print(f"[汇总] 完整图表集已保存: {agg_figures_dir}")

    print(f"\n[汇总] 完成 → {agg_dir}")


def main():
    """独立运行入口。"""
    import argparse
    parser = argparse.ArgumentParser(
        description="多轮实验评估汇总",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅汇总（各轮评估文件已存在）
  python multi_run_evaluate.py --variants 1,2,3 --runs 3

  # 重新评估每轮 + 画图 + 汇总
  python multi_run_evaluate.py --variants 1,2,3 --runs 3 --evaluate --plot

  # 只针对部分策略
  python multi_run_evaluate.py --strategies E0,E3,E5,E6 --variants 1,2,3 --runs 3
        """,
    )
    parser.add_argument("--variants",   type=str, default="1,2,3", help="变体编号, 逗号分隔")
    parser.add_argument("--runs",       type=int, default=3,        help="每变体轮数 (默认: 3)")
    parser.add_argument("--strategies", type=str, default=None,     help="策略列表, 逗号分隔 (默认: 全部)")
    parser.add_argument("--evaluate",   action="store_true",        help="对每轮调用 evaluate.main()")
    parser.add_argument("--plot",       action="store_true",        help="对每轮调用 plot.main()")
    args = parser.parse_args()

    variants   = [int(v.strip()) for v in args.variants.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")] if args.strategies else None

    aggregate_all(
        multi_root=MULTI_RESULTS_ROOT,
        variants=variants,
        n_runs=args.runs,
        strategies=strategies,
        run_evaluate=args.evaluate,
        run_plot=args.plot,
    )


if __name__ == "__main__":
    main()
