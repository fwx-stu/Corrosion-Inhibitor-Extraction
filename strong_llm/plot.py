"""
出版级图表生成

7 种图表，每个绘图函数先检测数据文件是否存在, 跳过缺失的。
支持阶段 A (仅 E0/E1) 和阶段 B (E0-E4) 增量适配。

风格: matplotlib + seaborn, Nature 风格, 300 DPI, PDF + PNG 双格式。
"""

from __future__ import annotations

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import config


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_fig(fig, name: str, output_dir: Path):
    """保存为 PDF + PNG 双格式。"""
    _ensure_dir(output_dir)
    fig.savefig(output_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    print(f"  [plot] 保存: {name}.pdf / .png")


def _load_csv(path: Path):
    """加载 CSV 为字典列表。"""
    import csv
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _setup_style():
    """设置 Nature 风格。"""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 1.0,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
    })


# ============================================================
# Fig.1 — 策略对比柱状图
# ============================================================
def plot_strategy_comparison(output_dir: Path, eval_dir: Path):
    """Record F1 + Field macro F1 柱状图。"""
    csv_path = eval_dir / "summary_metrics.csv"
    rows = _load_csv(csv_path)
    if not rows:
        print("  [plot] 跳过 Fig.1: summary_metrics.csv 不存在")
        return

    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    strategies = [r["strategy"] for r in rows]
    record_f1 = [float(r.get("F1", 0)) for r in rows]
    field_f1 = [float(r.get("field_macro_f1", 0)) for r in rows]

    x = np.arange(len(strategies))
    width = 0.35

    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, record_f1, width, label="Record F1", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, field_f1, width, label="Field Macro F1", color="#DD8452", edgecolor="white")

    ax.set_ylabel("F1 Score")
    ax.set_title("Strategy Comparison: Record F1 vs Field Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 标注数值
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save_fig(fig, "fig1_strategy_comparison", output_dir)
    plt.close(fig)


# ============================================================
# Fig.1b — Precision / Recall / F1 三指标对比
# ============================================================
def plot_prf_comparison(output_dir: Path, eval_dir: Path):
    """Precision, Recall, F1 三组柱状图对比。"""
    csv_path = eval_dir / "summary_metrics.csv"
    rows = _load_csv(csv_path)
    if not rows:
        print("  [plot] 跳过 Fig.1b: summary_metrics.csv 不存在")
        return

    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    strategies = [r["strategy"] for r in rows]
    precision = [float(r.get("precision", 0)) for r in rows]
    recall    = [float(r.get("recall", 0)) for r in rows]
    f1        = [float(r.get("F1", 0)) for r in rows]

    x = np.arange(len(strategies))
    width = 0.25
    colors = ["#4C72B0", "#55A868", "#DD8452"]

    fig, ax = plt.subplots(figsize=(max(6, len(strategies) * 2), 5))
    b1 = ax.bar(x - width, precision, width, label="Precision", color=colors[0], edgecolor="white")
    b2 = ax.bar(x,         recall,    width, label="Recall",    color=colors[1], edgecolor="white")
    b3 = ax.bar(x + width, f1,        width, label="F1",        color=colors[2], edgecolor="white")

    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 by Strategy (Record-level, Partial Match)")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    _save_fig(fig, "fig1b_prf_comparison", output_dir)
    plt.close(fig)


# ============================================================
# Fig.1c — 逐篇论文覆盖率散点图
# ============================================================
def plot_per_paper_coverage(output_dir: Path, eval_dir: Path):
    """每篇论文的 gold vs pred 记录数散点图，气泡大小=TP。"""
    json_path = eval_dir.parent / "evaluation" / "evaluation_details.json"
    if not json_path.exists():
        # 尝试同目录
        json_path = eval_dir / "evaluation_details.json"
    if not json_path.exists():
        print("  [plot] 跳过 Fig.1c: evaluation_details.json 不存在")
        return

    import matplotlib.pyplot as plt
    _setup_style()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    strategies = list(data.keys())
    n_strats = len(strategies)

    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#E377C2"]
    fig, axes = plt.subplots(1, n_strats, figsize=(5 * n_strats, 5), sharey=True)
    if n_strats == 1:
        axes = [axes]

    for ax, (strategy, color) in zip(axes, zip(strategies, palette)):
        per_paper = data[strategy].get("per_paper", [])
        gold_counts = [p["gold_record_count"] for p in per_paper]
        pred_counts = [p["pred_record_count"] for p in per_paper]
        tp_counts   = [p.get("record_tp", p.get("record_partial_tp", 0)) for p in per_paper]

        sizes = [max(20, t * 8) for t in tp_counts]
        ax.scatter(gold_counts, pred_counts, s=sizes, alpha=0.6, color=color, edgecolors="white", linewidths=0.5)

        # 对角线（完美预测线）
        max_val = max(max(gold_counts, default=1), max(pred_counts, default=1)) + 2
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=1)

        ax.set_xlabel("Gold Records")
        ax.set_ylabel("Predicted Records")
        ax.set_title(f"{strategy}\n(bubble size ~ TP)")
        ax.grid(alpha=0.2)

    fig.suptitle("Per-paper: Gold vs Predicted Record Count", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "fig1c_per_paper_coverage", output_dir)
    plt.close(fig)


# ============================================================
# Fig.2 — 字段级热力图
# ============================================================
def plot_field_heatmap(output_dir: Path, eval_dir: Path):
    """策略 × 字段 F1 热力图。"""
    csv_path = eval_dir / "per_field_metrics.csv"
    rows = _load_csv(csv_path)
    if not rows:
        print("  [plot] 跳过 Fig.2: per_field_metrics.csv 不存在")
        return

    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    strategies = sorted(set(r["strategy"] for r in rows))
    fields = config.EVAL_FIELDS

    matrix = np.zeros((len(strategies), len(fields)))
    for r in rows:
        if r["strategy"] in strategies and r["field"] in fields:
            si = strategies.index(r["strategy"])
            fi = fields.index(r["field"])
            matrix[si, fi] = float(r.get("f1", 0))

    fig, ax = plt.subplots(figsize=(12, max(3, len(strategies) * 0.8 + 1)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels([f.replace("_", "\n") for f in fields], rotation=45, ha="right")
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies)
    ax.set_title("Field-level F1 by Strategy")

    # 标注数值
    for i in range(len(strategies)):
        for j in range(len(fields)):
            color = "white" if matrix[i, j] > 0.6 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="F1")
    fig.tight_layout()
    _save_fig(fig, "fig2_field_heatmap", output_dir)
    plt.close(fig)


# ============================================================
# Fig.3 — BO 收敛曲线
# ============================================================
def plot_bo_convergence(output_dir: Path, bo_dir: Path):
    """贝叶斯优化目标值收敛曲线。"""
    db_path = bo_dir / "study.db"
    if not db_path.exists():
        print("  [plot] 跳过 Fig.3: study.db 不存在")
        return

    try:
        import optuna
    except ImportError:
        print("  [plot] 跳过 Fig.3: optuna 未安装")
        return

    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    storage = f"sqlite:///{db_path}"
    studies = optuna.study.get_all_study_names(storage=storage)
    if not studies:
        return
    study = optuna.load_study(study_name=studies[0], storage=storage)

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        return

    values = [t.value for t in trials]
    best_so_far = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(values) + 1), values, "o-", alpha=0.4, markersize=4, label="Trial value")
    ax.plot(range(1, len(best_so_far) + 1), best_so_far, "r-", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Record Partial F1")
    ax.set_title("Bayesian Optimization Convergence")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, "fig3_bo_convergence", output_dir)
    plt.close(fig)


# ============================================================
# Fig.4 — 超参数重要度
# ============================================================
def plot_param_importance(output_dir: Path, bo_dir: Path):
    """Optuna 超参数重要度。"""
    db_path = bo_dir / "study.db"
    if not db_path.exists():
        print("  [plot] 跳过 Fig.4: study.db 不存在")
        return

    try:
        import optuna
        from optuna.importance import get_param_importances
    except ImportError:
        print("  [plot] 跳过 Fig.4: optuna 未安装")
        return

    import matplotlib.pyplot as plt
    _setup_style()

    storage = f"sqlite:///{db_path}"
    studies = optuna.study.get_all_study_names(storage=storage)
    if not studies:
        return
    study = optuna.load_study(study_name=studies[0], storage=storage)

    try:
        importances = get_param_importances(study)
    except Exception:
        print("  [plot] 跳过 Fig.4: 不够 trials 计算重要度")
        return

    names = list(importances.keys())
    values = list(importances.values())

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.5)))
    ax.barh(names, values, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title("Hyperparameter Importance")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, "fig4_param_importance", output_dir)
    plt.close(fig)


# ============================================================
# Fig.5 — 消融实验柱状图
# ============================================================
def plot_ablation(output_dir: Path, ablation_dir: Path):
    """消融实验对比柱状图。"""
    json_path = ablation_dir / "ablation_results.json"
    if not json_path.exists():
        print("  [plot] 跳过 Fig.5: ablation_results.json 不存在")
        return

    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    names = [d["ablation_name"] for d in data]
    f1_values = [d.get("F1", d.get("record_partial_f1", 0)) for d in data]

    colors = ["#4C72B0" if n == "baseline" else "#DD8452" for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), f1_values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Record F1")
    ax.set_title("Ablation Study")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save_fig(fig, "fig5_ablation", output_dir)
    plt.close(fig)


# ============================================================
# Fig.6 — Cross-LLM 雷达图
# ============================================================
def plot_cross_llm_radar(output_dir: Path, ablation_dir: Path):
    """多 LLM 对比雷达图。"""
    json_path = ablation_dir / "cross_llm_results.json"
    if not json_path.exists():
        print("  [plot] 跳过 Fig.6: cross_llm_results.json 不存在")
        return

    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if len(data) < 2:
        print("  [plot] 跳过 Fig.6: 少于 2 个 LLM 结果")
        return

    # 用简易柱状图代替雷达图（更稳健）
    providers = [d.get("provider", d.get("model", "unknown")) for d in data]
    f1_values = [d.get("F1", d.get("record_partial_f1", 0)) for d in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(range(len(providers)))
    ax.bar(providers, f1_values, color=colors, edgecolor="white")
    ax.set_ylabel("Record F1")
    ax.set_title("Cross-LLM Comparison")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(f1_values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    _save_fig(fig, "fig6_cross_llm", output_dir)
    plt.close(fig)


# ============================================================
# Fig.7 — 泛化验证 Dev vs Test
# ============================================================
def plot_generalization(output_dir: Path, results_dir: Path):
    """Dev 集 vs Test 集性能对比。"""
    dev_path = results_dir / "evaluation" / "summary_metrics.csv"
    test_path = results_dir / "test_evaluation" / "summary_metrics.csv"

    dev_rows = _load_csv(dev_path)
    test_rows = _load_csv(test_path)

    if not dev_rows or not test_rows:
        print("  [plot] 跳过 Fig.7: dev/test 评估结果不完整")
        return

    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    # 按策略配对
    dev_map = {r["strategy"]: float(r.get("F1", r.get("record_partial_f1", 0))) for r in dev_rows}
    test_map = {r["strategy"]: float(r.get("F1", r.get("record_partial_f1", 0))) for r in test_rows}
    strategies = sorted(set(dev_map) & set(test_map))

    if not strategies:
        print("  [plot] 跳过 Fig.7: 无匹配策略")
        return

    x = np.arange(len(strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, [dev_map[s] for s in strategies], width, label="Dev", color="#4C72B0")
    ax.bar(x + width / 2, [test_map[s] for s in strategies], width, label="Test", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylabel("Record F1")
    ax.set_title("Generalization: Dev vs Test")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, "fig7_generalization", output_dir)
    plt.close(fig)


# ============================================================
# Fig.8 — P-R 散点图（策略作为点）
# ============================================================
def plot_pr_scatter(output_dir: Path, eval_dir: Path):
    """各策略在 P-R 空间中的分布散点图，附 iso-F1 等高线。"""
    csv_path = eval_dir / "summary_metrics.csv"
    rows = _load_csv(csv_path)
    if not rows:
        print("  [plot] 跳过 Fig.8: summary_metrics.csv 不存在")
        return

    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    strategies = [r["strategy"] for r in rows]
    precisions = [float(r.get("precision", 0)) for r in rows]
    recalls = [float(r.get("recall", 0)) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 6))

    # 绘制 iso-F1 等高线
    f_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    p_grid = np.linspace(0.01, 1.0, 300)
    for fv in f_values:
        r_grid = fv * p_grid / (2 * p_grid - fv)
        mask = (r_grid >= 0) & (r_grid <= 1)
        ax.plot(p_grid[mask], r_grid[mask], color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
        # 标注 F1 值
        idx = np.argmin(np.abs(p_grid[mask] - 0.75))
        ax.text(p_grid[mask][idx], r_grid[mask][idx], f"F1={fv:.1f}",
                fontsize=7, color="gray", ha="center")

    # 绘制策略点
    colors = plt.cm.tab10(range(len(strategies)))
    for i, (s, p, r) in enumerate(zip(strategies, precisions, recalls)):
        ax.scatter(p, r, color=colors[i], s=100, zorder=5, label=s)
        ax.annotate(s, (p, r), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Precision-Recall Tradeoff (Partial Match)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, "fig8_pr_scatter", output_dir)
    plt.close(fig)


# ============================================================
# Fig.9 — 字段 P/R 分离热图
# ============================================================
def plot_field_pr_heatmap(output_dir: Path, eval_dir: Path):
    """每个字段的 Precision 和 Recall 分离热图（两个子图）。"""
    csv_path = eval_dir / "per_field_pr_split.csv"
    rows = _load_csv(csv_path)
    if not rows:
        # 回退到 per_field_metrics.csv
        csv_path = eval_dir / "per_field_metrics.csv"
        rows = _load_csv(csv_path)
    if not rows:
        print("  [plot] 跳过 Fig.9: per_field_pr_split.csv 不存在")
        return

    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    strategies = sorted(set(r["strategy"] for r in rows))
    fields = config.FIELD_NAMES

    p_matrix = np.zeros((len(fields), len(strategies)))
    r_matrix = np.zeros((len(fields), len(strategies)))

    for row in rows:
        s = row["strategy"]
        f = row["field"]
        if s in strategies and f in fields:
            si = strategies.index(s)
            fi = fields.index(f)
            p_matrix[fi, si] = float(row.get("precision", 0))
            r_matrix[fi, si] = float(row.get("recall", 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    for ax, matrix, title in [(ax1, p_matrix, "Precision"), (ax2, r_matrix, "Recall")]:
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(fields)))
        ax.set_yticklabels(fields, fontsize=8)
        ax.set_title(f"Field {title}")
        for i in range(len(fields)):
            for j in range(len(strategies)):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if matrix[i,j] > 0.6 else "black")

    fig.tight_layout()
    _save_fig(fig, "fig9_field_pr_heatmap", output_dir)
    plt.close(fig)


# ============================================================
# Fig.15 — 字段存在性二分类混淆矩阵（每策略一个 2×2 子图）
# ============================================================
def plot_confusion_matrix(output_dir: Path, eval_dir: Path):
    """
    每策略一个 2×2 混淆矩阵子图，统计字段 presence / absence 二分类：
      - Label 1 (present): 字段有值
      - Label 0 (absent) : 字段为空
    各 cell 值为跨所有字段汇总的计数：
      TP = sum(tp)                               → 字段有值且正确抽取
      FP = sum(pred_count - tp)                  → 字段为空却预测有值（幻觉）
      FN = sum(gold_count - tp)                  → 字段有值却漏掉
      TN = sum(record_gold_total - gold_count - FP_per_field)  → 字段为空且正确为空
    数据来源：per_field_metrics.csv + summary_metrics.csv
    """
    field_csv  = eval_dir / "per_field_metrics.csv"
    summ_csv   = eval_dir / "summary_metrics.csv"
    field_rows = _load_csv(field_csv)
    summ_rows  = _load_csv(summ_csv)
    if not field_rows or not summ_rows:
        print("  [plot] 跳过 Fig.15: per_field_metrics.csv 或 summary_metrics.csv 不存在")
        return

    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    # record_gold_total 按策略索引
    gold_totals = {r["strategy"]: float(r.get("record_gold_total", 0)) for r in summ_rows}

    # 汇总各策略的 TP / FP / FN / TN
    stats: dict = {}
    for row in field_rows:
        s = row["strategy"]
        try:
            tp   = float(row.get("tp",         0) or 0)
            pred = float(row.get("pred_count", 0) or 0)
            gold = float(row.get("gold_count", 0) or 0)
        except (ValueError, TypeError):
            continue
        fp = max(0.0, pred - tp)
        fn = max(0.0, gold - tp)
        tn = max(0.0, gold_totals.get(s, 0) - gold - fp)
        d = stats.setdefault(s, {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0})
        d["tp"] += tp
        d["fp"] += fp
        d["fn"] += fn
        d["tn"] += tn

    strategies = sorted(stats.keys())
    n = len(strategies)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0))
    if n == 1:
        axes = [axes]

    for ax, strategy in zip(axes, strategies):
        d = stats[strategy]
        # 矩阵排列：行=True label(0/1), 列=Predicted label(0/1)
        # [TN, FP]
        # [FN, TP]
        cm = np.array([
            [d["tn"], d["fp"]],
            [d["fn"], d["tp"]],
        ])
        total = cm.sum()

        im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 格子内标注：整数计数 + 行百分比
        thresh = cm.max() / 2.0
        row_sums = cm.sum(axis=1, keepdims=True)
        for i in range(2):
            for j in range(2):
                count = int(round(cm[i, j]))
                pct   = cm[i, j] / row_sums[i, 0] * 100 if row_sums[i, 0] > 0 else 0
                color = "white" if cm[i, j] < thresh else "black"
                ax.text(j, i, f"{count}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=9, color=color, fontweight="bold")

        tick_labels = ["Absent (0)", "Present (1)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_xlabel("Predicted label", fontsize=9)
        ax.set_ylabel("True label",      fontsize=9)
        ax.set_title(f"{strategy}\n(n={int(round(total))})", fontsize=10, fontweight="bold")

    fig.suptitle("Field Presence / Absence Confusion Matrix by Strategy", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "fig15_confusion_matrix", output_dir)
    plt.close(fig)


# ============================================================
# Fig.10 — top-k sweep 折线图
# ============================================================
def plot_topk_sweep(output_dir: Path, ablation_dir: Path):
    """field_retrieval_top_k 消融结果折线图。"""
    json_path = ablation_dir / "structured_ablation_results.json"
    if not json_path.exists():
        print("  [plot] 跳过 Fig.10: structured_ablation_results.json 不存在")
        return

    data = json.loads(json_path.read_text(encoding="utf-8"))
    sweep_data = data.get("field_topk_sweep", [])
    if not sweep_data:
        print("  [plot] 跳过 Fig.10: field_topk_sweep 数据为空")
        return

    import matplotlib.pyplot as plt
    _setup_style()

    topk_vals = [r.get("field_retrieval_top_k", i) for i, r in enumerate(sweep_data)]
    f1_vals = [r.get("F1", r.get("record_partial_f1", 0)) for r in sweep_data]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(topk_vals, f1_vals, marker="o", linewidth=2, color="#4C72B0")
    for x, y in zip(topk_vals, f1_vals):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("Field Retrieval Top-K")
    ax.set_ylabel("Record F1")
    ax.set_title("Effect of Field-Level Top-K on E3 Performance")
    ax.set_xticks(topk_vals)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, "fig10_topk_sweep", output_dir)
    plt.close(fig)


# ============================================================
# Fig.11 — window size sweep 折线图
# ============================================================
def plot_window_sweep(output_dir: Path, ablation_dir: Path):
    """window_size 消融结果折线图。"""
    json_path = ablation_dir / "structured_ablation_results.json"
    if not json_path.exists():
        print("  [plot] 跳过 Fig.11: structured_ablation_results.json 不存在")
        return

    data = json.loads(json_path.read_text(encoding="utf-8"))
    sweep_data = data.get("window_size_sweep", [])
    if not sweep_data:
        print("  [plot] 跳过 Fig.11: window_size_sweep 数据为空")
        return

    import matplotlib.pyplot as plt
    _setup_style()

    ws_vals = [r.get("window_size", i) for i, r in enumerate(sweep_data)]
    f1_vals = [r.get("F1", r.get("record_partial_f1", 0)) for r in sweep_data]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ws_vals, f1_vals, marker="s", linewidth=2, color="#DD8452")
    for x, y in zip(ws_vals, f1_vals):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("Window Size (±chunks)")
    ax.set_ylabel("Record Partial F1")
    ax.set_title("Effect of Context Window Size")
    ax.set_xticks(ws_vals)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, "fig11_window_sweep", output_dir)
    plt.close(fig)


# ============================================================
# Fig.12 — 文档长度分桶性能
# ============================================================
def plot_doc_length_binning(output_dir: Path, eval_dir: Path):
    """不同文档长度区间下各策略的性能变化折线图。"""
    csv_path = eval_dir / "per_doc_length_metrics.csv"
    rows = _load_csv(csv_path)
    if not rows:
        print("  [plot] 跳过 Fig.12: per_doc_length_metrics.csv 不存在")
        return

    import matplotlib.pyplot as plt
    _setup_style()

    strategies = sorted(set(r["strategy"] for r in rows))
    def _bin_sort_key(label: str) -> float:
        m = re.match(r"(\d+)", label)
        return float(m.group(1)) if m else float("inf")
    bins = sorted(set(r["bin"] for r in rows), key=_bin_sort_key)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(range(len(strategies)))
    for i, strat in enumerate(strategies):
        strat_rows = [r for r in rows if r["strategy"] == strat]
        bin_f1 = {r["bin"]: float(r.get("F1", r.get("partial_f1", 0))) for r in strat_rows}
        f1_vals = [bin_f1.get(b, 0) for b in bins]
        ax.plot(bins, f1_vals, marker="o", linewidth=2, color=colors[i], label=strat)

    ax.set_xlabel("Document Length Bin")
    ax.set_ylabel("Record F1")
    ax.set_title("Performance by Document Length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.xticks(rotation=20, ha="right")

    fig.tight_layout()
    _save_fig(fig, "fig12_doc_length_binning", output_dir)
    plt.close(fig)


# ============================================================
# Fig.13 — 记录数分桶性能
# ============================================================
def plot_record_count_binning(output_dir: Path, eval_dir: Path):
    """不同记录数的论文中各策略的 F1 对比。"""
    csv_path = eval_dir / "per_record_count_metrics.csv"
    rows = _load_csv(csv_path)
    if not rows:
        print("  [plot] 跳过 Fig.13: per_record_count_metrics.csv 不存在")
        return

    import matplotlib.pyplot as plt
    _setup_style()

    strategies = sorted(set(r["strategy"] for r in rows))
    def _bin_sort_key(label: str) -> float:
        m = re.match(r"(\d+)", label)
        return float(m.group(1)) if m else float("inf")
    bins = sorted(set(r["bin"] for r in rows), key=_bin_sort_key)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(range(len(strategies)))
    for i, strat in enumerate(strategies):
        strat_rows = [r for r in rows if r["strategy"] == strat]
        bin_f1 = {r["bin"]: float(r.get("F1", r.get("partial_f1", 0))) for r in strat_rows}
        f1_vals = [bin_f1.get(b, 0) for b in bins]
        ax.plot(bins, f1_vals, marker="o", linewidth=2, color=colors[i], label=strat)

    ax.set_xlabel("Gold Record Count per Paper")
    ax.set_ylabel("Record F1")
    ax.set_title("Performance by Number of Records per Paper")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, "fig13_record_count_binning", output_dir)
    plt.close(fig)

# ============================================================
# 主入口
# ============================================================
def main(output_dir: Path = None, results_dir: Path = None):
    output_dir = output_dir or config.FIGURES_DIR
    results_dir = results_dir or Path(config.RESULTS_DIR)
    eval_dir = results_dir / "evaluation"
    bo_dir = results_dir / "bo"
    ablation_dir = results_dir / "ablation"
    structured_ablation_dir = results_dir / "structured_ablation"

    print("[plot] 生成图表...")

    # 阶段 A/B 均可用
    plot_strategy_comparison(output_dir, eval_dir)
    plot_prf_comparison(output_dir, eval_dir)
    plot_per_paper_coverage(output_dir, eval_dir)
    plot_field_heatmap(output_dir, eval_dir)

    # 阶段 B 专用
    plot_bo_convergence(output_dir, bo_dir)
    plot_param_importance(output_dir, bo_dir)
    plot_ablation(output_dir, ablation_dir)
    plot_cross_llm_radar(output_dir, ablation_dir)
    plot_generalization(output_dir, results_dir)

    # 新增图表 (v4 计划)
    plot_confusion_matrix(output_dir, eval_dir)
    plot_pr_scatter(output_dir, eval_dir)
    plot_field_pr_heatmap(output_dir, eval_dir)
    plot_topk_sweep(output_dir, structured_ablation_dir)
    plot_window_sweep(output_dir, structured_ablation_dir)
    plot_doc_length_binning(output_dir, eval_dir)
    plot_record_count_binning(output_dir, eval_dir)

    print(f"[plot] 完成 → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成实验图表")
    parser.add_argument("--output", type=str, default=None, help="图表输出目录")
    parser.add_argument("--results", type=str, default=None, help="结果目录")
    args = parser.parse_args()

    main(
        output_dir=Path(args.output) if args.output else None,
        results_dir=Path(args.results) if args.results else None,
    )
