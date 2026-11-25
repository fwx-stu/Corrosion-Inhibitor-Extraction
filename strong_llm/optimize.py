"""
贝叶斯优化 (Optuna) + 消融实验

功能:
  - BO 超参数搜索 (TPE sampler + MedianPruner)
  - 消融实验: 逐一关闭组件
  - Cross-LLM 对比
  - 泛化验证 (dev → test)

  cd <local-project-path>
    python run.py --mode optimize --strategy E3 --trials 30 --limit 10

阶段 B 专用 (依赖检索基础设施)。
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import config


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# 辅助: 运行单策略并评估
# ============================================================
def _run_and_evaluate(
    strategy: str,
    params: Dict[str, Any],
    md_files: List[Path],
    gold_map: Dict[str, list],
    provider: str = None,
    model: str = None,
) -> Dict[str, Any]:
    """对一组文档运行策略并返回评估指标。"""
    import time
    from llm_client import LLMClient
    from pipeline import run_strategy
    from evaluate import greedy_match_records, extract_records, _prf

    llm = LLMClient(
        provider=provider or config.LLM_PROVIDER,
        model=model,
    )

    pred_map: Dict[str, list] = {}
    for idx, md_path in enumerate(md_files, 1):
        paper_id = md_path.stem
        print(f"  [{idx}/{len(md_files)}] {paper_id} ...", flush=True)
        try:
            out = run_strategy(strategy, paper_id, md_path, llm, params=params)
            pred_map[paper_id] = extract_records(out)
            print(f"  [{idx}/{len(md_files)}] {paper_id} -> {len(pred_map[paper_id])} 条记录", flush=True)
        except Exception as e:
            print(f"  [optimize] {paper_id} 失败: {e}", flush=True)
        time.sleep(3)  # 避免 BO 循环中触发速率限制

    # 简易评估 — 返回 record_partial_f1
    total_tp = total_pred = total_gold = 0
    for pid, gold_recs in gold_map.items():
        pred_recs = pred_map.get(pid, [])
        total_pred += len(pred_recs)
        total_gold += len(gold_recs)
        _, partial_tp, _ = greedy_match_records(pred_recs, gold_recs)
        total_tp += partial_tp

    _, _, f1 = _prf(total_tp, total_pred, total_gold)
    return {
        "record_partial_f1": f1,
        "paper_count": len(md_files),
        "pred_total": total_pred,
        "gold_total": total_gold,
        "partial_tp": total_tp,
        "token_stats": llm.get_stats(),
    }


def _get_md_files(limit: int = None) -> List[Path]:
    """获取 MD 文件列表。"""
    md_dir = config.MD_DIR
    if not md_dir.exists():
        # fallback 到已有文档目录
        md_dir = config.DOCUMENTS_DIR
    if not Path(md_dir).exists():
        return []
    files = sorted(Path(md_dir).glob("*.md"))
    if limit:
        files = files[:limit]
    return files


# ============================================================
# Part A: Optuna 贝叶斯优化
# ============================================================
def run_bo(
    strategy: str = "E4",
    n_trials: int = 50,
    limit: int = 10,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    使用 Optuna TPE sampler 搜索最优超参数。

    Args:
        strategy: 目标策略
        n_trials: 试验次数
        limit: 每次试验使用的文档数 (控制成本)
        output_dir: 输出目录
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("需要安装 optuna: pip install optuna")

    from prepare import load_gold

    output_dir = output_dir or Path(config.RESULTS_DIR) / "bo"
    _ensure_dir(output_dir)

    gold_map = load_gold()
    md_files = _get_md_files(limit)
    # 仅保留有 gold 标注的文档
    md_files = [f for f in md_files if f.stem in gold_map]

    if not md_files:
        print("[optimize] 无可用测试文档 (需 MD 文件 + gold 标注)")
        return {}

    print(f"[optimize] BO 搜索: strategy={strategy}, trials={n_trials}, docs={len(md_files)}")

    def objective(trial: optuna.Trial) -> float:
        print(f"\n[Trial {trial.number + 1}/{n_trials}] 开始...", flush=True)
        # 按策略选择搜索空间：E3 不用 dense_weight / global_top_k
        if strategy in ("E3",):
            space = config.BO_SEARCH_SPACE_E3
        else:
            space = config.BO_SEARCH_SPACE
        params = {}
        for name, spec in space.items():
            if spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])

        # 映射到 pipeline 参数
        pipeline_params = dict(config.DEFAULT_PARAMS)
        pipeline_params.update({
            "chunk_size": params.get("chunk_size", pipeline_params["chunk_size"]),
            "chunk_overlap": params.get("chunk_overlap", pipeline_params["chunk_overlap"]),
            "retrieval_top_k": params.get("local_top_k", pipeline_params["retrieval_top_k"]),
            "rerank_top_n": params.get("rerank_top_n", pipeline_params["rerank_top_n"]),
            "enable_reranker": params.get("enable_reranker", pipeline_params["enable_reranker"]),
            "window_size": params.get("window_size", pipeline_params["window_size"]),
        })
        if "dense_weight" in params:
            pipeline_params["dense_weight"] = params["dense_weight"]
        # BO temperature 覆盖 LLM 参数（由 pipeline 传给 llm_client）
        if "temperature" in params:
            pipeline_params["llm_temperature"] = params["temperature"]
        # global_top_k 用于 E2/E4 的 Milvus 检索
        if "global_top_k" in params:
            pipeline_params["global_top_k"] = params["global_top_k"]

        print(f"[Trial {trial.number + 1}/{n_trials}] params={params}", flush=True)
        result = _run_and_evaluate(
            strategy=strategy,
            params=pipeline_params,
            md_files=md_files,
            gold_map={pid: gold_map[pid] for pid in [f.stem for f in md_files] if pid in gold_map},
        )
        f1 = result["record_partial_f1"]
        print(f"[Trial {trial.number + 1}/{n_trials}] F1={f1:.4f}  (pred={result['pred_total']}, gold={result['gold_total']}, tp={result['partial_tp']})", flush=True)
        return f1

    # 创建 study
    storage = f"sqlite:///{output_dir / 'study.db'}"
    study = optuna.create_study(
        study_name=f"bo_{strategy}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials)

    # 保存最佳参数
    best = {
        "strategy": strategy,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    (output_dir / "best_params.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[optimize] 最优 F1={study.best_value:.4f}, params={study.best_params}")
    print(f"[optimize] 结果保存至 {output_dir}")
    return best


# ============================================================
# Part B: 消融实验
# ============================================================
def run_ablation(
    best_params_path: Path = None,
    strategy: str = "E4",
    limit: int = 10,
    output_dir: Path = None,
) -> List[Dict[str, Any]]:
    """
    基于最优参数做消融实验:
      - baseline: 最优配置
      - -reranker: 关闭重排序
      - -window: 关闭窗口扩展
      - chunk_size_sweep: 不同块大小
    """
    from prepare import load_gold

    output_dir = output_dir or Path(config.RESULTS_DIR) / "ablation"
    _ensure_dir(output_dir)

    # 加载最优参数
    if best_params_path and best_params_path.exists():
        best_data = json.loads(best_params_path.read_text(encoding="utf-8"))
        best_params = best_data.get("best_params", {})
    else:
        best_params = {}

    base = dict(config.DEFAULT_PARAMS)
    base.update(best_params)

    gold_map = load_gold()
    md_files = _get_md_files(limit)
    md_files = [f for f in md_files if f.stem in gold_map]
    if not md_files:
        print("[optimize] 无可用测试文档")
        return []

    filtered_gold = {pid: gold_map[pid] for pid in [f.stem for f in md_files] if pid in gold_map}

    # 消融配置
    ablation_configs = {
        "baseline": dict(base),
        "-reranker": {**base, "enable_reranker": False},
        "-window": {**base, "window_size": 0, "enable_window_expand": False},
    }

    # chunk_size sweep
    for cs in [400, 600, 800, 1000, 1200]:
        ablation_configs[f"chunk_{cs}"] = {**base, "chunk_size": cs}

    results = []
    for name, params in ablation_configs.items():
        print(f"[ablation] 运行: {name}")
        result = _run_and_evaluate(
            strategy=strategy, params=params,
            md_files=md_files, gold_map=filtered_gold,
        )
        result["ablation_name"] = name
        result["params"] = params
        results.append(result)
        print(f"  F1={result['record_partial_f1']:.4f}")

    # 保存
    (output_dir / "ablation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[ablation] 完成 → {output_dir}")
    return results


# ============================================================
# Part C: Cross-LLM 对比
# ============================================================
def run_cross_llm(
    strategy: str = "E4",
    providers: List[str] = None,
    limit: int = 10,
    output_dir: Path = None,
) -> List[Dict[str, Any]]:
    """对最优参数在不同 LLM 上运行对比。"""
    from prepare import load_gold

    providers = providers or ["zhipu", "deepseek"]
    output_dir = output_dir or Path(config.RESULTS_DIR) / "ablation"
    _ensure_dir(output_dir)

    gold_map = load_gold()
    md_files = _get_md_files(limit)
    md_files = [f for f in md_files if f.stem in gold_map]
    if not md_files:
        print("[optimize] 无可用测试文档")
        return []

    filtered_gold = {pid: gold_map[pid] for pid in [f.stem for f in md_files] if pid in gold_map}

    results = []
    for provider in providers:
        print(f"[cross-llm] {provider}")
        result = _run_and_evaluate(
            strategy=strategy,
            params=config.DEFAULT_PARAMS,
            md_files=md_files,
            gold_map=filtered_gold,
            provider=provider,
        )
        result["provider"] = provider
        result["model"] = config.LLM_PROVIDERS.get(provider, {}).get("model", "")
        results.append(result)
        print(f"  F1={result['record_partial_f1']:.4f}")

    (output_dir / "cross_llm_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[cross-llm] 完成 → {output_dir}")
    return results


# ============================================================
# Part D: 结构化消融实验（面向 E3/E5/E6 的专项对比）
# ============================================================
def run_structured_ablation(
    base_strategy: str = "E3",
    limit: int = 10,
    output_dir: Path = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    专为新策略体系设计的结构化消融，包含 5 组实验：

    1. field_topk_sweep   — E3 字段级 top_k 影响 (1 / 2 / 3 / 5)
    2. window_size_sweep  — 窗口扩展大小影响 (0 / 1 / 2)
    3. retrieval_style    — 字段级查询 vs 合并查询（诊断 E3 的改进来源）
    4. local_vs_hybrid    — E3 vs E6（全局知识的边际贡献）
    5. e2rag_vs_e5global  — Example-RAG vs Global-RAG（两种全局利用方式对比）
    """
    from prepare import load_gold

    output_dir = output_dir or Path(config.RESULTS_DIR) / "structured_ablation"
    _ensure_dir(output_dir)

    gold_map = load_gold()
    md_files = _get_md_files(limit)
    md_files = [f for f in md_files if f.stem in gold_map]
    if not md_files:
        print("[structured_ablation] 无可用测试文档")
        return {}

    filtered_gold = {pid: gold_map[pid] for pid in [f.stem for f in md_files] if pid in gold_map}
    base_params = dict(config.DEFAULT_PARAMS)
    all_results: Dict[str, List[Dict]] = {}

    # --- 1. field_top_k_sweep ---
    print("\n[structured_ablation] 1/5: field_topk_sweep")
    ftk_results = []
    for ftk in [1, 2, 3, 5]:
        print(f"  field_retrieval_top_k={ftk}")
        params = {**base_params, "field_retrieval_top_k": ftk}
        r = _run_and_evaluate("E3", params, md_files, filtered_gold)
        r["variant"] = f"field_topk={ftk}"
        r["field_retrieval_top_k"] = ftk
        ftk_results.append(r)
        print(f"    F1={r['record_partial_f1']:.4f}")
    all_results["field_topk_sweep"] = ftk_results

    # --- 2. window_size_sweep ---
    print("\n[structured_ablation] 2/5: window_size_sweep")
    ws_results = []
    for ws in [0, 1, 2]:
        print(f"  window_size={ws}")
        params = {**base_params, "window_size": ws, "enable_window_expand": ws > 0}
        r = _run_and_evaluate(base_strategy, params, md_files, filtered_gold)
        r["variant"] = f"window={ws}"
        r["window_size"] = ws
        ws_results.append(r)
        print(f"    F1={r['record_partial_f1']:.4f}")
    all_results["window_size_sweep"] = ws_results

    # --- 3. retrieval_style: field-level vs unified query ---
    print("\n[structured_ablation] 3/5: retrieval_style (field-level vs unified)")
    rs_results = []
    # E3 with field-level queries（新 E3 行为，已默认）
    r_field = _run_and_evaluate("E3", {**base_params}, md_files, filtered_gold)
    r_field["variant"] = "field_level_queries"
    rs_results.append(r_field)
    print(f"  field_level F1={r_field['record_partial_f1']:.4f}")

    # E3 with unified query（旧 E3 行为，通过 use_field_queries=False 参数禁用）
    r_unified = _run_and_evaluate("E3", {**base_params, "use_unified_query": True}, md_files, filtered_gold)
    r_unified["variant"] = "unified_query"
    rs_results.append(r_unified)
    print(f"  unified F1={r_unified['record_partial_f1']:.4f}")
    all_results["retrieval_style"] = rs_results

    # --- 4. local_vs_hybrid: E3 vs E6 ---
    print("\n[structured_ablation] 4/5: local_vs_hybrid (E3 vs E6)")
    lh_results = []
    for strat in ["E3", "E6"]:
        try:
            r = _run_and_evaluate(strat, base_params, md_files, filtered_gold)
        except Exception as e:
            print(f"  {strat} 跳过: {e}")
            r = {"record_partial_f1": 0.0, "error": str(e)}
        r["variant"] = strat
        lh_results.append(r)
        print(f"  {strat} F1={r.get('record_partial_f1', 'N/A')}")
    all_results["local_vs_hybrid"] = lh_results

    # --- 5. e2rag_vs_e5global ---
    print("\n[structured_ablation] 5/5: e2rag_vs_e5global (E2-rag vs E5)")
    eg_results = []
    for strat in ["E2-rag", "E5"]:
        try:
            r = _run_and_evaluate(strat, base_params, md_files, filtered_gold)
        except Exception as e:
            print(f"  {strat} 跳过: {e}")
            r = {"record_partial_f1": 0.0, "error": str(e)}
        r["variant"] = strat
        eg_results.append(r)
        print(f"  {strat} F1={r.get('record_partial_f1', 'N/A')}")
    all_results["e2rag_vs_e5global"] = eg_results

    # 保存结果
    (output_dir / "structured_ablation_results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n[structured_ablation] 完成 → {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="贝叶斯优化 & 消融实验")
    sub = parser.add_subparsers(dest="mode")

    # bo
    bo_parser = sub.add_parser("bo", help="贝叶斯优化")
    bo_parser.add_argument("--strategy", default="E4")
    bo_parser.add_argument("--trials", type=int, default=50)
    bo_parser.add_argument("--limit", type=int, default=10)

    # ablation
    abl_parser = sub.add_parser("ablation", help="消融实验")
    abl_parser.add_argument("--strategy", default="E4")
    abl_parser.add_argument("--best-params", type=str, default=None)
    abl_parser.add_argument("--limit", type=int, default=10)

    # cross-llm
    cross_parser = sub.add_parser("cross-llm", help="Cross-LLM 对比")
    cross_parser.add_argument("--strategy", default="E4")
    cross_parser.add_argument("--providers", nargs="+", default=["zhipu", "deepseek"])
    cross_parser.add_argument("--limit", type=int, default=10)

    # structured-ablation
    sabl_parser = sub.add_parser("structured-ablation", help="结构化消融实验 (E3/E5/E6)")
    sabl_parser.add_argument("--strategy", default="E3")
    sabl_parser.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    if args.mode == "bo":
        run_bo(strategy=args.strategy, n_trials=args.trials, limit=args.limit)
    elif args.mode == "ablation":
        bp = Path(args.best_params) if args.best_params else Path(config.RESULTS_DIR) / "bo" / "best_params.json"
        run_ablation(best_params_path=bp, strategy=args.strategy, limit=args.limit)
    elif args.mode == "cross-llm":
        run_cross_llm(strategy=args.strategy, providers=args.providers, limit=args.limit)
    elif args.mode == "structured-ablation":
        run_structured_ablation(base_strategy=args.strategy, limit=args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
