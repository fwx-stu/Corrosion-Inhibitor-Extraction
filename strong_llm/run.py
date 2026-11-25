"""
主入口 — 统一 CLI 调度

环境要求:
  - conda activate langchain
  - .env 中配置 API Key (ZHIPU_API_KEY / DEEPSEEK_API_KEY)
  - 本地模型: <local-project-path>, bge-reranker-v2-m3
  - 切换 LLM: 修改 config.py 的 LLM_PROVIDER 或命令行 --provider

所有命令:
  # === 单策略/多策略运行 ===
  python run.py --mode compare --strategies E0              # 单跑 E0
  python run.py --mode compare --strategies E0,E1           # E0+E1
  python run.py --mode compare --strategies E3,E4           # 需本地 embedding+reranker
  python run.py --mode compare --strategies E0 --limit 5    # 限制处理 5 篇

  # === 切换 LLM (无需改 config) ===
  python run.py --mode compare --strategies E0 --provider deepseek
  python run.py --mode compare --strategies E0 --provider zhipu --model glm-4-plus

  # === 断点续跑 & 重试 ===
  python run.py --mode compare --strategies E0              # 中断后重跑, 自动跳过已完成
  python run.py --mode compare --strategies E0 --retry-failed  # 重试之前失败的论文

  # === 评估 & 可视化 ===
  python run.py --mode evaluate                             # 评估所有已有结果
  python run.py --mode plot                                 # 画图

  # === 快捷模式 ===
  python run.py --mode full-a                               # E0+E1 → 评估 → 画图
  python run.py --mode full-b                               # E3+E4 → 评估 → BO → 消融 → 画图
  python run.py --mode full                                 # 全流程

  # === BO 优化 & 消融 ===
  python run.py --mode optimize --strategy E4 --trials 50 --limit 10
  python run.py --mode ablation --strategy E4 --limit 10
  python run.py --mode cross-llm --providers zhipu deepseek --limit 10

  # === Batch API (智谱半价, E0/E1/E3/E4/E5/E6) ===
  python run.py --mode compare --strategies E0,E1 --batch
  python run.py --mode compare --strategies E3 --batch       # 本地 embed+rerank 后批量提交
  python run.py --mode compare --strategies E4 --batch       # 两轮 batch
  python run.py --mode compare --strategies E6 --batch       # 本地+Milvus 后批量提交
  python run.py --mode optimize --strategy E3 --trials 30 --limit 10       # BO优化
"""

from __future__ import annotations

import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List

import config


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_jsonl(path: Path, obj: dict) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: dict) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# 策略对比运行
# ============================================================
def run_compare(
    strategies: List[str],
    limit: int = None,
    provider: str = None,
    model: str = None,
    batch: bool = False,
    retry_failed: bool = False,
):
    """运行指定策略并保存结果。"""
    # Batch 模式 (仅支持智谱 E0/E1)
    if batch:
        _run_batch_compare(strategies, limit=limit)
        return

    from llm_client import LLMClient
    from pipeline import run_strategy

    # 获取文档列表（若 MD 目录为空但有 PDF，自动转换）
    md_dir = config.MD_DIR
    if not list(md_dir.glob("*.md")) if md_dir.exists() else True:
        pdf_dir = config.PDF_DIR
        if pdf_dir.exists() and list(pdf_dir.glob("*.pdf")):
            print("[run] MD 目录为空，检测到 PDF 文件，自动转换...")
            from prepare import convert_pdfs
            convert_pdfs()

    if not md_dir.exists():
        md_dir = Path(config.DOCUMENTS_DIR)
    if not md_dir.exists():
        print(f"[run] 文档目录不存在: {md_dir}")
        print(f"[run] 请先将 MD 文件放入 {config.MD_DIR} 或运行 prepare.py")
        return

    md_files = sorted(md_dir.glob("*.md"))
    if limit:
        md_files = md_files[:limit]

    if not md_files:
        print(f"[run] 未找到 MD 文件: {md_dir}")
        return

    print(f"[run] 文档数: {len(md_files)}, 策略: {strategies}")

    for strategy in strategies:
        # 依赖检查
        missing = config.check_deps(strategy)
        if missing:
            print(f"[run] 跳过 {strategy}: 缺少依赖 {missing}")
            continue

        strategy_dir = _ensure_dir(Path(config.RESULTS_DIR) / strategy.lower())
        result_file = strategy_dir / f"{strategy.lower()}_results.jsonl"
        failed_file = strategy_dir / f"{strategy.lower()}_failed.jsonl"
        raw_dir = _ensure_dir(strategy_dir / "raw")

        # 检查已有结果, 跳过已处理的文档
        existing_ids = set()
        if result_file.exists():
            with result_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            pid = obj.get("paper_id")
                            if pid:
                                existing_ids.add(pid)
                        except Exception:
                            pass

        # 过滤待处理列表
        todo_files = [f for f in md_files if f.stem not in existing_ids]

        # --retry-failed: 把之前失败的论文也加入待处理
        if retry_failed and failed_file.exists():
            failed_ids = set()
            with failed_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            pid = obj.get("paper_id")
                            if pid and pid not in existing_ids:
                                failed_ids.add(pid)
                        except Exception:
                            pass
            if failed_ids:
                # 确保失败的文件在 todo 列表中
                todo_stems = {f.stem for f in todo_files}
                retry_files = [f for f in md_files if f.stem in failed_ids and f.stem not in todo_stems]
                todo_files.extend(retry_files)
                print(f"  [{strategy}] 重试之前失败的 {len(retry_files)} 篇")
                # 清空旧的失败记录
                failed_file.unlink()

        skipped = len(md_files) - len(todo_files)
        if skipped:
            print(f"  [{strategy}] 断点续跑: 跳过已完成 {skipped} 篇, 剩余 {len(todo_files)} 篇")
        if not todo_files:
            print(f"  [{strategy}] 全部已完成, 跳过")
            continue

        # 创建 LLM 客户端
        llm = LLMClient(
            provider=provider or config.LLM_PROVIDER,
            model=model,
        )

        success = 0
        failed = 0
        consecutive_fails = 0
        # MAX_CONSECUTIVE_FAILS 定义在 config.py
        start_time = time.time()

        for i, md_path in enumerate(todo_files):
            paper_id = md_path.stem

            try:
                out = run_strategy(strategy, paper_id, md_path, llm)
                _append_jsonl(result_file, out)
                _write_json(raw_dir / f"{paper_id}.json", out)
                success += 1
                consecutive_fails = 0  # 成功则重置计数
                print(f"  [{strategy}] ({skipped+success+failed}/{len(md_files)}) OK: {paper_id}")
                time.sleep(config.RATE_LIMIT_DELAY_S)
            except KeyboardInterrupt:
                print(f"\n  [{strategy}] 用户中断, 已保存 {success} 篇结果, 下次运行自动续跑")
                break
            except Exception as e:
                err_str = str(e)
                is_rate_limit = any(k in err_str for k in ("429", "1302", "rate", "Rate"))
                fail_obj = {
                    "paper_id": paper_id,
                    "source_file": md_path.name,
                    "strategy": strategy,
                    "error": err_str,
                }
                _append_jsonl(failed_file, fail_obj)
                failed += 1
                consecutive_fails += 1
                print(f"  [{strategy}] ({skipped+success+failed}/{len(md_files)}) FAIL: {paper_id} -> {err_str[:80]}")
                if is_rate_limit:
                    print(f"  [{strategy}] 触发速率限制，等待 120s 后继续...")
                    time.sleep(config.RATE_LIMIT_RECOVERY_S)

                if consecutive_fails >= config.MAX_CONSECUTIVE_FAILS:
                    print(f"\n  [{strategy}] 连续失败 {config.MAX_CONSECUTIVE_FAILS} 次, 疑似 API 额度耗尽, 自动停止")
                    print(f"  [{strategy}] 已保存 {success} 篇结果, 下次运行自动续跑")
                    break

        elapsed = time.time() - start_time
        token_stats = llm.get_stats()

        # 保存运行日志
        run_log = {
            "strategy": strategy,
            "model": llm.model,
            "provider": llm.provider,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "total_docs": len(md_files),
            "elapsed_seconds": round(elapsed, 1),
            "token_stats": token_stats,
        }
        log_file = strategy_dir / f"{strategy.lower()}_run_log.jsonl"
        _append_jsonl(log_file, run_log)

        print(f"[run] {strategy} 完成: 成功={success}, 失败={failed}, 跳过={skipped}, 耗时={elapsed:.1f}s")
        print(f"  Token 统计: {token_stats}")
        print(f"  日志已保存: {log_file}")


# ============================================================
# 快捷模式
# ============================================================
def _run_batch_compare(strategies: List[str], limit: int = None):
    """Batch 模式运行 (智谱 Batch API，半价)。"""
    from batch_client import (
        run_e0_batch,
        run_e1_batch,
        run_e3_batch, run_e4_batch,
        run_e5_batch, run_e6_batch,
    )

    md_dir = config.MD_DIR
    if not md_dir.exists():
        md_dir = Path(config.DOCUMENTS_DIR)
    md_files = sorted(md_dir.glob("*.md")) if md_dir.exists() else []

    if not md_files:
        print(f"[batch] 未找到 MD 文件: {md_dir}")
        return

    for strategy in strategies:
        s = strategy.upper() if "-" not in strategy else strategy
        if s == "E0":
            run_e0_batch(md_files, limit=limit)
        elif s == "E1":
            run_e1_batch(md_files, limit=limit)
        elif s == "E3":
            run_e3_batch(md_files, limit=limit)
        elif s == "E4":
            run_e4_batch(md_files, limit=limit)
        elif s == "E5":
            run_e5_batch(md_files, limit=limit)
        elif s == "E6":
            run_e6_batch(md_files, limit=limit)
        else:
            print(f"[batch] {strategy} 不支持 Batch 模式")


def run_full_a(limit: int = None, batch: bool = False):
    """阶段 A: E0 + E1 → 评估 → 画图。"""
    print("=" * 60)
    print("[Phase A] 开始")
    print("=" * 60)
    run_compare(["E0", "E1"], limit=limit, batch=batch)
    run_evaluate()
    run_plot()


def run_full_b(limit: int = None):
    """阶段 B: E3 + E4 → 评估 → BO → 消融 → 画图。"""
    print("=" * 60)
    print("[Phase B] 开始")
    print("=" * 60)
    run_compare(["E3", "E4"], limit=limit)
    run_evaluate()
    # BO + ablation 可选
    try:
        from optimize import run_bo, run_ablation
        run_bo(strategy="E4", n_trials=20, limit=limit or 10)
        run_ablation(strategy="E4", limit=limit or 10)
    except Exception as e:
        print(f"[Phase B] BO/消融跳过: {e}")
    run_plot()


def run_full_c(limit: int = None):
    """阶段 C: E5 + E6 → 评估 → 画图。"""
    print("=" * 60)
    print("[Phase C] 开始")
    print("=" * 60)
    run_compare(["E5", "E6"], limit=limit)
    run_evaluate()
    run_plot()


def run_evaluate(gold_path: Path = None):
    """运行评估。"""
    from evaluate import main as eval_main
    eval_main(gold_path=gold_path)


def run_plot():
    """生成图表。"""
    from plot import main as plot_main
    plot_main()


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="RAG 增强信息提取实验框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py --mode compare --strategies E0,E1 --limit 5
  python run.py --mode evaluate
  python run.py --mode plot
  python run.py --mode full-a --limit 5
  python run.py --mode optimize --strategy E4 --trials 10
        """,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["compare", "evaluate", "plot", "optimize", "ablation", "structured-ablation", "cross-llm", "full-a", "full-b", "full-c", "full"],
        help="运行模式",
    )
    parser.add_argument("--strategies", type=str, default="E0,E3", help="策略列表, 逗号分隔 (E0,E1,E3,E4,E5,E6)")
    parser.add_argument("--strategy", type=str, default="E4", help="单策略 (optimize/ablation)")
    parser.add_argument("--limit", type=int, default=None, help="处理文档数限制")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider")
    parser.add_argument("--model", type=str, default=None, help="LLM model")
    parser.add_argument("--trials", type=int, default=50, help="BO 试验次数")
    parser.add_argument("--providers", nargs="+", default=["zhipu", "deepseek"], help="Cross-LLM providers")
    parser.add_argument("--best-params", type=str, default=None, help="最优参数 JSON 路径")
    parser.add_argument("--batch", action="store_true", help="使用智谱 Batch API (半价, 仅 E0/E1)")
    parser.add_argument("--retry-failed", action="store_true", help="重试之前失败的论文")
    parser.add_argument("--gold", type=str, default=None, help="评估用 gold JSONL 路径（默认 train.jsonl，最终评估传 test.jsonl）")
    args = parser.parse_args()

    if args.mode == "compare":
        strategies = [s.strip() for s in args.strategies.split(",")]
        run_compare(strategies, limit=args.limit, provider=args.provider, model=args.model, batch=args.batch, retry_failed=args.retry_failed)

    elif args.mode == "evaluate":
        run_evaluate(gold_path=Path(args.gold) if args.gold else None)

    elif args.mode == "plot":
        run_plot()

    elif args.mode == "optimize":
        from optimize import run_bo
        run_bo(strategy=args.strategy, n_trials=args.trials, limit=args.limit or 10)

    elif args.mode == "ablation":
        from optimize import run_ablation
        bp = Path(args.best_params) if args.best_params else None
        run_ablation(best_params_path=bp, strategy=args.strategy, limit=args.limit or 10)

    elif args.mode == "structured-ablation":
        from optimize import run_structured_ablation
        run_structured_ablation(base_strategy=args.strategy, limit=args.limit or 10)

    elif args.mode == "cross-llm":
        from optimize import run_cross_llm
        run_cross_llm(strategy=args.strategy, providers=args.providers, limit=args.limit or 10)

    elif args.mode == "full-a":
        run_full_a(limit=args.limit, batch=args.batch)

    elif args.mode == "full-b":
        run_full_b(limit=args.limit)

    elif args.mode == "full-c":
        run_full_c(limit=args.limit)

    elif args.mode == "full":
        run_full_a(limit=args.limit)
        run_full_b(limit=args.limit)
        run_full_c(limit=args.limit)


if __name__ == "__main__":
    main()
