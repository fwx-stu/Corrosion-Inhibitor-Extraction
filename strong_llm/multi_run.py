"""
多轮自动化实验脚本

用法:
  # 试跑：只跑 E0，prompt v1，1轮，限5篇
  python multi_run.py --strategies E0 --variants 1 --runs 1 --limit 5

  # 正式跑所有策略、3个prompt变体、每变体1轮 (temp=0.15提供随机性)
  python multi_run.py --strategies E0,E1,E3,E4,E5,E6 --batch

  # 跑特定变体和轮次
  python multi_run.py --strategies E3,E4,E5,E6 --variants 1,2,3 --runs 3 --batch --evaluate
  python multi_run.py --strategies E1 --variants 1,2,3 --runs 3 --batch --evaluate
  python multi_run.py --strategies E3,E4,E5,E6 --variants 2,3 --runs 3 --batch --evaluate --resume-from 2:2
  # 跑完后自动评估+汇总
  python multi_run.py --strategies E1 --variants 1,2,3 --runs 3 --batch --evaluate --overwrite --resume-from 2:2
  python multi_run.py --strategies E0,E3,E5,E6 --batch --evaluate --overwrite
工作原理:
  对每个 (prompt_variant, run_index) 组合:
    1. 将 config.PROMPTS_DIR 切换到对应的 prompts{_v2,_v3}/ 目录
    2. 将 config.RESULTS_DIR 切换到 output/results_v{V}_r{R}/
    3. 运行所有指定策略 (batch 或实时模式)
    4. 运行评估
  最后调用 multi_run_evaluate.py 汇总所有结果。
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# 确保 experiment 目录在 sys.path
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import config

# prompt 变体目录映射
VARIANT_DIRS = {
    1: config.EXPERIMENT_DIR / "prompts",
    2: config.EXPERIMENT_DIR / "prompts_v2",
    3: config.EXPERIMENT_DIR / "prompts_v3",
}

BASE_RESULTS_DIR = config.OUTPUT_DIR / "results"          # 原始 results 目录
MULTI_RESULTS_ROOT = config.OUTPUT_DIR / "multi_run"      # 多轮结果根目录


def get_results_dir(variant: int, run: int) -> Path:
    """返回每轮的结果目录: output/multi_run/v{V}_r{R}/results/"""
    return MULTI_RESULTS_ROOT / f"v{variant}_r{run}" / "results"


def run_one_round(
    variant: int,
    run: int,
    strategies: list[str],
    batch: bool,
    limit: int | None,
    retry_failed: bool,
    overwrite: bool = False,
):
    """运行一轮: 切换 prompt 目录和结果目录，然后跑策略。"""
    prompt_dir = VARIANT_DIRS[variant]
    results_dir = get_results_dir(variant, run)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"[multi_run] variant={variant}, run={run}")
    print(f"  prompt_dir  = {prompt_dir}")
    print(f"  results_dir = {results_dir}")
    print(f"  strategies  = {strategies}")
    print(f"  batch={batch}, limit={limit}")
    print("=" * 70)

    # 动态切换 config 路径
    config.PROMPTS_DIR = prompt_dir
    config.RESULTS_DIR = results_dir
    config.CHUNKS_DIR = results_dir.parent / "chunks"
    config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # --overwrite: 删除指定策略的已有结果文件，使断点续跑失效
    if overwrite:
        import shutil
        for s in strategies:
            s_dir = results_dir / s.lower()
            result_file = s_dir / f"{s.lower()}_results.jsonl"
            failed_file = s_dir / f"{s.lower()}_failed.jsonl"
            for f in (result_file, failed_file):
                if f.exists():
                    f.unlink()
                    print(f"  [overwrite] 已删除: {f.relative_to(MULTI_RESULTS_ROOT)}")

    if batch:
        from run import _run_batch_compare
        _run_batch_compare(strategies, limit=limit)
    else:
        from run import run_compare
        run_compare(strategies, limit=limit, retry_failed=retry_failed)


def evaluate_one_round(variant: int, run: int):
    """评估一轮结果。"""
    results_dir = get_results_dir(variant, run)
    eval_output = results_dir.parent / "evaluation"

    if not results_dir.exists():
        print(f"[multi_run] 跳过评估 v{variant}_r{run}: 结果目录不存在")
        return

    print(f"[multi_run] 评估 v{variant}_r{run} ...")
    from evaluate import main as eval_main
    eval_main(results_dir=results_dir, output_dir=eval_output)
    print(f"[multi_run] 评估结果已保存: {eval_output}")


def main():
    parser = argparse.ArgumentParser(
        description="多 Prompt 变体 × 多轮自动化实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 试跑
  python multi_run.py --strategies E0 --variants 1 --runs 1 --limit 5

  # Batch 全量
  python multi_run.py --strategies E0,E3,E5,E6 --batch --evaluate

  # 全部策略、全部变体
  python multi_run.py --strategies E0,E1,E3,E4,E5,E6 --batch --evaluate

  # 从第 v2_r2 轮续跑 (跳过已完成的 v1_r* 和 v2_r1)
  python multi_run.py --strategies E3,E4,E5,E6 --variants 2,3 --runs 3 --batch --evaluate --resume-from 2:2
        """,
    )
    parser.add_argument(
        "--strategies", type=str, default="E0,E1,E3,E4,E5,E6",
        help="策略列表, 逗号分隔 (默认: 全部6个策略)",
    )
    parser.add_argument(
        "--variants", type=str, default="1,2,3",
        help="Prompt 变体编号, 逗号分隔 (默认: 1,2,3)",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="每个变体重复跑几轮 (默认: 1, temp=0.15 已提供随机性)",
    )
    parser.add_argument("--batch", action="store_true", help="使用智谱 Batch API (半价)")
    parser.add_argument("--limit", type=int, default=None, help="每策略处理文档数限制")
    parser.add_argument("--evaluate", action="store_true", help="每轮跑完后自动评估")
    parser.add_argument("--evaluate-only", action="store_true", help="只评估已有结果 (不跑策略)")
    parser.add_argument("--aggregate-only", action="store_true", help="只汇总所有评估结果")
    parser.add_argument("--retry-failed", action="store_true", help="重试之前失败的论文")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有结果: 跑前删除指定策略的结果文件")
    parser.add_argument(
        "--resume-from", type=str, default=None, metavar="V:R",
        help="从指定轮次续跑, 格式 V:R (如 2:2 表示从 variant=2, run=2 开始, 跳过之前的轮次)",
    )

    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",")]
    variants = [int(v.strip()) for v in args.variants.split(",")]
    n_runs = args.runs

    # 解析 --resume-from V:R
    resume_variant: int | None = None
    resume_run: int | None = None
    if args.resume_from:
        try:
            rv, rr = args.resume_from.split(":")
            resume_variant, resume_run = int(rv), int(rr)
            print(f"[multi_run] 续跑模式: 从 variant={resume_variant}, run={resume_run} 开始")
        except ValueError:
            print(f"[multi_run] 错误: --resume-from 格式应为 V:R (如 2:2), 收到: {args.resume_from!r}")
            sys.exit(1)

    # 校验 variant 编号
    for v in variants:
        if v not in VARIANT_DIRS:
            print(f"[multi_run] 错误: variant {v} 未定义, 可选: {list(VARIANT_DIRS.keys())}")
            sys.exit(1)
        if not VARIANT_DIRS[v].exists():
            print(f"[multi_run] 错误: prompt 目录不存在: {VARIANT_DIRS[v]}")
            sys.exit(1)

    total = len(variants) * n_runs
    print(f"[multi_run] 计划: {len(variants)} 变体 × {n_runs} 轮 = {total} 轮")
    print(f"  strategies = {strategies}")
    print(f"  variants   = {variants}")
    print(f"  temp       = {config.LLM_TEMPERATURE}")
    print()

    if args.aggregate_only:
        _run_aggregate(variants, n_runs, strategies)
        return

    round_idx = 0
    for v in variants:
        for r in range(1, n_runs + 1):
            round_idx += 1
            # --resume-from: 跳过指定轮次之前的所有轮次
            if resume_variant is not None and resume_run is not None:
                if (v, r) < (resume_variant, resume_run):
                    print(f"[multi_run] 跳过 variant={v}, run={r} (已完成)")
                    continue

            print(f"\n{'#' * 70}")
            print(f"# 第 {round_idx}/{total} 轮: variant={v}, run={r}")
            print(f"{'#' * 70}\n")

            if not args.evaluate_only:
                run_one_round(v, r, strategies, args.batch, args.limit, args.retry_failed, args.overwrite)

            if args.evaluate or args.evaluate_only:
                evaluate_one_round(v, r)

    print(f"\n[multi_run] 全部 {total} 轮完成!")
    print(f"  结果根目录: {MULTI_RESULTS_ROOT}")

    # 自动汇总
    if args.evaluate or args.evaluate_only or args.aggregate_only:
        _run_aggregate(variants, n_runs, strategies)


def _run_aggregate(variants: list[int], n_runs: int, strategies: list[str]):
    """汇总所有评估结果。"""
    print(f"\n[multi_run] 汇总所有评估结果...")
    try:
        from multi_run_evaluate import aggregate_all
        aggregate_all(
            multi_root=MULTI_RESULTS_ROOT,
            variants=variants,
            n_runs=n_runs,
            strategies=strategies,
        )
    except ImportError:
        print("[multi_run] 请确保 multi_run_evaluate.py 存在")
    except Exception as e:
        print(f"[multi_run] 汇总失败: {e}")


if __name__ == "__main__":
    main()
