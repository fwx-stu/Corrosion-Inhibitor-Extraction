#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import load_config
from corrosion_ie.runner import run_single_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one model/strategy/prompt/seed extraction.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--paper-id", default=None)
    ap.add_argument("--paper-ids", default=None, help="Comma-separated paper IDs.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--fail-on-quality-gate", action="store_true")
    ap.add_argument("--dry-run-client", action="store_true", help="Do not load an LLM; emit empty JSON for pipeline smoke tests.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paper_ids = [x.strip() for x in (args.paper_ids or "").split(",") if x.strip()]
    if args.paper_id and paper_ids:
        raise SystemExit("use either --paper-id or --paper-ids")
    run_single_experiment(
        cfg,
        model_key=args.model,
        strategy_name=args.strategy,
        prompt_variant=args.prompt,
        seed=args.seed,
        force=args.force,
        limit=args.limit,
        paper_id=args.paper_id,
        paper_ids=paper_ids or None,
        stop_on_error=args.stop_on_error,
        fail_on_quality_gate=(args.fail_on_quality_gate or bool((cfg.raw.get("run", {}) or {}).get("fail_on_quality_gate", False))),
        dry_run=args.dry_run_client,
    )

if __name__ == "__main__":
    main()
