#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import ensure_model_env, load_config, resolve_model_config
from corrosion_ie.retrieval.legacy_hybrid import LegacyHybridRetriever


def model_marker(path: str) -> tuple[bool, str]:
    p = Path(path).expanduser()
    if (p / "config.json").exists():
        return True, "config.json"
    snap = p / "snapshots"
    if snap.exists():
        hits = sorted(snap.glob("*/config.json"))
        if hits:
            return True, str(hits[-1].parent)
    if p.exists():
        return False, "path exists but config.json is missing"
    return False, "path missing"


def require_local_model(path: str, label: str) -> None:
    ok, detail = model_marker(path)
    print(f"{label:16s}: {path} [{detail}]")
    if not ok:
        raise SystemExit(f"[preflight-error] {label} is not a valid local HuggingFace model: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate local Qwen/BGE paths and execute embedding/reranker probes."
    )
    ap.add_argument("--config", default="configs/experiment_q0q1q2_dev37_balanced_opt.yaml")
    ap.add_argument("--model", default="qwen2_5_3b")
    ap.add_argument(
        "--probe-retriever",
        action="store_true",
        help="Actually load BGE-M3 and the reranker and run one inference probe.",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    models_dir = ensure_model_env(cfg_path)
    print(f"project_config  : {cfg_path}")
    print(f"models_dir      : {models_dir}")

    model_cfg = resolve_model_config(args.model, cfg_path)
    resolved_path = str(model_cfg.get("resolved_path") or "")
    if not resolved_path:
        print("candidate_paths:")
        for candidate in model_cfg.get("local_path_candidates", []) or []:
            ok, detail = model_marker(str(candidate))
            print(f"  - {candidate} [{detail}]")
        raise SystemExit(
            f"[preflight-error] no valid local path found for {args.model}; "
            f"set {model_cfg.get('local_path_env') or 'the model path environment variable'}"
        )
    require_local_model(resolved_path, "qwen_model")

    retrieval_cfg = dict(cfg.retrieval)
    embedding_path = str(retrieval_cfg.get("embedding_model") or "")
    reranker_path = str(retrieval_cfg.get("reranker_model") or "")
    require_local_model(embedding_path, "bge_embedding")
    if retrieval_cfg.get("enable_reranker", True):
        require_local_model(reranker_path, "bge_reranker")

    print(f"retrieval_backend: {retrieval_cfg.get('backend')}")
    print(f"chunking_backend : {retrieval_cfg.get('chunking_backend')}")
    print(f"scoring_protocol : {retrieval_cfg.get('scoring_protocol')}")
    print(f"retrieval_device : {retrieval_cfg.get('device')}")
    print(f"strict_loading   : {retrieval_cfg.get('strict_loading')}")

    if args.probe_retriever:
        print("[probe] loading and executing embedding/reranker models...")
        retriever = LegacyHybridRetriever(retrieval_cfg)
        diagnostics = retriever.preflight(
            require_reranker=bool(retrieval_cfg.get("enable_reranker", True))
        )
        for key, value in diagnostics.items():
            print(f"  {key}: {value}")
        if diagnostics.get("embedding_error") or diagnostics.get("rerank_error"):
            raise SystemExit("[preflight-error] retrieval diagnostics contain an error")
        print("[probe] retrieval models are operational")
    else:
        print("[warning] paths were checked, but model inference was not probed")


if __name__ == "__main__":
    main()
