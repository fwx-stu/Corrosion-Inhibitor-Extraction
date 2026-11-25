from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _load_legacy():
    root = Path(__file__).resolve().parents[2]
    path = root / "vendor" / "legacy_evaluate.py"
    spec = importlib.util.spec_from_file_location("legacy_evaluate_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["legacy_evaluate_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


def evaluate_prediction(gold_path: str | Path, pred_path: str | Path, output_dir: str | Path, strategy: str, docs_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    legacy = _load_legacy()
    return legacy.run_single_prediction(
        gold_path=str(gold_path),
        pred_file=str(pred_path),
        output_dir=str(output_dir),
        strategy=strategy,
        docs_dir=str(docs_dir) if docs_dir else None,
    )
