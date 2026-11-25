#!/usr/bin/env python3
"""Run a formally frozen test matrix without shell-specific environment syntax."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and run one frozen formal-test matrix.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--progress", choices=["auto", "tqdm", "plain", "none"], default="plain")
    args = parser.parse_args()

    config = Path(args.config).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    if not config.is_file():
        raise SystemExit(f"missing config: {config}")
    if not manifest.is_file():
        raise SystemExit(f"missing frozen protocol manifest: {manifest}")

    env = dict(os.environ)
    env["CONFIRM_FROZEN_TEST"] = "YES"
    env["PROTOCOL_MANIFEST"] = str(manifest)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_matrix.py"),
        "--config", str(config),
        "--progress", args.progress,
    ]
    completed = subprocess.run(command, cwd=ROOT, env=env)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
