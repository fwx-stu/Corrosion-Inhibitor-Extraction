#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, platform, subprocess, sys
from pathlib import Path


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="environment_lock")
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out/"requirements-lock.txt").write_text(run([sys.executable,"-m","pip","freeze"])+"\n",encoding="utf-8")
    meta={"python":platform.python_version(),"platform":platform.platform(),"executable":sys.executable}
    try:
        import torch
        meta.update({"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except Exception as exc:
        meta["torch_error"]=repr(exc)
    (out/"runtime.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"output_dir":str(out),**meta},indent=2,ensure_ascii=False))
if __name__=="__main__": main()
