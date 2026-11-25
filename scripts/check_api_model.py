#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from corrosion_ie.config import load_config, resolve_model_config

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--model',required=True); args=ap.parse_args()
    cfg=load_config(args.config); m=resolve_model_config(args.model,cfg.config_path)
    if m.get('backend')!='openai_compatible': raise SystemExit(f"{args.model} is not openai_compatible")
    base=os.getenv(m.get('base_url_env',''),m.get('base_url','')); key=os.getenv(m.get('api_key_env',''),m.get('api_key',''))
    if not base or not key: raise SystemExit(f"[api-preflight-error] set {m.get('base_url_env')} and {m.get('api_key_env')}")
    print(f"[api-preflight-ok] model={m.get('model_id')} base_url={base}")
if __name__=='__main__': main()
