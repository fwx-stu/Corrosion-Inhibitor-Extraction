#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from corrosion_ie.config import load_config
from corrosion_ie.retrieval.legacy_hybrid import LegacyHybridRetriever

def main():
    ap=argparse.ArgumentParser(description='Load BGE-M3 and reranker and run actual probes.')
    ap.add_argument('--config',required=True)
    args=ap.parse_args(); cfg=load_config(args.config)
    r=LegacyHybridRetriever(dict(cfg.retrieval))
    d=r.preflight(require_reranker=bool(cfg.retrieval.get('enable_reranker',True)))
    for k,v in d.items(): print(f'{k}: {v}')
    if d.get('embedding_error') or d.get('rerank_error'): raise SystemExit(2)
    print('[retrieval-probe-ok]')
if __name__=='__main__': main()
