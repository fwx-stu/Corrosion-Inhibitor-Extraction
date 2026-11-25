#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import load_config

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_title(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates = []
    for line in lines[:80]:
        clean = re.sub(r"^#{1,6}\s*", "", line).strip()
        if clean.lower() in {"abstract", "introduction", "keywords"}:
            continue
        if len(clean) >= 20 and not clean.startswith("<") and clean.count("|") < 2:
            candidates.append(clean)
    title = candidates[0] if candidates else (lines[0] if lines else "")
    return " ".join(TOKEN_RE.findall(title.lower()))


def simhash64(text: str, shingle_size: int = 5, max_shingles: int = 4096) -> int:
    """Return a deterministic, bounded-cost near-duplicate fingerprint.

    Corrosion papers share a large domain vocabulary, so word-frequency SimHash
    produces false positives. Five-token shingles retain local wording. For very
    long papers, evenly sampling at most ``max_shingles`` positions keeps the
    preflight check fast without biasing it toward only the beginning of a paper.
    Exact SHA256, DOI and normalized-title checks remain the hard leakage checks.
    """
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0

    total = max(1, len(tokens) - shingle_size + 1)
    sample_count = min(total, max(1, int(max_shingles)))
    if sample_count == 1:
        positions = [0]
    else:
        # Integer interpolation includes both the first and last available
        # shingle and is deterministic across Python versions.
        positions = [i * (total - 1) // (sample_count - 1) for i in range(sample_count)]

    # Start each bit at -N. Each set bit adds 2, which is equivalent to adding
    # +1 for set bits and -1 for unset bits while touching only set bits.
    vector = [-sample_count] * 64
    for start in positions:
        shingle = " ".join(tokens[start:start + shingle_size])
        value = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        while value:
            lsb = value & -value
            vector[lsb.bit_length() - 1] += 2
            value ^= lsb

    out = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            out |= 1 << bit
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def describe(md_dir: Path, *, label: str = "dataset") -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    paths = sorted(md_dir.glob("*.md"))
    print(f"[split-isolation] hashing {label}: {len(paths)} papers", flush=True)
    for index, path in enumerate(paths, start=1):
        text = path.read_text(encoding="utf-8", errors="ignore")
        dois = sorted({match.group(0).rstrip(".,;)").lower() for match in DOI_RE.finditer(text[:20000])})
        out[path.stem] = {
            "path": str(path),
            "sha256": sha256(path),
            "title": normalize_title(text),
            "dois": dois,
            "simhash": simhash64(text),
        }
        if index % 2 == 0 or index == len(paths):
            print(f"[split-isolation] {label}: {index}/{len(paths)}", flush=True)
    return out


def duplicates(left: Dict[str, Dict[str, object]], right: Dict[str, Dict[str, object]], max_hamming: int) -> Dict[str, object]:
    id_overlap = sorted(set(left) & set(right))
    exact_hash = []
    doi_overlap = []
    title_overlap = []
    near = []
    for lid, lv in left.items():
        for rid, rv in right.items():
            if lv["sha256"] == rv["sha256"]:
                exact_hash.append([lid, rid])
            ldois, rdois = set(lv["dois"]), set(rv["dois"])
            if ldois & rdois:
                doi_overlap.append({"left": lid, "right": rid, "dois": sorted(ldois & rdois)})
            if lv["title"] and lv["title"] == rv["title"]:
                title_overlap.append({"left": lid, "right": rid, "title": lv["title"]})
            distance = hamming(int(lv["simhash"]), int(rv["simhash"]))
            if distance <= max_hamming:
                near.append({"left": lid, "right": rid, "simhash_hamming": distance})
    return {
        "paper_id_overlap": id_overlap,
        "exact_document_hash_overlap": exact_hash,
        "doi_overlap": doi_overlap,
        "normalized_title_overlap": title_overlap,
        "near_duplicate_candidates": near,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Check dev/test split isolation and duplicate leakage.")
    ap.add_argument("--left-config", required=True)
    ap.add_argument("--right-config", required=True)
    ap.add_argument("--max-simhash-hamming", type=int, default=2)
    ap.add_argument("--fail-on-near", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    left_cfg = load_config(args.left_config)
    right_cfg = load_config(args.right_config)
    left = describe(left_cfg.md_dir, label=left_cfg.dataset_name)
    right = describe(right_cfg.md_dir, label=right_cfg.dataset_name)
    found = duplicates(left, right, args.max_simhash_hamming)
    hard = any(found[key] for key in [
        "paper_id_overlap", "exact_document_hash_overlap", "doi_overlap", "normalized_title_overlap"
    ])
    if args.fail_on_near and found["near_duplicate_candidates"]:
        hard = True
    result = {
        "left": {"name": left_cfg.dataset_name, "count": len(left), "md_dir": str(left_cfg.md_dir)},
        "right": {"name": right_cfg.dataset_name, "count": len(right), "md_dir": str(right_cfg.md_dir)},
        "max_simhash_hamming": args.max_simhash_hamming,
        "valid": not hard,
        **found,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if hard:
        raise SystemExit(2)
    print("[split-isolation-ok]")


if __name__ == "__main__":
    main()
