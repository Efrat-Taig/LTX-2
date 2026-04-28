#!/usr/bin/env python3
"""Emit a trainer-input CSV from a dataset's metadata.json, with brand-token
substitution applied at emit time. The CSV is ephemeral — a transient
artifact for `process_dataset.py`, not a persisted dataset asset.

Usage:
    python emit_trainer_csv.py --dataset DATASET_DIR \
        [--out DATASET_DIR/precomputed/_trainer_input.csv]

Default output is `<dataset>/precomputed/_trainer_input.csv` so it lives
alongside the latents the trainer will produce. Idempotent — overwrite OK.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "dataset_pipeline"))
from captions import apply_brand_tokens, load_brand_tokens  # noqa: E402


def emit(dataset_root: Path, out_path: Path) -> int:
    md = json.loads((dataset_root / "metadata.json").read_text())
    token_map = load_brand_tokens()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["index", "caption", "video_path"])
        for c in md.get("clips", []):
            if c.get("status", "active") != "active":
                continue
            if not (c.get("download_ok", True) and c.get("prompt")):
                continue
            caption = apply_brand_tokens(c["prompt"], token_map)
            w.writerow([c["index"], caption, c["video"]])
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.dataset).resolve()
    out = Path(args.out) if args.out else root / "precomputed" / "_trainer_input.csv"
    n = emit(root, out)
    token_map = load_brand_tokens()
    print(f"wrote {out} ({n} active clips)")
    print(f"brand-token substitutions applied: {token_map}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
