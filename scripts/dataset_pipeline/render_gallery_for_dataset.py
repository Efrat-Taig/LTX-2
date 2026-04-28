#!/usr/bin/env python3
"""Render gallery.mp4 for an existing dataset dir.

Reads metadata.json, calls gallery.render_per_clip for each active clip,
then concats with 0.5s black filler. Idempotent — skips per-clip renders
that already exist.

Usage:
  python render_gallery_for_dataset.py --dataset /path/to/dataset_dir
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "dataset_pipeline"))

from captions import apply_brand_tokens, load_brand_tokens
from gallery import (
    build_concat_list,
    check_ffmpeg,
    concat_gallery,
    render_per_clip,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Local dataset root dir")
    args = ap.parse_args()

    check_ffmpeg()
    root = Path(args.dataset).resolve()
    md_path = root / "metadata.json"
    if not md_path.exists():
        print(f"ERROR: {md_path} not found", file=sys.stderr)
        return 2

    md = json.loads(md_path.read_text())
    clips = md.get("clips", [])
    # Per-clip overlays + concat list stay LOCAL (regeneration cache, not uploaded).
    work_dir = root / "_qa_render_cache"
    per_clip_dir = work_dir / "per_clip"
    per_clip_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    skipped = 0
    failed = 0
    token_map = load_brand_tokens()
    print(f"=== Rendering gallery for {root.name} ({len(clips)} clips) ===", flush=True)
    print(f"    brand-token substitutions (just-in-time): {token_map}", flush=True)
    for c in clips:
        if not c.get("download_ok", True) or not c.get("prompt"):
            skipped += 1
            continue
        idx = c["index"]
        src = root / c["video"]
        dst = per_clip_dir / f"{idx:04d}.mp4"
        if dst.exists() and dst.stat().st_size > 0:
            skipped += 1
            continue
        if not src.exists():
            print(f"  [{idx:04d}] missing source: {src}", flush=True)
            failed += 1
            continue
        try:
            # Substitute character names → brand tokens at render time only;
            # metadata.json's `prompt` stays raw.
            display_prompt = apply_brand_tokens(c["prompt"], token_map)
            render_per_clip(src, display_prompt, idx, dst)
            print(f"  [{idx:04d}] rendered", flush=True)
            rendered += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [{idx:04d}] FAIL: {e}", flush=True)
            failed += 1

    active = [c["index"] for c in clips if c.get("download_ok", True) and c.get("prompt")]
    concat_list = work_dir / "concat.txt"
    gallery_mp4 = root / "qa_gallery.mp4"
    print(f"\n[concat] active={len(active)} clips, rendered_this_run={rendered}, skipped={skipped}, failed={failed}", flush=True)
    build_concat_list(active, per_clip_dir, concat_list)
    concat_gallery(concat_list, gallery_mp4)
    print(f"  → {gallery_mp4} ({gallery_mp4.stat().st_size/1024:.0f} KB)", flush=True)

    # Update metadata
    import datetime as dt
    md["qa_gallery"] = {
        "path": "qa_gallery.mp4",
        "interclip_black_s": 0.5,
        "rendered_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rendered_index_count": sum(1 for i in active if (per_clip_dir / f"{i:04d}.mp4").exists()),
    }
    md.pop("gallery", None)  # drop legacy field if present
    md_path.write_text(json.dumps(md, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
