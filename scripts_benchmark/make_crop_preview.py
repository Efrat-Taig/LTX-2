#!/usr/bin/env python3
"""
make_crop_preview.py

Creates two preview videos showing exactly what each training clip looks like
after the center-crop used during preprocessing — one per training resolution.

No GPU required. Uses ffmpeg (same resize+center-crop logic as process_videos.py).

Outputs:
  output/preview_crops_960x544.mp4    landscape — used by EXP-2, EXP-4, EXP-4-10K
  output/preview_crops_768x1024.mp4   portrait  — used by EXP-3-Highres, EXP-4-Highres

Usage:
    uv run python make_crop_preview.py
    uv run python make_crop_preview.py --dataset-json /path/to/dataset.json
    uv run python make_crop_preview.py --resolutions 960x544  # only one
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

DATASET_JSON = Path("/home/efrattaig/data/golden_skye_train/dataset.json")
OUTPUT_DIR   = Path(__file__).resolve().parent / "output"

RESOLUTIONS = [
    ("960x544",   960,  544),
    ("768x1024",  768, 1024),
]


def make_preview(clips: list[tuple[str, str]], width: int, height: int, output_path: Path) -> None:
    with tempfile.TemporaryDirectory() as _tmp:
        tmpdir = Path(_tmp)
        ok_clips: list[Path] = []

        for i, (media_path, caption) in enumerate(clips):
            out_clip = tmpdir / f"clip_{i:04d}.mp4"

            # Write caption to a temp file — avoids ALL shell/filter-string escaping issues
            caption_file = tmpdir / f"cap_{i:04d}.txt"
            caption_file.write_text(caption[:120])

            # Same logic as process_videos.py _resize_and_crop:
            #   scale so both dims are >= target (force_original_aspect_ratio=increase)
            #   then center-crop to exact target size
            vf = (
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},"
                f"drawtext=textfile={caption_file}:"
                f"fontsize=18:fontcolor=white:x=10:y=h-70:"
                f"box=1:boxcolor=black@0.65:boxborderw=5:expansion=none"
            )

            cmd = [
                "ffmpeg", "-y",
                "-i", str(media_path),
                "-vf", vf,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast",
                "-an",
                "-avoid_negative_ts", "make_zero",  # prevent B-frame DTS issues in concat
                "-t", "6",   # cap each clip at 6s so preview doesn't get too long
                str(out_clip),
            ]
            r = subprocess.run(cmd, capture_output=True)
            name = Path(media_path).name
            if r.returncode == 0:
                ok_clips.append(out_clip)
                print(f"  [{i+1:3d}/{len(clips)}] ✓  {name}")
            else:
                print(f"  [{i+1:3d}/{len(clips)}] ✗  {name}")
                print("    " + r.stderr.decode(errors="replace")[-200:])

        if not ok_clips:
            print("ERROR: no clips processed successfully", file=sys.stderr)
            return

        # Write concat list
        concat_txt = tmpdir / "concat.txt"
        concat_txt.write_text("".join(f"file '{p}'\n" for p in ok_clips))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "fast",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"\n  → saved: {output_path}  ({len(ok_clips)}/{len(clips)} clips)\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create center-crop preview videos for training data")
    parser.add_argument("--dataset-json", type=Path, default=DATASET_JSON)
    parser.add_argument("--output-dir",   type=Path, default=OUTPUT_DIR)
    parser.add_argument("--resolutions",  nargs="+", default=None,
                        help="Limit to these resolutions e.g. 960x544 768x1024")
    args = parser.parse_args()

    if not args.dataset_json.exists():
        print(f"ERROR: dataset.json not found: {args.dataset_json}", file=sys.stderr)
        return 1

    clips_raw = json.loads(args.dataset_json.read_text())
    clips = [(c["media_path"], c.get("caption", "")) for c in clips_raw]
    print(f"Loaded {len(clips)} clips from {args.dataset_json}\n")

    resolutions = RESOLUTIONS
    if args.resolutions:
        wanted = set(args.resolutions)
        resolutions = [r for r in RESOLUTIONS if r[0] in wanted]

    for res_name, w, h in resolutions:
        out = args.output_dir / f"preview_crops_{res_name}.mp4"
        print(f"=== {res_name} → {out.name} ===")
        make_preview(clips, w, h, out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
