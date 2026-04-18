#!/usr/bin/env python3
"""
create_comparison.py

For each benchmark scene, assembles all available experiment final-step videos
into a single side-by-side grid video with labels.

Output: lora_compar_1/<scene>_comparison.mp4

Usage:
    python3 create_comparison.py
    python3 create_comparison.py --scenes bey party   # specific scenes only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Config ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR  = SCRIPT_DIR / "lora_results_FIX" / "benchmarks"
OUTPUT_DIR = SCRIPT_DIR / "lora_compar_1"

# Each row: (bench_folder, final_step_or_None, short_label)
# None = base model (no step subfolder, no _lora suffix)
# Add EXP-3 here once its results land in lora_results_FIX/benchmarks/
ALL_EXPERIMENTS = [
    ("_base",               None,  "Base Model"),
    ("skye_exp1_baseline",  1000,  "EXP-1\nrank16 1k"),
    ("skye_exp2_standard",  5000,  "EXP-2\nrank32 5k"),
    ("skye_exp2_10k",      10000,  "EXP-2-10k\nrank32 10k"),
    ("skye_exp4_i2v",       2500,  "EXP-4\ni2v 2.5k"),
    ("skye_exp4_10k",      10000,  "EXP-4-10k\ni2v 10k"),
    # ("skye_exp3_highcap", XXXX,  "EXP-3\n+FFN"),  ← uncomment when ready
]

ALL_SCENES = [
    "bey", "crsms", "holoween", "party", "snow",
    "of_silly_goose", "of_eagle", "of_lifeguard", "of_lift", "of_everest",
]

# Grid cell size (portrait 3:4)
CELL_W = 320
CELL_H = 428  # must be even for yuv420p

# ── Helpers ──────────────────────────────────────────────────────────────────

def video_path(folder: str, step: int | None, scene: str) -> Path | None:
    if step is None:
        p = BENCH_DIR / folder / f"{scene}.mp4"
    else:
        p = BENCH_DIR / folder / f"step_{step:05d}" / f"{scene}_lora.mp4"
    return p if p.exists() else None


def make_label_image(label: str, width: int, height: int = 36) -> Path:
    """Render a label bar as a PNG, return path to temp file."""
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
    text = label.replace("\n", " | ")
    draw.text((6, 8), text, fill=(255, 255, 255), font=font)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return Path(tmp.name)


def make_grid(scene: str, exps: list[tuple[str, int | None, str]]) -> None:
    videos = []
    labels = []
    for folder, step, label in exps:
        p = video_path(folder, step, scene)
        if p:
            videos.append(p)
            labels.append(label)
        else:
            print(f"    skip {label.split(chr(10))[0]} — not found")

    if len(videos) < 2:
        print(f"  {scene}: only {len(videos)} video(s) — skipping")
        return

    n = len(videos)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    # Create label PNGs
    label_files = [make_label_image(lbl, CELL_W) for lbl in labels]
    label_h = 36

    # Each cell = label bar (36px) + video (CELL_H px)
    cell_total_h = label_h + CELL_H
    out_w = cols * CELL_W
    out_h = rows * cell_total_h

    # Build filter_complex:
    # Each video stream: scale to CELL_W x CELL_H
    # Each label image: loop as still frame, scale to CELL_W x label_h
    # Combine label + video vertically per cell, then xstack all cells
    n_inputs = n  # video inputs
    parts: list[str] = []

    # Scale each video
    for i in range(n):
        parts.append(f"[{i}:v]scale={CELL_W}:{CELL_H}[sv{i}]")

    # For each label PNG: loop 1 frame, scale to label bar
    label_input_offset = n
    for i, lf in enumerate(label_files):
        idx = label_input_offset + i
        parts.append(f"[{idx}:v]scale={CELL_W}:{label_h}[lb{i}]")

    # Stack label + video vertically per cell (shortest=1 trims looped label to video length)
    for i in range(n):
        parts.append(f"[lb{i}][sv{i}]vstack=inputs=2:shortest=1[cell{i}]")

    # xstack all cells
    layout = "|".join(
        f"{(i % cols) * CELL_W}_{(i // cols) * cell_total_h}" for i in range(n)
    )
    stack_inputs = "".join(f"[cell{i}]" for i in range(n))
    parts.append(
        f"{stack_inputs}xstack=inputs={n}:layout={layout}"
        f":fill=black:shortest=1[out]"
    )

    filter_complex = ";".join(parts)

    # Build input list: videos first, then label images (looped)
    input_args: list[str] = []
    for v in videos:
        input_args += ["-i", str(v)]
    for lf in label_files:
        input_args += ["-loop", "1", "-i", str(lf)]

    out_path = OUTPUT_DIR / f"{scene}_comparison.mp4"
    cmd = (
        ["ffmpeg", "-y"]
        + input_args
        + [
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:v", "libx264", "-crf", "20", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]
    )

    print(f"  {scene}: {n} exps → {out_path.name}  ({out_w}×{out_h})")
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Clean up temp label files
    for lf in label_files:
        lf.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr[-800:]}")
    else:
        size_mb = out_path.stat().st_size / 1e6
        print(f"    saved {size_mb:.1f} MB")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build comparison grid videos")
    parser.add_argument("--scenes", nargs="+", default=ALL_SCENES)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Detect which experiments are available
    available = []
    print("Checking available experiments:")
    for folder, step, label in ALL_EXPERIMENTS:
        p = video_path(folder, step, "bey")
        short = label.split("\n")[0]
        if p:
            available.append((folder, step, label))
            print(f"  ✓  {short}")
        else:
            print(f"  ✗  {short}  (not yet)")
    print()

    if len(available) < 2:
        print("Need at least 2 experiments. Aborting.")
        sys.exit(1)

    print(f"Building grid for {len(args.scenes)} scene(s)...\n")
    for scene in args.scenes:
        make_grid(scene, available)

    print(f"\nDone — videos in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
