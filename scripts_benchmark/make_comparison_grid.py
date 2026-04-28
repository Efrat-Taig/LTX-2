#!/usr/bin/env python3
"""
make_comparison_grid.py

Per-scene side-by-side comparison grid of every variant:

  baseline  |  t1_prompt  |  t2_stg0.5  |  t2_stg1.0  |  t2_stg2.0  [ + t3_acfg12 | t3_acfg18 ]

Each scene produces one MP4 with all variants playing in a single row, each cell
labeled. Audio is taken from the baseline cell (most representative of "what the
model was asked to do" — you can listen to individual variants directly).

All per-scene grids are also concatenated into one master comparison video.

Output:
  output/exp_no_lora_comparison/{scene}.mp4
  output/exp_no_lora_comparison/comparison_all.mp4
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).resolve().parent
CHASE_BASE  = SCRIPT_DIR / "lora_results" / "chase_lora" / "benchmarks" / "_base"
SKYE_BASE   = SCRIPT_DIR / "lora_results_FIX" / "benchmarks" / "_base"
HELI_BASE   = (
    SCRIPT_DIR / "output" / "output2"
    / "run_skye_like_api__skye_helicopter_birthday_gili__768x1024_15steps_8.04s_seed1660213644.mp4"
)
EXP_ROOT    = SCRIPT_DIR / "lora_results" / "exp_no_lora"
OUT_DIR     = SCRIPT_DIR / "output" / "exp_no_lora_comparison"

FONT_BOLD   = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

# ── Cell dimensions ────────────────────────────────────────────────────────────

CELL_W       = 384
CELL_H       = 512   # letterbox all clips to 384×512
LABEL_H      = 40    # label strip above each cell

# ── Scene definitions ──────────────────────────────────────────────────────────

def chase_scene(name: str, filename: str) -> dict:
    return {
        "name": f"chase_{name}",
        "display": f"Chase — {name}",
        "baseline": CHASE_BASE / filename,
        "variants": ["baseline_notext", "t1_prompt", "t2_stg0.5", "t2_stg1.0", "t2_stg2.0"],
        "exp_key": f"chase_{name}",
    }


def skye_scene(name: str, filename: str, *, has_acfg: bool = False) -> dict:
    extra = ["t3_acfg12", "t3_acfg18"] if has_acfg else []
    return {
        "name": f"skye_{name}",
        "display": f"Skye — {name}",
        "baseline": SKYE_BASE / filename,
        "variants": ["baseline_notext", "t1_prompt", "t2_stg0.5", "t2_stg1.0", "t2_stg2.0"] + extra,
        "exp_key": f"skye_{name}",
    }


SCENES = [
    chase_scene("normal",    "normal.mp4"),
    chase_scene("halloween", "halloween.mp4"),
    chase_scene("crms",      "crms.mp4"),
    chase_scene("snow",      "snow.mp4"),
    chase_scene("party",     "party.mp4"),
    skye_scene("bey",        "bey.mp4",       has_acfg=True),
    skye_scene("crsms",      "crsms.mp4",     has_acfg=True),
    skye_scene("holoween",   "holoween.mp4"),
    skye_scene("party",      "party.mp4"),
    skye_scene("snow",       "snow.mp4",      has_acfg=True),
    {
        "name": "skye_helicopter",
        "display": "Skye — helicopter",
        "baseline": HELI_BASE,
        "variants": ["baseline_notext", "t1_prompt", "t2_stg0.5", "t2_stg1.0", "t2_stg2.0", "t3_acfg12", "t3_acfg18"],
        "exp_key": "skye_helicopter",
    },
]


# ── ffmpeg helpers ─────────────────────────────────────────────────────────────

def run(cmd: list[str], *, label: str) -> None:
    print(f"  ▶ {label}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({label}):\n{result.stderr[-3000:]}")


def get_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    streams = json.loads(result.stdout)["streams"]
    return float(next(s["duration"] for s in streams if s["codec_type"] == "video"))


def make_label_png(text: str, w: int, h: int) -> Image.Image:
    img  = Image.new("RGB", (w, h), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_BOLD, 20)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (h - th) // 2), text, font=font, fill=(255, 220, 0))
    return img


def collect_clips(scene: dict) -> list[tuple[str, Path]]:
    """Return [(label, path)] for variants that exist on disk."""
    result: list[tuple[str, Path]] = []
    if scene["baseline"].exists():
        result.append(("baseline", scene["baseline"]))
    for v in scene["variants"]:
        p = EXP_ROOT / v / f"{scene['exp_key']}.mp4"
        if p.exists():
            result.append((v, p))
    return result


def build_scene_grid(scene: dict, tmp_dir: Path) -> Path | None:
    clips = collect_clips(scene)
    if len(clips) < 2:
        print(f"  · skip {scene['name']} — only {len(clips)} clip(s) available")
        return None

    out = OUT_DIR / f"{scene['name']}.mp4"
    n   = len(clips)
    grid_w = CELL_W * n
    grid_h = CELL_H + LABEL_H

    # Render label strip as one PNG spanning the whole grid.
    label_img = Image.new("RGB", (grid_w, LABEL_H), color=(20, 20, 20))
    for i, (label, _) in enumerate(clips):
        strip = make_label_png(label, CELL_W, LABEL_H)
        label_img.paste(strip, (i * CELL_W, 0))
    label_png = tmp_dir / f"{scene['name']}_labels.png"
    label_img.save(str(label_png))

    # Determine shortest clip duration — truncate all to match.
    min_dur = min(get_duration(p) for _, p in clips)
    dur = round(min_dur - 0.1, 2)  # small safety margin

    # Build filter_complex: scale/letterbox each clip → hstack → overlay labels.
    n_inputs = len(clips)
    parts = []
    for i in range(n_inputs):
        parts.append(
            f"[{i}:v]scale={CELL_W}:{CELL_H}:force_original_aspect_ratio=decrease,"
            f"pad={CELL_W}:{CELL_H}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,trim=duration={dur},setpts=PTS-STARTPTS[v{i}]"
        )
    stack_inputs = "".join(f"[v{i}]" for i in range(n_inputs))
    parts.append(f"{stack_inputs}hstack=inputs={n_inputs}[videos]")
    # Overlay label strip (loaded as last input, n_inputs index) at top
    parts.append(f"[videos]pad={grid_w}:{grid_h}:0:{LABEL_H}:black[padded]")
    parts.append(f"[padded][{n_inputs}:v]overlay=0:0[out]")
    fc = ";".join(parts)

    # Audio: take the baseline's audio (clip 0) so we have one coherent track.
    cmd = ["ffmpeg", "-y"]
    for _, p in clips:
        cmd += ["-i", str(p)]
    cmd += ["-loop", "1", "-i", str(label_png)]
    cmd += [
        "-filter_complex", fc,
        "-map", "[out]",
        "-map", "0:a?",
        "-t", str(dur),
        "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        str(out),
    ]
    run(cmd, label=f"grid {scene['name']} ({n} cells × {CELL_W}×{CELL_H}, dur {dur}s)")
    return out


def concat_grids(paths: list[Path], out_path: Path) -> None:
    """Concatenate per-scene grids vertically-in-time (sequential playback)."""
    # Normalize all to the widest grid width by padding.
    max_w = max_h = 0
    info: list[tuple[Path, int, int]] = []
    for p in paths:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(p)],
            capture_output=True, text=True,
        )
        v = next(s for s in json.loads(result.stdout)["streams"] if s["codec_type"] == "video")
        w, h = int(v["width"]), int(v["height"])
        info.append((p, w, h))
        max_w, max_h = max(max_w, w), max(max_h, h)

    pad_dir = out_path.parent / "tmp" / "padded"
    pad_dir.mkdir(parents=True, exist_ok=True)
    padded: list[Path] = []
    for p, w, h in info:
        pp = pad_dir / p.name
        if not pp.exists():
            run(
                [
                    "ffmpeg", "-y", "-i", str(p),
                    "-vf", f"pad={max_w}:{max_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
                    "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    str(pp),
                ],
                label=f"pad {p.name} → {max_w}×{max_h}",
            )
        padded.append(pp)

    list_file = out_path.parent / "concat_list.txt"
    with open(list_file, "w") as f:
        for p in padded:
            f.write(f"file '{p}'\n")
    run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            str(out_path),
        ],
        label=f"concatenate {len(padded)} grids → {out_path.name}",
    )
    list_file.unlink(missing_ok=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_DIR / "tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    grids: list[Path] = []
    print(f"\nBuilding per-scene comparison grids → {OUT_DIR}\n")
    for scene in SCENES:
        g = build_scene_grid(scene, tmp_dir)
        if g is not None:
            grids.append(g)

    if not grids:
        print("No grids built. Is lora_results/exp_no_lora/ populated?")
        return

    master = OUT_DIR / "comparison_all.mp4"
    print(f"\nConcatenating {len(grids)} scene grids → {master.name}")
    concat_grids(grids, master)
    print(f"\nDone → {master}")


if __name__ == "__main__":
    main()
