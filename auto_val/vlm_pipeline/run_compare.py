#!/usr/bin/env python3
"""
Run the VLM pipeline on a scene across all available generation models.
Emits a manifest.json consumed by viewer.html.

Usage:
    python -m vlm_pipeline.run_compare [--scene SCENE] [--out-dir DIR]

Default scene: skye_chase_trampoline_backflip
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

BM_ROOT = Path("/Users/efrattaig/projects/tiny_stories_bm/outputs_BM_v1")
REPO = Path("/Users/efrattaig/projects/sm/LTX-2/auto_val")

MODELS = [
    ("BM_v1_Kling_V3_Pro_I2V", "Kling-V3-Pro-I2V"),
    ("BM_v1_Seedance_V15_Pro_I2V", "Seedance-1.5-Pro-I2V"),
    ("BM_v1_LTX23_Fast", "LTX-2.3-Fast"),
    ("BM_v1_Grok_Imagine_Video", "Grok-Imagine-I2V"),
    ("BM_v1_Veo31_Fast_FLF", "Veo-3.1-Fast-FLF"),
    ("BM_v1_Wan26_Flash_I2V", "Wan-2.6-I2V-Flash"),
]


def _reset(out_dir: Path) -> None:
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        return
    for p in out_dir.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        elif p.name != "viewer.html":
            p.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="skye_chase_trampoline_backflip",
                   help="Scene folder name under BM_v1_<model>/<scene>/")
    p.add_argument("--out-dir", default=None,
                   help="Output root. Defaults to outputs_vlm_compare_<scene>/")
    p.add_argument("--enable-prompt-reasoning", action="store_true",
                   help="Forward --enable-prompt-reasoning to each per-model run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scene = args.scene
    out = Path(args.out_dir) if args.out_dir else REPO / f"outputs_vlm_compare_{scene}"
    _reset(out)

    # Ensure the viewer is in place
    viewer_src = REPO / "vlm_pipeline" / "viewer.html"
    if viewer_src.exists():
        shutil.copy2(viewer_src, out / "viewer.html")

    entries: list[tuple[str, str]] = []

    for bucket, subdir in MODELS:
        scene_dir = BM_ROOT / bucket / scene / subdir
        if not scene_dir.exists():
            print(f"SKIP {subdir}: not found")
            continue
        videos = sorted(scene_dir.glob(f"{scene}_P1_*.mp4"))
        if not videos:
            continue
        video = videos[-1]
        meta = video.with_name(video.stem + "_metadata.json")
        prompt = json.loads(meta.read_text()).get("prompt", "") if meta.exists() else ""
        prompt_file = Path(f"/tmp/vlm_prompt_{scene}_{subdir}.txt")
        prompt_file.write_text(prompt)

        print(f"\n{'=' * 70}\n  RUN: {scene} · {subdir}\n{'=' * 70}")
        cmd = [
            sys.executable, "-m", "vlm_pipeline.run",
            str(video),
            "--prompt-file", str(prompt_file),
            "--out-dir", str(out),
        ]
        if args.enable_prompt_reasoning:
            cmd.append("--enable-prompt-reasoning")
        try:
            subprocess.run(cmd, check=True, cwd=str(REPO))
            entries.append((subdir, video.name))
        except subprocess.CalledProcessError as e:
            print(f"FAILED {subdir}: {e}")

    manifest = []
    for subdir, video_name in entries:
        for d in sorted(out.iterdir()):
            if not d.is_dir():
                continue
            mp4s = list(d.glob("*.mp4"))
            if not mp4s or mp4s[0].name != video_name:
                continue
            if any(m["folder"] == d.name for m in manifest):
                continue
            report = json.loads((d / "report.json").read_text())
            agg = report.get("aggregator", {})
            manifest.append({
                "scene": scene,
                "model": subdir,
                "folder": d.name,
                "video": mp4s[0].name,
                "score": int(agg.get("score", 0)),
                "verdict": agg.get("overall_verdict", "FAIL"),
                "headline": agg.get("headline", ""),
            })
            break

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n=== {scene}: {len(manifest)} models ===")
    for m in sorted(manifest, key=lambda x: -x["score"]):
        print(f"  {m['score']:3d}  {m['verdict']:18}  {m['model']}")
    print(f"Manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
