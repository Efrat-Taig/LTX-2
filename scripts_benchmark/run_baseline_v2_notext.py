#!/usr/bin/env python3
"""
run_baseline_v2_notext.py

Re-runs the baseline (untrained) generation with the ORIGINAL prompts and
ORIGINAL settings, but with an extended negative prompt that explicitly
forbids burned-in text:

  , text, watermark, subtitles, captions, letters, alphabet, words.

Scenes (11 total):
  • 5 Chase — prompts + neg from run_chase_benchmark.BENCHMARK_SCENES
  • 5 Skye  — prompts + neg from run_skye_benchmark.BENCHMARK_SCENES
  • 1 skye_helicopter — prompts + neg from inputs/BM_v1/benchmark_v1/
                           skye_helicopter_birthday_gili/config.json

Output: lora_results/exp_no_lora/baseline_notext/<char>_<scene>.mp4
        (so the viewer picks it up as a new variant column next to baseline)

Usage:
    uv run python run_baseline_v2_notext.py
    uv run python run_baseline_v2_notext.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent

MODELS_DIR        = Path.home() / "models" / "LTX-2.3"
CHECKPOINT        = MODELS_DIR / "ltx-2.3-22b-dev.safetensors"
DISTILLED_LORA    = MODELS_DIR / "ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSAMPLER = MODELS_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT        = Path.home() / "models" / "gemma-3-12b-it-qat-q4_0-unquantized"

HELI_CFG          = SCRIPT_DIR / "inputs" / "BM_v1" / "benchmark_v1" / "skye_helicopter_birthday_gili" / "config.json"
HELI_IMG          = SCRIPT_DIR / "inputs" / "BM_v1" / "benchmark_v1" / "skye_helicopter_birthday_gili" / "start_frame.png"

OUT_DIR = SCRIPT_DIR / "lora_results" / "exp_no_lora" / "baseline_notext"

# Baseline generation params (match run_chase_benchmark.py / run_skye_benchmark.py)
DURATION_S = 10.0
SEED       = 42
FPS        = 25.0
STEPS      = 15
VIDEO_CFG  = 3.0
AUDIO_CFG  = 7.0
LORA_S1    = 0.25
LORA_S2    = 0.50

CHASE_W, CHASE_H = 832, 960
SKYE_W,  SKYE_H  = 768, 1024

# ── Extended negative: banned on-frame text ───────────────────────────────────
NEG_EXTRA = ", text, watermark, subtitles, captions, letters, alphabet, words."


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def frames_for_duration(seconds: float, fps: float) -> int:
    raw = round(seconds * fps)
    k = max(1, (raw - 1 + 7) // 8)
    return max(17, 8 * k + 1)


def build_pipeline():
    import torch
    torch.backends.cudnn.enabled = False
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
    return TI2VidTwoStagesHQPipeline(
        checkpoint_path=str(CHECKPOINT),
        distilled_lora=[LoraPathStrengthAndSDOps(
            path=str(DISTILLED_LORA), strength=1.0,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        )],
        distilled_lora_strength_stage_1=LORA_S1,
        distilled_lora_strength_stage_2=LORA_S2,
        spatial_upsampler_path=str(SPATIAL_UPSAMPLER),
        gemma_root=str(GEMMA_ROOT),
        loras=(),
    )


def unload(pipeline) -> None:
    import torch
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


# ── Scene assembly ─────────────────────────────────────────────────────────────

def assemble_scenes() -> list[dict]:
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT

    # Original per-character negatives match what run_chase/skye_benchmark.py sends.
    chase_neg = (
        DEFAULT_NEGATIVE_PROMPT
        + ", waving paw, raised paw, paw in the air, arm raised, hand gesture, "
          "waving hand, pointing, gesturing, paw movement, arm movement"
        + NEG_EXTRA
    )
    skye_neg = DEFAULT_NEGATIVE_PROMPT + NEG_EXTRA

    chase_mod = _load(SCRIPT_DIR / "run_chase_benchmark.py", "_chase_bm")
    skye_mod  = _load(SCRIPT_DIR / "run_skye_benchmark.py",  "_skye_bm")

    scenes: list[dict] = []

    for s in chase_mod.BENCHMARK_SCENES:
        scenes.append({
            "key":    f"chase_{s['name']}",
            "image":  s["image"],
            "prompt": s["prompt"],
            "neg":    chase_neg,
            "w": CHASE_W, "h": CHASE_H, "fps": FPS, "strength": 1.0,
        })

    for s in skye_mod.BENCHMARK_SCENES:
        scenes.append({
            "key":    f"skye_{s['name']}",
            "image":  s["image"],
            "prompt": s["prompt"],
            "neg":    skye_neg,
            "w": SKYE_W, "h": SKYE_H, "fps": FPS, "strength": 1.0,
        })

    # Helicopter: use its own JSON config (own neg, own dims, own fps/strength).
    if HELI_CFG.exists():
        cfg = json.loads(HELI_CFG.read_text())
        heli_neg = cfg.get("negative_prompt", "") + NEG_EXTRA
        heli_prompt = cfg["global_prompt"] + "\n\n" + cfg["specific_prompts"][0]
        dims = cfg.get("dimensions", {})
        # Round config 698 up to nearest multiple of 32 for pipeline safety (704).
        w = int(dims.get("width", 704))
        if w % 32:
            w = ((w + 31) // 32) * 32
        h = int(dims.get("height", 1280))
        if h % 32:
            h = ((h + 31) // 32) * 32
        scenes.append({
            "key":      "skye_helicopter",
            "image":    HELI_IMG,
            "prompt":   heli_prompt,
            "neg":      heli_neg,
            "w": w, "h": h,
            "fps":      float(cfg.get("fps", 24)),
            "strength": float(cfg.get("images", [{}])[0].get("strength", 0.9)),
        })

    return scenes


# ── Generation ─────────────────────────────────────────────────────────────────

def generate(pipeline, *, scene: dict, out_path: Path) -> None:
    import torch
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video

    fps = scene["fps"]
    num_frames    = frames_for_duration(DURATION_S, fps)
    tiling_config = TilingConfig.default()
    video_chunks  = get_video_chunks_number(num_frames, tiling_config)

    video_guider = MultiModalGuiderParams(
        cfg_scale=VIDEO_CFG, stg_scale=0.0, rescale_scale=0.45,
        modality_scale=3.0, skip_step=0, stg_blocks=[],
    )
    audio_guider = MultiModalGuiderParams(
        cfg_scale=AUDIO_CFG, stg_scale=0.0, rescale_scale=1.0,
        modality_scale=3.0, skip_step=0, stg_blocks=[],
    )

    video, audio = pipeline(
        prompt=scene["prompt"],
        negative_prompt=scene["neg"],
        seed=SEED,
        height=scene["h"],
        width=scene["w"],
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=STEPS,
        video_guider_params=video_guider,
        audio_guider_params=audio_guider,
        images=[ImageConditioningInput(path=str(scene["image"]), frame_idx=0, strength=scene["strength"])],
        tiling_config=tiling_config,
        enhance_prompt=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        encode_video(
            video=video, fps=int(fps), audio=audio,
            output_path=str(out_path),
            video_chunks_number=video_chunks,
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenes = assemble_scenes()
    work   = [s for s in scenes if not (OUT_DIR / f"{s['key']}.mp4").exists()]

    print(f"\nrun_baseline_v2_notext — extended negative: {NEG_EXTRA.strip(', .')}")
    print(f"  {len(work)} video(s) to generate (already done: {len(scenes) - len(work)})")
    for s in work:
        print(f"  · {s['key']}  ({s['w']}×{s['h']} @ {s['fps']}fps)")
    print()

    if not work:
        print("Nothing to do.")
        return 0
    if args.dry_run:
        return 0

    for p in (CHECKPOINT, DISTILLED_LORA, SPATIAL_UPSAMPLER, GEMMA_ROOT):
        if not p.exists():
            print(f"ERROR: missing weight: {p}", file=sys.stderr)
            return 1

    print("Loading pipeline…")
    pipeline = build_pipeline()
    print("Pipeline ready.\n")

    errors = 0
    for i, scene in enumerate(work):
        label = f"[{i+1}/{len(work)}] {scene['key']}"
        print(label)
        out = OUT_DIR / f"{scene['key']}.mp4"
        try:
            generate(pipeline, scene=scene, out_path=out)
            print(f"  saved → {out}")
        except Exception:
            log.exception("Generation failed: %s", label)
            errors += 1
        print()

    unload(pipeline)
    print(f"Done. {len(work)-errors}/{len(work)} succeeded.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
