#!/usr/bin/env python3
"""
run_skye_like_api.py

Closest local equivalent to fal-ai/ltx-2.3/image-to-video/fast.
All scene parameters are hardcoded below — edit this file to change them.

Pipeline: TI2VidTwoStagesHQPipeline
  Stage 1 — full 22B model at 960×544, 15 Res2s steps, CFG guidance
  Stage 2 — 2× spatial upsampler → 1920×1088, 4-step distilled denoising

Usage
-----
    uv run python run_skye_like_api.py
    uv run python run_skye_like_api.py --durations 6 8 10
    uv run python run_skye_like_api.py --seed 42 --no-audio
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  EDIT THESE — scene inputs
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent

# Start frame image
DEFAULT_IMAGE = SCRIPT_DIR / "inputs/BM_v1/benchmark_v1/skye_helicopter_birthday_gili/start_frame.png"

# Prompt — edit freely
DEFAULT_PROMPT = (
    "High-quality 3D CGI children's cartoon style, vibrant colors, clean CGI, "
    "cinematic lighting, smooth character animation, preschool-friendly aesthetic, "
    "4k render, detailed textures, expressive facial expressions. "
    "Pink pup in pilot helmet and flight gear inside helicopter cockpit, hovering in place. "
    "Both paws gripping the helicopter controls at all times — paws never leave the controls, no waving. "
    "Rotors spinning overhead. Camera locked on cockpit — no camera movement.\n\n"
    "[DIALOGUE - WHAT SHE SAYS]\n"
    "Pink aviator pup (in a warm, cheerful voice, speaking directly to camera): "
    '"Hi Gili — I heard it\'s your birthday, so I came to wish you a happy birthday!" '
    "Expressive face, mouth moving clearly with the words. Both paws remain on the controls.\n\n"
    "[DIALOGUE - WHAT SHE SAYS]\n"
    "Pink aviator pup (warmly, sincerely): "
    '"Your mom, Efrat, asked me to tell you that you\'re a wonderful girl! '
    "You're an amazing person! A wonderful human — you're perfect just the way you are.\" "
    "Warm smile, expressive eyes, mouth moving with the speech. Paws steady on the controls.\n\n"
    "[DIALOGUE - WHAT SHE SAYS]\n"
    "Pink aviator pup (enthusiastic, upbeat): "
    '"Keep being a great big sister to your little sister Aya, and most importantly — '
    'stay happy and joyful! From all of us here at Adventure Bay." '
    "Big smile, excited expression, mouth moving animatedly. Both paws on the helicopter controls throughout."
)

# ══════════════════════════════════════════════════════════════════════════════
#  EDIT THESE — model weights paths
# ══════════════════════════════════════════════════════════════════════════════

MODELS_DIR        = Path("/home/efrattaig/models/LTX-2.3")
CHECKPOINT        = MODELS_DIR / "ltx-2.3-22b-dev.safetensors"
DISTILLED_LORA    = MODELS_DIR / "ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSAMPLER = MODELS_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT        = Path("/home/efrattaig/models/gemma-3-12b-it-qat-q4_0-unquantized")

# ══════════════════════════════════════════════════════════════════════════════
#  EDIT THESE — generation defaults (can also be overridden via CLI flags)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_DURATIONS  = [15.0]   # seconds — passed as --durations to override
DEFAULT_SEED       = 300                 # seed 300: no hand movement at 15s
DEFAULT_FPS        = 25.0
DEFAULT_WIDTH      = 768                 # stage-2 output; stage-1 uses half (384) — must be divisible by 64
DEFAULT_HEIGHT     = 1024                # stage-2 output; stage-1 uses half (512) — must be divisible by 64
DEFAULT_STEPS      = 15                  # Res2s second-order ≈ 30 standard Euler steps
DEFAULT_VIDEO_CFG  = 3.0
DEFAULT_AUDIO_CFG  = 7.0
DEFAULT_LORA_S1    = 0.25               # distilled LoRA strength in stage 1
DEFAULT_LORA_S2    = 0.50               # distilled LoRA strength in stage 2
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

# ══════════════════════════════════════════════════════════════════════════════
#  Nothing below here should need editing
# ══════════════════════════════════════════════════════════════════════════════

def frames_for_duration(seconds: float, fps: float) -> int:
    raw = round(seconds * fps)
    k = max(1, (raw - 1 + 7) // 8)
    return max(17, 8 * k + 1)


def build_pipeline(lora_path: Path | None = None, lora_strength: float = 1.0):
    import torch
    torch.backends.cudnn.enabled = False  # prevents BigVGAN cuBLASLt crash

    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline

    global _skye_loras
    if lora_path is not None:
        _skye_loras = (LoraPathStrengthAndSDOps(
            path=str(lora_path),
            strength=lora_strength,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        ),)
        print(f"  skye lora : {lora_path}  (strength={lora_strength})")
    else:
        _skye_loras = ()

    distilled_lora = [LoraPathStrengthAndSDOps(
        path=str(DISTILLED_LORA),
        strength=1.0,
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
    )]

    return TI2VidTwoStagesHQPipeline(
        checkpoint_path=str(CHECKPOINT),
        distilled_lora=distilled_lora,
        distilled_lora_strength_stage_1=DEFAULT_LORA_S1,
        distilled_lora_strength_stage_2=DEFAULT_LORA_S2,
        spatial_upsampler_path=str(SPATIAL_UPSAMPLER),
        gemma_root=str(GEMMA_ROOT),
        loras=_skye_loras,
    )


def generate_one(pipeline, *, prompt, negative_prompt, seed, num_frames,
                 width, height, fps, steps, video_guider_params, audio_guider_params,
                 image_path, output_path, generate_audio) -> None:
    import torch
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.types import Audio
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    if not generate_audio:
        class _SilentAudioDecoder:
            def __call__(self, latent: torch.Tensor) -> Audio:
                return Audio(waveform=torch.zeros(2, 0), sampling_rate=24000)
        pipeline.audio_decoder = _SilentAudioDecoder()

    video, audio = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=steps,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        images=[ImageConditioningInput(path=str(image_path), frame_idx=0, strength=1.0)],
        tiling_config=tiling_config,
    )

    with torch.inference_mode():
        encode_video(
            video=video,
            fps=int(fps),
            audio=audio if generate_audio else None,
            output_path=str(output_path),
            video_chunks_number=video_chunks_number,
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT

    negative_prompt = (
        DEFAULT_NEGATIVE_PROMPT
        + ", waving paw, raised paw, paw in the air, arm raised, hand gesture, "
        "waving hand, pointing, gesturing, paw movement, arm movement"
    )

    parser = argparse.ArgumentParser(
        description="run_skye_like_api — LTX-2.3 HQ local inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--durations", type=float, nargs="+", default=DEFAULT_DURATIONS,
                        help="Video length(s) in seconds.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed.")
    parser.add_argument("--no-audio", action="store_true",
                        help="Skip audio generation (faster, same video quality).")
    parser.add_argument("--video-cfg", type=float, default=DEFAULT_VIDEO_CFG,
                        help="Video guidance scale (CFG).")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                        help="Start frame image path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lora", type=Path, default=None,
                        help="Path to Skye LoRA .safetensors checkpoint to evaluate.")
    parser.add_argument("--lora-strength", type=float, default=1.0,
                        help="LoRA blend strength (0=base model, 1=full LoRA).")
    args = parser.parse_args()

    image_path = args.image
    prompt = DEFAULT_PROMPT

    # Validate
    if not image_path.exists():
        print(f"ERROR: start frame not found: {image_path}", file=sys.stderr)
        return 1
    missing = [p for p in (CHECKPOINT, DISTILLED_LORA, SPATIAL_UPSAMPLER, GEMMA_ROOT)
               if not p.exists()]
    if missing:
        for m in missing:
            print(f"ERROR: weight not found: {m}", file=sys.stderr)
        return 1

    video_guider_params = MultiModalGuiderParams(
        cfg_scale=args.video_cfg,
        stg_scale=0.0,
        rescale_scale=0.45,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[],
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=DEFAULT_AUDIO_CFG,
        stg_scale=0.0,
        rescale_scale=1.0,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_audio = not args.no_audio

    print("\nrun_skye_like_api — TI2VidTwoStagesHQPipeline")
    print(f"  image     : {image_path}")
    print(f"  stage-1   : {DEFAULT_WIDTH // 2} × {DEFAULT_HEIGHT // 2}  ({DEFAULT_STEPS} Res2s steps)")
    print(f"  stage-2   : {DEFAULT_WIDTH} × {DEFAULT_HEIGHT}  (2× upsample + distilled LoRA)")
    print(f"  fps       : {DEFAULT_FPS}  |  durations: {args.durations} s")
    print(f"  audio     : {'yes' if generate_audio else 'no'}")
    print()

    print("Loading pipeline…")
    pipeline = build_pipeline(lora_path=args.lora, lora_strength=args.lora_strength)

    errors = 0
    for dur in args.durations:
        seed = args.seed
        nf = frames_for_duration(dur, DEFAULT_FPS)
        actual_dur = round(nf / DEFAULT_FPS, 2)
        img_tag = image_path.stem.replace(" ", "_")[:40]
        lora_tag = f"_lora-{args.lora.stem[:20]}" if args.lora else "_base"
        out_name = (
            f"run_skye_like_api__{img_tag}__{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}_"
            f"{DEFAULT_STEPS}steps_cfg{args.video_cfg}_{actual_dur}s_seed{seed}{lora_tag}.mp4"
        )
        output_path = args.output_dir / out_name

        print(f"Generating {dur}s → {nf} frames ({actual_dur}s) | seed={seed}")
        try:
            generate_one(
                pipeline,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                num_frames=nf,
                width=DEFAULT_WIDTH,
                height=DEFAULT_HEIGHT,
                fps=DEFAULT_FPS,
                steps=DEFAULT_STEPS,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                image_path=image_path,
                output_path=output_path,
                generate_audio=generate_audio,
            )
            print(f"  saved: {output_path}")
        except Exception:
            logging.exception("Generation failed for %ss", dur)
            errors += 1
        print()

    total = len(args.durations)
    print(f"Done. {total - errors}/{total} generation(s) succeeded.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
