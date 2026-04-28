#!/usr/bin/env python3
"""
run_skye_v0.py

Local LTX-2.3 inference using TI2VidTwoStagesHQPipeline — the same two-stage
Res2s pipeline that the Fal API (fal-ai/ltx-2.3/image-to-video/fast) almost
certainly runs under the hood.

Pipeline flow (matches LTX_2_3_HQ_PARAMS):
  Stage 1 — Full model at half resolution (960×544), 15 steps, Res2s sampler,
             CFG=3.0, STG disabled.
  Stage 2 — Spatial upsampler 2× → 1920×1088, distilled LoRA, 3-step fixed
             sigma schedule, SimpleDenoiser (no CFG / STG).

Required weights (all in /home/efrattaig/models/LTX-2.3/):
  ltx-2.3-22b-dev.safetensors               (full model)
  ltx-2.3-22b-distilled-lora-384.safetensors (stage-2 LoRA adapter)
  ltx-2.3-spatial-upscaler-x2-1.0.safetensors (2× upsampler)

Usage (run from repo root):
    uv run python efrat_run/run_skye_v0.py
    uv run python efrat_run/run_skye_v0.py --durations 6 8
    uv run python efrat_run/run_skye_v0.py --prompt-type 2 --seed 42
    uv run python efrat_run/run_skye_v0.py --no-audio
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LTX_ROOT = SCRIPT_DIR.parent

MODELS_DIR = Path("/home/efrattaig/models/LTX-2.3")
CHECKPOINT = MODELS_DIR / "ltx-2.3-22b-dev.safetensors"
DISTILLED_LORA = MODELS_DIR / "ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSAMPLER = MODELS_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT = Path("/home/efrattaig/models/gemma-3-12b-it-qat-q4_0-unquantized")

DEFAULT_IMAGE = LTX_ROOT / "input" / "scenes" / "skye_helicopter" / "start_frame.png"
DEFAULT_OUTPUT_DIR = LTX_ROOT / "output"

# ── HQ pipeline defaults (mirrors LTX_2_3_HQ_PARAMS + Fal Fast API) ──────────
HQ_WIDTH = 1920  # stage-2 output width  (stage-1: 960)
HQ_HEIGHT = 1088 # stage-2 output height (stage-1: 544) — nearest multiple of 64 to 1080p
HQ_FPS = 25.0
HQ_STEPS = 15            # Res2s second-order sampler needs fewer steps
HQ_VIDEO_CFG = 3.0       # CFG guidance scale for video (API estimated default)
HQ_AUDIO_CFG = 7.0       # CFG guidance scale for audio
HQ_LORA_STR_S1 = 0.25   # distilled LoRA strength in stage 1
HQ_LORA_STR_S2 = 0.5    # distilled LoRA strength in stage 2
HQ_DURATIONS = (6, 8, 10)  # seconds; API supports 6–20 s, default 6

# ── prompt ─────────────────────────────────────────────────────────────────────

# DEFAULT_PROMPT = (
#     "High-quality 3D animation in the style of Paw Patrol, vibrant colors, clean CGI, "
#     "cinematic lighting, bright and cheerful atmosphere, smooth character animation, "
#     "Nick Jr aesthetic, 4k render, detailed textures, expressive facial expressions.\n\n"
#     "Skye from Paw Patrol, the cute female cockapoo rescue puppy pilot, wearing her pink "
#     "pilot goggles and pink vest, sitting in the cockpit of her small pink helicopter "
#     "hovering in a bright blue sky with soft fluffy clouds.\n\n"
#     "Skye looks directly into the camera and talks to the viewer like a cheerful kids TV "
#     "host. She speaks clearly and expressively, with natural mouth movements that match "
#     "every word. She is warm, happy, and sincere.\n\n"
#     "She says:\n"
#     '"Hi Gili — I heard it\'s your birthday, so I came to wish you a happy birthday! '
#     "Your mom Efrat asked me to tell you that you're a wonderful girl. "
#     "You're an amazing person — a wonderful human, perfect just the way you are. "
#     "Keep being a great big sister to your little sister Aya, and most importantly — "
#     'stay happy and joyful! From all of us here at Adventure Bay."\n\n'
#     "Her mouth movements precisely match the speech. She smiles warmly and talks with "
#     "gentle expression throughout.\n\n"
#     "Camera: locked, stable medium close-up inside the helicopter cockpit, Skye centered "
#     "and facing the camera. No camera movement, no cuts, no angle changes.\n"
#     "Motion: only the rotors spin; the helicopter holds a steady hover with no forward "
#     "flight, no banking, no drift. Subtle head movement and blinking while speaking.\n"
#     "Mood: warm, heartfelt, joyful children's show.\n"
#     "Environment: bright daylight sky, soft clouds."
# )

DEFAULT_PROMPT = (
      "High-quality 3D CGI cartoon, bright colors, preschool-friendly. Fixed camera — same framing as the reference, no camera movement. The helicopter stays in a stable hover: no forward or backward flight, no banking away, no sideways drift, no climb or dive, no change in distance — same apparent size and position in frame as the reference. Only the main and tail rotors spin with blur; the aircraft fuselage does not translate, drift, or rotate as if flying. Subtle face and mouth for speech only. A matching end frame is provided (same as the start): the final frame must match that image — end in the same hover pose and framing, not farther away. Clear steady delivery. [SHOT]\nStatic shot; helicopter locked in hover (no flight motion). Identical framing to the reference. Cockpit, pink pup in flight gear, rotors spinning.\n\n[SHE SAYS]\n\"Hi Gili — I heard it's your birthday, so I came to wish you a happy birthday!\" Same static framing; helicopter still hovering — no flight motion.\n\n[SHE SAYS]\n\"Your mom, Efrat, asked me to tell you that you're a wonderful girl! You're an amazing person! A wonderful human — you're perfect just the way you are.\" [SHOT]\nSame static framing; helicopter still hovering — no flight motion.\n\n[SHE SAYS]\n\"Keep being a great big sister to your little sister Aya, and most importantly — stay happy and joyful! From all of us here at Adventure Bay.\""
)
# ── helpers ────────────────────────────────────────────────────────────────────

def frames_for_duration(seconds: float, fps: float) -> int:
    """Return smallest num_frames satisfying k*8+1 that covers `seconds` at `fps`."""
    raw = round(seconds * fps)
    k = max(1, (raw - 1 + 7) // 8)
    return max(17, 8 * k + 1)


def build_pipeline(*, lora_str_s1: float, lora_str_s2: float, generate_audio: bool):
    """Construct TI2VidTwoStagesHQPipeline with the locally downloaded weights."""
    import torch
    # cuDNN triggers bitsandbytes' cuBLASLt hook on Conv ops (upsampler, vocoder)
    # which calls abort(). Disabling cuDNN makes those convolutions fall back to
    # the plain cuBLAS path, which works. The transformer uses flash-attention and
    # is unaffected by this flag.
    torch.backends.cudnn.enabled = False

    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline

    distilled_lora = [
        LoraPathStrengthAndSDOps(
            path=str(DISTILLED_LORA),
            strength=1.0,  # per-stage strengths are set separately below
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        )
    ]

    return TI2VidTwoStagesHQPipeline(
        checkpoint_path=str(CHECKPOINT),
        distilled_lora=distilled_lora,
        distilled_lora_strength_stage_1=lora_str_s1,
        distilled_lora_strength_stage_2=lora_str_s2,
        spatial_upsampler_path=str(SPATIAL_UPSAMPLER),
        gemma_root=str(GEMMA_ROOT),
        loras=(),
    )


def generate_one(
    pipeline,
    *,
    prompt: str,
    seed: int,
    num_frames: int,
    width: int,
    height: int,
    fps: float,
    steps: int,
    video_cfg: float,
    audio_cfg: float,
    image_path: Path,
    output_path: Path,
    generate_audio: bool,
) -> None:
    import torch
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.types import Audio
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    # The HQ pipeline always calls audio_decoder at the end (no skip option).
    # The vocoder (BigVGAN v2) crashes with cublasLtCreate when audio isn't needed.
    # Monkey-patch with a silent decoder when audio output isn't requested.
    if not generate_audio:
        class _SilentAudioDecoder:
            def __call__(self, latent: torch.Tensor) -> Audio:
                return Audio(waveform=torch.zeros(2, 0), sampling_rate=24000)
        pipeline.audio_decoder = _SilentAudioDecoder()

    video, audio = pipeline(
        prompt=prompt,
        negative_prompt="",   # API Fast schema has no negative_prompt
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=video_cfg,
            stg_scale=0.0,        # STG disabled — matches HQ params and API behaviour
            rescale_scale=0.45,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[],
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=audio_cfg,
            stg_scale=0.0,
            rescale_scale=1.0,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[],
        ),
        images=[ImageConditioningInput(
            path=str(image_path),
            frame_idx=0,
            strength=1.0,
        )],
        tiling_config=tiling_config,
    )

    # The HQ pipeline's __call__ is decorated with @torch.inference_mode().
    # VideoDecoder returns a lazy iterator that is consumed here, outside that
    # context. Wrapping encode_video in inference_mode avoids the "inference
    # tensors cannot be saved for backward" error.
    import torch
    with torch.inference_mode():
        encode_video(
            video=video,
            fps=int(fps),
            audio=audio if generate_audio else None,
            output_path=str(output_path),
            video_chunks_number=video_chunks_number,
        )


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Local LTX-2.3 HQ two-stage inference that mirrors "
            "fal-ai/ltx-2.3/image-to-video/fast."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--distilled-lora", type=Path, default=DISTILLED_LORA)
    parser.add_argument("--spatial-upsampler", type=Path, default=SPATIAL_UPSAMPLER)
    parser.add_argument("--gemma-root", type=Path, default=GEMMA_ROOT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, dest="image_path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-base", type=str, default="skye_v0")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Override the default prompt. Omit to use DEFAULT_PROMPT.")
    parser.add_argument("--durations", type=float, nargs="+", default=list(HQ_DURATIONS),
                        help="Duration(s) in seconds. API supports 6–20 s.")
    parser.add_argument("--fps", type=float, default=HQ_FPS)
    parser.add_argument("--width", type=int, default=HQ_WIDTH,
                        help="Stage-2 output width (stage-1 uses half).")
    parser.add_argument("--height", type=int, default=HQ_HEIGHT,
                        help="Stage-2 output height (stage-1 uses half). "
                             "1088 is nearest 32-divisible value to 1080.")
    parser.add_argument("--steps", type=int, default=HQ_STEPS,
                        help="Stage-1 diffusion steps (Res2s, so 15 ≈ 30 Euler steps).")
    parser.add_argument("--video-cfg", type=float, default=HQ_VIDEO_CFG,
                        help="CFG guidance scale for video in stage 1.")
    parser.add_argument("--audio-cfg", type=float, default=HQ_AUDIO_CFG,
                        help="CFG guidance scale for audio in stage 1.")
    parser.add_argument("--lora-str-s1", type=float, default=HQ_LORA_STR_S1,
                        help="Distilled LoRA strength for stage 1.")
    parser.add_argument("--lora-str-s2", type=float, default=HQ_LORA_STR_S2,
                        help="Distilled LoRA strength for stage 2.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fix seed. Omit for random (mirrors API behaviour).")
    parser.add_argument("--no-audio", action="store_true",
                        help="Skip audio generation (audio is on by default, like the API).")
    args = parser.parse_args()
    args.with_audio = not args.no_audio

    # Validate
    missing = [p for p in (args.checkpoint, args.distilled_lora, args.spatial_upsampler,
                            args.gemma_root, args.image_path) if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: not found: {p}", file=sys.stderr)
        return 1

    prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPT
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nrun_skye_v0 — TI2VidTwoStagesHQPipeline (API-matching)")
    print(f"  checkpoint      : {args.checkpoint.name}")
    print(f"  distilled_lora  : {args.distilled_lora.name}")
    print(f"  spatial_upsamp  : {args.spatial_upsampler.name}")
    print(f"  image           : {args.image_path}")
    print(f"  stage-1 res     : {args.width // 2}x{args.height // 2}  (half, Res2s, {args.steps} steps)")
    print(f"  stage-2 res     : {args.width}x{args.height}  (2x upsample + distilled LoRA)")
    print(f"  fps             : {args.fps}")
    print(f"  video_cfg       : {args.video_cfg}  (STG disabled)")
    print(f"  lora strengths  : s1={args.lora_str_s1}  s2={args.lora_str_s2}")
    print(f"  durations       : {args.durations} s")
    print(f"  audio           : {'yes (default)' if args.with_audio else 'no (--no-audio)'}")
    print()

    print("Loading pipeline (this takes ~1–2 min the first time)...")
    pipeline = build_pipeline(
        lora_str_s1=args.lora_str_s1,
        lora_str_s2=args.lora_str_s2,
        generate_audio=args.with_audio,
    )

    errors = 0
    for dur in args.durations:
        seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
        nf = frames_for_duration(dur, args.fps)
        actual_dur = round(nf / args.fps, 2)
        out_name = (
            f"{args.output_base}_{actual_dur}s_"
            f"{args.width}x{args.height}_{args.steps}steps_seed{seed}.mp4"
        )
        output_path = args.output_dir / out_name

        print(f"Generating {dur}s → {nf} frames ({actual_dur}s) | seed={seed}")
        try:
            generate_one(
                pipeline,
                prompt=prompt,
                seed=seed,
                num_frames=nf,
                width=args.width,
                height=args.height,
                fps=args.fps,
                steps=args.steps,
                video_cfg=args.video_cfg,
                audio_cfg=args.audio_cfg,
                image_path=args.image_path,
                output_path=output_path,
                generate_audio=args.with_audio,
            )
            print(f"  -> saved: {output_path}")
        except Exception:
            logging.exception("Generation failed for %ss", dur)
            errors += 1
        print()

    total = len(args.durations)
    print(f"Done. {total - errors}/{total} generation(s) succeeded.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
