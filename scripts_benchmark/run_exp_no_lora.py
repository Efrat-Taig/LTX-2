#!/usr/bin/env python3
"""
run_exp_no_lora.py

Three-track experiment addressing baseline issues without LoRA training.

Track 1 — Prompt Surgery  (all 5 Chase + 5 standard Skye scenes)
  · Reformatted dialogue: no [LABEL] blocks, single line, inline motion description
  · Negative prompt: anti-subtitle + anti-scene-transition additions
  · Positive prompt: explicit scene-stability instruction

Track 2 — STG Sweep  (Chase normal + halloween only)
  · Varies stg_scale: 0.5 / 1.0 / 2.0  with stg_blocks=[29]
  · Uses Track-1 prompts

Track 3 — Audio CFG Sweep  (Skye bey + crsms + snow only)
  · Varies audio_cfg: 12.0 / 18.0
  · Uses Track-1 prompts

Usage
-----
    uv run python run_exp_no_lora.py                   # all three tracks
    uv run python run_exp_no_lora.py --tracks 1        # Track 1 only
    uv run python run_exp_no_lora.py --tracks 2 3      # Tracks 2 and 3
    uv run python run_exp_no_lora.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR        = Path(__file__).resolve().parent
MODELS_DIR        = Path.home() / "models" / "LTX-2.3"
CHECKPOINT        = MODELS_DIR / "ltx-2.3-22b-dev.safetensors"
DISTILLED_LORA    = MODELS_DIR / "ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSAMPLER = MODELS_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT        = Path.home() / "models" / "gemma-3-12b-it-qat-q4_0-unquantized"

CHASE_IMAGES = SCRIPT_DIR / "inputs" / "chase_bm"
SKYE_IMAGES  = SCRIPT_DIR / "inputs" / "sky_bm"
HELI_IMAGES  = SCRIPT_DIR / "inputs" / "BM_v1" / "benchmark_v1" / "skye_helicopter_birthday_gili"

OUT_ROOT = SCRIPT_DIR / "lora_results" / "exp_no_lora"

# ── Fixed generation params (same as existing benchmarks) ─────────────────────

DURATION_S = 10.0
SEED       = 42
FPS        = 25.0
STEPS      = 15
LORA_S1    = 0.25
LORA_S2    = 0.50

CHASE_W, CHASE_H = 832, 960
SKYE_W,  SKYE_H  = 768, 1024

# ── Negative prompts ───────────────────────────────────────────────────────────

_ANTI_SUBTITLE = (
    ", subtitles, captions, on-screen text, text overlay, burned-in text, watermark"
    ", text, letters, alphabet, words"
)
_ANTI_TRANSITION = (
    ", scene change, environment change, transition, cut to new location, "
    "new background, different setting, camera cut"
)
_EXTRA_NEG = _ANTI_SUBTITLE + _ANTI_TRANSITION

CHASE_NEG_BASE = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, "
    "distorted proportions, unnatural skin tones, deformed facial features, "
    "asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, "
    "incorrect depth of field, background too sharp, background clutter, "
    "distracting reflections, harsh shadows, inconsistent lighting direction, "
    "color banding, cartoonish rendering, 3D CGI look, unrealistic materials, "
    "uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, "
    "wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, "
    "robotic voice, echo, background noise, off-sync audio, incorrect dialogue, "
    "added dialogue, repetitive speech, jittery movement, awkward pauses, "
    "incorrect timing, unnatural transitions, inconsistent framing, tilted camera, "
    "flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, "
    "or AI artifacts"
    ", waving paw, raised paw, paw in the air, arm raised, hand gesture, "
    "waving hand, pointing, gesturing, paw movement, arm movement"
)
CHASE_NEG = CHASE_NEG_BASE + _EXTRA_NEG

SKYE_NEG_BASE = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, "
    "distorted proportions, unnatural skin tones, deformed facial features, "
    "asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, "
    "incorrect depth of field, background too sharp, background clutter, "
    "distracting reflections, harsh shadows, inconsistent lighting direction, "
    "color banding, cartoonish rendering, 3D CGI look, unrealistic materials, "
    "uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, "
    "wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, "
    "robotic voice, echo, background noise, off-sync audio, incorrect dialogue, "
    "added dialogue, repetitive speech, jittery movement, awkward pauses, "
    "incorrect timing, unnatural transitions, inconsistent framing, tilted camera, "
    "flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, "
    "or AI artifacts"
)
SKYE_NEG = SKYE_NEG_BASE + _EXTRA_NEG

# Helicopter scene has its own negative prompt (from its config.json)
HELI_NEG_BASE = (
    "low quality, bad anatomy, deformed bodies, distorted faces, scary expressions, "
    "extra limbs, impossible positions, glitch, artifacts, darkness, grim atmosphere, "
    "static rotor blades, frozen helicopter rotors, out of focus face, characters merged, "
    "melting, bad composition, overly stylized, outdated cartoon style, wrong vehicle, "
    "cockpit closed incorrectly"
)
HELI_NEG = HELI_NEG_BASE + _EXTRA_NEG

# ── Stability suffix added to every positive prompt ───────────────────────────

_STABLE = (
    " Continuous single locked shot. Same environment throughout. No scene changes."
)

# ── Dialogue — single line, no label format ────────────────────────────────────

_CHASE_SPEECH = (
    " Speaking directly to camera with a confident, warm voice, mouth moving naturally "
    "with the words: \"Chase is on the case! Whatever the mission, I'm ready!\" "
    "Expressive animated face, ears perked up with energy."
)

_SKYE_SPEECH = (
    " Speaking warmly and directly to camera, mouth moving clearly with the words: "
    "\"Hi Gili — I heard it's your birthday, so I came to wish you a happy birthday!\" "
    "Warm smile, expressive eyes, animated face."
)

_HELI_SPEECH = (
    " She speaks in an enthusiastic, high-pitched voice over muffled engine noise, "
    "mouth moving clearly with the words: "
    "\"Hi Gili! Look where I am! I flew all the way over the coast to tell you — "
    "Congratulations! I'm so excited for you! Happy Birthday, Gili!\" "
    "Warm smile, expressive eyes, waving her paw with joyful energy."
)

# ── Scene definitions ──────────────────────────────────────────────────────────

CHASE_SCENES = [
    {
        "name": "normal",
        "char": "chase",
        "image": CHASE_IMAGES / "chase_normal.png",
        "w": CHASE_W, "h": CHASE_H,
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "CHASE, a brown German Shepherd puppy in his blue police uniform and cap, stands "
            "confidently at the PAW Patrol Lookout on a bright sunny day. He faces the camera "
            "with alert, focused eyes, ears perked up, tail wagging steadily. "
            "His blue badge gleams in the sunlight. Proud, professional energy."
            + _STABLE + _CHASE_SPEECH
        ),
    },
    {
        "name": "halloween",
        "char": "chase",
        "image": CHASE_IMAGES / "chase_halloween.png",
        "w": CHASE_W, "h": CHASE_H,
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "CHASE, the German Shepherd police pup from PAW Patrol, wearing a fun Halloween "
            "costume over his police gear — orange pumpkin hat, sitting on a spooky porch "
            "surrounded by glowing jack-o-lanterns. He gives a big excited grin at the camera, "
            "ears perking up with delight. Moonlit Halloween night, warm orange pumpkin glow."
            + _STABLE + _CHASE_SPEECH
        ),
    },
    {
        "name": "crms",
        "char": "chase",
        "image": CHASE_IMAGES / "chase_crms.png",
        "w": CHASE_W, "h": CHASE_H,
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "CHASE, the German Shepherd police pup from PAW Patrol, wearing a festive Christmas "
            "bandana and Santa hat, sitting in a cozy snowy village surrounded by decorated "
            "Christmas trees and colorful gift boxes. He smiles with festive excitement, "
            "ears perking up with joy. Northern lights glowing softly above."
            + _STABLE + _CHASE_SPEECH
        ),
    },
    {
        "name": "snow",
        "char": "chase",
        "image": CHASE_IMAGES / "chase_snow.png",
        "w": CHASE_W, "h": CHASE_H,
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "CHASE, the German Shepherd police pup from PAW Patrol, standing in a snowy "
            "Adventure Bay, snowflakes drifting gently around him. He smiles warmly at the camera. "
            "His blue police uniform dusted with snow. Peaceful winter night, crescent moon above."
            + _STABLE + _CHASE_SPEECH
        ),
    },
    {
        "name": "party",
        "char": "chase",
        "image": CHASE_IMAGES / "party_chase.png",
        "w": CHASE_W, "h": CHASE_H,
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "CHASE, the German Shepherd police pup from PAW Patrol, at a fun birthday party "
            "surrounded by colorful balloons, streamers and a cake. He looks directly at the "
            "camera with wide sparkling eyes and a big excited smile, tail wagging with energy. "
            "Bright celebratory lighting, lively party energy."
            + _STABLE + _CHASE_SPEECH
        ),
    },
]

SKYE_SCENES = [
    {
        "name": "bey",
        "char": "skye",
        "image": SKYE_IMAGES / "bey.png",
        "w": SKYE_W, "h": SKYE_H,
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "Pink pup in pink pilot helmet and flight gear, standing confidently in front of the "
            "PAW Patrol lookout tower on a bright sunny day. She turns her head toward the camera "
            "with a warm, proud smile, tail wagging gently. "
            "Mountains and ocean visible in the background."
            + _STABLE + _SKYE_SPEECH
        ),
    },
    {
        "name": "crsms",
        "char": "skye",
        "image": SKYE_IMAGES / "crsms.png",
        "w": SKYE_W, "h": SKYE_H,
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "Pink pup in a Santa hat and red Christmas sweater, sitting in a snowy Christmas "
            "village surrounded by decorated trees and colorful gift boxes. She smiles with "
            "festive excitement, ears perking up with joy. "
            "Northern lights glowing in the night sky behind her."
            + _STABLE + _SKYE_SPEECH
        ),
    },
    {
        "name": "holoween",
        "char": "skye",
        "image": SKYE_IMAGES / "holoween.png",
        "w": SKYE_W, "h": SKYE_H,
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "Pink pup wearing a purple witch hat and Halloween outfit, sitting on a spooky porch "
            "surrounded by glowing carved jack-o-lanterns. She gives a mischievous grin at the "
            "camera, wiggling her ears. Moonlit Halloween night, warm orange pumpkin glow."
            + _STABLE + _SKYE_SPEECH
        ),
    },
    {
        "name": "party",
        "char": "skye",
        "image": SKYE_IMAGES / "party.png",
        "w": SKYE_W, "h": SKYE_H,
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "Pink pup surrounded by glowing jack-o-lanterns at a festive Halloween party. "
            "She looks directly at the camera with wide sparkling eyes and a big excited smile, "
            "tail wagging with energy. Warm orange candlelit glow from the pumpkins."
            + _STABLE + _SKYE_SPEECH
        ),
    },
    {
        "name": "snow",
        "char": "skye",
        "image": SKYE_IMAGES / "snow.png",
        "w": SKYE_W, "h": SKYE_H,
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "Pink pup in pink pilot gear standing beside the snow-covered PAW Patrol lookout "
            "tower decorated with Christmas lights. Snowflakes drifting gently around her. "
            "She smiles warmly at the camera. Peaceful winter night, crescent moon above."
            + _STABLE + _SKYE_SPEECH
        ),
    },
    {
        "name": "helicopter",
        "char": "skye",
        "image": HELI_IMAGES / "start_frame.png",
        # Config dims 698x1280 rounded up to nearest multiple of 32 for pipeline safety
        "w": 704, "h": 1280,
        "fps": 24.0,
        "image_strength": 0.9,
        "neg": HELI_NEG,
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, clean CGI, "
            "cinematic lighting, smooth character animation, preschool-friendly aesthetic, "
            "4k render, detailed textures, expressive facial expressions. "
            "A small pink pup in pink flight gear sits in the cockpit of a pink cartoon "
            "helicopter hovering at the viewer's eye level. The main rotor and tail rotor "
            "spin at high speed with realistic motion blur. Below, a sunny coastline and "
            "harbor town spread out with a tall tower on a green hill; clouds drift gently."
            + _STABLE + _HELI_SPEECH
        ),
    },
]

# Scenes used in Track 2 STG sweep (visual quality focus) — all benchmark scenes
CHASE_STG_SCENES = list(CHASE_SCENES)
SKYE_STG_SCENES  = list(SKYE_SCENES)

# Scenes used in Track 3 audio CFG sweep (Skye script fidelity focus)
SKYE_CFG_SCENES = [s for s in SKYE_SCENES if s["name"] in ("bey", "crsms", "snow", "helicopter")]

# ── Experiment variants ────────────────────────────────────────────────────────

TRACK1_VARIANT = {
    "label": "t1_prompt",
    "video_cfg": 3.0, "audio_cfg": 7.0,
    "stg_scale": 0.0, "stg_blocks": [],
}

TRACK2_VARIANTS = [
    {"label": "t2_stg0.5", "video_cfg": 3.0, "audio_cfg": 7.0, "stg_scale": 0.5, "stg_blocks": [29]},
    {"label": "t2_stg1.0", "video_cfg": 3.0, "audio_cfg": 7.0, "stg_scale": 1.0, "stg_blocks": [29]},
    {"label": "t2_stg2.0", "video_cfg": 3.0, "audio_cfg": 7.0, "stg_scale": 2.0, "stg_blocks": [29]},
]

TRACK3_VARIANTS = [
    {"label": "t3_acfg12", "video_cfg": 3.0, "audio_cfg": 12.0, "stg_scale": 0.0, "stg_blocks": []},
    {"label": "t3_acfg18", "video_cfg": 3.0, "audio_cfg": 18.0, "stg_scale": 0.0, "stg_blocks": []},
]


# ── Pipeline ───────────────────────────────────────────────────────────────────

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


def generate(pipeline, *, scene: dict, variant: dict, output_path: Path) -> None:
    import torch
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video

    neg = scene.get("neg")
    if neg is None:
        neg = CHASE_NEG if scene["char"] == "chase" else SKYE_NEG

    fps            = scene.get("fps", FPS)
    image_strength = scene.get("image_strength", 1.0)
    num_frames     = frames_for_duration(DURATION_S, fps)
    tiling_config  = TilingConfig.default()
    video_chunks   = get_video_chunks_number(num_frames, tiling_config)

    video_guider = MultiModalGuiderParams(
        cfg_scale=variant["video_cfg"],
        stg_scale=variant["stg_scale"],
        rescale_scale=0.45,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=variant["stg_blocks"],
    )
    audio_guider = MultiModalGuiderParams(
        cfg_scale=variant["audio_cfg"],
        stg_scale=0.0,
        rescale_scale=1.0,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[],
    )

    video, audio = pipeline(
        prompt=scene["prompt"],
        negative_prompt=neg,
        seed=SEED,
        height=scene["h"],
        width=scene["w"],
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=STEPS,
        video_guider_params=video_guider,
        audio_guider_params=audio_guider,
        images=[ImageConditioningInput(path=str(scene["image"]), frame_idx=0, strength=image_strength)],
        tiling_config=tiling_config,
        enhance_prompt=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        encode_video(
            video=video, fps=int(fps), audio=audio,
            output_path=str(output_path),
            video_chunks_number=video_chunks,
        )


def out_path(variant_label: str, scene_name: str) -> Path:
    return OUT_ROOT / variant_label / f"{scene_name}.mp4"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tracks", type=int, nargs="+", default=[1, 2, 3],
                        choices=[1, 2, 3], help="Which tracks to run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Build work list: (scene, variant, output_path)
    work: list[tuple[dict, dict, Path]] = []

    if 1 in args.tracks:
        for scene in CHASE_SCENES + SKYE_SCENES:
            p = out_path(TRACK1_VARIANT["label"], f"{scene['char']}_{scene['name']}")
            if not p.exists():
                work.append((scene, TRACK1_VARIANT, p))

    if 2 in args.tracks:
        for variant in TRACK2_VARIANTS:
            for scene in CHASE_STG_SCENES + SKYE_STG_SCENES:
                p = out_path(variant["label"], f"{scene['char']}_{scene['name']}")
                if not p.exists():
                    work.append((scene, variant, p))

    if 3 in args.tracks:
        for variant in TRACK3_VARIANTS:
            for scene in SKYE_CFG_SCENES:
                p = out_path(variant["label"], f"skye_{scene['name']}")
                if not p.exists():
                    work.append((scene, variant, p))

    print(f"\nrun_exp_no_lora — tracks {args.tracks}")
    print(f"  {len(work)} video(s) to generate  (already done: skipped)")
    for scene, variant, p in work:
        print(f"  [{variant['label']}]  {scene['char']}_{scene['name']}  "
              f"stg={variant['stg_scale']}  acfg={variant['audio_cfg']}")
    print()

    if not work:
        print("Nothing to do.")
        return 0

    if args.dry_run:
        return 0

    # Validate weights
    for p in (CHECKPOINT, DISTILLED_LORA, SPATIAL_UPSAMPLER, GEMMA_ROOT):
        if not p.exists():
            print(f"ERROR: missing weight: {p}", file=sys.stderr)
            return 1

    print("Loading pipeline…")
    pipeline = build_pipeline()
    print("Pipeline ready.\n")

    errors = 0
    for i, (scene, variant, p) in enumerate(work):
        label = f"[{i+1}/{len(work)}] [{variant['label']}] {scene['char']}_{scene['name']}"
        print(label)
        try:
            generate(pipeline, scene=scene, variant=variant, output_path=p)
            print(f"  saved → {p}")
        except Exception:
            log.exception("Generation failed: %s", label)
            errors += 1
        print()

    unload(pipeline)
    print(f"Done. {len(work)-errors}/{len(work)} succeeded.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
