#!/usr/bin/env python3
"""
run_skye_benchmark.py

Auto-discovers every LoRA checkpoint across all Skye experiment output dirs
and runs all 5 benchmark scenes for each new checkpoint.

Idempotent — safe to re-run at any time.  Already-generated outputs are
skipped so this can be called after every training checkpoint.

Usage
-----
    uv run python run_skye_benchmark.py               # scan all default exp dirs
    uv run python run_skye_benchmark.py --dry-run     # print plan, generate nothing
    uv run python run_skye_benchmark.py --exp-dirs outputs/skye_exp1_baseline
    uv run python run_skye_benchmark.py --scenes bey snow   # only these scenes
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  EDIT THESE — paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent

# Where the 5 benchmark start-frame images live (scp'd once to the server)
BM_IMAGES_DIR = SCRIPT_DIR / "inputs" / "sky_bm"

# Model weights — same as run_skye_like_api.py
MODELS_DIR        = Path("/home/efrattaig/models/LTX-2.3")
CHECKPOINT        = MODELS_DIR / "ltx-2.3-22b-dev.safetensors"
DISTILLED_LORA    = MODELS_DIR / "ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSAMPLER = MODELS_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT        = Path("/home/efrattaig/models/gemma-3-12b-it-qat-q4_0-unquantized")

# Base dir that contains experiment output subdirs
DEFAULT_OUTPUTS_BASE = SCRIPT_DIR / "outputs"

# ══════════════════════════════════════════════════════════════════════════════
#  EDIT THESE — benchmark generation params (kept fixed for fair comparison)
# ══════════════════════════════════════════════════════════════════════════════

BM_DURATION_S  = 5.0    # short enough to be fast, long enough to show character
BM_SEED        = 42
BM_FPS         = 25.0
BM_WIDTH       = 768    # stage-2; stage-1 uses half
BM_HEIGHT      = 1024
BM_STEPS       = 15
BM_VIDEO_CFG   = 3.0
BM_AUDIO_CFG   = 7.0
BM_LORA_S1     = 0.25
BM_LORA_S2     = 0.50
BM_ENHANCE_PROMPT = False  # matches run_skye_like_api.py which gives best results without it

# Where the 5 overfit test frames live (extracted from training clips)
OVERFIT_IMAGES_DIR = SCRIPT_DIR / "inputs" / "overfit_bm"

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED DIALOGUE — appended to every scene prompt
# ══════════════════════════════════════════════════════════════════════════════

_DIALOGUE = (
    "\n\n[DIALOGUE - WHAT SHE SAYS]\n"
    "Pink pup (in a warm, cheerful voice, speaking directly to camera): "
    '"Hi Gili — I heard it\'s your birthday, so I came to wish you a happy birthday!" '
    "Expressive face, mouth moving clearly with the words.\n\n"
    "[DIALOGUE - WHAT SHE SAYS]\n"
    "Pink pup (warmly, sincerely): "
    '"Your mom, Efrat, asked me to tell you that you\'re a wonderful girl! '
    "You're an amazing person! A wonderful human — you're perfect just the way you are.\" "
    "Warm smile, expressive eyes, mouth moving with the speech.\n\n"
    "[DIALOGUE - WHAT SHE SAYS]\n"
    "Pink pup (enthusiastic, upbeat): "
    '"Keep being a great big sister to your little sister Aya, and most importantly — '
    'stay happy and joyful! From all of us here at Adventure Bay." '
    "Big smile, excited expression, mouth moving animatedly."
)

# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK SCENES
# ══════════════════════════════════════════════════════════════════════════════

BENCHMARK_SCENES = [
    {
        "name": "bey",
        "image": BM_IMAGES_DIR / "bey.png",
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "Pink pup in pink pilot helmet and flight gear, standing confidently in front of the "
            "PAW Patrol lookout tower on a bright sunny day. She turns her head toward the camera "
            "with a warm, proud smile, tail wagging gently. Expressive eyes, slight head tilt, "
            "charming and confident energy. Mountains and ocean visible in the background."
            + _DIALOGUE
        ),
    },
    {
        "name": "crsms",
        "image": BM_IMAGES_DIR / "crsms.png",
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "Pink pup in a Santa hat and red Christmas sweater with a wreath, sitting in a snowy "
            "Christmas village surrounded by decorated trees and colorful gift boxes. She bobs her "
            "head cheerfully, smiles with festive excitement, ears perking up with joy. "
            "Northern lights glowing in the night sky behind her. Warm, cozy holiday energy."
            + _DIALOGUE
        ),
    },
    {
        "name": "holoween",
        "image": BM_IMAGES_DIR / "holoween.png",
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "Pink pup wearing a purple witch hat and Halloween outfit, sitting on a spooky porch "
            "surrounded by glowing carved jack-o-lanterns and a 'Trick or Treat' sign. "
            "She looks around the scene with wide, playful eyes, then gives a mischievous grin "
            "at the camera, wiggling her ears. Moonlit Halloween night, warm orange pumpkin glow. "
            "Fun, spooky-cute energy."
            + _DIALOGUE
        ),
    },
    {
        "name": "party",
        "image": BM_IMAGES_DIR / "party.png",
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "Pink pup in a purple witch hat surrounded by glowing jack-o-lanterns at a festive "
            "Halloween party. She looks directly at the camera with wide, sparkling eyes and a big "
            "excited smile, tail wagging with energy. She does a happy little bounce in place. "
            "Warm orange candlelit glow from the pumpkins, lively party energy."
            + _DIALOGUE
        ),
    },
    {
        "name": "snow",
        "image": BM_IMAGES_DIR / "snow.png",
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "Pink pup in pink pilot gear standing beside the snow-covered PAW Patrol lookout tower "
            "decorated with Christmas lights and green garland. Snowflakes drifting gently around "
            "her. She looks up at the falling snow with wide, wonder-filled eyes, then turns to "
            "smile warmly at the camera. Peaceful winter night, crescent moon glowing softly above."
            + _DIALOGUE
        ),
    },
    # ── Overfitting test scenes ───────────────────────────────────────────────
    # These use exact training captions + the birthday dialogue.
    # If the LoRA has overfit, it should reproduce the motion it was trained on.
    # enhance_prompt=False so Gemma doesn't alter the training text.
    {
        "name": "of_silly_goose",
        "image": OVERFIT_IMAGES_DIR / "scene_41_episode_1_scene_segment_41.png",
        "enhance_prompt": False,
        "prompt": (
            'SKYE says "Where are you going, silly goose?". '
            "A fixed medium close-up shot frames Skye, a golden-brown pup, as she faces the camera. "
            "At the start, her eyes are almost closed, conveying a mischievous, slightly smug expression, "
            "with her perked ears framing her face. Her purple irises are barely visible beneath her heavy "
            "eyelids. Her head remains perfectly still as her eyes slowly open, revealing her full, bright "
            "purple irises, and her facial expression transforms into an alert, playful look, accompanied "
            "by a small, closed-mouth smile. Her gaze remains fixed on the camera throughout this subtle shift."
            + _DIALOGUE
        ),
    },
    {
        "name": "of_eagle",
        "image": OVERFIT_IMAGES_DIR / "scene_66_episode_2_scene_segment_66.png",
        "enhance_prompt": False,
        "prompt": (
            'SKYE says "That eagle can be dangerous to small flying creatures like Chickaletta and me!". '
            "The camera holds a fixed medium close-up on Skye, her face etched with a worried expression. "
            "Her wide, purple eyes dart slightly, initially looking to the right of the camera, then shifting "
            "upwards and further to the right, conveying deep concern. Her mouth is slightly agape, revealing "
            "her teeth as she speaks with trepidation, while her ears are slightly lowered, emphasizing her distress."
            + _DIALOGUE
        ),
    },
    {
        "name": "of_lifeguard",
        "image": OVERFIT_IMAGES_DIR / "scene_58_episode_1_scene_segment_58.png",
        "enhance_prompt": False,
        "prompt": (
            'SKYE says "You can\'t be a lifeguard without getting wet, Rocky.". '
            "A fixed medium close-up shot captures Skye, the tan and white Cockapoo, wearing her signature "
            "pink aviator visor and gear. Initially, her wide, purple eyes gaze slightly upwards and to the "
            "right with a thoughtful, concerned expression. Her head then subtly tilts downwards as her eyes "
            "close, and her mouth opens wide in a sigh of exasperation. Suddenly, her head lifts, and her eyes "
            "snap open, widening dramatically, while her mouth forms a surprised 'O' shape, as she looks "
            "directly forward, facing the camera with an expression of shock."
            + _DIALOGUE
        ),
    },
    {
        "name": "of_lift",
        "image": OVERFIT_IMAGES_DIR / "scene_189_episode_1_scene_segment_189.png",
        "enhance_prompt": False,
        "prompt": (
            'SKYE says "Did someone call for a lift?  Harness!". '
            "The camera maintains a fixed medium shot on Skye, the cockapoo, seated in her pink helicopter. "
            "Initially, she faces the camera with a wide, open-mouthed smile, her bright eyes sparkling behind "
            "pink goggles, and her head tilted slightly, conveying a cheerful and energetic demeanor. As the "
            "moment unfolds, her expression subtly transforms; her smile softens into a slight frown, and her "
            "eyes slowly close, her ears drooping slightly, transitioning her cheerful disposition to one of "
            "gentle weariness or sadness."
            + _DIALOGUE
        ),
    },
    {
        "name": "of_everest",
        "image": OVERFIT_IMAGES_DIR / "scene_27_episode_1_scene_segment_27.png",
        "enhance_prompt": False,
        "prompt": (
            'SKYE says "It\'s gonna be so much fun visiting Everest and Captain Turban!". '
            "A fixed medium shot centers on Skye, the golden-brown cockapoo, sitting upright on a teal surface. "
            "She faces the camera with wide, bright pink eyes and an enthusiastic, open-mouthed smile, her head "
            "slightly tilted as she speaks, conveying excitement. Her pink collar and badge are visible, and her "
            "ears are perked. As she continues to speak, her eyes briefly close and reopen, maintaining her "
            "cheerful demeanor. Finally, her eyes close again, and her smile softens into a peaceful, contented "
            "expression, her head remaining still, as if savoring a pleasant thought."
            + _DIALOGUE
        ),
    },
]

# ══════════════════════════════════════════════════════════════════════════════
#  Nothing below here should need editing
# ══════════════════════════════════════════════════════════════════════════════

log = logging.getLogger(__name__)


def frames_for_duration(seconds: float, fps: float) -> int:
    raw = round(seconds * fps)
    k = max(1, (raw - 1 + 7) // 8)
    return max(17, 8 * k + 1)


def discover_checkpoints(exp_dirs: list[Path]) -> list[tuple[str, int, Path]]:
    """Return list of (exp_name, step, ckpt_path) sorted by (exp_name, step)."""
    results = []
    for exp_dir in exp_dirs:
        ckpt_dir = exp_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            continue
        for p in ckpt_dir.glob("lora_weights_step_*.safetensors"):
            m = re.search(r"step_(\d+)", p.stem)
            if m:
                results.append((exp_dir.name, int(m.group(1)), p))
    results.sort(key=lambda t: (t[0], t[1]))
    return results


def output_path_for(base_dir: Path, exp_name: str, step: int, scene_name: str) -> Path:
    return base_dir / "benchmarks" / exp_name / f"step_{step:05d}" / f"{scene_name}.mp4"


def build_pipeline(lora_path: Path):
    import torch
    torch.backends.cudnn.enabled = False

    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline

    skye_loras = (LoraPathStrengthAndSDOps(
        path=str(lora_path),
        strength=1.0,
        sd_ops=None,
    ),)
    distilled_lora = [LoraPathStrengthAndSDOps(
        path=str(DISTILLED_LORA),
        strength=1.0,
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
    )]
    return TI2VidTwoStagesHQPipeline(
        checkpoint_path=str(CHECKPOINT),
        distilled_lora=distilled_lora,
        distilled_lora_strength_stage_1=BM_LORA_S1,
        distilled_lora_strength_stage_2=BM_LORA_S2,
        spatial_upsampler_path=str(SPATIAL_UPSAMPLER),
        gemma_root=str(GEMMA_ROOT),
        loras=skye_loras,
    )


def generate_one(pipeline, *, scene, num_frames, video_guider_params,
                 audio_guider_params, output_path) -> None:
    import torch
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.types import Audio
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    negative_prompt = (
        DEFAULT_NEGATIVE_PROMPT
        + ", waving paw, raised paw, paw in the air, arm raised, hand gesture, "
        "waving hand, pointing, gesturing, paw movement, arm movement"
    )

    enhance = scene.get("enhance_prompt", BM_ENHANCE_PROMPT)

    video, audio = pipeline(
        prompt=scene["prompt"],
        negative_prompt=negative_prompt,
        seed=BM_SEED,
        height=BM_HEIGHT,
        width=BM_WIDTH,
        num_frames=num_frames,
        frame_rate=BM_FPS,
        num_inference_steps=BM_STEPS,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        images=[ImageConditioningInput(path=str(scene["image"]), frame_idx=0, strength=1.0)],
        tiling_config=tiling_config,
        enhance_prompt=enhance,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        encode_video(
            video=video,
            fps=int(BM_FPS),
            audio=audio,
            output_path=str(output_path),
            video_chunks_number=video_chunks_number,
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="run_skye_benchmark — evaluate every LoRA checkpoint on all benchmark scenes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exp-dirs", type=Path, nargs="+", default=None,
        help="Explicit experiment output dirs to scan. Default: all subdirs of outputs/",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR / "output",
        help="Root output dir. Benchmark videos go under {output-dir}/benchmarks/.",
    )
    parser.add_argument(
        "--scenes", nargs="+", default=None,
        choices=[s["name"] for s in BENCHMARK_SCENES] + ["overfit"],
        help="Run only these scene(s). Default: all.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be generated without actually running.",
    )
    args = parser.parse_args()

    # Resolve experiment dirs
    if args.exp_dirs:
        exp_dirs = [d.resolve() for d in args.exp_dirs]
    else:
        if not DEFAULT_OUTPUTS_BASE.is_dir():
            print(f"ERROR: outputs base dir not found: {DEFAULT_OUTPUTS_BASE}", file=sys.stderr)
            return 1
        exp_dirs = sorted(
            d for d in DEFAULT_OUTPUTS_BASE.iterdir()
            if d.is_dir() and d.name.startswith("skye_")
        )

    if not exp_dirs:
        print("No experiment dirs found.", file=sys.stderr)
        return 1

    # Filter scenes  ("overfit" is a shorthand for all of_ scenes)
    scenes = BENCHMARK_SCENES
    if args.scenes:
        names = set(args.scenes)
        if "overfit" in names:
            names.discard("overfit")
            names |= {s["name"] for s in BENCHMARK_SCENES if s["name"].startswith("of_")}
        scenes = [s for s in BENCHMARK_SCENES if s["name"] in names]

    # Validate benchmark images exist
    missing_images = [s["image"] for s in scenes if not s["image"].exists()]
    if missing_images:
        for p in missing_images:
            print(f"ERROR: benchmark image not found: {p}", file=sys.stderr)
        print("  → scp them to the server:  scp -r 'inputs/sky bm/' server:~/projects/LTX-2/inputs/sky_bm/",
              file=sys.stderr)
        return 1

    # Validate model weights
    if not args.dry_run:
        missing = [p for p in (CHECKPOINT, DISTILLED_LORA, SPATIAL_UPSAMPLER, GEMMA_ROOT) if not p.exists()]
        if missing:
            for m in missing:
                print(f"ERROR: weight not found: {m}", file=sys.stderr)
            return 1

    # Discover checkpoints
    checkpoints = discover_checkpoints(exp_dirs)
    if not checkpoints:
        print("No checkpoints found in any experiment dir yet.")
        return 0

    # Build work list — skip already-done
    work = []
    skip_count = 0
    for exp_name, step, ckpt_path in checkpoints:
        for scene in scenes:
            out = output_path_for(args.output_dir, exp_name, step, scene["name"])
            if out.exists():
                skip_count += 1
            else:
                work.append((exp_name, step, ckpt_path, scene, out))

    total = len(work) + skip_count
    print(f"\nrun_skye_benchmark")
    print(f"  checkpoints : {len(checkpoints)}")
    print(f"  scenes      : {[s['name'] for s in scenes]}")
    print(f"  total jobs  : {total}  ({skip_count} already done, {len(work)} to run)")
    print()

    if not work:
        print("Nothing to do — all outputs already exist.")
        return 0

    if args.dry_run:
        for exp_name, step, ckpt_path, scene, out in work:
            print(f"  WOULD RUN  {exp_name}  step={step:05d}  scene={scene['name']}")
            print(f"             → {out}")
        return 0

    from ltx_core.components.guiders import MultiModalGuiderParams

    video_guider_params = MultiModalGuiderParams(
        cfg_scale=BM_VIDEO_CFG,
        stg_scale=0.0,
        rescale_scale=0.45,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[],
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=BM_AUDIO_CFG,
        stg_scale=0.0,
        rescale_scale=1.0,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[],
    )

    num_frames = frames_for_duration(BM_DURATION_S, BM_FPS)
    actual_dur = round(num_frames / BM_FPS, 2)
    print(f"  duration    : {BM_DURATION_S}s → {num_frames} frames ({actual_dur}s actual)")
    print(f"  resolution  : {BM_WIDTH}×{BM_HEIGHT}  |  steps={BM_STEPS}  seed={BM_SEED}")
    print()

    errors = 0
    current_ckpt = None
    pipeline = None

    for exp_name, step, ckpt_path, scene, out in work:
        # Load a new pipeline only when the checkpoint changes
        if (exp_name, step) != current_ckpt:
            print(f"── Loading checkpoint: {exp_name}  step={step:05d} ──")
            print(f"   {ckpt_path}")
            pipeline = build_pipeline(ckpt_path)
            current_ckpt = (exp_name, step)
            print()

        print(f"  Scene: {scene['name']:12s}  → {out.relative_to(args.output_dir)}")
        try:
            generate_one(
                pipeline,
                scene=scene,
                num_frames=num_frames,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                output_path=out,
            )
            print(f"    saved ✓")
        except Exception:
            log.exception("Generation failed: %s / step %d / %s", exp_name, step, scene["name"])
            errors += 1
        print()

    done = len(work) - errors
    print(f"Done. {done}/{len(work)} generation(s) succeeded.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
