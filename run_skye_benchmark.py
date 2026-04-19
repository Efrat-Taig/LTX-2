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
MODELS_DIR        = Path.home() / "models" / "LTX-2.3"
CHECKPOINT        = MODELS_DIR / "ltx-2.3-22b-dev.safetensors"
DISTILLED_LORA    = MODELS_DIR / "ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSAMPLER = MODELS_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT        = Path.home() / "models" / "gemma-3-12b-it-qat-q4_0-unquantized"

# Base dir that contains experiment output subdirs
DEFAULT_OUTPUTS_BASE = SCRIPT_DIR / "outputs"

# ══════════════════════════════════════════════════════════════════════════════
#  EDIT THESE — benchmark generation params (kept fixed for fair comparison)
# ══════════════════════════════════════════════════════════════════════════════

BM_DURATION_S  = 10.0   # each clip; comparison video = 3 × 10s = 30s total
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

import gc
import subprocess

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


# ── Output paths ─────────────────────────────────────────────────────────────
# base video:        benchmarks/_base/{scene}.mp4          (generated once, reused)
# lora video:        benchmarks/{exp}/step_{N}/{scene}_lora.mp4
# comparison video:  benchmarks/{exp}/step_{N}/{scene}.mp4  ← what the gallery shows

def base_path_for(output_dir: Path, scene_name: str) -> Path:
    return output_dir / "benchmarks" / "_base" / f"{scene_name}.mp4"

def lora_path_for(output_dir: Path, exp_name: str, step: int, scene_name: str) -> Path:
    return output_dir / "benchmarks" / exp_name / f"step_{step:05d}" / f"{scene_name}_lora.mp4"

def comparison_path_for(output_dir: Path, exp_name: str, step: int, scene_name: str) -> Path:
    return output_dir / "benchmarks" / exp_name / f"step_{step:05d}" / f"{scene_name}.mp4"


# ── Pipeline builders ─────────────────────────────────────────────────────────

def _distilled_lora():
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    return [LoraPathStrengthAndSDOps(
        path=str(DISTILLED_LORA),
        strength=1.0,
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
    )]

def build_pipeline_base():
    """Base LTX-2.3 model — no Skye LoRA."""
    import torch
    torch.backends.cudnn.enabled = False
    from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
    return TI2VidTwoStagesHQPipeline(
        checkpoint_path=str(CHECKPOINT),
        distilled_lora=_distilled_lora(),
        distilled_lora_strength_stage_1=BM_LORA_S1,
        distilled_lora_strength_stage_2=BM_LORA_S2,
        spatial_upsampler_path=str(SPATIAL_UPSAMPLER),
        gemma_root=str(GEMMA_ROOT),
        loras=(),
    )

def build_pipeline_lora(lora_path: Path):
    """Base model + Skye LoRA checkpoint."""
    import torch
    torch.backends.cudnn.enabled = False
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
    return TI2VidTwoStagesHQPipeline(
        checkpoint_path=str(CHECKPOINT),
        distilled_lora=_distilled_lora(),
        distilled_lora_strength_stage_1=BM_LORA_S1,
        distilled_lora_strength_stage_2=BM_LORA_S2,
        spatial_upsampler_path=str(SPATIAL_UPSAMPLER),
        gemma_root=str(GEMMA_ROOT),
        loras=(LoraPathStrengthAndSDOps(path=str(lora_path), strength=1.0, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP),),
    )

def unload_pipeline(pipeline) -> None:
    """Free GPU memory after a pipeline is no longer needed."""
    import torch
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


# ── Generation ────────────────────────────────────────────────────────────────

def generate_one(pipeline, *, scene, num_frames, video_guider_params,
                 audio_guider_params, output_path) -> None:
    import torch
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
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
        enhance_prompt=scene.get("enhance_prompt", BM_ENHANCE_PROMPT),
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


# ── Concatenation ─────────────────────────────────────────────────────────────

def concat_comparison(base_path: Path, lora_path: Path, out_path: Path,
                      lora_label: str) -> None:
    """Build a 3-clip side-by-side comparison video using ffmpeg.

    Clip 1 (0–10s):  [Base | LoRA]  — both side by side, mixed audio
    Clip 2 (10–20s): [Base | black] — base only, base audio
    Clip 3 (20–30s): [black | LoRA] — LoRA only, LoRA audio

    Output: 1536×1024 (768×1024 per half), 30 seconds total.
    """
    def esc(s: str) -> str:
        return s.replace("'", "\\'").replace(":", "\\:").replace("[", "\\[").replace("]", "\\]")

    base_lbl = esc("Base LTX-2.3")
    lora_lbl = esc(lora_label)

    fs_small = 28   # per-model label font size
    fs_large = 38   # section title font size
    box = "box=1:boxcolor=black@0.6:boxborderw=10"

    # Each source video is 768×1024.  We need a 768×1024 black silent clip
    # that exactly matches duration/fps/audio-rate of the sources.
    W, H = BM_WIDTH, BM_HEIGHT   # 768 × 1024 per half
    fps   = BM_FPS                # 25
    dur   = BM_DURATION_S         # 10.0 s
    arate = 44100                 # audio sample rate used by the pipeline

    # ── filter_complex ────────────────────────────────────────────────────────
    # Inputs:  [0] = base,  [1] = lora
    # We generate two silent-black pads inline:
    #   [2] = black video for clip-2 right side (base solo)
    #   [3] = black video for clip-3 left side  (LoRA solo)
    # silent audio is synthesised with aevalsrc=0.
    fc = (
        # ── Split sources so we can use each twice ──────────────────────────
        f"[0:v]split=2[base_v1][base_v2];"
        f"[0:a]asplit=2[base_a1][base_a2];"
        f"[1:v]split=2[lora_v1][lora_v2];"
        f"[1:a]asplit=2[lora_a1][lora_a2];"

        # ── Black silent pad (shared geometry / duration) ───────────────────
        f"color=black:s={W}x{H}:r={fps}:d={dur}[black_v1];"
        f"color=black:s={W}x{H}:r={fps}:d={dur}[black_v2];"
        f"aevalsrc=0:s={arate}:d={dur}[sil1];"
        f"aevalsrc=0:s={arate}:d={dur}[sil2];"

        # ── Clip 1: both side by side ───────────────────────────────────────
        # Add per-model labels at bottom of each half
        f"[base_v1]drawtext=text='{base_lbl}':fontsize={fs_small}:fontcolor=white"
        f":x=10:y=h-th-10:{box}[bv1];"
        f"[lora_v1]drawtext=text='{lora_lbl}':fontsize={fs_small}:fontcolor=yellow"
        f":x=10:y=h-th-10:{box}[lv1];"
        # Stack horizontally
        f"[bv1][lv1]hstack=inputs=2[side1];"
        # Section label centred at top of combined 1536-wide frame
        f"[side1]drawtext=text='[ Base | LoRA ]':fontsize={fs_large}:fontcolor=white"
        f":x=(w-tw)/2:y=18:{box}[clip1v];"
        # Mix both audio streams
        f"[base_a1][lora_a1]amix=inputs=2:duration=first:normalize=0[clip1a];"

        # ── Clip 2: base left, black right ──────────────────────────────────
        f"[base_v2]drawtext=text='{base_lbl}':fontsize={fs_small}:fontcolor=white"
        f":x=10:y=h-th-10:{box}[bv2];"
        f"[bv2][black_v1]hstack=inputs=2[side2];"
        f"[side2]drawtext=text='[ Base only ]':fontsize={fs_large}:fontcolor=white"
        f":x=(w-tw)/2:y=18:{box}[clip2v];"
        f"[base_a2][sil1]amix=inputs=2:duration=first:normalize=0[clip2a];"

        # ── Clip 3: black left, LoRA right ──────────────────────────────────
        f"[lora_v2]drawtext=text='{lora_lbl}':fontsize={fs_small}:fontcolor=yellow"
        f":x=10:y=h-th-10:{box}[lv3];"
        f"[black_v2][lv3]hstack=inputs=2[side3];"
        f"[side3]drawtext=text='[ LoRA only ]':fontsize={fs_large}:fontcolor=yellow"
        f":x=(w-tw)/2:y=18:{box}[clip3v];"
        f"[lora_a2][sil2]amix=inputs=2:duration=first:normalize=0[clip3a];"

        # ── Concatenate 3 clips ──────────────────────────────────────────────
        f"[clip1v][clip1a][clip2v][clip2a][clip3v][clip3a]concat=n=3:v=1:a=1[outv][outa]"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(base_path),
            "-i", str(lora_path),
            "-filter_complex", fc,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed:\n{result.stderr[-2000:]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="run_skye_benchmark — base vs LoRA comparison for every checkpoint",
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
        "--steps", type=int, nargs="+", default=None,
        help="Only benchmark these checkpoint step numbers. Default: all found.",
    )
    parser.add_argument(
        "--last-only", action="store_true",
        help="Only run the highest-step checkpoint per experiment. Useful for quick cross-experiment comparison.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without generating anything.",
    )
    args = parser.parse_args()

    # ── Resolve experiment dirs ───────────────────────────────────────────────
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

    # ── Filter scenes ─────────────────────────────────────────────────────────
    scenes = BENCHMARK_SCENES
    if args.scenes:
        names = set(args.scenes)
        if "overfit" in names:
            names.discard("overfit")
            names |= {s["name"] for s in BENCHMARK_SCENES if s["name"].startswith("of_")}
        scenes = [s for s in BENCHMARK_SCENES if s["name"] in names]

    # ── Validate images ───────────────────────────────────────────────────────
    missing_images = [s["image"] for s in scenes if not s["image"].exists()]
    if missing_images:
        for p in missing_images:
            print(f"ERROR: benchmark image not found: {p}", file=sys.stderr)
        return 1

    # ── Validate weights ──────────────────────────────────────────────────────
    if not args.dry_run:
        missing = [p for p in (CHECKPOINT, DISTILLED_LORA, SPATIAL_UPSAMPLER, GEMMA_ROOT)
                   if not p.exists()]
        if missing:
            for m in missing:
                print(f"ERROR: weight not found: {m}", file=sys.stderr)
            return 1

    # ── Discover checkpoints ──────────────────────────────────────────────────
    checkpoints = discover_checkpoints(exp_dirs)
    if not checkpoints:
        print("No checkpoints found in any experiment dir yet.")
        return 0

    # ── Filter to requested steps ─────────────────────────────────────────────
    if args.steps:
        allowed = set(args.steps)
        checkpoints = [(exp, step, p) for exp, step, p in checkpoints if step in allowed]
        if not checkpoints:
            print(f"No checkpoints found at steps {sorted(allowed)}.", file=sys.stderr)
            return 1

    if args.last_only:
        last_per_exp: dict[str, tuple] = {}
        for exp_name, step, ckpt_path in checkpoints:
            if exp_name not in last_per_exp or step > last_per_exp[exp_name][1]:
                last_per_exp[exp_name] = (exp_name, step, ckpt_path)
        checkpoints = sorted(last_per_exp.values(), key=lambda t: t[0])
        print(f"  --last-only: keeping {[f'{e}@{s}' for e,s,_ in checkpoints]}")

    # ── Build work lists ──────────────────────────────────────────────────────
    # base_needed:   scenes where base video doesn't exist yet
    # lora_work:     (exp, step, ckpt, scene) where comparison doesn't exist yet
    base_needed = [
        (scene, base_path_for(args.output_dir, scene["name"]))
        for scene in scenes
        if not base_path_for(args.output_dir, scene["name"]).exists()
    ]
    # Deduplicate (same scene can appear multiple times across checkpoints)
    seen_scenes = set()
    base_needed_dedup = []
    for scene, p in base_needed:
        if scene["name"] not in seen_scenes:
            base_needed_dedup.append((scene, p))
            seen_scenes.add(scene["name"])
    base_needed = base_needed_dedup

    lora_work = []
    skip_count = 0
    for exp_name, step, ckpt_path in checkpoints:
        for scene in scenes:
            comp = comparison_path_for(args.output_dir, exp_name, step, scene["name"])
            if comp.exists():
                skip_count += 1
            else:
                lora_out = lora_path_for(args.output_dir, exp_name, step, scene["name"])
                lora_work.append((exp_name, step, ckpt_path, scene, lora_out, comp))

    total = len(lora_work) + skip_count
    print(f"\nrun_skye_benchmark")
    print(f"  checkpoints  : {len(checkpoints)}")
    print(f"  scenes       : {[s['name'] for s in scenes]}")
    print(f"  base videos  : {len(base_needed)} to generate  "
          f"({len(scenes) - len(base_needed)} already done)")
    print(f"  comparisons  : {total} total  ({skip_count} already done, {len(lora_work)} to run)")
    print()

    if not base_needed and not lora_work:
        print("Nothing to do — all outputs already exist.")
        return 0

    if args.dry_run:
        for scene, p in base_needed:
            print(f"  WOULD GEN BASE  scene={scene['name']}")
            print(f"                  → {p}")
        for exp_name, step, _, scene, lora_out, comp in lora_work:
            print(f"  WOULD GEN LORA  {exp_name}  step={step:05d}  scene={scene['name']}")
            print(f"                  → {comp}")
        return 0

    from ltx_core.components.guiders import MultiModalGuiderParams

    video_guider_params = MultiModalGuiderParams(
        cfg_scale=BM_VIDEO_CFG, stg_scale=0.0, rescale_scale=0.45,
        modality_scale=3.0, skip_step=0, stg_blocks=[],
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=BM_AUDIO_CFG, stg_scale=0.0, rescale_scale=1.0,
        modality_scale=3.0, skip_step=0, stg_blocks=[],
    )

    num_frames = frames_for_duration(BM_DURATION_S, BM_FPS)
    actual_dur = round(num_frames / BM_FPS, 2)
    print(f"  duration    : {BM_DURATION_S}s → {num_frames} frames ({actual_dur}s actual)")
    print(f"  resolution  : {BM_WIDTH}×{BM_HEIGHT}  |  steps={BM_STEPS}  seed={BM_SEED}")
    print()

    gen_kwargs = dict(num_frames=num_frames, video_guider_params=video_guider_params,
                      audio_guider_params=audio_guider_params)
    errors = 0

    # ── PASS 1: Base model — generate all missing base videos ─────────────────
    if base_needed:
        print("── Loading BASE pipeline ──")
        base_pipeline = build_pipeline_base()
        print()
        for scene, base_out in base_needed:
            print(f"  [BASE] {scene['name']:20s} → {base_out.name}")
            try:
                generate_one(base_pipeline, scene=scene, output_path=base_out, **gen_kwargs)
                print(f"    saved ✓")
            except Exception:
                log.exception("Base generation failed: %s", scene["name"])
                errors += 1
            print()
        print("── Unloading BASE pipeline ──\n")
        unload_pipeline(base_pipeline)

    # ── PASS 2: LoRA model — generate + concatenate ───────────────────────────
    current_ckpt = None
    lora_pipeline = None

    for exp_name, step, ckpt_path, scene, lora_out, comp_out in lora_work:
        # Reload pipeline only when checkpoint changes
        if (exp_name, step) != current_ckpt:
            if lora_pipeline is not None:
                print(f"── Unloading {current_ckpt[0]} step {current_ckpt[1]:05d} ──\n")
                unload_pipeline(lora_pipeline)
            print(f"── Loading LoRA: {exp_name}  step={step:05d} ──")
            print(f"   {ckpt_path}")
            lora_pipeline = build_pipeline_lora(ckpt_path)
            current_ckpt = (exp_name, step)
            print()

        lora_label = f"LoRA: {exp_name.replace('skye_', '')}  step {step:,}"
        print(f"  [LORA] {scene['name']:20s} → {lora_out.name}")

        # Generate LoRA video (skip if already exists from a previous interrupted run)
        if not lora_out.exists():
            try:
                generate_one(lora_pipeline, scene=scene, output_path=lora_out, **gen_kwargs)
                print(f"    lora saved ✓")
            except Exception:
                log.exception("LoRA generation failed: %s / step %d / %s",
                              exp_name, step, scene["name"])
                errors += 1
                print()
                continue

        # Concatenate base + lora → comparison
        base_out = base_path_for(args.output_dir, scene["name"])
        if base_out.exists():
            try:
                concat_comparison(base_out, lora_out, comp_out, lora_label)
                print(f"    comparison saved ✓  ({comp_out.name})")
            except Exception:
                log.exception("Concat failed: %s / step %d / %s", exp_name, step, scene["name"])
                errors += 1
        else:
            print(f"    WARNING: base video missing for {scene['name']} — skipping concat")
        print()

    if lora_pipeline is not None:
        unload_pipeline(lora_pipeline)

    done = len(lora_work) - errors
    print(f"Done. {done}/{len(lora_work)} comparison(s) succeeded.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
