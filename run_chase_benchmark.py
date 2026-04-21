#!/usr/bin/env python3
"""
run_chase_benchmark.py

Auto-discovers every LoRA checkpoint across all Chase experiment output dirs
and runs all 5 benchmark scenes for each new checkpoint.

Idempotent — safe to re-run at any time.  Already-generated outputs are
skipped so this can be called after every training checkpoint.

Usage
-----
    uv run python run_chase_benchmark.py               # scan all default exp dirs
    uv run python run_chase_benchmark.py --dry-run     # print plan, generate nothing
    uv run python run_chase_benchmark.py --exp-dirs outputs/chase_exp1_highcap
    uv run python run_chase_benchmark.py --scenes normal snow   # only these scenes
    uv run python run_chase_benchmark.py --steps 2000 5000 10000
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
BM_IMAGES_DIR = SCRIPT_DIR / "inputs" / "chase_bm"

# Model weights
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

BM_DURATION_S  = 10.0
BM_SEED        = 42
BM_FPS         = 25.0
BM_WIDTH       = 832    # Chase trained at 960×832; use portrait-ish for 2-stage
BM_HEIGHT      = 960
BM_STEPS       = 15
BM_VIDEO_CFG   = 3.0
BM_AUDIO_CFG   = 7.0
BM_LORA_S1     = 0.25
BM_LORA_S2     = 0.50
BM_ENHANCE_PROMPT = False

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED DIALOGUE — appended to every scene prompt
# ══════════════════════════════════════════════════════════════════════════════

_DIALOGUE = (
    "\n\n[DIALOGUE - WHAT CHASE SAYS]\n"
    "Brown German Shepherd police pup (confident, clear voice, speaking directly to camera): "
    '"Chase is on the case! Whatever the mission, I\'m ready!" '
    "Expressive face, mouth moving clearly with the words, ears perked up.\n\n"
    "[DIALOGUE - WHAT CHASE SAYS]\n"
    "Brown German Shepherd police pup (earnest, looking at the camera): "
    '"PAW Patrol, ready for action! Ryder needs us!" '
    "Alert expression, mouth moving with speech, tail wagging.\n\n"
    "[DIALOGUE - WHAT CHASE SAYS]\n"
    "Brown German Shepherd police pup (enthusiastic, proud): "
    '"This pup\'s gotta fly! No job too big, no pup too small!" '
    "Big smile, excited expression, mouth moving animatedly."
)

# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK SCENES
# ══════════════════════════════════════════════════════════════════════════════

BENCHMARK_SCENES = [
    {
        "name": "normal",
        "image": BM_IMAGES_DIR / "chase_normal.png",
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "CHASE, a brown German Shepherd puppy in his blue police uniform and cap, stands "
            "confidently at the PAW Patrol Lookout on a bright sunny day. He faces the camera "
            "with alert, focused eyes, ears perked up, tail wagging steadily. His blue badge "
            "gleams in the sunlight. Proud, professional energy."
            + _DIALOGUE
        ),
    },
    {
        "name": "halloween",
        "image": BM_IMAGES_DIR / "chase_halloween.png",
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "CHASE, the German Shepherd police pup from PAW Patrol, wearing a fun Halloween "
            "costume over his police gear — orange pumpkin hat, sitting on a spooky porch "
            "surrounded by glowing jack-o-lanterns. He looks around with wide playful eyes, "
            "then gives a big excited grin at the camera, ears perking up with delight. "
            "Moonlit Halloween night, warm orange pumpkin glow, spooky-cute energy."
            + _DIALOGUE
        ),
    },
    {
        "name": "crms",
        "image": BM_IMAGES_DIR / "chase_crms.png",
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "CHASE, the German Shepherd police pup from PAW Patrol, wearing a festive Christmas "
            "bandana and Santa hat, sitting in a cozy snowy village surrounded by decorated "
            "Christmas trees and colorful gift boxes. He bobs his head cheerfully, smiles with "
            "festive excitement, ears perking up with joy. Northern lights glowing softly above. "
            "Warm, cozy holiday energy."
            + _DIALOGUE
        ),
    },
    {
        "name": "snow",
        "image": BM_IMAGES_DIR / "chase_snow.png",
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "CHASE, the German Shepherd police pup from PAW Patrol, standing in a snowy "
            "Adventure Bay, snowflakes drifting gently around him. He looks up at the falling "
            "snow with wide wonder-filled eyes, then turns to smile warmly at the camera. "
            "His blue police uniform dusted with snow. Peaceful winter night, crescent moon "
            "glowing softly above."
            + _DIALOGUE
        ),
    },
    {
        "name": "party",
        "image": BM_IMAGES_DIR / "party_chase.png",
        "prompt": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "CHASE, the German Shepherd police pup from PAW Patrol, at a fun birthday party "
            "surrounded by colorful balloons, streamers and a cake. He looks directly at the "
            "camera with wide sparkling eyes and a big excited smile, tail wagging with energy. "
            "He does a happy little bounce in place. Bright celebratory lighting, lively "
            "party energy."
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
# base video:        benchmarks/_base/{scene}.mp4
# lora video:        benchmarks/{exp}/step_{N}/{scene}_lora.mp4
# comparison video:  benchmarks/{exp}/step_{N}/{scene}.mp4

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
    def esc(s: str) -> str:
        return s.replace("'", "\\'").replace(":", "\\:").replace("[", "\\[").replace("]", "\\]")

    base_lbl = esc("Base LTX-2.3")
    lora_lbl = esc(lora_label)

    fs_small = 28
    fs_large = 38
    box = "box=1:boxcolor=black@0.6:boxborderw=10"

    W, H = BM_WIDTH, BM_HEIGHT
    fps   = BM_FPS
    dur   = BM_DURATION_S
    arate = 44100

    fc = (
        f"[0:v]split=2[base_v1][base_v2];"
        f"[0:a]asplit=2[base_a1][base_a2];"
        f"[1:v]split=2[lora_v1][lora_v2];"
        f"[1:a]asplit=2[lora_a1][lora_a2];"
        f"color=black:s={W}x{H}:r={fps}:d={dur}[black_v1];"
        f"color=black:s={W}x{H}:r={fps}:d={dur}[black_v2];"
        f"aevalsrc=0:s={arate}:d={dur}[sil1];"
        f"aevalsrc=0:s={arate}:d={dur}[sil2];"
        f"[base_v1]drawtext=text='{base_lbl}':fontsize={fs_small}:fontcolor=white"
        f":x=10:y=h-th-10:{box}[bv1];"
        f"[lora_v1]drawtext=text='{lora_lbl}':fontsize={fs_small}:fontcolor=yellow"
        f":x=10:y=h-th-10:{box}[lv1];"
        f"[bv1][lv1]hstack=inputs=2[side1];"
        f"[side1]drawtext=text='[ Base | LoRA ]':fontsize={fs_large}:fontcolor=white"
        f":x=(w-tw)/2:y=18:{box}[clip1v];"
        f"[base_a1][lora_a1]amix=inputs=2:duration=first:normalize=0[clip1a];"
        f"[base_v2]drawtext=text='{base_lbl}':fontsize={fs_small}:fontcolor=white"
        f":x=10:y=h-th-10:{box}[bv2];"
        f"[bv2][black_v1]hstack=inputs=2[side2];"
        f"[side2]drawtext=text='[ Base only ]':fontsize={fs_large}:fontcolor=white"
        f":x=(w-tw)/2:y=18:{box}[clip2v];"
        f"[base_a2][sil1]amix=inputs=2:duration=first:normalize=0[clip2a];"
        f"[lora_v2]drawtext=text='{lora_lbl}':fontsize={fs_small}:fontcolor=yellow"
        f":x=10:y=h-th-10:{box}[lv3];"
        f"[black_v2][lv3]hstack=inputs=2[side3];"
        f"[side3]drawtext=text='[ LoRA only ]':fontsize={fs_large}:fontcolor=yellow"
        f":x=(w-tw)/2:y=18:{box}[clip3v];"
        f"[lora_a2][sil2]amix=inputs=2:duration=first:normalize=0[clip3a];"
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
        description="run_chase_benchmark — base vs LoRA comparison for every Chase checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exp-dirs", type=Path, nargs="+", default=None,
        help="Explicit experiment output dirs to scan. Default: all chase_* subdirs of outputs/",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR / "lora_results" / "chase_lora",
        help="Root output dir. Benchmark videos go under {output-dir}/benchmarks/.",
    )
    parser.add_argument(
        "--scenes", nargs="+", default=None,
        choices=[s["name"] for s in BENCHMARK_SCENES],
        help="Run only these scene(s). Default: all.",
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", default=None,
        help="Only benchmark these checkpoint step numbers. Default: all found.",
    )
    parser.add_argument(
        "--last-only", action="store_true",
        help="Only run the highest-step checkpoint per experiment.",
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
            if d.is_dir() and d.name.startswith("chase_")
        )
    if not exp_dirs:
        print("No experiment dirs found.", file=sys.stderr)
        return 1

    # ── Filter scenes ─────────────────────────────────────────────────────────
    scenes = BENCHMARK_SCENES
    if args.scenes:
        names = set(args.scenes)
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
    base_needed = [
        (scene, base_path_for(args.output_dir, scene["name"]))
        for scene in scenes
        if not base_path_for(args.output_dir, scene["name"]).exists()
    ]
    seen_scenes: set[str] = set()
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
    print(f"\nrun_chase_benchmark")
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

    # ── PASS 1: Base model ────────────────────────────────────────────────────
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

    # ── PASS 2: LoRA model ────────────────────────────────────────────────────
    current_ckpt = None
    lora_pipeline = None

    for exp_name, step, ckpt_path, scene, lora_out, comp_out in lora_work:
        if (exp_name, step) != current_ckpt:
            if lora_pipeline is not None:
                print(f"── Unloading {current_ckpt[0]} step {current_ckpt[1]:05d} ──\n")
                unload_pipeline(lora_pipeline)
            print(f"── Loading LoRA: {exp_name}  step={step:05d} ──")
            print(f"   {ckpt_path}")
            lora_pipeline = build_pipeline_lora(ckpt_path)
            current_ckpt = (exp_name, step)
            print()

        lora_label = f"LoRA: {exp_name.replace('chase_', '')}  step {step:,}"
        print(f"  [LORA] {scene['name']:20s} → {lora_out.name}")

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
