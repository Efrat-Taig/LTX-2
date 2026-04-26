#!/usr/bin/env python3
"""
run_bm_v1_lora.py

Evaluate LoRA checkpoints against BM_v1 benchmark scenes using ValidationSampler.
Works with ltx-trainer only — no ltx-pipelines required.

Auto-discovers checkpoints in the experiment output dir.  Idempotent: already-
generated outputs are skipped so this can be run at any time during/after training.

Usage
-----
    uv run python run_bm_v1_lora.py --exp-dir outputs/paw_patrol_s01e03a_exp1
    uv run python run_bm_v1_lora.py --exp-dir outputs/paw_patrol_s01e03a_exp1 --dry-run
    uv run python run_bm_v1_lora.py --exp-dir outputs/paw_patrol_s01e03a_exp1 --steps 5000 10000
    uv run python run_bm_v1_lora.py --exp-dir outputs/paw_patrol_s01e03a_exp1 --last-only
    uv run python run_bm_v1_lora.py --exp-dir outputs/paw_patrol_s01e03a_exp1 --scenes skye_chase_crosswalk_safety
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  Paths (server paths — edit only if layout changes)
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = Path.home() / "models" / "LTX-2.3" / "ltx-2.3-22b-dev.safetensors"
GEMMA_ROOT      = Path.home() / "models" / "gemma-3-12b-it-qat-q4_0-unquantized"
BM_V1_ROOT      = SCRIPT_DIR / "inputs" / "BM_v1" / "benchmark_v1"
DEFAULT_OUT_BASE = SCRIPT_DIR / "lora_results" / "full_episods"

# ══════════════════════════════════════════════════════════════════════════════
#  Generation parameters (fixed for fair cross-checkpoint comparison)
# ══════════════════════════════════════════════════════════════════════════════

BM_WIDTH  = 960
BM_HEIGHT = 544
BM_FRAMES = 97       # ~4 s at 24 fps  (satisfies frames % 8 == 1)
BM_FPS    = 24.0
BM_STEPS  = 30
BM_CFG    = 4.0
BM_STG    = 1.0
BM_STG_BLOCKS = [29]
BM_STG_MODE   = "stg_av"
BM_SEED   = 42
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"
AUDIO_SAMPLE_RATE = 44100   # default; overridden at runtime from vocoder.output_sampling_rate

# ══════════════════════════════════════════════════════════════════════════════
#  Scene discovery
# ══════════════════════════════════════════════════════════════════════════════

def discover_scenes(bm_root: Path) -> list[dict]:
    """Load all BM_v1 scene configs that have a valid config.json."""
    scenes = []
    for scene_dir in sorted(bm_root.iterdir()):
        cfg_path = scene_dir / "config.json"
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        prompts = cfg.get("specific_prompts", [])
        if not prompts:
            log.warning(f"Scene {scene_dir.name}: no specific_prompts, skipping")
            continue
        images = cfg.get("images", [])
        start_frame = None
        for img in images:
            img_path = scene_dir / img["path"]
            if img_path.exists():
                start_frame = img_path
                break
        if start_frame is None:
            log.warning(f"Scene {scene_dir.name}: no start frame image found, skipping")
            continue
        global_prompt = cfg.get("global_prompt", "")
        full_prompt = (global_prompt + "\n\n" + prompts[0]).strip() if global_prompt else prompts[0]
        scenes.append({
            "name": scene_dir.name,
            "prompt": full_prompt,
            "image": start_frame,
            "negative_prompt": cfg.get("negative_prompt", NEGATIVE_PROMPT),
        })
    return scenes


# ══════════════════════════════════════════════════════════════════════════════
#  Checkpoint discovery
# ══════════════════════════════════════════════════════════════════════════════

def discover_checkpoints(exp_dir: Path) -> list[tuple[int, Path]]:
    """Return list of (step, ckpt_path) sorted by step."""
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    results = []
    for p in ckpt_dir.glob("lora_weights_step_*.safetensors"):
        m = re.search(r"step_(\d+)", p.stem)
        if m:
            results.append((int(m.group(1)), p))
    results.sort(key=lambda t: t[0])
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Output paths
# ══════════════════════════════════════════════════════════════════════════════

def base_path_for(out_dir: Path, scene_name: str) -> Path:
    return out_dir / "_base" / f"{scene_name}.mp4"

def lora_path_for(out_dir: Path, step: int, scene_name: str) -> Path:
    return out_dir / f"step_{step:05d}" / f"{scene_name}_lora.mp4"

def comparison_path_for(out_dir: Path, step: int, scene_name: str) -> Path:
    return out_dir / f"step_{step:05d}" / f"{scene_name}.mp4"


# ══════════════════════════════════════════════════════════════════════════════
#  LoRA config from training YAML
# ══════════════════════════════════════════════════════════════════════════════

def _load_lora_config_from_exp(exp_dir: Path):
    """Read PEFT LoraConfig from the training_config.yaml saved in the output dir."""
    from peft import LoraConfig

    training_cfg_path = exp_dir / "training_config.yaml"
    if not training_cfg_path.exists():
        log.warning(f"training_config.yaml not found in {exp_dir}; using PAW Patrol EXP-1 defaults")
        return LoraConfig(
            r=32, lora_alpha=32, lora_dropout=0.0,
            target_modules=["to_k", "to_q", "to_v", "to_out.0",
                            "ff.net.0.proj", "ff.net.2",
                            "audio_ff.net.0.proj", "audio_ff.net.2"],
        )
    import yaml
    # The trainer's saved training_config.yaml contains `!!python/tuple` tags (e.g. video_dims),
    # which yaml.safe_load rejects. Use unsafe_load — file is trainer-generated, not user input.
    with open(training_cfg_path) as f:
        cfg = yaml.unsafe_load(f)
    lora_cfg = cfg.get("lora", {})
    return LoraConfig(
        r=lora_cfg.get("rank", 32),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.0),
        target_modules=lora_cfg.get("target_modules", [
            "to_k", "to_q", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2",
            "audio_ff.net.0.proj", "audio_ff.net.2",
        ]),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading / unloading
# ══════════════════════════════════════════════════════════════════════════════

def load_components():
    """Load base model components (transformer, VAEs, audio, vocoder)."""
    import torch
    from ltx_trainer.model_loader import load_model
    log.info("Loading base model components...")
    components = load_model(
        checkpoint_path=str(CHECKPOINT_PATH),
        device="cpu",
        dtype=torch.bfloat16,
        with_video_vae_encoder=True,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=True,
        with_vocoder=True,
        with_text_encoder=False,
    )
    components.transformer = components.transformer.to(dtype=torch.bfloat16)
    components.video_vae_encoder = components.video_vae_encoder.to(dtype=torch.bfloat16)
    components.video_vae_decoder = components.video_vae_decoder.to(dtype=torch.bfloat16)
    components.transformer.requires_grad_(False)
    components.video_vae_encoder.requires_grad_(False)
    components.video_vae_decoder.requires_grad_(False)
    if components.audio_vae_decoder:
        components.audio_vae_decoder.requires_grad_(False)
    if components.vocoder:
        components.vocoder.requires_grad_(False)
    log.info("Base model components loaded.")
    return components


def apply_lora(transformer, lora_config):
    """Wrap transformer with PEFT LoRA."""
    from peft import get_peft_model
    return get_peft_model(transformer, lora_config)


def load_lora_weights(peft_transformer, ckpt_path: Path) -> None:
    """Load LoRA weights from a safetensors checkpoint into peft_transformer."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    state_dict = load_file(str(ckpt_path))
    # Strip ComfyUI "diffusion_model." prefix added by trainer at save time
    state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}
    base_model = peft_transformer.get_base_model()
    set_peft_model_state_dict(base_model, state_dict)
    log.info(f"Loaded LoRA weights from {ckpt_path.name}")


def unload_model(obj) -> None:
    import torch
    del obj
    gc.collect()
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
#  Text embedding pre-computation
# ══════════════════════════════════════════════════════════════════════════════

def precompute_embeddings(scenes: list[dict], device: str = "cuda") -> list[dict]:
    """Load Gemma (8-bit) + embeddings processor, encode all prompts, return cached embeddings."""
    import torch
    from ltx_trainer.model_loader import load_text_encoder, load_embeddings_processor
    from ltx_trainer.validation_sampler import CachedPromptEmbeddings

    log.info("Loading text encoder (8-bit) + embeddings processor...")
    text_encoder = load_text_encoder(str(GEMMA_ROOT), device=device, load_in_8bit=True)
    text_encoder.eval()
    embeddings_processor = load_embeddings_processor(
        checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
        dtype=torch.bfloat16,
    )

    cached = []
    log.info(f"Encoding prompts for {len(scenes)} scenes...")
    with torch.inference_mode():
        for scene in scenes:
            log.info(f"  Encoding: {scene['name']}")
            pos_hs, pos_mask = text_encoder.encode(scene["prompt"])
            pos_out = embeddings_processor.process_hidden_states(pos_hs, pos_mask)

            neg_hs, neg_mask = text_encoder.encode(scene.get("negative_prompt", NEGATIVE_PROMPT))
            neg_out = embeddings_processor.process_hidden_states(neg_hs, neg_mask)

            cached.append(CachedPromptEmbeddings(
                video_context_positive=pos_out.video_encoding.cpu(),
                audio_context_positive=(
                    pos_out.audio_encoding.cpu() if pos_out.audio_encoding is not None else None
                ),
                video_context_negative=neg_out.video_encoding.cpu(),
                audio_context_negative=(
                    neg_out.audio_encoding.cpu() if neg_out.audio_encoding is not None else None
                ),
            ))

    log.info("Unloading text encoder + embeddings processor...")
    unload_model(text_encoder)
    unload_model(embeddings_processor)
    return cached


# ══════════════════════════════════════════════════════════════════════════════
#  Inference
# ══════════════════════════════════════════════════════════════════════════════

def generate_one(
    sampler,
    *,
    scene: dict,
    cached_embeddings,
    output_path: Path,
    audio_sample_rate: int,
    device: str = "cuda",
) -> None:
    import torch
    from torchvision.transforms.functional import to_tensor
    from PIL import Image
    from ltx_trainer.validation_sampler import GenerationConfig
    from ltx_trainer.video_utils import save_video

    # Load conditioning image
    img = Image.open(scene["image"]).convert("RGB")
    condition_image = to_tensor(img)  # [C, H, W] in [0, 1]

    gen_config = GenerationConfig(
        prompt=scene["prompt"],
        negative_prompt=scene.get("negative_prompt", NEGATIVE_PROMPT),
        height=BM_HEIGHT,
        width=BM_WIDTH,
        num_frames=BM_FRAMES,
        frame_rate=BM_FPS,
        num_inference_steps=BM_STEPS,
        guidance_scale=BM_CFG,
        seed=BM_SEED,
        condition_image=condition_image,
        generate_audio=True,
        cached_embeddings=cached_embeddings,
        stg_scale=BM_STG,
        stg_blocks=BM_STG_BLOCKS,
        stg_mode=BM_STG_MODE,
    )

    video, audio = sampler.generate(config=gen_config, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(
        video_tensor=video,
        output_path=output_path,
        fps=BM_FPS,
        audio=audio,
        audio_sample_rate=audio_sample_rate if audio is not None else None,
    )
    log.info(f"  Saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Comparison video (ffmpeg hstack)
# ══════════════════════════════════════════════════════════════════════════════

def concat_comparison(
    base_path: Path,
    lora_path: Path,
    out_path: Path,
    lora_label: str,
) -> None:
    """Build a side-by-side [Base | LoRA] comparison using ffmpeg."""
    def esc(s: str) -> str:
        return s.replace("'", "\\'").replace(":", "\\:").replace("[", "\\[").replace("]", "\\]")

    W, H = BM_WIDTH, BM_HEIGHT
    fps = BM_FPS
    arate = AUDIO_SAMPLE_RATE

    base_lbl = esc("Base LTX-2.3")
    lora_lbl = esc(lora_label)
    box = "box=1:boxcolor=black@0.6:boxborderw=8"
    fs = 26

    fc = (
        f"[0:v]split=2[bv1][bv2];"
        f"[0:a]asplit=2[ba1][ba2];"
        f"[1:v]split=2[lv1][lv2];"
        f"[1:a]asplit=2[la1][la2];"
        f"color=black:s={W}x{H}:r={fps}:d=99[blk1];"
        f"color=black:s={W}x{H}:r={fps}:d=99[blk2];"
        f"aevalsrc=0:s={arate}:d=99[sil1];"
        f"aevalsrc=0:s={arate}:d=99[sil2];"
        # clip 1: both side by side
        f"[bv1]drawtext=text='{base_lbl}':fontsize={fs}:fontcolor=white:x=10:y=h-th-10:{box}[bv1t];"
        f"[lv1]drawtext=text='{lora_lbl}':fontsize={fs}:fontcolor=yellow:x=10:y=h-th-10:{box}[lv1t];"
        f"[bv1t][lv1t]hstack=inputs=2[side1];"
        f"[side1]drawtext=text='[ Base  |  LoRA ]':fontsize=34:fontcolor=white:x=(w-tw)/2:y=16:{box}[c1v];"
        f"[ba1][la1]amix=inputs=2:duration=first:normalize=0[c1a];"
        # clip 2: base only
        f"[bv2]drawtext=text='{base_lbl}':fontsize={fs}:fontcolor=white:x=10:y=h-th-10:{box}[bv2t];"
        f"[bv2t][blk1]hstack=inputs=2[side2];"
        f"[side2]drawtext=text='[ Base only ]':fontsize=34:fontcolor=white:x=(w-tw)/2:y=16:{box}[c2v];"
        f"[ba2][sil1]amix=inputs=2:duration=first:normalize=0[c2a];"
        # clip 3: lora only
        f"[lv2]drawtext=text='{lora_lbl}':fontsize={fs}:fontcolor=yellow:x=10:y=h-th-10:{box}[lv2t];"
        f"[blk2][lv2t]hstack=inputs=2[side3];"
        f"[side3]drawtext=text='[ LoRA only ]':fontsize=34:fontcolor=yellow:x=(w-tw)/2:y=16:{box}[c3v];"
        f"[la2][sil2]amix=inputs=2:duration=first:normalize=0[c3a];"
        # concat
        f"[c1v][c1a][c2v][c2a][c3v][c3a]concat=n=3:v=1:a=1[outv][outa]"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y",
         "-i", str(base_path),
         "-i", str(lora_path),
         "-filter_complex", fc,
         "-map", "[outv]", "-map", "[outa]",
         "-c:v", "libx264", "-crf", "18",
         "-c:a", "aac", "-b:a", "192k",
         str(out_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="BM_v1 LoRA benchmark — base vs LoRA comparison for every checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exp-dir", type=Path, required=True,
        help="Experiment output dir (e.g. outputs/paw_patrol_s01e03a_exp1)")
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE,
        help="Root dir for results. Results go under {out_base}/{exp_name}/")
    parser.add_argument("--scenes", nargs="+", default=None,
        help="Only run these scene names. Default: all BM_v1 scenes.")
    parser.add_argument("--steps", type=int, nargs="+", default=None,
        help="Only benchmark these step numbers. Default: all found.")
    parser.add_argument("--last-only", action="store_true",
        help="Only run the highest-step checkpoint.")
    parser.add_argument("--dry-run", action="store_true",
        help="Print plan without generating anything.")
    parser.add_argument("--device", default="cuda",
        help="Device for inference.")
    args = parser.parse_args()

    exp_dir = args.exp_dir.resolve()
    if not exp_dir.is_dir():
        print(f"ERROR: exp-dir not found: {exp_dir}", file=sys.stderr)
        return 1

    exp_name = exp_dir.name
    out_dir = args.out_base / exp_name

    # ── Validate required paths ────────────────────────────────────────────────
    if not args.dry_run:
        missing = [p for p in (CHECKPOINT_PATH, GEMMA_ROOT) if not p.exists()]
        if missing:
            for m in missing:
                print(f"ERROR: required path not found: {m}", file=sys.stderr)
            return 1

    # ── Discover scenes ────────────────────────────────────────────────────────
    all_scenes = discover_scenes(BM_V1_ROOT)
    if not all_scenes:
        print(f"ERROR: no BM_v1 scenes found in {BM_V1_ROOT}", file=sys.stderr)
        return 1

    if args.scenes:
        allowed = set(args.scenes)
        scenes = [s for s in all_scenes if s["name"] in allowed]
        missing_names = allowed - {s["name"] for s in scenes}
        if missing_names:
            print(f"WARNING: requested scenes not found: {missing_names}", file=sys.stderr)
    else:
        scenes = all_scenes

    # ── Discover checkpoints ───────────────────────────────────────────────────
    checkpoints = discover_checkpoints(exp_dir)
    if not checkpoints:
        print(f"No checkpoints found in {exp_dir / 'checkpoints'}")
        return 0

    if args.steps:
        allowed_steps = set(args.steps)
        checkpoints = [(s, p) for s, p in checkpoints if s in allowed_steps]
        if not checkpoints:
            print(f"No checkpoints found at steps {sorted(args.steps)}", file=sys.stderr)
            return 1

    if args.last_only:
        checkpoints = [checkpoints[-1]]
        print(f"  --last-only: using step {checkpoints[0][0]}")

    # ── Build work lists ───────────────────────────────────────────────────────
    base_needed = []
    seen = set()
    for scene in scenes:
        bp = base_path_for(out_dir, scene["name"])
        if not bp.exists() and scene["name"] not in seen:
            base_needed.append((scene, bp))
            seen.add(scene["name"])

    lora_work = []
    skip_count = 0
    for step, ckpt_path in checkpoints:
        for scene in scenes:
            comp = comparison_path_for(out_dir, step, scene["name"])
            if comp.exists():
                skip_count += 1
            else:
                lora_out = lora_path_for(out_dir, step, scene["name"])
                lora_work.append((step, ckpt_path, scene, lora_out, comp))

    total = len(lora_work) + skip_count
    print(f"\nrun_bm_v1_lora — {exp_name}")
    print(f"  BM_v1 root  : {BM_V1_ROOT}")
    print(f"  Output dir  : {out_dir}")
    print(f"  Scenes      : {[s['name'] for s in scenes]}")
    print(f"  Checkpoints : {[s for s, _ in checkpoints]}")
    print(f"  Resolution  : {BM_WIDTH}×{BM_HEIGHT}  frames={BM_FRAMES}  steps={BM_STEPS}")
    print(f"  Base videos : {len(base_needed)} to generate  ({len(scenes) - len(base_needed)} already done)")
    print(f"  LoRA comps  : {total} total  ({skip_count} already done, {len(lora_work)} to run)")
    print()

    if not base_needed and not lora_work:
        print("Nothing to do — all outputs already exist.")
        return 0

    if args.dry_run:
        for scene, p in base_needed:
            print(f"  WOULD GEN BASE  {scene['name']:40s} → {p}")
        for step, _, scene, lora_out, comp in lora_work:
            print(f"  WOULD GEN LORA  step={step:05d}  {scene['name']:35s} → {comp}")
        return 0

    # ── Pre-compute text embeddings ────────────────────────────────────────────
    print("── Pre-computing text embeddings ──")
    scene_embeddings = precompute_embeddings(scenes, device=args.device)
    print()

    # ── Load base model components ─────────────────────────────────────────────
    print("── Loading model components ──")
    components = load_components()
    audio_sample_rate = getattr(components.vocoder, "output_sampling_rate", AUDIO_SAMPLE_RATE)
    print(f"  Audio sample rate: {audio_sample_rate}")
    print()

    from ltx_trainer.validation_sampler import ValidationSampler

    errors = 0

    # ── PASS 1: Base model — no LoRA ───────────────────────────────────────────
    if base_needed:
        print("── Generating BASE videos (no LoRA) ──")
        sampler = ValidationSampler(
            transformer=components.transformer,
            vae_decoder=components.video_vae_decoder,
            vae_encoder=components.video_vae_encoder,
            audio_decoder=components.audio_vae_decoder,
            vocoder=components.vocoder,
        )
        for idx, (scene, base_out) in enumerate(base_needed):
            emb = scene_embeddings[scenes.index(scene)]
            print(f"  [BASE] {scene['name']}")
            try:
                generate_one(sampler, scene=scene, cached_embeddings=emb,
                             output_path=base_out, audio_sample_rate=audio_sample_rate,
                             device=args.device)
            except Exception:
                log.exception("Base generation failed: %s", scene["name"])
                errors += 1
        del sampler
        print()

    # ── PASS 2: LoRA checkpoints ───────────────────────────────────────────────
    if lora_work:
        print("── Loading LoRA config ──")
        lora_config = _load_lora_config_from_exp(exp_dir)
        print(f"  rank={lora_config.r}  alpha={lora_config.lora_alpha}")
        print(f"  targets={lora_config.target_modules}")

        print("── Applying PEFT LoRA to transformer ──")
        peft_transformer = apply_lora(components.transformer, lora_config)
        print()

        current_step = None
        lora_sampler = None

        for step, ckpt_path, scene, lora_out, comp_out in lora_work:
            if step != current_step:
                print(f"── Loading checkpoint step={step:05d}: {ckpt_path.name} ──")
                try:
                    load_lora_weights(peft_transformer, ckpt_path)
                except Exception:
                    log.exception("Failed to load checkpoint: %s", ckpt_path)
                    # Skip all scenes for this step
                    errors += sum(1 for s, c, sc, lo, co in lora_work if s == step)
                    current_step = step
                    lora_sampler = None
                    continue

                lora_sampler = ValidationSampler(
                    transformer=peft_transformer,
                    vae_decoder=components.video_vae_decoder,
                    vae_encoder=components.video_vae_encoder,
                    audio_decoder=components.audio_vae_decoder,
                    vocoder=components.vocoder,
                )
                current_step = step
                print()

            if lora_sampler is None:
                continue

            emb = scene_embeddings[scenes.index(scene)]
            print(f"  [LORA] step={step:05d}  {scene['name']}")

            if not lora_out.exists():
                try:
                    generate_one(lora_sampler, scene=scene, cached_embeddings=emb,
                                 output_path=lora_out, audio_sample_rate=audio_sample_rate,
                                 device=args.device)
                except Exception:
                    log.exception("LoRA generation failed: step=%d  %s", step, scene["name"])
                    errors += 1
                    print()
                    continue

            # Build comparison video
            base_out = base_path_for(out_dir, scene["name"])
            if base_out.exists():
                lora_label = f"LoRA {exp_name}  step {step:,}"
                try:
                    concat_comparison(base_out, lora_out, comp_out, lora_label)
                    print(f"    comparison saved ✓  ({comp_out.name})")
                except Exception:
                    log.exception("ffmpeg concat failed: step=%d  %s", step, scene["name"])
                    errors += 1
            else:
                print(f"    WARNING: base video missing for {scene['name']} — skipping concat")
            print()

        # Clean up
        del lora_sampler
        del peft_transformer

    unload_model(components)

    done = len(lora_work) - errors
    print(f"Done. {done}/{len(lora_work)} LoRA comparison(s) succeeded.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
