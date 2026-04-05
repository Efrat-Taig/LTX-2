#!/usr/bin/env python3
"""
Batch LTX inference over benchmark scenarios: each prompt gets **one base-model render** (shared across
all LoRA steps), then **one LoRA render per checkpoint** in `LORA_STEPS`.

  cd packages/ltx-trainer
  uv run python scripts/efrat_run_lora.py

Edit CONFIG below. Set `SCENARIO_NAMES` to restrict which benchmark subfolders run; leave empty
to process every directory under BENCHMARK_DIR that contains a `config.json` (recursive).
Outputs: `OUTPUT_BASE_PARENT/base/<scenario_key>/prompt_XX_base.*` (once each), and
`OUTPUT_BASE_PARENT/step_<N>/<scenario_key>/prompt_XX_lora.*` per checkpoint.
Use `USE_CONDITION_IMAGE_ASPECT` to size output from each scenario’s conditioning image (e.g. `start_frame.png`)
while keeping aspect ratio; `CONDITION_IMAGE_MAX_SIDE` caps the long edge. Or set `INFER_WIDTH`/`INFER_HEIGHT`
for fixed resolution. `NUM_INFERENCE_STEPS` / `INFER_DURATION_SEC` tune speed vs quality.
Nested paths use `__` in the folder name (e.g. `veo__skye_chase_crosswalk_safety`).

Per scenario: prompts, negative prompt, seed, fps, duration, and dimensions come from that folder’s
`config.json`. The conditioning image is `start_frame.png` in that same folder (or `images[0].path`
under the config if `start_frame.png` is absent). Optional `PROMPT_SUFFIX` / `FALLBACK_CONDITION_IMAGE`
at the bottom of CONFIG are off by default so behavior matches the benchmark files only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

# =============================================================================
# CONFIG — edit these values
# =============================================================================

# Repository root (LTX-2). Default: three levels up from this file.
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

# Base transformer checkpoint (same as training `model_path`).
CHECKPOINT: Path = REPO_ROOT / "models" / "ltx-2.3-22b-dev.safetensors"

# Gemma text encoder directory (same as training `text_encoder_path`).
TEXT_ENCODER_PATH: Path = REPO_ROOT / "models" / "gemma-3-12b-it-qat-q4_0-unquantized"

# Training run output dir — must match `output_dir` in the training YAML (checkpoints:
# <TRAINING_OUTPUT_DIR>/checkpoints/lora_weights_step_XXXXX.safetensors). Override with env
# LTX_TRAINING_OUTPUT_DIR if your run used a different folder.
def _training_output_dir() -> Path:
    env = os.environ.get("LTX_TRAINING_OUTPUT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT / "outputs" / "paw_patrol_s01e01_lora_10k"


TRAINING_OUTPUT_DIR: Path = _training_output_dir()
CHECKPOINTS_DIR: Path = TRAINING_OUTPUT_DIR / "checkpoints"

# Run these LoRA steps (each run compares LoRA vs base; outputs go under OUTPUT_BASE_PARENT / step_<N>/).
LORA_STEPS: list[int] = [10_000, 5000, 3000]

# Benchmark scenarios (each subdir may contain config.json + optional start_frame.png, etc.).
BENCHMARK_DIR: Path = REPO_ROOT / "input" / "benchmark_v1"

# Only these paths under BENCHMARK_DIR are run (order preserved). Use POSIX-style segments
# for nested dirs, e.g. "veo/skye_chase_crosswalk_safety".
# Leave empty to run every directory under BENCHMARK_DIR that contains config.json (recursive).
SCENARIO_NAMES: list[str] = []

# Root for this batch: base videos under OUTPUT_BASE_PARENT/base/...; LoRA under step_<N>/...
OUTPUT_BASE_PARENT: Path = TRAINING_OUTPUT_DIR / "efrat_benchmark_v1"

# If True, skip base inference when prompt_XX_base.mp4 already exists under `base/<scenario>/` **or**
# under any legacy path `step_<N>/<scenario>/` (older layout duplicated base per step).
SKIP_EXISTING_BASE: bool = True

# If True, width/height follow the conditioning image aspect ratio (scaled so the long edge is at most
# CONDITION_IMAGE_MAX_SIDE; both sides snapped to multiples of 32). Falls back if there is no image.
USE_CONDITION_IMAGE_ASPECT: bool = True
CONDITION_IMAGE_MAX_SIDE: int = 960

# Fixed resolution (multiples of 32). Used only when USE_CONDITION_IMAGE_ASPECT is False.
# Set both to None to use each benchmark's config.json dimensions instead.
INFER_WIDTH: int | None = None
INFER_HEIGHT: int | None = None

# Wall-clock target in seconds. None = use each benchmark `config.json` `duration`. Set to ~10 for full clips.
INFER_DURATION_SEC: float | None = 10.0

# Inference defaults — aligned with repo `run_skye_video.sh` (same CLI as scripts/inference.py).
NUM_INFERENCE_STEPS: int = 20
GUIDANCE_SCALE: float = 4.0
STG_SCALE: float = 1.0
STG_BLOCKS: list[int] = [29]
STG_MODE: str = "stg_av"  # "stg_av" | "stg_v"
ENHANCE_PROMPT: bool = False
SKIP_AUDIO: bool = False
INCLUDE_REFERENCE_IN_OUTPUT: bool = False
REFERENCE_VIDEO: Path | None = None  # set for V2V; benchmark configs do not use this by default
DEVICE: str = "cuda"

# If set, overrides benchmark `fps` (run_skye_video.sh uses 25.0; many benchmark JSONs use 24).
FRAME_RATE_OVERRIDE: float | None = None

# Extra text appended after the JSON prompt (empty = prompts are exactly from config.json).
# Example (run_skye_video.sh style): " static camera composition maintained for the entire clip"
PROMPT_SUFFIX: str = ""

# If set, used only when `start_frame.png` and config `images[0]` paths are missing. None = no fallback.
FALLBACK_CONDITION_IMAGE: Path | None = None

# If True, combine each entry in `specific_prompts` with `global_prompt` (recommended).
COMBINE_GLOBAL_AND_SPECIFIC_PROMPTS: bool = True

# =============================================================================
# Benchmark parsing helpers
# =============================================================================


def snap_multiple_of_32(n: int, *, minimum: int = 32) -> int:
    """LTX requires height/width divisible by 32."""
    return max(minimum, ((int(n) + 31) // 32) * 32)


def find_existing_base_video(output_parent: Path, scenario_key: str, tag: str) -> Path | None:
    """Return path to an existing base mp4 if present in canonical or legacy `step_*` folders."""
    name = f"prompt_{tag}_base.mp4"
    canonical = output_parent / "base" / scenario_key / name
    if canonical.is_file():
        return canonical
    for step_dir in sorted(output_parent.glob("step_*")):
        if not step_dir.is_dir():
            continue
        legacy = step_dir / scenario_key / name
        if legacy.is_file():
            return legacy
    return None


def dimensions_from_condition_image(path: Path, max_side: int) -> tuple[int, int]:
    """Preserve source aspect ratio; scale so long edge <= max_side; W/H divisible by 32."""
    with Image.open(path) as im:
        w0, h0 = im.size
    if w0 < 1 or h0 < 1:
        raise ValueError(f"Invalid image dimensions {w0}x{h0}: {path}")
    scale = float(max_side) / float(max(w0, h0))
    w_float = w0 * scale
    h_float = h0 * scale
    width = snap_multiple_of_32(int(round(w_float)))
    height = snap_multiple_of_32(int(round(width * h0 / w0)))
    return width, height


def num_frames_from_duration(duration_sec: float, fps: float) -> int:
    """Same as get_frames() in run_skye_video.sh: raw=round(sec*fps), k=int((raw-1)/8+0.5), n=8*k+1, min 17."""
    raw = max(1, int(round(float(duration_sec) * float(fps))))
    k = int((raw - 1) / 8 + 0.5)
    n = 8 * k + 1
    return max(17, n)


def load_benchmark_config(scenario_dir: Path) -> dict:
    path = scenario_dir / "config.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_condition_image(scenario_dir: Path, cfg: dict, fallback: Path | None) -> Path | None:
    """Use each scenario's `start_frame.png` first, then config `images[0].path`, then optional fallback."""
    start_default = scenario_dir / "start_frame.png"
    if start_default.is_file():
        return start_default.resolve()

    images = cfg.get("images") or []
    if images:
        rel = images[0].get("path")
        if rel:
            p = (scenario_dir / rel).resolve()
            if p.is_file():
                return p
            print(f"  Warning: conditioning image missing: {p}")

    if fallback is not None:
        fb = fallback.expanduser().resolve()
        if fb.is_file():
            print(f"  Using fallback conditioning image: {fb}")
            return fb

    print("  No usable conditioning image; continuing with text-to-video.")
    return None


def build_full_prompt(cfg: dict, specific: str) -> str:
    global_p = (cfg.get("global_prompt") or "").strip()
    spec = specific.strip()
    if COMBINE_GLOBAL_AND_SPECIFIC_PROMPTS and global_p:
        text = f"{global_p}\n\n{spec}"
    else:
        text = spec
    if PROMPT_SUFFIX:
        text = text.rstrip() + PROMPT_SUFFIX
    return text


def discover_scenario_dirs_with_config(benchmark_dir: Path) -> list[Path]:
    """Every directory under benchmark_dir that contains config.json (recursive, sorted)."""
    return sorted({p.parent for p in benchmark_dir.rglob("config.json")}, key=lambda p: str(p))


def scenario_output_key(scenario_dir: Path, benchmark_root: Path) -> str:
    """Unique folder name under OUTPUT_BASE; nested paths use __ instead of /."""
    rel = scenario_dir.resolve().relative_to(benchmark_root.resolve())
    return rel.as_posix().replace("/", "__")


def iter_scenario_dirs(benchmark_dir: Path, only_names: list[str]) -> list[Path]:
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(f"BENCHMARK_DIR not found: {benchmark_dir}")
    if only_names:
        out: list[Path] = []
        for name in only_names:
            p = (benchmark_dir / name).resolve()
            if not p.is_dir():
                raise FileNotFoundError(f"Scenario folder not found: {p}")
            out.append(p)
        return out
    return discover_scenario_dirs_with_config(benchmark_dir)


def run_one_inference(
    *,
    inference_script: Path,
    lora: Path | None,
    prompt: str,
    negative_prompt: str,
    output_mp4: Path,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    condition_image: Path | None,
    audio_output: Path | None,
) -> int:
    cmd: list[str] = [
        sys.executable,
        str(inference_script),
        "--checkpoint",
        str(CHECKPOINT),
        "--text-encoder-path",
        str(TEXT_ENCODER_PATH),
    ]
    if lora is not None:
        cmd.extend(["--lora-path", str(lora)])
    cmd.extend(
        [
            "--prompt",
            prompt,
            "--negative-prompt",
            negative_prompt,
            "--output",
            str(output_mp4),
            "--height",
            str(height),
            "--width",
            str(width),
            "--num-frames",
            str(num_frames),
            "--frame-rate",
            str(frame_rate),
            "--num-inference-steps",
            str(NUM_INFERENCE_STEPS),
            "--guidance-scale",
            str(GUIDANCE_SCALE),
            "--stg-scale",
            str(STG_SCALE),
            "--stg-mode",
            STG_MODE,
            "--seed",
            str(seed),
            "--device",
            DEVICE,
        ]
    )
    cmd.append("--stg-blocks")
    cmd.extend(str(b) for b in STG_BLOCKS)

    if SKIP_AUDIO:
        cmd.append("--skip-audio")
    if ENHANCE_PROMPT:
        cmd.append("--enhance-prompt")
    if condition_image is not None:
        cmd.extend(["--condition-image", str(condition_image)])
    if REFERENCE_VIDEO is not None:
        cmd.extend(["--reference-video", str(REFERENCE_VIDEO)])
    if INCLUDE_REFERENCE_IN_OUTPUT:
        cmd.append("--include-reference-in-output")
    if audio_output is not None and not SKIP_AUDIO:
        cmd.extend(["--audio-output", str(audio_output)])

    return subprocess.call(cmd)


def main() -> None:
    inference_script = Path(__file__).resolve().parent / "inference.py"
    if not inference_script.is_file():
        raise FileNotFoundError(f"Expected inference.py next to this script: {inference_script}")

    if not LORA_STEPS:
        raise ValueError("LORA_STEPS is empty; set at least one checkpoint step.")

    OUTPUT_BASE_PARENT.mkdir(parents=True, exist_ok=True)
    base_root = OUTPUT_BASE_PARENT / "base"
    base_root.mkdir(parents=True, exist_ok=True)

    scenarios = iter_scenario_dirs(BENCHMARK_DIR, SCENARIO_NAMES)
    if not scenarios:
        raise FileNotFoundError(f"No scenario subdirectories under {BENCHMARK_DIR}")

    failed: list[str] = []

    for scenario_dir in scenarios:
        cfg_path = scenario_dir / "config.json"
        if not cfg_path.is_file():
            print(f"Skip (no config.json): {scenario_dir.name}")
            continue

        cfg = load_benchmark_config(scenario_dir)
        name = scenario_output_key(scenario_dir, BENCHMARK_DIR)
        label = str(scenario_dir.relative_to(BENCHMARK_DIR))
        prompts: list[str] = cfg.get("specific_prompts") or []
        if not prompts:
            print(f"Skip (no specific_prompts): {label}")
            continue

        neg = (cfg.get("negative_prompt") or "").strip()
        seed = int(cfg.get("seed", 42))
        condition_image = resolve_condition_image(scenario_dir, cfg, FALLBACK_CONDITION_IMAGE)

        if USE_CONDITION_IMAGE_ASPECT and condition_image is not None:
            width, height = dimensions_from_condition_image(condition_image, CONDITION_IMAGE_MAX_SIDE)
            print(
                f"  Resolution from conditioning image ({condition_image.name}): "
                f"{width}x{height} (aspect ratio preserved, max {CONDITION_IMAGE_MAX_SIDE}px long edge)"
            )
        elif INFER_WIDTH is not None and INFER_HEIGHT is not None:
            raw_w, raw_h = INFER_WIDTH, INFER_HEIGHT
            width = snap_multiple_of_32(raw_w)
            height = snap_multiple_of_32(raw_h)
            if width != raw_w or height != raw_h:
                print(f"  Note: snapped INFER dims {raw_w}x{raw_h} -> {width}x{height}")
        else:
            dim = cfg.get("dimensions") or {}
            raw_w = int(dim.get("width", 1280))
            raw_h = int(dim.get("height", 720))
            width = snap_multiple_of_32(raw_w)
            height = snap_multiple_of_32(raw_h)
            if width != raw_w or height != raw_h:
                print(f"  Note: snapped dimensions {raw_w}x{raw_h} -> {width}x{height} (multiple of 32)")

        duration = float(INFER_DURATION_SEC if INFER_DURATION_SEC is not None else cfg.get("duration", 10))
        fps = float(FRAME_RATE_OVERRIDE if FRAME_RATE_OVERRIDE is not None else cfg.get("fps", 24))
        num_frames = num_frames_from_duration(duration, fps)

        scenario_base_dir = base_root / name
        scenario_base_dir.mkdir(parents=True, exist_ok=True)

        for i, spec in enumerate(prompts):
            prompt = build_full_prompt(cfg, spec)
            tag = f"{i:02d}"

            base_mp4 = scenario_base_dir / f"prompt_{tag}_base.mp4"
            base_wav = scenario_base_dir / f"prompt_{tag}_base.wav"

            existing_base = (
                find_existing_base_video(OUTPUT_BASE_PARENT, name, tag) if SKIP_EXISTING_BASE else None
            )
            if SKIP_EXISTING_BASE and existing_base is not None:
                print(f"  Skip base (exists): {existing_base.relative_to(OUTPUT_BASE_PARENT)}")
            else:
                print("=" * 80)
                print(f"Base LTX-2 | {label}  [{i + 1}/{len(prompts)}]  -> {base_mp4.name}")
                print("=" * 80)
                code = run_one_inference(
                    inference_script=inference_script,
                    lora=None,
                    prompt=prompt,
                    negative_prompt=neg,
                    output_mp4=base_mp4,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=fps,
                    condition_image=condition_image,
                    audio_output=base_wav,
                )
                if code != 0:
                    failed.append(f"base/{label}/prompt_{tag}_base (exit {code})")

            for step in LORA_STEPS:
                lora = CHECKPOINTS_DIR / f"lora_weights_step_{step:05d}.safetensors"
                lora = lora.expanduser().resolve()
                if not lora.is_file():
                    print(f"  Skip LoRA step {step}: missing {lora.name}")
                    continue

                step_out = OUTPUT_BASE_PARENT / f"step_{step}" / name
                step_out.mkdir(parents=True, exist_ok=True)
                lora_mp4 = step_out / f"prompt_{tag}_lora.mp4"
                lora_wav = step_out / f"prompt_{tag}_lora.wav"

                print("=" * 80)
                print(
                    f"LoRA step {step} | Scenario: {label}  [{i + 1}/{len(prompts)}]  -> {lora_mp4.name}"
                )
                print("=" * 80)

                code = run_one_inference(
                    inference_script=inference_script,
                    lora=lora,
                    prompt=prompt,
                    negative_prompt=neg,
                    output_mp4=lora_mp4,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=fps,
                    condition_image=condition_image,
                    audio_output=lora_wav,
                )
                if code != 0:
                    failed.append(f"step_{step}/{label}/prompt_{tag}_lora (exit {code})")

    if failed:
        print("\nFailures:")
        for f in failed:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nAll benchmark runs finished OK.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
