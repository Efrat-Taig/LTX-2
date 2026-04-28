#!/usr/bin/env python3
"""
Run BM_v1 benchmarks through Fal **LTX-2.3 Image-to-Video (Fast)**.

API reference: https://fal.ai/models/fal-ai/ltx-2.3/image-to-video/fast/api

This mirrors ``run_bm_v1_kling_v3_pro_i2v.py`` / ``run_bm_v1_wan26_i2v.py`` (same inputs under ``inputs/BM_v1/benchmark_v1/``)
but targets LTX-2.3 Fast. Requires ``FAL_KEY`` (env or ``api_config.json``).

LTX-2.3 Fast constraints (enforced here)
----------------------------------------
- **duration**: **6, 8, 10, 12, 14, 16, 18, or 20** seconds—``config.json`` ``duration`` is
  clamped to the nearest supported value (default API: 6s).
- **Long clips**: Durations **> 10s** are only supported with **25 FPS** and **1080p**;
  if your scene asks for longer video with other FPS/resolution, those are forced and a
  warning is logged.
- **fps**: **24, 25, 48, or 50**—scene ``fps`` is clamped to the nearest value.
- **resolution**: **1080p**, **1440p**, or **2160p** (CLI ``--resolution``, default 1080p).
- **prompt** / **image_url**: required; start frame is uploaded via Fal (resized like other
  runners using ``dimensions`` from config).
- **end_image_url**: optional—if ``end_frame.png`` exists in the scene folder, it is uploaded
  and passed for start→end transitions.

The published Fast I2V schema does **not** include ``negative_prompt`` or ``seed``; those
are not sent by default. Use each scene's ``extra_params`` in ``config.json`` to pass any
additional API fields your account supports.

There is **no** ``audio_url`` (or similar) on this image-to-video endpoint: the model either
generates audio (``generate_audio=true``) or not. To reuse LTX's audio with **Wan 2.6 Flash**,
run ``extract_ltx_audio_to_inputs.py`` and then ``run_bm_v1_wan26_flash_i2v.py`` (Wan accepts
``audio_url``).

Outputs go to ``outputs/BM_v1_LTX23_Fast/<scene>/LTX-2.3-Fast/*.mp4`` plus metadata JSON.

Optional overrides via ``extra_params`` (merged into the Fal request), e.g. ``aspect_ratio``,
``generate_audio``, ``fps``, ``resolution``, ``duration``.

Usage
-----
    python run_bm_v1_ltx23_fast.py
    python run_bm_v1_ltx23_fast.py --scenes skye_chase_crosswalk_safety
    python run_bm_v1_ltx23_fast.py --resolution 1440p --outputs ./my_outputs
    python run_bm_v1_ltx23_fast.py --no-audio
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import fal_client
except ImportError:
    fal_client = None  # type: ignore[assignment, misc]

from benchmark_runner import (
    GenerationResult,
    _build_metadata,
    _clamp_duration,
    _get_fal_key,
    _resize_for_scene,
    _timestamp_suffix,
    add_label_to_video,
    discover_scenes,
    download_video,
    iter_scene_prompt_jobs,
    load_scene_config,
    logger,
)

PROJECT_ROOT = Path(__file__).resolve().parent
BM_V1_SCENES_DIR = PROJECT_ROOT / "inputs" / "BM_v1" / "benchmark_v1"
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs" / "BM_v1_LTX23_Fast"

FAL_LTX_MODEL_ID_DEFAULT = "fal-ai/ltx-2.3/image-to-video/fast"

LTX_DURATIONS = (6, 8, 10, 12, 14, 16, 18, 20)
LTX_FPS_CHOICES = (24, 25, 48, 50)
MODEL_LABEL = "LTX-2.3-Fast"


def _clamp_fps(fps: int) -> int:
    return min(LTX_FPS_CHOICES, key=lambda x: abs(x - fps))


def _apply_long_duration_constraints(
    duration: int,
    fps: int,
    resolution: str,
) -> tuple[int, str, list[str]]:
    """
    Durations > 10s require 25 FPS and 1080p per LTX-2.3 Fast docs.
    Returns (fps, resolution, list of warning messages).
    """
    warnings: list[str] = []
    if duration <= 10:
        return fps, resolution, warnings
    new_fps, new_res = fps, resolution
    if fps != 25:
        new_fps = 25
        warnings.append(
            f"duration={duration}s (>10) requires 25 FPS; adjusted from {fps} to 25."
        )
    if resolution != "1080p":
        new_res = "1080p"
        warnings.append(
            f"duration={duration}s (>10) requires 1080p; adjusted resolution from "
            f"{resolution} to 1080p."
        )
    return new_fps, new_res, warnings


def run_ltx23_fast(
    scene,
    prompt: str,
    output_dir: Path,
    prompt_id: str,
    *,
    fal_model_id: str,
    resolution: str,
    generate_audio: bool,
    ltx_extra: dict[str, Any] | None = None,
    duration_override_sec: int | None = None,
) -> GenerationResult:
    """Single Fal LTX-2.3 Fast image-to-video job."""
    if fal_client is None:
        return GenerationResult(success=False, error="fal_client not installed")

    api_key = _get_fal_key()
    if not api_key:
        return GenerationResult(success=False, error="FAL_KEY not set (env or api_config.json)")

    os.environ["FAL_KEY"] = api_key

    scene_id = scene.scene_name
    ts = _timestamp_suffix()
    output_name = f"{scene_id}_{prompt_id}_{MODEL_LABEL}_{ts}"
    output_path = output_dir / f"{output_name}.mp4"
    metadata_path = output_dir / f"{output_name}_metadata.json"

    start_time = time.perf_counter()
    api_response: dict[str, Any] = {}

    requested_sec = (
        int(duration_override_sec) if duration_override_sec is not None else int(scene.duration)
    )
    dur = _clamp_duration(requested_sec, LTX_DURATIONS)
    fps = _clamp_fps(int(scene.fps))
    res = resolution
    constraint_warnings: list[str] = []
    adj_fps, adj_res, constraint_warnings = _apply_long_duration_constraints(dur, fps, res)
    fps, res = adj_fps, adj_res
    for w in constraint_warnings:
        logger.warning(w)

    print(
        f"🎬 {MODEL_LABEL} (Fal) | Scene: {scene.scene_name} | Prompt: {prompt_id} | {dur}s ... Generating...",
        flush=True,
    )

    try:
        import tempfile

        img_bytes = _resize_for_scene(scene, scene.start_frame_path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        try:
            image_url = fal_client.upload_file(tmp_path)
        finally:
            os.unlink(tmp_path)

        args: dict[str, Any] = {
            "prompt": prompt,
            "image_url": image_url,
            "duration": dur,
            "resolution": res,
            "aspect_ratio": "auto",
            "fps": fps,
            "generate_audio": generate_audio,
        }

        if scene.end_frame_path:
            logger.info("Uploading end frame for LTX-2.3 Fast...")
            end_bytes = _resize_for_scene(scene, scene.end_frame_path)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(end_bytes)
                end_tmp = tmp.name
            try:
                args["end_image_url"] = fal_client.upload_file(end_tmp)
            finally:
                os.unlink(end_tmp)

        args.update(scene.extra_params)
        if ltx_extra:
            args.update(ltx_extra)

        # Re-apply long-clip rules if extra_params overrode duration/fps/resolution.
        final_dur = int(args.get("duration", dur))
        if final_dur not in LTX_DURATIONS:
            final_dur = _clamp_duration(final_dur, LTX_DURATIONS)
            args["duration"] = final_dur
        final_fps = int(args.get("fps", fps))
        final_res = str(args.get("resolution", res))
        adj2_fps, adj2_res, extra_warns = _apply_long_duration_constraints(
            final_dur, final_fps, final_res
        )
        for w in extra_warns:
            logger.warning(w)
        args["fps"] = adj2_fps
        args["resolution"] = adj2_res

        def on_queue_update(update):
            if hasattr(update, "logs") and update.logs:
                for log in update.logs:
                    msg = log.get("message") if isinstance(log, dict) else str(log)
                    if msg:
                        print(f"  📋 {msg}", flush=True)

        result = fal_client.subscribe(
            fal_model_id,
            arguments=args,
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        api_response = dict(result) if isinstance(result, dict) else {"raw": str(result)}

        video_data = result.get("video")
        if not video_data:
            raise RuntimeError(f"No 'video' in response: {result}")

        video_url = video_data.get("url")
        if not video_url:
            raise RuntimeError(f"No 'url' in video: {video_data}")

        download_video(video_url, output_path)
        has_audio = bool(args.get("generate_audio", True))
        add_label_to_video(output_path, MODEL_LABEL, has_audio)

        elapsed = time.perf_counter() - start_time
        print(f"✅ Success! Video saved to: {output_path} (Took: {elapsed:.1f} seconds).", flush=True)

        meta = _build_metadata(
            scene=scene,
            prompt=prompt,
            model_name=MODEL_LABEL,
            prompt_id=prompt_id,
            api_response=api_response,
            elapsed=elapsed,
            run_timestamp=ts,
        )
        meta["ltx_fal_model_id"] = fal_model_id
        meta["ltx_duration"] = args["duration"]
        meta["ltx_resolution"] = args["resolution"]
        meta["ltx_fps"] = args["fps"]
        meta["ltx_generate_audio"] = args.get("generate_audio", True)
        meta["ltx_aspect_ratio"] = args.get("aspect_ratio", "auto")
        meta["ltx_constraint_warnings"] = constraint_warnings + extra_warns
        meta["ltx_end_frame_used"] = bool(scene.end_frame_path)
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

        return GenerationResult(
            success=True,
            output_path=output_path,
            metadata_path=metadata_path,
            api_response=api_response,
            prompt=prompt,
            seed=scene.seed,
            execution_time_sec=elapsed,
        )

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.exception("%s failed: %s", MODEL_LABEL, e)
        meta = _build_metadata(
            scene=scene,
            prompt=prompt,
            model_name=MODEL_LABEL,
            prompt_id=prompt_id,
            api_response=api_response,
            elapsed=elapsed,
            error=str(e),
            run_timestamp=ts,
        )
        meta["ltx_fal_model_id"] = fal_model_id
        meta["ltx_duration"] = dur
        meta["ltx_resolution"] = res
        meta["ltx_fps"] = fps
        meta["ltx_generate_audio"] = generate_audio
        meta["ltx_constraint_warnings"] = constraint_warnings
        meta["ltx_end_frame_used"] = bool(scene.end_frame_path)
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)
        return GenerationResult(
            success=False,
            error=str(e),
            metadata_path=metadata_path,
            prompt=prompt,
            seed=scene.seed,
            execution_time_sec=elapsed,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run BM_v1 benchmarks with Fal LTX-2.3 I2V Fast."
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=BM_V1_SCENES_DIR,
        help="Benchmark scene root (default: inputs/BM_v1/benchmark_v1).",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        metavar="SCENE",
        help="Scene folder names (default: all under --scenes-dir).",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=DEFAULT_OUTPUTS,
        help="Output root (default: outputs/BM_v1_LTX23_Fast).",
    )
    parser.add_argument(
        "--resolution",
        choices=("1080p", "1440p", "2160p"),
        default="1080p",
        help="Video resolution (default: 1080p).",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Set generate_audio=false (default: model generates audio).",
    )
    parser.add_argument(
        "--fal-model-id",
        default=FAL_LTX_MODEL_ID_DEFAULT,
        help=f"Fal queue model id (default: {FAL_LTX_MODEL_ID_DEFAULT}).",
    )
    args = parser.parse_args()
    fal_model_id: str = args.fal_model_id
    generate_audio = not args.no_audio

    scenes_dir = Path(args.scenes_dir).resolve()
    if not scenes_dir.is_dir():
        print(f"ERROR: Scenes directory not found: {scenes_dir}", file=sys.stderr)
        return 1

    scenes = discover_scenes(scenes_dir)
    if args.scenes:
        allow = set(args.scenes)
        scenes = [p for p in scenes if p.name in allow]
    if not scenes:
        print("ERROR: No matching scenes found.", file=sys.stderr)
        return 1

    out_root = Path(args.outputs).resolve()
    results: list[GenerationResult] = []

    print(
        f"\n🚀 LTX-2.3 Fast | {len(scenes)} scene(s) | Fal model: {fal_model_id}\n",
        flush=True,
    )

    for scene_path in scenes:
        try:
            scene = load_scene_config(scene_path)
        except Exception as e:
            logger.error("Failed to load %s: %s", scene_path.name, e)
            continue

        output_dir = out_root / scene.scene_name / MODEL_LABEL
        output_dir.mkdir(parents=True, exist_ok=True)

        for job in iter_scene_prompt_jobs(scene):
            r = run_ltx23_fast(
                scene,
                job.full_prompt,
                output_dir,
                job.prompt_id,
                fal_model_id=fal_model_id,
                resolution=args.resolution,
                generate_audio=generate_audio,
            )
            results.append(r)
            if not r.success and r.error:
                short = r.error[:120] + ("..." if len(r.error) > 120 else "")
                print(f"❌ ERROR: {short}", flush=True)

    success_count = sum(1 for r in results if r.success)
    total = len(results)
    print(
        f"\n{'✅' if success_count == total else '⚠️'} "
        f"Done. {success_count}/{total} generation(s) succeeded.",
        flush=True,
    )
    return 0 if success_count == total else 1


if __name__ == "__main__":
    logging.getLogger(__name__).setLevel(logging.INFO)
    raise SystemExit(main())
