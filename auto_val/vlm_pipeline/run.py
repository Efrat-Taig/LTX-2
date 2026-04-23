#!/usr/bin/env python3
"""
VLM Video QA Pipeline — 10 Gemini 2.5 Pro agents on a video, heavily
parallelised: grid-independent agents and Gemini-Flash character detection
run concurrently, so grid-dependent agents start as soon as detection returns.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from . import agents as A
from . import detection as DET
from . import keyframes as KF
from .perception import transcribe_with_segments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM Video QA Pipeline")
    p.add_argument("video_path")
    p.add_argument("--prompt", default="")
    p.add_argument("--prompt-file", default="")
    p.add_argument("--out-dir", default="./outputs_vlm")
    p.add_argument(
        "--enable-prompt-reasoning",
        action="store_true",
        help="Run the downstream Prompt Reasoning Agent (Deduction Engine) to "
             "propose a rewritten prompt that removes semantic triggers behind "
             "the observed failures. Off by default; adds one Gemini 2.5 Pro "
             "call after the aggregator.",
    )
    return p.parse_args()


# Agents that need nothing from character detection — they can start immediately,
# overlapping with the Flash detection call.
GRID_INDEPENDENT = {"motion_weight", "audio_quality", "speech_coherence", "prompt_fidelity"}


def run_video(
    video_path: str,
    prompt_text: str,
    out_dir: Path,
    enable_prompt_reasoning: bool = False,
) -> dict:
    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    stem = Path(video_path).stem
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"{stem[:40]}__{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    local_video = run_dir / Path(video_path).name
    shutil.copy2(video_path, local_video)

    t_total_start = time.time()

    # ── Perception: Whisper transcript + timestamped segments ───────────────
    logger.info("Transcribing audio …")
    transcript, segments = transcribe_with_segments(str(local_video))
    logger.info(f"Transcript: {transcript[:120]!r} ({len(segments)} segments)")

    # ── Extract 16 keyframes + up to 4×2 speech-pair frames (all at once) ───
    logger.info("Extracting keyframes + speech-pair frames …")
    frames16 = KF.extract_evenly_spaced(str(local_video), n=16)
    chosen_segs = segments[:4] if segments else []
    speech_frames_flat: list = []
    for s in chosen_segs:
        t_start = s.start + (s.end - s.start) * 0.15
        t_end   = s.start + (s.end - s.start) * 0.85
        speech_frames_flat.extend(KF.extract_at_timestamps(str(local_video), [t_start, t_end]))
    all_frames = frames16 + speech_frames_flat

    speech_captions: list[str] = [s.text.strip()[:80] for s in chosen_segs]

    if segments:
        segments_block = "\n".join(
            f"  [{s.start:.2f}s – {s.end:.2f}s] {s.text.strip()}" for s in segments
        )
    else:
        segments_block = "(no segments)"

    # Read video bytes (used by every agent)
    with open(local_video, "rb") as f:
        video_bytes = f.read()

    A.prewarm_client()

    # ── PARALLEL PHASE ──────────────────────────────────────────────────────
    # 1. Start Gemini-Flash detection on ALL frames (main + speech) in one batch.
    # 2. Start the 4 grid-independent agents immediately (they don't need bboxes).
    # 3. Once detection returns, build grids and launch the 6 grid-dependent agents.
    # 4. Collect all 10 agent reports.

    # Shared state
    reports_lock = threading.Lock()
    reports: list[A.AgentReport] = []

    pool = cf.ThreadPoolExecutor(max_workers=max(len(A.AGENTS) + 2, 14))
    agent_futures: dict = {}

    # Agent runner factory (shared by grid-independent and grid-dependent)
    def _runner(name, fn, kwargs):
        try:
            return fn(video_bytes, **kwargs)
        except Exception as exc:
            logger.error(f"{name} failed: {exc}")
            return A.AgentReport(
                agent_id=name, title=name, focus="",
                verdict="great", summary=f"(agent error: {exc})",
                findings=str(exc), elapsed_s=0.0,
            )

    # Submit grid-independent agents FIRST (so they run concurrently with detection)
    grid_indep_kwargs_pool = {
        "transcript":     transcript,
        "segments_block": segments_block,
        "prompt_text":    prompt_text,
    }
    for name, fn, needs in A.AGENTS:
        if name in GRID_INDEPENDENT:
            kwargs = {k: grid_indep_kwargs_pool[k] for k in needs}
            agent_futures[pool.submit(_runner, name, fn, kwargs)] = name

    # Start Flash detection in parallel (single big batch of up to 24 frames)
    t_det0 = time.time()
    logger.info(f"Starting Flash detection on {len(all_frames)} frames (parallel with {len(agent_futures)} agents) …")
    det_future = pool.submit(DET.detect_batch, all_frames, 16)  # max_workers=16

    # Block on detection (other threads keep running agents)
    bboxes_per_frame = det_future.result()
    main_bboxes = bboxes_per_frame[:len(frames16)]
    speech_bboxes = bboxes_per_frame[len(frames16):]
    logger.info(
        f"Detection done in {time.time()-t_det0:.1f}s "
        f"({sum(1 for b in main_bboxes if b)}/{len(main_bboxes)} main frames detected)"
    )

    # ── Compose grids (fast, CPU-only) ──────────────────────────────────────
    keyframe_grid_img = KF.compose_grid(frames16, cols=4, panel_width=480)
    keyframe_grid_img.save(run_dir / "keyframe_grid.png")
    keyframe_grid_bytes = KF.image_to_png_bytes(keyframe_grid_img)

    body_pairs = []
    for (ts, img), bboxes in zip(frames16, main_bboxes):
        body_bbox = DET.largest_body_bbox(bboxes)
        body_pairs.append((ts, DET.crop_with_padding(img, body_bbox, padding=0.15)))
    body_grid_img = KF.compose_grid(body_pairs, cols=4, panel_width=480)
    body_grid_img.save(run_dir / "body_grid.png")
    body_grid_bytes = KF.image_to_png_bytes(body_grid_img)

    face_pairs = []
    for (ts, img), bboxes in zip(frames16, main_bboxes):
        face_bbox = DET.largest_face_bbox(bboxes)
        face_pairs.append((
            ts,
            DET.crop_with_padding(
                img, face_bbox, padding=0.25,
                fallback_region=(0.25, 0.15, 0.75, 0.55),
            ),
        ))
    face_grid_img = KF.compose_grid(face_pairs, cols=4, panel_width=384)
    face_grid_img.save(run_dir / "face_grid.png")
    face_grid_bytes = KF.image_to_png_bytes(face_grid_img)

    speech_grid_bytes = None
    if speech_frames_flat:
        speech_pair_images = []
        for (ts, img), bboxes in zip(speech_frames_flat, speech_bboxes):
            face_bbox = DET.largest_face_bbox(bboxes)
            speech_pair_images.append((
                ts,
                DET.crop_with_padding(img, face_bbox, padding=0.3,
                                      fallback_region=(0.25, 0.15, 0.75, 0.55)),
            ))
        speech_grid_img = KF.compose_grid(speech_pair_images, cols=2, panel_width=640)
        speech_grid_img.save(run_dir / "speech_grid.png")
        speech_grid_bytes = KF.image_to_png_bytes(speech_grid_img)
        logger.info(f"Built speech grid with {len(speech_pair_images)} panels ({len(chosen_segs)} rows × 2)")

    # Submit grid-dependent agents now that grids are ready
    grid_dep_kwargs_pool = {
        "keyframe_grid":   keyframe_grid_bytes,
        "body_grid":       body_grid_bytes,
        "face_grid":       face_grid_bytes,
        "speech_grid":     speech_grid_bytes,
        "speech_captions": speech_captions,
    }
    for name, fn, needs in A.AGENTS:
        if name in GRID_INDEPENDENT:
            continue
        kwargs = {k: grid_dep_kwargs_pool[k] for k in needs}
        agent_futures[pool.submit(_runner, name, fn, kwargs)] = name

    # Collect all 10 agent results
    t_agents0 = time.time()
    for fut in cf.as_completed(agent_futures):
        r = fut.result()
        with reports_lock:
            reports.append(r)
        logger.info(f"✓ {r.title} — {r.verdict} ({r.elapsed_s}s)")
    pool.shutdown(wait=True)
    logger.info(f"All agents finished (wall-clock from detection start: {time.time()-t_det0:.1f}s)")

    order = {name: i for i, (name, _, _) in enumerate(A.AGENTS)}
    reports.sort(key=lambda r: order.get(r.agent_id, 99))

    # ── Aggregator ──────────────────────────────────────────────────────────
    logger.info("Running aggregator …")
    t0 = time.time()
    agg = A.run_aggregator(reports, prompt_text)
    logger.info(f"Aggregator done in {time.time()-t0:.1f}s — {agg['overall_verdict']} {agg['score']}")

    # ── Optional Prompt Reasoner ────────────────────────────────────────────
    prompt_reasoner_output = None
    if enable_prompt_reasoning:
        logger.info("Running Prompt Reasoner (Deduction Engine) …")
        t0 = time.time()
        prompt_reasoner_output = A.run_prompt_reasoner(
            video_bytes, reports, agg, prompt_text,
        )
        n_edits = len(prompt_reasoner_output.get("counterfactual_edits", []))
        skipped = prompt_reasoner_output.get("skipped_reason")
        if skipped:
            logger.info(f"Prompt Reasoner skipped ({time.time()-t0:.1f}s) — {skipped}")
        else:
            logger.info(f"Prompt Reasoner done in {time.time()-t0:.1f}s — {n_edits} edit(s) proposed")

    # Persist detected bboxes
    detections_dump = [
        [{"name": b.name, "body": list(b.body), "face": list(b.face) if b.face else None}
         for b in per_frame]
        for per_frame in bboxes_per_frame
    ]

    payload = {
        "video_file": local_video.name,
        "prompt": prompt_text,
        "transcript": transcript,
        "speech_captions": speech_captions,
        "timestamp": tag,
        "detections": detections_dump,
        "agents": [r.to_dict() for r in reports],
        "aggregator": agg,
        "prompt_reasoner": prompt_reasoner_output,
        "wall_clock_seconds": round(time.time() - t_total_start, 1),
    }
    (run_dir / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    (run_dir / "report.md").write_text(_markdown_report(payload))
    logger.info(f"Total wall-clock: {payload['wall_clock_seconds']}s")
    return payload


def _markdown_report(p: dict) -> str:
    agg = p["aggregator"]
    out = [
        f"# {p['video_file']}",
        "",
        f"**Verdict:** {agg['overall_verdict']}  ·  **Score:** {agg['score']}/100  ·  **Wall-clock:** {p.get('wall_clock_seconds', '?')}s",
        "",
        "### Headline", agg["headline"], "",
        "### What works", agg["what_works"], "",
        "### What breaks", agg["what_breaks"], "",
        "### Top issue", agg["top_issue"], "",
        "---",
        "## Agent analyses", "",
    ]
    for r in p["agents"]:
        out += [
            f"### {r['title']}  _[{r['verdict']}]_",
            f"**{r['summary']}**",
            "",
            r["findings"],
            "",
        ]
    pr = p.get("prompt_reasoner")
    if pr:
        out += ["---", "## Prompt Reasoning (Deduction Engine)", ""]
        if pr.get("skipped_reason"):
            out += [f"_Skipped: {pr['skipped_reason']}_", ""]
        else:
            if pr.get("rationale"):
                out += ["### Rationale", pr["rationale"], ""]
            conflicts = pr.get("diagnosed_conflicts") or []
            if conflicts:
                out += ["### Diagnosed conflicts (intent vs outcome)", ""]
                for c in conflicts:
                    cap = " _(capacity-limited — prompt fix won't help)_" if c.get("is_capacity_limited") else ""
                    out += [
                        f"- **Intent:** {c.get('intent','')}",
                        f"  **Outcome:** {c.get('outcome','')} _(source: {c.get('source_agent','')})_{cap}",
                    ]
                out += [""]
            triggers = pr.get("trigger_attributions") or []
            if triggers:
                out += ["### Trigger attributions", ""]
                for t in triggers:
                    out += [
                        f"- **Visible element:** {t.get('visible_element','')}  "
                        f"(confidence: {t.get('confidence','')})",
                        f"  **Prompt phrase:** \"{t.get('prompt_phrase','')}\"",
                        f"  **Hypothesised bias:** {t.get('hypothesized_bias','')}",
                    ]
                out += [""]
            edits = pr.get("counterfactual_edits") or []
            if edits:
                out += ["### Counterfactual edits", ""]
                for e in edits:
                    out += [
                        f"- **{e.get('operation','').upper()}** \"{e.get('from','')}\" → \"{e.get('to','')}\"",
                        f"  _{e.get('reasoning','')}_",
                    ]
                out += [""]
            if pr.get("rewritten_prompt"):
                out += ["### Rewritten prompt", "```", pr["rewritten_prompt"], "```", ""]
    out += ["---", "### Transcript", "```", p["transcript"], "```"]
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    prompt_text = args.prompt
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  VLM Video QA — {os.path.basename(args.video_path)}")
    print(f"{'=' * 60}\n")
    run_video(
        args.video_path,
        prompt_text,
        out_dir,
        enable_prompt_reasoning=args.enable_prompt_reasoning,
    )


if __name__ == "__main__":
    main()
