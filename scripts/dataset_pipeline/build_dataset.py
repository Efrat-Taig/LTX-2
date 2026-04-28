#!/usr/bin/env python3
"""Build a unified training dataset from a Golden Set manifest.

PILOT SCOPE: stages 1 (fetch manifest), 2 (download pre-cut clips),
3 (write prompts with brand-token substitution), 5 (render gallery),
6 (write metadata.json). Stage 4 (latent encoding) is intentionally
skipped — gallery is the deliverable for review.

Idempotent: re-running skips clips whose video + prompt + per-clip
gallery render already exist. Delete a clip's outputs to force re-build.

Example:
  python build_dataset.py \\
      --manifest gs://video_gen_dataset/dataset/labelbox/chase_golden.json \\
      --character chase \\
      --version 1 \\
      --output /home/efrattaig/training_data_pilot/chase_golden_v1_20260428
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from captions import build_caption_index, apply_brand_tokens
from gallery import (
    build_concat_list,
    check_ffmpeg,
    concat_gallery,
    render_per_clip,
)
from schema import (
    Captioning,
    Clip,
    DatasetMetadata,
    Encoding,
    Gallery,
    Source,
    Stats,
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def md5_of_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gcloud_cp(src_uri: str, dst: Path) -> tuple[bool, str]:
    """gcloud storage cp with one retry. Returns (ok, error_msg)."""
    for attempt in range(2):
        r = subprocess.run(
            ["gcloud", "storage", "cp", src_uri, str(dst), "--verbosity=error"],
            capture_output=True, text=True, check=False,
        )
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            return True, ""
        if attempt == 0:
            time.sleep(2)
    return False, (r.stderr or "").strip().splitlines()[-1] if r.stderr else "unknown"


def ffprobe_video(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        return {}
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        return {}
    vstreams = [s for s in d.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        return {}
    v = vstreams[0]
    fps = 0.0
    if v.get("avg_frame_rate"):
        try:
            num, den = v["avg_frame_rate"].split("/")
            if int(den) != 0:
                fps = int(num) / int(den)
        except (ValueError, ZeroDivisionError):
            pass
    return {
        "duration_s": float(d.get("format", {}).get("duration", 0)),
        "width": int(v.get("width", 0)),
        "height": int(v.get("height", 0)),
        "fps": fps,
    }


def git_commit(repo_root: Path) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return r.stdout.strip() if r.returncode == 0 else ""


def fetch_manifest(uri: str, dst: Path) -> tuple[Path, str]:
    """Download the manifest JSON. Returns (local_path, md5)."""
    if not dst.exists():
        ok, err = gcloud_cp(uri, dst)
        if not ok:
            raise RuntimeError(f"Failed to fetch manifest {uri}: {err}")
    return dst, md5_of_file(dst)


def log_line(audit_log: Path, msg: str) -> None:
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    with audit_log.open("a") as f:
        f.write(f"[{utc_now()}] {msg}\n")


def build(args: argparse.Namespace) -> int:
    check_ffmpeg()
    out_root = Path(args.output).resolve()
    videos_dir = out_root / "videos"
    prompts_dir = out_root / "prompts"
    gallery_dir = out_root / "gallery"
    per_clip_dir = gallery_dir / "per_clip"
    audit_log = out_root / "audit" / "build.log"
    cache_dir = out_root / ".cache"

    for d in (videos_dir, prompts_dir, per_clip_dir, audit_log.parent, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_line(audit_log, f"build start args={vars(args)}")

    # Stage 1: fetch manifest
    print(f"[1/5] Fetch manifest: {args.manifest}", flush=True)
    manifest_local, manifest_md5 = fetch_manifest(args.manifest, cache_dir / "manifest.json")
    entries = json.loads(manifest_local.read_text())
    print(f"      → {len(entries)} entries; md5={manifest_md5[:12]}", flush=True)

    # Stage 1b: build caption index from all known parquets
    print("[1b]  Build caption index from parquets…", flush=True)
    cap_idx = build_caption_index(cache_dir / "parquets")
    print(f"      → {len(cap_idx)} unique URL→caption pairs across parquets", flush=True)

    # Stage 2 + 3 (interleaved per clip): download mp4, ffprobe, write prompt
    print(f"[2-3] Download {len(entries)} clips + write prompts…", flush=True)
    clips: list[Clip] = []
    missing_caption = 0
    download_fail = 0
    for i, entry in enumerate(entries, start=1):
        idx = i  # 1-based, stable from manifest order
        data = entry.get("data", {})
        meta = data.get("meta", {})
        url = data.get("video", "")
        speaker = meta.get("speaker", "")
        scene_fn = meta.get("scene_filename") or ""
        clip = Clip(
            index=idx,
            source_url=url or "",
            source_episode=scene_fn.rsplit("-Scene-", 1)[0] if scene_fn else "",
            source_season=int(meta.get("season_number") or 0),
            source_scene_number=float(meta.get("scene_number") or 0),
            speaker=speaker or "",
            video=f"videos/{idx:04d}.mp4",
            prompt_path=f"prompts/{idx:04d}.txt",
        )
        video_local = videos_dir / f"{idx:04d}.mp4"
        prompt_local = prompts_dir / f"{idx:04d}.txt"

        # Caption resolution
        cap_entry = cap_idx.get(url)
        if cap_entry is None:
            clip.status = "active"
            clip.prompt = ""
            missing_caption += 1
        else:
            clip.prompt = apply_brand_tokens(cap_entry.caption)
        if not prompt_local.exists() and clip.prompt:
            prompt_local.write_text(clip.prompt + "\n")

        # Download (skip if cached)
        if not video_local.exists() or video_local.stat().st_size == 0:
            ok, err = gcloud_cp(url, video_local)
            if not ok:
                clip.status = "active"
                clip.video_md5 = ""
                download_fail += 1
                log_line(audit_log, f"DOWNLOAD_FAIL idx={idx} url={url} err={err}")
                clips.append(clip)
                if i % 25 == 0 or i == len(entries):
                    print(f"      [{i}/{len(entries)}] missing_caption={missing_caption} dl_fail={download_fail}", flush=True)
                continue

        # ffprobe
        meta_v = ffprobe_video(video_local)
        clip.duration_s = meta_v.get("duration_s", 0.0)
        clip.width = meta_v.get("width", 0)
        clip.height = meta_v.get("height", 0)
        clip.fps = meta_v.get("fps", 0.0)
        clip.video_md5 = md5_of_file(video_local)
        clips.append(clip)
        if i % 25 == 0 or i == len(entries):
            print(f"      [{i}/{len(entries)}] missing_caption={missing_caption} dl_fail={download_fail}", flush=True)

    # Stage 5: per-clip gallery render
    print(f"[5a]  Render per-clip gallery overlays…", flush=True)
    for i, clip in enumerate(clips, start=1):
        if clip.status != "active":
            continue
        if not clip.prompt:
            continue  # no caption → exclude from gallery for now
        src = videos_dir / f"{clip.index:04d}.mp4"
        if not src.exists():
            continue
        dst = per_clip_dir / f"{clip.index:04d}.mp4"
        if dst.exists() and dst.stat().st_size > 0:
            continue  # idempotent skip
        try:
            render_per_clip(src, clip.prompt, clip.index, dst)
        except subprocess.CalledProcessError as e:
            log_line(audit_log, f"GALLERY_RENDER_FAIL idx={clip.index} err={e}")
        if i % 25 == 0 or i == len(clips):
            print(f"      [{i}/{len(clips)}] rendered", flush=True)

    print("[5b]  Concat gallery…", flush=True)
    active_idxs = [c.index for c in clips if c.status == "active" and c.prompt]
    concat_list = gallery_dir / "concat.txt"
    build_concat_list(active_idxs, per_clip_dir, concat_list)
    gallery_path = gallery_dir / "gallery.mp4"
    try:
        concat_gallery(concat_list, gallery_path)
        rendered_idx_count = sum(1 for c in clips if (per_clip_dir / f"{c.index:04d}.mp4").exists())
    except subprocess.CalledProcessError as e:
        log_line(audit_log, f"GALLERY_CONCAT_FAIL err={e}")
        rendered_idx_count = 0

    # Stage 6: metadata.json
    print("[6]   Write metadata.json…", flush=True)
    md = DatasetMetadata(
        dataset_id=Path(out_root).name,
        character=args.character,
        version=args.version,
        created_at=utc_now(),
        git_commit=git_commit(Path(__file__).resolve().parents[2]),
        source=Source(manifest=args.manifest, manifest_md5=manifest_md5),
        encoding=Encoding(
            resolution=(args.target_w, args.target_h),
            fps=args.target_fps,
            frames_per_clip=args.frames,
        ),
        captioning=Captioning(),
        stats=Stats(
            active_clips=sum(1 for c in clips if c.status == "active"),
            deleted_clips=0,
            total_duration_s=sum(c.duration_s for c in clips),
            missing_captions=missing_caption,
            download_failures=download_fail,
        ),
        clips=clips,
        gallery=Gallery(
            path="gallery/gallery.mp4",
            interclip_black_s=0.5,
            rendered_at=utc_now(),
            rendered_index_count=rendered_idx_count,
        ),
    )
    md.write(out_root / "metadata.json")

    # prompts/all.csv (active only) for bulk view
    all_csv = prompts_dir / "all.csv"
    with all_csv.open("w") as f:
        f.write("index,prompt\n")
        for c in clips:
            if c.status == "active" and c.prompt:
                p = c.prompt.replace('"', '""').replace("\n", " ").strip()
                f.write(f'{c.index},"{p}"\n')

    log_line(audit_log, f"build complete clips={len(clips)} active={md.stats.active_clips} "
                        f"missing_cap={missing_caption} dl_fail={download_fail} "
                        f"gallery_idx={rendered_idx_count}")
    print()
    print(f"DONE.  {out_root}")
    print(f"  clips:           {len(clips)} ({md.stats.active_clips} active)")
    print(f"  missing caption: {missing_caption}")
    print(f"  download fail:   {download_fail}")
    print(f"  gallery:         {gallery_path} ({rendered_idx_count} indices rendered)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="gs:// URL of the Golden Set JSON manifest")
    ap.add_argument("--character", required=True, choices=["chase", "skye"])
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--output", required=True, help="Local output directory")
    ap.add_argument("--target-w", type=int, default=960)
    ap.add_argument("--target-h", type=int, default=544)
    ap.add_argument("--target-fps", type=int, default=24)
    ap.add_argument("--frames", type=int, default=49)
    args = ap.parse_args()
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
