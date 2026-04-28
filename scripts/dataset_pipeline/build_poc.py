#!/usr/bin/env python3
"""POC build — 5 clips per character, parquet-as-source-of-truth.

Per user direction (2026-04-28): the parquet drives everything; videos/ is
reconstructed from scratch by downloading each row's source URL into
0NNN.mp4.

Skye: skye_golden_dataset_v2.parquet (111 rows) gives the curation + caption.
      Its own URLs are stale, so we join to skye_all_seasons_…augmented on
      (season_number, episode_number, scene_number, internal_episode,
       speaker='SKYE') and take the first matching aug row's URL.

Chase: chase_golden.json (605 rows) is the curation. Captions come from
       chase_all_seasoned_results_filtered.parquet keyed by URL.
       (chase_golden_dataset.parquet at the labelbox root is mis-named:
        contains only Skye rows — do not use.)

Output layout (per `docs/training_data_pipeline.md`):
    {LOCAL_OUT}/{character}_golden_v{N}/
        videos/0001.mp4 … 000N.mp4
        prompts/0001.txt … 000N.txt
        prompts/all.csv
        metadata.json

After local writes, optionally rsync to:
    gs://video_gen_dataset/TinyStories/training_data/{character}_golden_v{N}/
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "dataset_pipeline"))

GCS_TARGET_ROOT = "gs://video_gen_dataset/TinyStories/training_data"
LOCAL_PARQUET_CACHE = Path("/tmp/manifests")  # already populated locally


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gcs_cp(src: str, dst: Path) -> tuple[bool, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["gcloud", "storage", "cp", src, str(dst), "--verbosity=error"],
        capture_output=True, text=True, check=False,
    )
    ok = r.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    return ok, (r.stderr.strip() if r.stderr else "")


def ffprobe_video(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
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
            if int(den):
                fps = int(num) / int(den)
        except (ValueError, ZeroDivisionError):
            pass
    return {
        "duration_s": float(d.get("format", {}).get("duration", 0)),
        "width": int(v.get("width", 0)),
        "height": int(v.get("height", 0)),
        "fps": fps,
    }


@dataclass
class POCClip:
    index: int
    source_url: str = ""
    source_episode: str = ""
    source_season: int = 0
    source_scene_number: float = 0.0
    speaker: str = ""
    video: str = ""
    video_md5: str = ""
    prompt: str = ""
    prompt_path: str = ""
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    download_ok: bool = True
    note: str = ""


def build_skye(out_root: Path, limit: int) -> list[POCClip]:
    v2 = pq.read_table(LOCAL_PARQUET_CACHE / "skye_v2.parquet").to_pylist()
    aug = pq.read_table(LOCAL_PARQUET_CACHE / "skye_all_seasons_results_filte.parquet").to_pylist()
    if limit <= 0:
        limit = len(v2)
    aug_by_key: dict[tuple, list[dict]] = {}
    for r in aug:
        k = (r.get("season_number"), r.get("episode_number"),
             r.get("scene_number"), r.get("internal_episode"), r.get("speaker"))
        aug_by_key.setdefault(k, []).append(r)

    clips: list[POCClip] = []
    for i, row in enumerate(v2[:limit], start=1):
        key = (row.get("season_number"), row.get("episode_number"),
               row.get("scene_number"), row.get("internal_episode"), "SKYE")
        aug_matches = aug_by_key.get(key, [])
        if not aug_matches:
            clips.append(POCClip(index=i, note=f"no aug match for key={key}"))
            print(f"  [{i}] NO_AUG_MATCH key={key}", flush=True)
            continue
        url = aug_matches[0]["output_video_path"]
        # Persist the raw caption (with the original character name). Brand-
        # token substitution is applied just-in-time at gallery render and at
        # trainer-CSV emit. See scripts/dataset_pipeline/brand_tokens.yaml.
        caption = row.get("scene_caption") or ""

        clip = POCClip(
            index=i,
            source_url=url,
            source_episode=(row.get("scene_filename") or "").rsplit("-Scene-", 1)[0],
            source_season=int(row.get("season_number") or 0),
            source_scene_number=float(row.get("scene_number") or 0),
            speaker="SKYE",
            video=f"videos/{i:04d}.mp4",
            prompt=caption,
            prompt_path=f"prompts/{i:04d}.txt",
        )
        video_local = out_root / clip.video
        ok, err = gcs_cp(url, video_local)
        clip.download_ok = ok
        if not ok:
            clip.note = f"download fail: {err[:120]}"
            print(f"  [{i}] DL_FAIL  url={url}  err={err[:80]}", flush=True)
        else:
            meta = ffprobe_video(video_local)
            clip.duration_s = meta.get("duration_s", 0.0)
            clip.width = meta.get("width", 0)
            clip.height = meta.get("height", 0)
            clip.fps = meta.get("fps", 0.0)
            clip.video_md5 = md5_file(video_local)
            print(f"  [{i}] OK  {clip.width}x{clip.height}@{clip.fps:.2f} {clip.duration_s:.2f}s  "
                  f"src={url.split('/')[-1]}", flush=True)
        clips.append(clip)
    return clips


def build_chase(out_root: Path, limit: int) -> list[POCClip]:
    js = json.loads((LOCAL_PARQUET_CACHE / "chase.json").read_text())
    chase_aug = pq.read_table(LOCAL_PARQUET_CACHE / "chase_all_seasoned_results_fil.parquet").to_pylist()
    if limit <= 0:
        limit = len(js)
    cap_by_url = {r["output_video_path"]: r["scene_caption"]
                  for r in chase_aug if r.get("output_video_path") and r.get("scene_caption")}

    clips: list[POCClip] = []
    for i, entry in enumerate(js[:limit], start=1):
        data = entry.get("data", {})
        meta = data.get("meta", {}) or {}
        url = data.get("video", "") or ""
        caption = cap_by_url.get(url, "")
        if not caption:
            clips.append(POCClip(index=i, source_url=url, note="no caption found"))
            print(f"  [{i}] NO_CAPTION  url={url}", flush=True)
            continue
        # Caption stays raw ("Chase ..."); substitution is just-in-time.

        scene_fn = meta.get("scene_filename") or ""
        clip = POCClip(
            index=i,
            source_url=url,
            source_episode=scene_fn.rsplit("-Scene-", 1)[0] if scene_fn else "",
            source_season=int(meta.get("season_number") or 0),
            source_scene_number=float(meta.get("scene_number") or 0),
            speaker="CHASE",
            video=f"videos/{i:04d}.mp4",
            prompt=caption,
            prompt_path=f"prompts/{i:04d}.txt",
        )
        video_local = out_root / clip.video
        ok, err = gcs_cp(url, video_local)
        clip.download_ok = ok
        if not ok:
            clip.note = f"download fail: {err[:120]}"
            print(f"  [{i}] DL_FAIL  url={url}  err={err[:80]}", flush=True)
        else:
            meta_v = ffprobe_video(video_local)
            clip.duration_s = meta_v.get("duration_s", 0.0)
            clip.width = meta_v.get("width", 0)
            clip.height = meta_v.get("height", 0)
            clip.fps = meta_v.get("fps", 0.0)
            clip.video_md5 = md5_file(video_local)
            print(f"  [{i}] OK  {clip.width}x{clip.height}@{clip.fps:.2f} {clip.duration_s:.2f}s  "
                  f"src={url.split('/')[-1]}", flush=True)
        clips.append(clip)
    return clips


def write_metadata(out_root: Path, character: str, version: int, source_label: str,
                   clips: list[POCClip]) -> None:
    md = {
        "dataset_id": out_root.name,
        "character": character,
        "version": version,
        "poc": True,
        "limit_used": len(clips),
        "created_at": utc_now(),
        "source": {
            "manifest_label": source_label,
            "raw_footage_root": "gs://video_gen_dataset/raw/paw_patrol/",
        },
        "captioning": {
            "version": "v1",
            "brand_token_format": "CHASE_PP / SKYE_PP",
            "substitution_at_build_time": True,
        },
        "stats": {
            "active_clips": sum(1 for c in clips if c.download_ok and c.prompt),
            "missing_clips": sum(1 for c in clips if not c.download_ok or not c.prompt),
        },
        "clips": [asdict(c) for c in clips],
    }
    (out_root / "metadata.json").write_text(json.dumps(md, indent=2, ensure_ascii=False))

    # dataset.csv — single rich CSV. Doubles as trainer input (LTX-2's
    # process_dataset.py reads `caption` + `video_path` columns) and the
    # human-readable bulk review. Other columns are inert to the trainer
    # but kept for traceability.
    csv = out_root / "dataset.csv"
    with csv.open("w") as f:
        f.write("index,caption,video_path,speaker,duration_s,width,height,fps,source_episode,source_url,video_md5\n")
        for c in clips:
            if not (c.download_ok and c.prompt):
                continue
            cap = c.prompt.replace('"', '""').replace("\n", " ").strip()
            ep = (c.source_episode or "").replace('"', '""')
            url = (c.source_url or "").replace('"', '""')
            f.write(
                f'{c.index},"{cap}","{c.video}","{c.speaker}",{c.duration_s:.3f},'
                f'{c.width},{c.height},{c.fps:.3f},"{ep}","{url}",{c.video_md5}\n'
            )


def upload_to_gcs(local_root: Path, gcs_dst: str) -> None:
    print(f"\nUploading {local_root} → {gcs_dst} …", flush=True)
    r = subprocess.run(
        ["gcloud", "storage", "rsync", str(local_root), gcs_dst,
         "--recursive", "--verbosity=error"],
        capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        print(f"  upload error: {r.stderr.strip().splitlines()[-1] if r.stderr else '?'}")
    else:
        print(f"  upload OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", required=True, choices=["chase", "skye", "both"])
    ap.add_argument("--limit", type=int, default=5, help="Cap clips processed (0 or negative = all entries in manifest)")
    ap.add_argument("--local-out", default="/tmp/training_data_poc")
    ap.add_argument("--no-upload", action="store_true",
                    help="Skip GCS upload (just produce locally)")
    args = ap.parse_args()

    targets: list[tuple[str, int, str, str]] = []
    if args.character in ("skye", "both"):
        targets.append(("skye", 2, "skye_golden_v2",
                        "skye_golden_dataset_v2.parquet (curation+captions) joined to skye_all_seasons_aug.parquet (URLs)"))
    if args.character in ("chase", "both"):
        targets.append(("chase", 1, "chase_golden_v1",
                        "chase_golden.json (URLs) + chase_all_seasoned_results_filtered.parquet (captions)"))

    for character, version, dataset_dir, source_label in targets:
        out_root = Path(args.local_out) / dataset_dir
        (out_root / "videos").mkdir(parents=True, exist_ok=True)
        print(f"\n=== POC build: {dataset_dir} (limit={args.limit}) ===", flush=True)
        if character == "skye":
            clips = build_skye(out_root, args.limit)
        else:
            clips = build_chase(out_root, args.limit)
        write_metadata(out_root, character, version, source_label, clips)
        ok = sum(1 for c in clips if c.download_ok and c.prompt)
        print(f"\n  → {ok}/{len(clips)} ready, written to {out_root}", flush=True)
        if not args.no_upload:
            upload_to_gcs(out_root, f"{GCS_TARGET_ROOT}/{dataset_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
