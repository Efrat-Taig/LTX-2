#!/usr/bin/env python3
"""Phase 0.1 — Data Audit (active_plan §3.1).

Walks local paths and GCS prefixes from docs/pre_flight.yaml. Emits:
  - docs/phase0/data_inventory.csv: one row per file
  - docs/phase0/duplicates.csv: hash-collision groups
  - docs/phase0/coverage.md: per-source counts, run-times, and skipped paths

Determinism: rows sorted by (source_kind, source_root, relative_path) so
two back-to-back runs produce byte-identical CSVs.

Hash policy: md5 for everything. Local files: streamed md5 in Python.
GCS files: use the object's md5Hash from the listing API (no download).
This trades sha256 for cross-source dedup that actually works.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import yaml
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent.parent
PRE_FLIGHT = REPO / "docs" / "pre_flight.yaml"
OUT_DIR = REPO / "docs" / "phase0"
INVENTORY_CSV = OUT_DIR / "data_inventory.csv"
DUPLICATES_CSV = OUT_DIR / "duplicates.csv"
COVERAGE_MD = OUT_DIR / "coverage.md"

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tiff", ".gif"}
CAPTION_EXTS = {".txt", ".json", ".csv", ".jsonl", ".srt", ".vtt", ".yaml", ".yml"}
CHECKPOINT_EXTS = {".safetensors", ".ckpt", ".pt", ".sft", ".bin", ".pth"}

CHARACTERS = [
    "chase", "skye", "marshall", "rubble", "rocky", "zuma", "everest",
    "ryder", "tracker", "tuck", "ella", "liberty", "wildcat",
]

SCENE_TYPE_HINTS = {
    "static": ["static", "still", "blink", "breath", "idle", "anchor"],
    "dynamic": ["dynamic", "march", "run", "running", "fly", "flying", "lift", "bubble", "backflip", "sneeze", "jump"],
    "closeup": ["closeup", "close_up", "face", "paw", "paws", "head", "eye", "snout"],
}

EPISODE_RE = re.compile(r"(?i)(?:s|season)[\s_-]?(\d{1,2})[\s_-]?(?:e|ep|episode)[\s_-]?(\d{1,3})")
EPISODE_FALLBACK_RE = re.compile(r"(?i)episode[\s_-]?(\d{1,3})")
HASH_BUF = 1 << 20  # 1 MiB streaming buffer


@dataclass
class Row:
    path: str
    type: str
    duration_s: str = ""
    width: str = ""
    height: str = ""
    fps: str = ""
    codec: str = ""
    file_hash: str = ""
    size_mb: str = ""
    last_modified: str = ""
    guessed_character: str = ""
    guessed_scene_type: str = ""
    source_episode_guess: str = ""
    notes: str = ""

    @staticmethod
    def header() -> list[str]:
        return [
            "path", "type", "duration_s", "width", "height", "fps", "codec",
            "file_hash", "size_mb", "last_modified",
            "guessed_character", "guessed_scene_type", "source_episode_guess", "notes",
        ]


@dataclass
class CoverageEntry:
    source_kind: str           # "local" | "gcs"
    source_root: str           # path or gs://bucket/[prefix]
    file_count: int = 0
    bytes_total: int = 0
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)


def classify(path: str) -> str:
    p = path.lower()
    suffix = Path(p).suffix
    if suffix in VIDEO_EXTS:
        return "video"
    if suffix in IMAGE_EXTS:
        return "image"
    if suffix in CHECKPOINT_EXTS:
        return "checkpoint"
    if suffix in CAPTION_EXTS:
        return "caption"
    return "other"


def guess_character(path: str) -> str:
    p = path.lower()
    hits = [c for c in CHARACTERS if c in p]
    if not hits:
        if "pawpatrol" in p or "paw_patrol" in p or "/pp/" in p or "_pp_" in p:
            return "paw_patrol_other"
        return ""
    return ",".join(sorted(hits))


def guess_scene_type(path: str) -> str:
    p = path.lower()
    for label, kws in SCENE_TYPE_HINTS.items():
        if any(kw in p for kw in kws):
            return label
    return ""


def guess_episode(path: str) -> str:
    m = EPISODE_RE.search(path)
    if m:
        return f"S{int(m.group(1)):02d}E{int(m.group(2)):03d}"
    m = EPISODE_FALLBACK_RE.search(path)
    if m:
        return f"E{int(m.group(1)):03d}"
    return ""


# --- Local file scanning -----------------------------------------------------

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            buf = f.read(HASH_BUF)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def ffprobe_video(path: Path) -> dict:
    try:
        cmd = [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        if out.returncode != 0:
            return {"error": out.stderr.strip().splitlines()[-1] if out.stderr else "ffprobe failed"}
        data = json.loads(out.stdout)
        vstreams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
        if not vstreams:
            return {"error": "no video stream"}
        v = vstreams[0]
        fps = ""
        if v.get("avg_frame_rate"):
            try:
                num, den = v["avg_frame_rate"].split("/")
                if int(den) != 0:
                    fps = f"{int(num) / int(den):.3f}"
            except (ValueError, ZeroDivisionError):
                pass
        return {
            "duration_s": data.get("format", {}).get("duration", ""),
            "width": str(v.get("width", "")),
            "height": str(v.get("height", "")),
            "fps": fps,
            "codec": v.get("codec_name", ""),
        }
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return {"error": f"ffprobe exception: {type(e).__name__}"}


def image_dims(path: Path) -> dict:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return {"width": str(im.width), "height": str(im.height)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"PIL: {type(e).__name__}"}


def walk_local(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts):
            yield p


def scan_local(root: Path, cov: CoverageEntry, do_hash: bool) -> Iterator[Row]:
    if not root.exists():
        cov.notes.append(f"path does not exist: {root}")
        return
    files = list(walk_local(root))
    for fp in tqdm(files, desc=f"local: {root}", unit="file", leave=False):
        try:
            stat = fp.stat()
        except OSError as e:
            yield Row(path=f"file://{fp}", type="other", notes=f"stat error: {e.strerror}")
            continue
        cov.file_count += 1
        cov.bytes_total += stat.st_size
        type_ = classify(str(fp))
        row = Row(
            path=f"file://{fp}",
            type=type_,
            size_mb=f"{stat.st_size / 1_048_576:.3f}",
            last_modified=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
            guessed_character=guess_character(str(fp)),
            guessed_scene_type=guess_scene_type(str(fp)),
            source_episode_guess=guess_episode(str(fp)),
        )
        if type_ == "video":
            meta = ffprobe_video(fp)
            if "error" in meta:
                row.notes = meta["error"]
            else:
                row.duration_s = str(meta["duration_s"])
                row.width = meta["width"]
                row.height = meta["height"]
                row.fps = meta["fps"]
                row.codec = meta["codec"]
        elif type_ == "image":
            meta = image_dims(fp)
            if "error" in meta:
                row.notes = meta["error"]
            else:
                row.width = meta["width"]
                row.height = meta["height"]
        if do_hash:
            try:
                row.file_hash = md5_file(fp)
            except OSError as e:
                row.notes = (row.notes + "; " if row.notes else "") + f"md5 failed: {e.strerror}"
        yield row


# --- GCS scanning ------------------------------------------------------------

def gcs_b64_md5_to_hex(b64: str) -> str:
    if not b64:
        return ""
    try:
        return base64.b64decode(b64).hex()
    except Exception:  # noqa: BLE001
        return ""


def _gcs_list_top_prefixes(uri: str) -> list[str]:
    """Non-recursive list of immediate children under uri (subdirs + objects).
    Used to subdivide a big bucket into smaller per-prefix listings so that no
    single `gcloud storage objects list` call has to stream a million objects
    (which we've observed stalls in practice)."""
    proc = subprocess.run(
        ["gcloud", "storage", "ls", uri, "--verbosity=error"],
        capture_output=True, text=True, check=False, timeout=120,
    )
    if proc.returncode != 0:
        return []
    out = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line == uri or line.endswith("/:") or not line.startswith("gs://"):
            continue
        out.append(line)
    return out


def _scan_gcs_chunk(pattern: str, bucket: str, cov: CoverageEntry, pbar: tqdm) -> Iterator[Row]:
    """Stream one `gcloud storage objects list <pattern>` invocation. stderr
    is drained to a tempfile to avoid the pipe-buffer deadlock that silently
    truncated earlier runs."""
    import tempfile

    cmd = [
        "gcloud", "storage", "objects", "list", pattern,
        "--verbosity=error",
        "--format=value(name,size,md5_hash,update_time,crc32c_hash,creation_time)",
    ]
    stderr_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, prefix="audit_stderr_", suffix=".log")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr_file, text=True, bufsize=1)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            while len(parts) < 6:
                parts.append("")
            name, size_s, md5_b64, updated, crc32c, created = parts[:6]
            try:
                size = int(size_s) if size_s else 0
            except ValueError:
                size = 0
            url = f"gs://{bucket}/{name}"
            cov.file_count += 1
            cov.bytes_total += size
            pbar.update(1)
            notes_parts: list[str] = []
            md5 = gcs_b64_md5_to_hex(md5_b64)
            if not md5 and crc32c:
                md5 = f"crc32c:{crc32c}"
                notes_parts.append("composite-upload (no md5)")
            yield Row(
                path=url,
                type=classify(url),
                file_hash=md5,
                size_mb=f"{size / 1_048_576:.3f}",
                last_modified=updated or created or "",
                guessed_character=guess_character(url),
                guessed_scene_type=guess_scene_type(url),
                source_episode_guess=guess_episode(url),
                notes="; ".join(notes_parts),
            )
        proc.wait(timeout=120)
        stderr_file.flush()
        stderr_file.seek(0)
        stderr_text = stderr_file.read()
        if proc.returncode != 0:
            tail = stderr_text.strip().splitlines()[-1] if stderr_text.strip() else "?"
            cov.notes.append(f"gcloud exit {proc.returncode} for {pattern}: {tail}")
    finally:
        if proc.poll() is None:
            proc.kill()
        stderr_file.close()


def scan_gcs(uri: str, cov: CoverageEntry) -> Iterator[Row]:
    """Scan a GCS uri (gs://bucket/ or gs://bucket/prefix/).

    Strategy: split into per-top-level-child invocations. Single mega-listings
    of huge buckets stall in practice (observed: >1h hang in a 562k-object
    bucket). Per-prefix listings bound each invocation's runtime and let us
    progress past a single bad prefix.
    """
    if not uri.startswith("gs://"):
        cov.notes.append(f"skipping non-gs uri: {uri}")
        return
    rest = uri[len("gs://"):]
    bucket = rest.split("/", 1)[0]

    # Discover top-level children. If empty (rare bucket-level edge), fall
    # back to a single recursive listing.
    children = _gcs_list_top_prefixes(uri)
    pbar = tqdm(desc=f"gcs:  {uri}", unit="obj", leave=False)
    try:
        if not children:
            yield from _scan_gcs_chunk(f"{uri}**" if uri.endswith("/") else f"{uri}/**", bucket, cov, pbar)
            return
        for child in children:
            # If child is itself a prefix (ends with '/'), list recursively
            # under it; if it's a leaf object, capture it as a single chunk.
            if child.endswith("/"):
                pattern = child + "**"
            else:
                # Single object: list just that name.
                pattern = child
            chunk_start = cov.file_count
            yield from _scan_gcs_chunk(pattern, bucket, cov, pbar)
            chunk_added = cov.file_count - chunk_start
            pbar.set_postfix_str(f"last={child.split('/')[-2 if child.endswith('/') else -1]}={chunk_added}")
    finally:
        pbar.close()


def gcs_count_truth(uri: str) -> int:
    """Independent ground-truth count of objects under uri. Same listing API,
    but only the name field — used for post-scan integrity verification."""
    if not uri.startswith("gs://"):
        return -1
    rest = uri[len("gs://"):]
    bucket = rest.split("/", 1)[0]
    prefix = rest[len(bucket) + 1:] if "/" in rest else ""
    pattern = f"gs://{bucket}/{prefix}**" if prefix else f"gs://{bucket}/**"
    cmd = ["gcloud", "storage", "objects", "list", pattern, "--verbosity=error", "--format=value(name)"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=1800)
    except subprocess.TimeoutExpired:
        return -1
    if proc.returncode != 0:
        return -1
    return sum(1 for line in proc.stdout.splitlines() if line.strip())


# --- Orchestration -----------------------------------------------------------

def load_pre_flight() -> dict:
    return yaml.safe_load(PRE_FLIGHT.read_text())


def collect_local_roots(cfg: dict) -> list[Path]:
    dp = cfg.get("data_paths", {}) or {}
    roots: list[Path] = []
    if dp.get("local_inputs_root") and dp["local_inputs_root"] != "TBD":
        roots.append(Path(dp["local_inputs_root"]).expanduser())
    for k in ("golden_set_150", "high_res_stills_partial"):
        v = dp.get(k)
        if v and v != "TBD":
            p = Path(v).expanduser()
            if p.exists():
                roots.append(p)
    for v in dp.get("extra_paths", []) or []:
        if v and v != "TBD":
            p = Path(v).expanduser()
            if p.exists():
                roots.append(p)
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        rr = r.resolve()
        if rr not in seen:
            seen.add(rr)
            out.append(rr)
    return out


def collect_gcs_uris(cfg: dict) -> list[str]:
    gcp = cfg.get("gcp", {}) or {}
    uris: list[str] = [f"gs://{b}/" for b in gcp.get("buckets_to_audit", []) or []]
    for u in gcp.get("gcs_known_data_prefixes", []) or []:
        if u not in uris and not any(u.startswith(p) for p in uris):
            uris.append(u)
    return uris


def write_inventory(rows: list[Row]) -> None:
    rows_sorted = sorted(rows, key=lambda r: (r.path.split("://", 1)[0], r.path))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with INVENTORY_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=Row.header())
        w.writeheader()
        for r in rows_sorted:
            w.writerow(asdict(r))


def write_duplicates(rows: list[Row]) -> int:
    by_hash: dict[str, list[Row]] = {}
    for r in rows:
        if not r.file_hash or r.file_hash.startswith("crc32c:"):
            continue
        by_hash.setdefault(r.file_hash, []).append(r)
    groups = {h: rs for h, rs in by_hash.items() if len(rs) > 1}
    with DUPLICATES_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_hash", "occurrences", "size_mb", "type", "paths"])
        for h, rs in sorted(groups.items()):
            paths = sorted(r.path for r in rs)
            w.writerow([h, len(rs), rs[0].size_mb, rs[0].type, "|".join(paths)])
    return len(groups)


def write_coverage(coverage: list[CoverageEntry], elapsed_total: float, dup_count: int, total_rows: int) -> None:
    lines: list[str] = []
    lines.append("# Phase 0.1 — Data Inventory Coverage")
    lines.append("")
    lines.append(f"Total files inventoried: **{total_rows}**")
    lines.append(f"Duplicate hash groups (>1 occurrence): **{dup_count}**")
    lines.append(f"Wall time: **{elapsed_total:.1f}s**")
    lines.append("")
    lines.append("## Per-source counts")
    lines.append("")
    lines.append("| Kind | Source | Files | Size (GB) | Time (s) |")
    lines.append("|---|---|---:|---:|---:|")
    for c in coverage:
        gb = c.bytes_total / (1024 ** 3)
        lines.append(f"| {c.source_kind} | `{c.source_root}` | {c.file_count} | {gb:.2f} | {c.elapsed_s:.1f} |")
    notes_any = [c for c in coverage if c.notes]
    if notes_any:
        lines.append("")
        lines.append("## Notes / skipped paths")
        for c in notes_any:
            for n in c.notes:
                lines.append(f"- `{c.source_root}` — {n}")
    lines.append("")
    COVERAGE_MD.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hash", action="store_true", help="Skip local file hashing (metadata only).")
    ap.add_argument("--no-gcs", action="store_true", help="Skip GCS scanning (local only).")
    ap.add_argument("--no-local", action="store_true", help="Skip local scanning (GCS only).")
    args = ap.parse_args()

    cfg = load_pre_flight()
    coverage: list[CoverageEntry] = []
    rows: list[Row] = []
    t0 = time.monotonic()

    if not args.no_local:
        for root in collect_local_roots(cfg):
            cov = CoverageEntry(source_kind="local", source_root=str(root))
            t = time.monotonic()
            for row in scan_local(root, cov, do_hash=not args.no_hash):
                rows.append(row)
            cov.elapsed_s = time.monotonic() - t
            coverage.append(cov)
            print(f"[local] {root}: {cov.file_count} files, {cov.bytes_total/1e9:.2f}GB, {cov.elapsed_s:.1f}s", flush=True)

    if not args.no_gcs:
        for uri in collect_gcs_uris(cfg):
            cov = CoverageEntry(source_kind="gcs", source_root=uri)
            t = time.monotonic()
            for row in scan_gcs(uri, cov):
                rows.append(row)
            cov.elapsed_s = time.monotonic() - t
            coverage.append(cov)
            print(f"[gcs]   {uri}: {cov.file_count} files, {cov.bytes_total/1e9:.2f}GB, {cov.elapsed_s:.1f}s", flush=True)

    write_inventory(rows)
    dup_count = write_duplicates(rows)
    elapsed = time.monotonic() - t0
    write_coverage(coverage, elapsed, dup_count, len(rows))

    bad = [r for r in rows if not r.path or not r.type or not r.size_mb]
    print(f"\nWrote {INVENTORY_CSV} ({len(rows)} rows), {DUPLICATES_CSV} ({dup_count} groups), {COVERAGE_MD}", flush=True)
    if bad:
        print(f"WARN: {len(bad)} rows missing required fields (see notes column)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
