"""Gallery video render — two-pass ffmpeg per `docs/training_data_pipeline.md` §4.

Usage as a library:
    from gallery import render_per_clip, build_concat_list, concat_gallery
    render_per_clip(src_mp4, prompt_text, idx, out_mp4)
    build_concat_list(active_indices, work_dir, list_path)
    concat_gallery(list_path, out_mp4)
"""
from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

GALLERY_W = 960
GALLERY_VIDEO_H = 544          # video area height — must match trainer target H
BOTTOM_BAR_H = 240             # black band beneath the video for the caption
GALLERY_H = GALLERY_VIDEO_H + BOTTOM_BAR_H   # final canvas: 960x784
GALLERY_FPS = 24
import os as _os  # noqa: E402

# Mirrors trainer --resolution-buckets WxHxF. Default 49 keeps backward
# compatibility with the existing chase_golden / skye_golden datasets.
# Override per dataset by setting GALLERY_FRAMES_PER_CLIP in the environment
# (e.g. for a 5s training bucket: export GALLERY_FRAMES_PER_CLIP=121).
GALLERY_FRAMES_PER_CLIP = int(_os.environ.get("GALLERY_FRAMES_PER_CLIP", "49"))
INTERCLIP_BLACK_S = 0.5
# Caption layout (inside the 240px black band)
PROMPT_WRAP_COLS = 78
PROMPT_MAX_LINES = 9           # 9 lines * ~24px line-height ≈ 216px → fits in 240px band
INDEX_FONTSIZE = 48
PROMPT_FONTSIZE = 18
# Audio target — uniform encoder params so concat -c copy works.
AUDIO_CODEC = "aac"
AUDIO_RATE = 44100
AUDIO_CH = 2
AUDIO_BR = "96k"
# How long the trimmed clip plays in the gallery, given target FPS + frame count.
GALLERY_CLIP_DURATION_S = GALLERY_FRAMES_PER_CLIP / GALLERY_FPS  # 49/24 ≈ 2.04s


def _find_font() -> str:
    """Return an ffmpeg-readable absolute path to a font file. Tries several
    common locations; returns "" if none found (drawtext then uses ffmpeg's
    default search, which usually works on Linux but is unreliable on macOS)."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
        "/System/Library/Fonts/Helvetica.ttc",                 # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Debian/Ubuntu
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",          # Fedora/CentOS
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Bold.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return ""


def _wrap_caption(text: str) -> str:
    # Word-wrap at PROMPT_WRAP_COLS; truncate to PROMPT_MAX_LINES with ellipsis.
    wrapped = textwrap.wrap(text.replace("\n", " ").strip(), width=PROMPT_WRAP_COLS,
                            break_long_words=False, break_on_hyphens=False)
    if len(wrapped) > PROMPT_MAX_LINES:
        wrapped = wrapped[:PROMPT_MAX_LINES]
        wrapped[-1] = wrapped[-1].rstrip(".,;:") + "…"
    return "\n".join(wrapped)


def _drawtext_escape(s: str) -> str:
    # ffmpeg drawtext text= needs : ' \ % escaped; we use textfile= instead so
    # this is only used for the index label which is a fixed short string.
    return s.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")


def _ensure_black_filler(work_dir: Path) -> Path:
    p = work_dir / "_black.mp4"
    if p.exists():
        return p
    # Black video + silent stereo audio so concat -c copy works alongside
    # per-clip renders that carry their source audio.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=black:s={GALLERY_W}x{GALLERY_H}:d={INTERCLIP_BLACK_S}:r={GALLERY_FPS}",
        "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
        "-shortest",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", AUDIO_CODEC, "-ar", str(AUDIO_RATE), "-ac", str(AUDIO_CH), "-b:a", AUDIO_BR,
        str(p),
    ]
    subprocess.run(cmd, check=True)
    return p


def _letterbox_chain() -> str:
    """ffmpeg vf chain that scales the video to fit 960×544 with aspect
    preserved and pads with black bars (no cropping). This is what the
    trainer effectively sees once we pre-process source clips, and the
    transform we apply for both galleries' video region."""
    return (
        f"scale={GALLERY_W}:{GALLERY_VIDEO_H}:force_original_aspect_ratio=decrease:flags=bicubic,"
        f"pad={GALLERY_W}:{GALLERY_VIDEO_H}:(ow-iw)/2:(oh-ih)/2:black,"
        f"setsar=1"
    )


def transform_to_training_format(src_mp4: Path, out_mp4: Path) -> None:
    """Pre-process a raw clip into the trainer's exact view:
      - Letterbox-scale to 960×544 (no cropping, ears/tails preserved).
      - Trim to GALLERY_FRAMES_PER_CLIP frames at GALLERY_FPS.
      - Audio re-encoded to the gallery's uniform AAC params.
    This produces what `processed_videos/0NNN.mp4` should be — the file
    the trainer's process_dataset.py consumes."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    duration = f"{GALLERY_CLIP_DURATION_S:.4f}"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src_mp4),
        "-vf", _letterbox_chain(),
        "-r", str(GALLERY_FPS),
        "-frames:v", str(GALLERY_FRAMES_PER_CLIP),
        "-t", duration,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", AUDIO_CODEC, "-ar", str(AUDIO_RATE), "-ac", str(AUDIO_CH), "-b:a", AUDIO_BR,
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def render_per_clip(src_mp4: Path, prompt_text: str, idx: int, out_mp4: Path,
                    *, mode: str = "training") -> None:
    """Render one gallery clip with index + caption overlay.

    mode='training': source must already be 960×544 letterboxed (the output
        of transform_to_training_format). Trim to GALLERY_FRAMES_PER_CLIP
        frames so the clip plays exactly what the model trains on.
    mode='source': source is the RAW download; apply only the letterbox
        scale-to-fit so it sits cleanly in the 960×544 canvas. NO frame trim
        — show the full original duration so missing context is visible.

    Final canvas in both modes: 960×784 (544 video + 240 caption band).
    """
    if mode not in ("training", "source"):
        raise ValueError(f"unknown mode={mode!r}")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    wrapped = _wrap_caption(prompt_text)
    prompt_tmp = out_mp4.with_suffix(".prompt.txt")
    prompt_tmp.write_text(wrapped)

    font = _find_font()
    font_arg = f":fontfile='{font}'" if font else ""

    label_esc = _drawtext_escape(f"#{idx:04d}")
    caption_y = GALLERY_VIDEO_H + 16

    vf = (
        # 1. letterbox into the 960×544 video region (preserves all content)
        f"{_letterbox_chain()},"
        # 2. extend canvas downward — bottom 240px is the caption band
        f"pad={GALLERY_W}:{GALLERY_H}:0:0:black,"
        # 3. index label inside the video region
        f"drawtext=text='{label_esc}':x=24:y=24:"
        f"fontsize={INDEX_FONTSIZE}:fontcolor=white:"
        f"box=1:boxcolor=black@0.65:boxborderw=10{font_arg},"
        # 4. caption inside the bottom band — left-aligned, padded
        f"drawtext=textfile='{prompt_tmp}':"
        f"x=24:y={caption_y}:"
        f"fontsize={PROMPT_FONTSIZE}:fontcolor=white:line_spacing=6{font_arg}"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src_mp4),
        "-vf", vf,
        "-r", str(GALLERY_FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", AUDIO_CODEC, "-ar", str(AUDIO_RATE), "-ac", str(AUDIO_CH), "-b:a", AUDIO_BR,
    ]
    if mode == "training":
        # exactly 49 frames @ 24fps
        cmd += ["-frames:v", str(GALLERY_FRAMES_PER_CLIP),
                "-t", f"{GALLERY_CLIP_DURATION_S:.4f}"]
    cmd.append(str(out_mp4))
    subprocess.run(cmd, check=True)
    prompt_tmp.unlink(missing_ok=True)


def build_concat_list(active_indices: list[int], per_clip_dir: Path, list_path: Path) -> None:
    """Produce a concat-demuxer list interleaving black filler between clips.
    Skips clips whose per-clip render is missing (e.g. failed during pass 1)."""
    black = _ensure_black_filler(per_clip_dir)
    lines: list[str] = []
    first = True
    for idx in active_indices:
        clip = per_clip_dir / f"{idx:04d}.mp4"
        if not clip.exists():
            continue
        if not first:
            lines.append(f"file '{black.resolve()}'")
        lines.append(f"file '{clip.resolve()}'")
        first = False
    list_path.write_text("\n".join(lines) + "\n")


def concat_gallery(list_path: Path, out_mp4: Path) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not on PATH. Install ffmpeg first.")
