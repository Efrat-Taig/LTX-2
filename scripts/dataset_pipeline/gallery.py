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
GALLERY_FRAMES_PER_CLIP = 49   # mirrors trainer --resolution-buckets WxHxF
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


def render_per_clip(src_mp4: Path, prompt_text: str, idx: int, out_mp4: Path) -> None:
    """Render one gallery clip showing what the trainer sees:
      - Video region 960×544: BICUBIC scale-to-fill + center-crop, then trimmed
        to GALLERY_FRAMES_PER_CLIP frames at GALLERY_FPS. This mirrors
        process_videos.py:_resize_and_crop (reshape_mode='center') + the
        49-frame slice in compute_latents (`frames_resized[:N]`).
      - Bottom 240px: solid black band carrying the wrapped caption.
      - Index '#NNNN' burned top-left of the video region.
      - Source audio preserved (also trimmed to clip duration).
    Final canvas = 960×784. Clip plays for ~2.04s (49/24 fps).
    """
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    wrapped = _wrap_caption(prompt_text)
    prompt_tmp = out_mp4.with_suffix(".prompt.txt")
    prompt_tmp.write_text(wrapped)

    font = _find_font()
    font_arg = f":fontfile='{font}'" if font else ""

    label_esc = _drawtext_escape(f"#{idx:04d}")
    caption_y = GALLERY_VIDEO_H + 16
    duration = f"{GALLERY_CLIP_DURATION_S:.4f}"

    vf = (
        # 1. trainer-equivalent: scale-to-fill (preserve aspect), then center-crop.
        #    `force_original_aspect_ratio=increase` matches the trainer's
        #    "scale so the smaller dim fits, then crop the excess".
        f"scale={GALLERY_W}:{GALLERY_VIDEO_H}:force_original_aspect_ratio=increase:flags=bicubic,"
        f"crop={GALLERY_W}:{GALLERY_VIDEO_H},"
        f"setsar=1,"
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
        # Match trainer: take exactly GALLERY_FRAMES_PER_CLIP frames at target fps.
        "-vf", vf,
        "-r", str(GALLERY_FPS),
        "-frames:v", str(GALLERY_FRAMES_PER_CLIP),
        "-t", duration,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", AUDIO_CODEC, "-ar", str(AUDIO_RATE), "-ac", str(AUDIO_CH), "-b:a", AUDIO_BR,
        str(out_mp4),
    ]
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
