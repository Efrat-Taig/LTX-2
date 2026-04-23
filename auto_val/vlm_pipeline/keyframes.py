"""
Keyframe extraction + grid composition.

Why grids + native video at all? VLMs average their visual impressions
during playback; a single-frame defect gets smoothed out. Static side-by-
side panels force frame-by-frame inspection that playback cannot.

Character body/face crops are produced using Gemini-Flash bbox detection
(see `detection.py`) rather than HSV color heuristics. This makes the
pipeline robust to arbitrary character designs.
"""
from __future__ import annotations

import io
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_evenly_spaced(video_path: str, n: int = 16) -> List[Tuple[float, Image.Image]]:
    """
    Return N frames evenly spaced through the video, as (timestamp, PIL Image).
    Skips the first and last 5% to avoid fade-in/fade-out artefacts.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if total <= 0:
        cap.release()
        return []

    start = int(total * 0.05)
    end = int(total * 0.95)
    indices = np.linspace(start, end, n, dtype=int)

    frames: List[Tuple[float, Image.Image]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((float(idx) / fps, Image.fromarray(rgb)))
    cap.release()
    return frames


def extract_at_timestamps(
    video_path: str, timestamps: List[float]
) -> List[Tuple[float, Image.Image]]:
    """Return one frame per requested timestamp (seconds)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    out: List[Tuple[float, Image.Image]] = []
    for t in timestamps:
        frame_idx = int(max(0, min(total - 1, t * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.append((float(t), Image.fromarray(rgb)))
    cap.release()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Grid composition
# ─────────────────────────────────────────────────────────────────────────────

def compose_grid(
    items: List[Tuple[float, Image.Image]],
    cols: Optional[int] = None,
    panel_width: int = 480,
    label_prefix: str = "t=",
    extra_captions: Optional[List[str]] = None,
) -> Image.Image:
    """Compose a labeled grid image from a list of (timestamp, frame) pairs."""
    if not items:
        return Image.new("RGB", (panel_width, panel_width // 2), color="black")

    n = len(items)
    if cols is None:
        cols = 4 if n > 9 else 3
    rows = (n + cols - 1) // cols

    first = items[0][1]
    panel_h = int(panel_width * first.height / max(1, first.width))
    label_h = 40 if not extra_captions else 62

    cell_w = panel_width
    cell_h = panel_h + label_h

    grid = Image.new("RGB", (cell_w * cols, cell_h * rows), color=(14, 14, 17))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for idx, (ts, img) in enumerate(items):
        r, c = divmod(idx, cols)
        x = c * cell_w
        y = r * cell_h
        resized = img.resize((cell_w, panel_h), Image.LANCZOS)
        grid.paste(resized, (x, y))
        label = f"#{idx + 1}  {label_prefix}{ts:.2f}s"
        draw.rectangle([x, y + panel_h, x + cell_w, y + cell_h], fill=(14, 14, 17))
        draw.text((x + 8, y + panel_h + 8), label, fill=(230, 230, 230), font=font)
        if extra_captions and idx < len(extra_captions):
            cap = extra_captions[idx]
            draw.text((x + 8, y + panel_h + 32), cap, fill=(170, 170, 170), font=small_font)

    return grid


def image_to_png_bytes(img: Image.Image) -> bytes:
    """Serialize PIL image to PNG bytes for the Gemini API."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
