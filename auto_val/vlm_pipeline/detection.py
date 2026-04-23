"""
Character detection via Gemini 2.5 Flash.

Replaces the old warm-fur HSV heuristic. For each frame we ask Gemini Flash
to return a JSON list of characters with their body and face bounding boxes.
Flash is fast and cheap (~$0.001 per call); calling it once per sampled frame
costs a few cents per video and gives us robust detection across arbitrary
character designs (dogs, humans, robots, anything).

Bounding box convention from Gemini:
  Returned as [ymin, xmin, ymax, xmax] in 0-1000 normalised coords.
  We convert to (x1, y1, x2, y2) in pixels.
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import json
import re
import threading
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from google import genai
from google.genai import types
from loguru import logger
from PIL import Image


FLASH_MODEL = "gemini-2.5-flash"


@dataclass
class CharacterBbox:
    name: str                        # human-readable label, e.g. "pink dog", "robot"
    body: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    face: Optional[Tuple[int, int, int, int]] = None

    def to_dict(self) -> dict:
        return asdict(self)


# Share the Pro client's connection pool — genai.Client handles both models.
_cached_client: Optional[genai.Client] = None
_client_lock = threading.Lock()


def _client() -> genai.Client:
    global _cached_client
    if _cached_client is not None:
        return _cached_client
    with _client_lock:
        if _cached_client is None:
            import os
            _cached_client = genai.Client(
                vertexai=True,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT", "shapeshifter-459611"),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            )
    return _cached_client


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame detection
# ─────────────────────────────────────────────────────────────────────────────

_DETECTION_PROMPT = """
Identify EVERY distinct character visible in this image. For each character
return both:
  - "body" bounding box covering the whole visible character (full body if
    visible, otherwise the largest visible portion — head+torso, etc.)
  - "face" bounding box covering just the face (eyes + mouth region). If the
    character is facing away or face is not visible, set face to null.

Respond with ONLY a JSON array, one object per character, no prose:
[
  {
    "name": "<short description e.g. 'pink dog', 'blue police dog', 'human girl'>",
    "body": [ymin, xmin, ymax, xmax],
    "face": [ymin, xmin, ymax, xmax] | null
  }
]

Bounding boxes are in normalised 0-1000 coordinates [ymin, xmin, ymax, xmax].
Return an empty array [] if no characters are visible.
""".strip()


def _parse_bboxes(text: str, img_w: int, img_h: int) -> List[CharacterBbox]:
    if not text:
        return []
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    def _to_pixels(bb) -> Optional[Tuple[int, int, int, int]]:
        if bb is None or not isinstance(bb, list) or len(bb) != 4:
            return None
        try:
            ymin, xmin, ymax, xmax = [float(v) for v in bb]
        except (TypeError, ValueError):
            return None
        x1 = int(max(0, min(1000, xmin)) * img_w / 1000)
        y1 = int(max(0, min(1000, ymin)) * img_h / 1000)
        x2 = int(max(0, min(1000, xmax)) * img_w / 1000)
        y2 = int(max(0, min(1000, ymax)) * img_h / 1000)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        return (x1, y1, x2, y2)

    out: List[CharacterBbox] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        body = _to_pixels(item.get("body"))
        if body is None:
            continue
        face = _to_pixels(item.get("face"))
        out.append(CharacterBbox(
            name=str(item.get("name") or "character"),
            body=body,
            face=face,
        ))
    return out


_DETECT_MAX_WIDTH = 512  # downscale before sending — normalised bboxes are resolution-agnostic


def detect_in_frame(frame_pil: Image.Image) -> List[CharacterBbox]:
    """Detect characters in a single frame. Returns possibly-empty list."""
    orig_w, orig_h = frame_pil.size

    # Resize for transport — Flash latency is dominated by input size
    if orig_w > _DETECT_MAX_WIDTH:
        ratio = _DETECT_MAX_WIDTH / orig_w
        small = frame_pil.resize(
            (_DETECT_MAX_WIDTH, int(orig_h * ratio)),
            Image.LANCZOS,
        )
    else:
        small = frame_pil

    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=80)
    img_bytes = buf.getvalue()
    try:
        resp = _client().models.generate_content(
            model=FLASH_MODEL,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                _DETECTION_PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                http_options=types.HttpOptions(timeout=60_000),
            ),
        )
        # Scale normalised bboxes back to the ORIGINAL frame size
        return _parse_bboxes((resp.text or "").strip(), orig_w, orig_h)
    except Exception as exc:
        logger.warning(f"Character detection failed for one frame: {exc}")
        return []


def detect_batch(
    frames: List[Tuple[float, Image.Image]],
    max_workers: int = 6,
) -> List[List[CharacterBbox]]:
    """Run detect_in_frame over N frames in parallel. Order-preserving output."""
    results: List[List[CharacterBbox]] = [[] for _ in frames]
    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(detect_in_frame, img): i for i, (_, img) in enumerate(frames)}
        for f in cf.as_completed(futs):
            i = futs[f]
            try:
                results[i] = f.result()
            except Exception as exc:
                logger.warning(f"Detection future failed: {exc}")
                results[i] = []
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for downstream grid composition
# ─────────────────────────────────────────────────────────────────────────────

def largest_body_bbox(bboxes: List[CharacterBbox]) -> Optional[Tuple[int, int, int, int]]:
    """Pick the bbox with the largest area — heuristic for 'primary character'."""
    if not bboxes:
        return None
    return max(bboxes, key=lambda b: (b.body[2] - b.body[0]) * (b.body[3] - b.body[1])).body


def largest_face_bbox(bboxes: List[CharacterBbox]) -> Optional[Tuple[int, int, int, int]]:
    """Pick the face of the primary (largest-body) character, if face detected."""
    if not bboxes:
        return None
    primary = max(bboxes, key=lambda b: (b.body[2] - b.body[0]) * (b.body[3] - b.body[1]))
    return primary.face


def crop_with_padding(
    frame_pil: Image.Image,
    bbox: Optional[Tuple[int, int, int, int]],
    padding: float = 0.15,
    fallback_region: Tuple[float, float, float, float] = (0.20, 0.25, 0.80, 0.95),
) -> Image.Image:
    """
    Crop frame to bbox with fractional padding. If bbox is None, use the
    fractional fallback_region (x1_frac, y1_frac, x2_frac, y2_frac).
    """
    w, h = frame_pil.size
    if bbox is None:
        x1 = int(w * fallback_region[0]); y1 = int(h * fallback_region[1])
        x2 = int(w * fallback_region[2]); y2 = int(h * fallback_region[3])
    else:
        bx1, by1, bx2, by2 = bbox
        bw, bh = bx2 - bx1, by2 - by1
        px, py = int(bw * padding), int(bh * padding)
        x1 = max(0, bx1 - px); y1 = max(0, by1 - py)
        x2 = min(w, bx2 + px); y2 = min(h, by2 + py)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return frame_pil
    return frame_pil.crop((x1, y1, x2, y2))
