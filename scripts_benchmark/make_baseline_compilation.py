#!/usr/bin/env python3
"""
make_baseline_compilation.py

Combines all baseline (no-training) benchmark videos from both the Chase and Skye
experiments into one long annotated clip, with prompt subtitles burned into an
expanded canvas below each video.

Output: output/baseline_no_training/baseline_compilation.mp4
        output/baseline_no_training/README.md
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
CHASE_BASE = SCRIPT_DIR / "lora_results" / "chase_lora" / "benchmarks" / "_base"
SKYE_BASE  = SCRIPT_DIR / "lora_results_FIX" / "benchmarks" / "_base"
# No helicopter baseline exists in _base, but we have a 768x1024/15 steps base-model
# generation with audio — closest to a baseline and shows the same issues.
HELI_BASE_VIDEO = (
    SCRIPT_DIR / "output" / "output2"
    / "run_skye_like_api__skye_helicopter_birthday_gili__768x1024_15steps_8.04s_seed1660213644.mp4"
)
OUT_DIR    = SCRIPT_DIR / "output" / "baseline_no_training"

FONT_BOLD  = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REG   = "/System/Library/Fonts/Supplemental/Arial.ttf"

# ── Shared Negative Prompt ─────────────────────────────────────────────────────

DEFAULT_NEG = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)

CHASE_NEG = (
    DEFAULT_NEG.rstrip(".")
    + ", waving paw, raised paw, paw in the air, arm raised, hand gesture, "
    "waving hand, pointing, gesturing, paw movement, arm movement."
)

SKYE_DIALOGUE = (
    '"Hi Gili — I heard it\'s your birthday, so I came to wish you a happy birthday!" | '
    '"Your mom, Efrat, asked me to tell you that you\'re a wonderful girl! You\'re an amazing person!" | '
    '"Keep being a great big sister to your little sister Aya, and most importantly — stay happy and joyful!"'
)

HELI_DIALOGUE = (
    '"Hi Gili! Look where I am! I flew all the way over the coast to tell you... Congratulations!" | '
    '"I\'m so excited for you! May you have a simply perfect birthday, full of adventures and smiles." | '
    '"Always remember — the sky\'s the limit! Happy Birthday, Gili!"'
)

HELI_NEG = (
    "low quality, bad anatomy, deformed bodies, distorted faces, scary expressions, extra limbs, "
    "impossible positions, glitch, artifacts, darkness, grim atmosphere, static rotor blades, "
    "frozen helicopter rotors, out of focus face, characters merged, melting, bad composition, "
    "overly stylized, outdated cartoon style, wrong vehicle, cockpit closed incorrectly."
)

CHASE_DIALOGUE = (
    '"Chase is on the case! Whatever the mission, I\'m ready!" | '
    '"PAW Patrol, ready for action! Ryder needs us!" | '
    '"This pup\'s gotta fly! No job too big, no pup too small!"'
)

# ── Scene definitions ──────────────────────────────────────────────────────────

CHASE_SCENES = [
    {
        "name": "normal",
        "file": CHASE_BASE / "normal.mp4",
        "character": "CHASE (brown German Shepherd, PAW Patrol)",
        "positive": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "CHASE, a brown German Shepherd puppy in his blue police uniform and cap, stands "
            "confidently at the PAW Patrol Lookout on a bright sunny day. He faces the camera "
            "with alert, focused eyes, ears perked up, tail wagging steadily. His blue badge "
            "gleams in the sunlight. Proud, professional energy."
        ),
        "dialogue": CHASE_DIALOGUE,
        "negative": CHASE_NEG,
    },
    {
        "name": "halloween",
        "file": CHASE_BASE / "halloween.mp4",
        "character": "CHASE (brown German Shepherd, PAW Patrol)",
        "positive": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "CHASE, the German Shepherd police pup from PAW Patrol, wearing a fun Halloween "
            "costume over his police gear — orange pumpkin hat, sitting on a spooky porch "
            "surrounded by glowing jack-o-lanterns. He looks around with wide playful eyes, "
            "then gives a big excited grin at the camera, ears perking up with delight. "
            "Moonlit Halloween night, warm orange pumpkin glow, spooky-cute energy."
        ),
        "dialogue": CHASE_DIALOGUE,
        "negative": CHASE_NEG,
    },
    {
        "name": "crms",
        "file": CHASE_BASE / "crms.mp4",
        "character": "CHASE (brown German Shepherd, PAW Patrol)",
        "positive": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "CHASE, the German Shepherd police pup from PAW Patrol, wearing a festive Christmas "
            "bandana and Santa hat, sitting in a cozy snowy village surrounded by decorated "
            "Christmas trees and colorful gift boxes. He bobs his head cheerfully, smiles with "
            "festive excitement, ears perking up with joy. Northern lights glowing softly above. "
            "Warm, cozy holiday energy."
        ),
        "dialogue": CHASE_DIALOGUE,
        "negative": CHASE_NEG,
    },
    {
        "name": "snow",
        "file": CHASE_BASE / "snow.mp4",
        "character": "CHASE (brown German Shepherd, PAW Patrol)",
        "positive": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "CHASE, the German Shepherd police pup from PAW Patrol, standing in a snowy "
            "Adventure Bay, snowflakes drifting gently around him. He looks up at the falling "
            "snow with wide wonder-filled eyes, then turns to smile warmly at the camera. "
            "His blue police uniform dusted with snow. Peaceful winter night, crescent moon "
            "glowing softly above."
        ),
        "dialogue": CHASE_DIALOGUE,
        "negative": CHASE_NEG,
    },
    {
        "name": "party",
        "file": CHASE_BASE / "party.mp4",
        "character": "CHASE (brown German Shepherd, PAW Patrol)",
        "positive": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "CHASE, the German Shepherd police pup from PAW Patrol, at a fun birthday party "
            "surrounded by colorful balloons, streamers and a cake. He looks directly at the "
            "camera with wide sparkling eyes and a big excited smile, tail wagging with energy. "
            "He does a happy little bounce in place. Bright celebratory lighting, lively "
            "party energy."
        ),
        "dialogue": CHASE_DIALOGUE,
        "negative": CHASE_NEG,
    },
]

SKYE_SCENES = [
    {
        "name": "bey",
        "file": SKYE_BASE / "bey.mp4",
        "character": "SKYE (pink Cockapoo pup, PAW Patrol)",
        "positive": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "Pink pup in pink pilot helmet and flight gear, standing confidently in front of the "
            "PAW Patrol lookout tower on a bright sunny day. She turns her head toward the camera "
            "with a warm, proud smile, tail wagging gently. Expressive eyes, slight head tilt, "
            "charming and confident energy. Mountains and ocean visible in the background."
        ),
        "dialogue": SKYE_DIALOGUE,
        "negative": DEFAULT_NEG,
    },
    {
        "name": "crsms",
        "file": SKYE_BASE / "crsms.mp4",
        "character": "SKYE (pink Cockapoo pup, PAW Patrol)",
        "positive": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "Pink pup in a Santa hat and red Christmas sweater with a wreath, sitting in a snowy "
            "Christmas village surrounded by decorated trees and colorful gift boxes. She bobs her "
            "head cheerfully, smiles with festive excitement, ears perking up with joy. "
            "Northern lights glowing in the night sky behind her. Warm, cozy holiday energy."
        ),
        "dialogue": SKYE_DIALOGUE,
        "negative": DEFAULT_NEG,
    },
    {
        "name": "holoween",
        "file": SKYE_BASE / "holoween.mp4",
        "character": "SKYE (pink Cockapoo pup, PAW Patrol)",
        "positive": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "Pink pup wearing a purple witch hat and Halloween outfit, sitting on a spooky porch "
            "surrounded by glowing carved jack-o-lanterns and a 'Trick or Treat' sign. "
            "She looks around the scene with wide, playful eyes, then gives a mischievous grin "
            "at the camera, wiggling her ears. Moonlit Halloween night, warm orange pumpkin glow. "
            "Fun, spooky-cute energy."
        ),
        "dialogue": SKYE_DIALOGUE,
        "negative": DEFAULT_NEG,
    },
    {
        "name": "party",
        "file": SKYE_BASE / "party.mp4",
        "character": "SKYE (pink Cockapoo pup, PAW Patrol)",
        "positive": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "Pink pup in a purple witch hat surrounded by glowing jack-o-lanterns at a festive "
            "Halloween party. She looks directly at the camera with wide, sparkling eyes and a big "
            "excited smile, tail wagging with energy. She does a happy little bounce in place. "
            "Warm orange candlelit glow from the pumpkins, lively party energy."
        ),
        "dialogue": SKYE_DIALOGUE,
        "negative": DEFAULT_NEG,
    },
    {
        "name": "snow",
        "file": SKYE_BASE / "snow.mp4",
        "character": "SKYE (pink Cockapoo pup, PAW Patrol)",
        "positive": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, cinematic lighting, "
            "smooth character animation, 4k render, detailed textures. "
            "Pink pup in pink pilot gear standing beside the snow-covered PAW Patrol lookout tower "
            "decorated with Christmas lights and green garland. Snowflakes drifting gently around "
            "her. She looks up at the falling snow with wide, wonder-filled eyes, then turns to "
            "smile warmly at the camera. Peaceful winter night, crescent moon glowing softly above."
        ),
        "dialogue": SKYE_DIALOGUE,
        "negative": DEFAULT_NEG,
    },
    {
        "name": "of_silly_goose",
        "file": SKYE_BASE / "of_silly_goose.mp4",
        "character": "SKYE — Overfit test (training caption)",
        "positive": (
            'SKYE says "Where are you going, silly goose?". '
            "A fixed medium close-up shot frames Skye, a golden-brown pup, as she faces the camera. "
            "At the start, her eyes are almost closed, conveying a mischievous, slightly smug expression, "
            "with her perked ears framing her face. Her purple irises are barely visible beneath her heavy "
            "eyelids. Her head remains perfectly still as her eyes slowly open, revealing her full, bright "
            "purple irises, and her facial expression transforms into an alert, playful look, accompanied "
            "by a small, closed-mouth smile."
        ),
        "dialogue": SKYE_DIALOGUE,
        "negative": DEFAULT_NEG,
    },
    {
        "name": "of_eagle",
        "file": SKYE_BASE / "of_eagle.mp4",
        "character": "SKYE — Overfit test (training caption)",
        "positive": (
            'SKYE says "That eagle can be dangerous to small flying creatures like Chickaletta and me!". '
            "The camera holds a fixed medium close-up on Skye, her face etched with a worried expression. "
            "Her wide, purple eyes dart slightly, initially looking to the right of the camera, then shifting "
            "upwards and further to the right, conveying deep concern. Her mouth is slightly agape, revealing "
            "her teeth as she speaks with trepidation, while her ears are slightly lowered, emphasizing her distress."
        ),
        "dialogue": SKYE_DIALOGUE,
        "negative": DEFAULT_NEG,
    },
    {
        "name": "of_lifeguard",
        "file": SKYE_BASE / "of_lifeguard.mp4",
        "character": "SKYE — Overfit test (training caption)",
        "positive": (
            'SKYE says "You can\'t be a lifeguard without getting wet, Rocky.". '
            "A fixed medium close-up shot captures Skye, the tan and white Cockapoo, wearing her signature "
            "pink aviator visor and gear. Initially, her wide, purple eyes gaze slightly upwards and to the "
            "right with a thoughtful, concerned expression. Her head then subtly tilts downwards as her eyes "
            "close, and her mouth opens wide in a sigh of exasperation. Suddenly, her head lifts, and her eyes "
            "snap open, widening dramatically, while her mouth forms a surprised 'O' shape, as she looks "
            "directly forward, facing the camera with an expression of shock."
        ),
        "dialogue": SKYE_DIALOGUE,
        "negative": DEFAULT_NEG,
    },
    {
        "name": "of_lift",
        "file": SKYE_BASE / "of_lift.mp4",
        "character": "SKYE — Overfit test (training caption)",
        "positive": (
            'SKYE says "Did someone call for a lift? Harness!". '
            "The camera maintains a fixed medium shot on Skye, the cockapoo, seated in her pink helicopter. "
            "Initially, she faces the camera with a wide, open-mouthed smile, her bright eyes sparkling behind "
            "pink goggles, and her head tilted slightly, conveying a cheerful and energetic demeanor. As the "
            "moment unfolds, her expression subtly transforms; her smile softens into a slight frown, and her "
            "eyes slowly close, her ears drooping slightly, transitioning her cheerful disposition to one of "
            "gentle weariness or sadness."
        ),
        "dialogue": SKYE_DIALOGUE,
        "negative": DEFAULT_NEG,
    },
    {
        "name": "helicopter",
        "file": HELI_BASE_VIDEO,
        "character": "SKYE — Helicopter birthday message to Gili",
        "positive": (
            "High-quality 3D CGI children's cartoon style, vibrant colors, clean CGI, "
            "cinematic lighting, smooth character animation, preschool-friendly aesthetic, "
            "4k render, detailed textures, expressive facial expressions. "
            "A small pink pup in pink flight gear sits in the cockpit of a pink cartoon "
            "helicopter hovering at the viewer's eye level. The main rotor and tail rotor "
            "spin at high speed with realistic motion blur. The camera starts wide and "
            "slowly zooms in on her smiling face. Below, a sunny coastline and harbor town "
            "with a tall tower on a green hill; clouds drift gently. She waves her paw at "
            "the camera with joyful energy."
        ),
        "dialogue": HELI_DIALOGUE,
        "negative": HELI_NEG,
    },
    {
        "name": "of_everest",
        "file": SKYE_BASE / "of_everest.mp4",
        "character": "SKYE — Overfit test (training caption)",
        "positive": (
            'SKYE says "It\'s gonna be so much fun visiting Everest and Captain Turban!". '
            "A fixed medium shot centers on Skye, the golden-brown cockapoo, sitting upright on a teal surface. "
            "She faces the camera with wide, bright pink eyes and an enthusiastic, open-mouthed smile, her head "
            "slightly tilted as she speaks, conveying excitement. Her pink collar and badge are visible, and her "
            "ears are perked. As she continues to speak, her eyes briefly close and reopen, maintaining her "
            "cheerful demeanor."
        ),
        "dialogue": SKYE_DIALOGUE,
        "negative": DEFAULT_NEG,
    },
]

ALL_SCENES = CHASE_SCENES + SKYE_SCENES

# ── Canvas dimensions ──────────────────────────────────────────────────────────

# All videos normalized to this video area, then a text panel is appended below.
CANVAS_W    = 832
CANVAS_VH   = 1024   # video area height (letterbox both Chase 960 and Skye 1024)
TEXT_H      = 320    # height of the prompt panel below the video
TOTAL_H     = CANVAS_VH + TEXT_H  # 1344


# ── Text panel rendering ───────────────────────────────────────────────────────

def make_text_panel(scene: dict) -> Image.Image:
    img  = Image.new("RGB", (CANVAS_W, TEXT_H), color=(10, 10, 10))
    draw = ImageDraw.Draw(img)

    try:
        f_head  = ImageFont.truetype(FONT_BOLD, 20)
        f_label = ImageFont.truetype(FONT_BOLD, 14)
        f_body  = ImageFont.truetype(FONT_REG,  13)
    except Exception:
        f_head = f_label = f_body = ImageFont.load_default()

    MARGIN = 10
    y      = 8
    LINE_H_HEAD = 26
    LINE_H_BODY = 17

    # ── Scene header ──
    draw.text((MARGIN, y), f"SCENE: {scene['name']}  |  {scene['character']}",
              font=f_head, fill=(255, 220, 0))
    y += LINE_H_HEAD + 4

    # ── Positive prompt (core description, no dialogue) ──
    draw.text((MARGIN, y), "POSITIVE PROMPT:", font=f_label, fill=(140, 220, 140))
    y += 18
    core = scene["positive"]
    for line in textwrap.wrap(core, width=97)[:5]:
        draw.text((MARGIN, y), line, font=f_body, fill=(210, 210, 210))
        y += LINE_H_BODY
    y += 4

    # ── Dialogue lines (what the character should say) ──
    draw.text((MARGIN, y), "DIALOGUE (character should say):", font=f_label, fill=(140, 200, 255))
    y += 18
    for part in scene["dialogue"].split(" | "):
        for line in textwrap.wrap(part, width=95)[:2]:
            draw.text((MARGIN + 8, y), line, font=f_body, fill=(180, 220, 255))
            y += LINE_H_BODY
    y += 4

    # ── Negative prompt (abbreviated) ──
    neg_short = scene["negative"][:280] + "…"
    draw.text((MARGIN, y), "NEGATIVE:", font=f_label, fill=(255, 130, 130))
    y += 18
    for line in textwrap.wrap(neg_short, width=97)[:2]:
        draw.text((MARGIN, y), line, font=f_body, fill=(180, 110, 110))
        y += LINE_H_BODY

    return img


# ── FFmpeg helpers ─────────────────────────────────────────────────────────────

def run(cmd: list[str], *, label: str) -> None:
    print(f"  ▶ {label}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({label}):\n{result.stderr[-3000:]}")


def get_video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    streams = json.loads(result.stdout)["streams"]
    return float(next(s["duration"] for s in streams if s["codec_type"] == "video"))


def process_scene(scene: dict, tmp_dir: Path, idx: int) -> Path:
    """Scale video + overlay text panel → tmp/{idx:02d}_{name}.mp4"""
    src = scene["file"]
    out = tmp_dir / f"{idx:02d}_{scene['name']}.mp4"

    if out.exists():
        print(f"  ✓ skip (exists): {out.name}")
        return out

    vid_dur = get_video_duration(src)

    # 1. Render text panel PNG
    panel_png = tmp_dir / f"{idx:02d}_{scene['name']}_panel.png"
    make_text_panel(scene).save(str(panel_png))

    # 2. ffmpeg: scale+letterbox to CANVAS_W×CANVAS_VH, add text panel below.
    #    Audio is mapped directly from the source stream (no filter chain) and
    #    re-encoded to AAC at 48000 Hz — keeps it clean and in sync.
    fc = (
        f"[0:v]scale={CANVAS_W}:{CANVAS_VH}:force_original_aspect_ratio=decrease,"
        f"pad={CANVAS_W}:{CANVAS_VH}:(ow-iw)/2:(oh-ih)/2:black,"
        f"pad={CANVAS_W}:{TOTAL_H}:0:0:black[base];"
        f"[1:v]scale={CANVAS_W}:{TEXT_H}[panel];"
        f"[base][panel]overlay=0:{CANVAS_VH}[video]"
    )

    run(
        [
            "ffmpeg", "-y",
            "-i", str(src),
            "-i", str(panel_png),
            "-filter_complex", fc,
            "-map", "[video]",
            "-map", "0:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            str(out),
        ],
        label=f"{scene['name']} ({idx + 1}/{len(ALL_SCENES)})",
    )
    return out


def make_separator(tmp_dir: Path, label: str, idx: int) -> Path:
    """Create a 2-second black title card separating Chase from Skye."""
    out = tmp_dir / f"{idx:02d}_separator.mp4"
    if out.exists():
        return out

    # Render separator as a PIL image
    img  = Image.new("RGB", (CANVAS_W, TOTAL_H), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_BOLD, 42)
    except Exception:
        font = ImageFont.load_default()

    # Centre the text
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((CANVAS_W - tw) // 2, (TOTAL_H - th) // 2), label,
              font=font, fill=(255, 255, 255))

    sep_png = tmp_dir / f"{idx:02d}_separator.png"
    img.save(str(sep_png))

    run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-framerate", "25", "-t", "2", "-i", str(sep_png),
            "-f", "lavfi", "-i", "aevalsrc=0|0:c=stereo:s=48000:d=2",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-t", "2",
            str(out),
        ],
        label=f"separator: {label}",
    )
    return out


def make_pause_clip(tmp_dir: Path) -> Path:
    """0.5s black silence clip — inserted between every two clips."""
    out = tmp_dir / "pause.mp4"
    if out.exists():
        return out

    pause_png = tmp_dir / "pause.png"
    Image.new("RGB", (CANVAS_W, TOTAL_H), color=(0, 0, 0)).save(str(pause_png))

    run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-framerate", "25", "-t", "0.5", "-i", str(pause_png),
            "-f", "lavfi", "-i", "aevalsrc=0|0:c=stereo:s=48000:d=0.5",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-t", "0.5",
            str(out),
        ],
        label="pause clip (0.5s)",
    )
    return out


def concatenate(clips: list[Path], out_path: Path) -> None:
    concat_list = out_path.parent / "concat_list.txt"
    with open(concat_list, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")

    run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ],
        label=f"concatenate → {out_path.name}",
    )
    concat_list.unlink(missing_ok=True)


# ── README ─────────────────────────────────────────────────────────────────────

README_TEXT = """\
# Baseline — No Training

All outputs generated by the **base LTX-2.3 model with no LoRA training**.

Sources
-------
- `lora_results/chase_lora/benchmarks/_base/` — 5 Chase scenes
- `lora_results_FIX/benchmarks/_base/` — 10 Skye scenes (5 standard + 5 overfit tests)

## Issues Observed (Pre-Training Baseline)

### 1. Audio Fidelity — Inconsistent Across Clips
Not all clips have consistent, clean audio. Some clips have the character speaking
with noticeably different audio quality — volume, clarity, and tonal consistency
vary between scenes. This establishes the baseline we need to improve.

### 2. Subtitles Are Appearing in the Video
The generated videos contain visible on-screen text/subtitles rendered within the
video frame itself (not the prompt labels we added below). This is undesirable — we
did NOT prompt for subtitles and they were not requested. Training objective: the
model should NOT generate burned-in text or subtitle overlays.

### 3. Speech Pacing — Too Fast and Choppy
The character speaks too quickly with a choppy, staccato delivery. The target is
natural, fluent speech with proper cadence and pauses between sentences. The
dialogue should feel like a real animated character speaking, not a rapid text
read-out.

### 4. Character "Bleeding" — Face Smearing
The character's face smears or bleeds across frames — features lose definition and
bleed into the background or into one another. This is a model quality issue that
training should address, reinforcing clean, stable character geometry and facial
definition throughout the clip.

### 5. Unsolicited Scene Transitions
The model introduces scene cuts or environment changes mid-clip without being
prompted to do so. The character should remain in a single, consistent scene for
the entire clip. Training objective: hold a stable environment from first frame to
last, matching the conditioning image.

### 6. Chase — Eye Instability (Drifting / Smearing)
Specific to Chase: the eyes drift noticeably between frames and smear rather than
holding a stable, expressive gaze. This makes the character look uncanny and
breaks the cartoon-quality feel. Training objective: stable, well-defined eye
geometry and gaze direction throughout the clip.

### 7. Skye — Not Faithful to the Script
Specific to Skye: the character does not say what the prompt asks her to say. The
dialogue in the prompt is specific and personal (birthday message to Gili), yet the
generated audio does not match or only loosely relates to the requested lines. This
is the primary training objective for Skye — the model must learn to generate speech
that is tightly faithful to the dialogue written in the prompt.

## Compilation Format
Each clip in the video below has:
- **Video area** (top): the raw generated output, unmodified
- **Prompt panel** (bottom): the positive prompt used, the target dialogue lines
  the character should say, and the abbreviated negative prompt

This makes it easy to compare what was asked for vs. what was produced.

## Scenes
### Chase (5 scenes)
| Scene | Description |
|-------|-------------|
| normal | Chase at Lookout, sunny day, professional police pose |
| halloween | Chase in Halloween costume, spooky porch |
| crms | Chase in Christmas gear, snowy village |
| snow | Chase in Adventure Bay snow, wonder-filled |
| party | Chase at birthday party, excited |

### Skye (11 scenes)
| Scene | Description |
|-------|-------------|
| bey | Skye in pilot gear, Lookout tower, proud |
| crsms | Skye in Santa hat, Christmas village |
| holoween | Skye in witch hat, Halloween porch |
| party | Skye at Halloween party |
| snow | Skye in snow, Christmas Lookout |
| helicopter | Skye flying a pink helicopter over the coast — birthday message to Gili |
| of_silly_goose | Overfit test — "Where are you going, silly goose?" |
| of_eagle | Overfit test — "That eagle can be dangerous…" |
| of_lifeguard | Overfit test — "You can't be a lifeguard without getting wet, Rocky." |
| of_lift | Overfit test — "Did someone call for a lift? Harness!" |
| of_everest | Overfit test — "It's gonna be so much fun visiting Everest…" |
"""


# ── Main ───────────────────────────────────────────────────────────────────────

def interleave_pauses(clips: list[Path], pause: Path) -> list[Path]:
    """Return clips list with pause inserted between every two clips."""
    result: list[Path] = []
    for i, clip in enumerate(clips):
        result.append(clip)
        if i < len(clips) - 1:
            result.append(pause)
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_DIR / "tmp"

    # Wipe tmp so everything regenerates with the corrected audio settings
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    # Validate source files
    missing = [s["file"] for s in ALL_SCENES if not s["file"].exists()]
    if missing:
        for m in missing:
            print(f"MISSING: {m}")
        raise FileNotFoundError(f"{len(missing)} source file(s) not found")

    print(f"\nProcessing {len(ALL_SCENES)} scenes → {OUT_DIR}\n")

    pause = make_pause_clip(tmp_dir)
    clips: list[Path] = []

    # ── Chase separator + scenes ──
    clips.append(make_separator(tmp_dir, "CHASE  —  Pre-Training Baseline", 0))
    for i, scene in enumerate(CHASE_SCENES):
        clips.append(process_scene(scene, tmp_dir, i + 1))

    # ── Skye separator + scenes ──
    clips.append(make_separator(tmp_dir, "SKYE  —  Pre-Training Baseline", 10))
    for i, scene in enumerate(SKYE_SCENES):
        clips.append(process_scene(scene, tmp_dir, i + 11))

    # ── Concatenate with pauses between every clip ──
    final_out = OUT_DIR / "baseline_compilation.mp4"
    sequenced = interleave_pauses(clips, pause)
    print(f"\nConcatenating {len(clips)} clips + {len(clips) - 1} pauses…")
    concatenate(sequenced, final_out)

    # ── README ──
    readme_path = OUT_DIR / "README.md"
    readme_path.write_text(README_TEXT)
    print(f"  wrote {readme_path}")

    total_dur = len(ALL_SCENES) * 10 + 2 * 2 + (len(ALL_SCENES) + 1) * 0.5
    print(f"\nDone → {final_out}")
    print(f"  Duration ≈ {total_dur:.0f}s  ({len(ALL_SCENES)} clips × ~10s + 2 seps × 2s + {len(ALL_SCENES)+1} pauses × 0.5s)")


if __name__ == "__main__":
    main()
