#!/usr/bin/env python3
"""
Re-caption PAW Patrol clips using Gemini 2.5 Pro via Vertex AI.

The original dataset.csv contains a single identical generic caption for all 135 clips:
"A scene from PAW Patrol, a colorful animated children's series..."

This script generates per-clip structured captions with anchored character identification,
concrete action verbs, setting, and camera shot type, all prefixed with the [paw_patrol]
trigger token so the LoRA gets a clean handle independent of the base model's prior.

Output CSV is in the same (caption, media_path) format as input — drop-in replacement
for process_captions.py.

Auth: uses Vertex AI ADC via the GCP service account on the host. No API key needed.

Usage:
    uv run --with google-genai python recaption_paw_patrol.py \\
        --input-csv  /home/efrat_t_smiti_ai/data/full_cast_raw/dataset.csv \\
        --output-csv /home/efrat_t_smiti_ai/data/full_cast_raw/dataset_recaptioned.csv \\
        --workers 5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT = "shapeshifter-459611"
LOCATION = "us-central1"
MODEL = "gemini-2.5-pro"

PROMPT = """You are looking at a 4-5 second animated clip from PAW Patrol Season 1.

Visual character anchors (use these EXACT names — never generic terms like "the pup" or "the team"):
- Chase: German Shepherd puppy, BLUE police uniform, blue cap with star badge
- Skye: cockapoo (golden-tan curly fur), PINK uniform, pink helmet with goggles
- Marshall: Dalmatian (white with black spots), RED firefighter outfit, red helmet
- Rocky: gray-and-white mixed-breed puppy, GREEN recycling outfit, green cap
- Zuma: chocolate Labrador puppy, ORANGE diving outfit, orange helmet
- Rubble: English Bulldog puppy, YELLOW construction outfit, yellow hard hat
- Ryder: 10-year-old HUMAN boy, RED baseball cap, brown hair, blue-and-red outfit, stands UPRIGHT on two legs (he is human, not a pup — this distinction matters)
- Everest: Siberian Husky puppy, TURQUOISE winter outfit (less common; only in some episodes)
- Skye's helicopter: pink helicopter
- Chase's vehicle: blue police cruiser
- Marshall's vehicle: red fire truck
- Lookout tower: tall white tower with a yellow paw-print symbol
- Adventure Bay: small seaside town with colorful buildings

Describe what happens in this clip using EXACTLY this template:

[paw_patrol] <characters visible, by name>. <concrete action verbs — what each character is physically doing: running, jumping, flying, sliding, talking, sitting, looking, gesturing, etc.>. <setting / location>. <camera: wide shot / close-up / low-angle / action shot / etc.>. <dialogue, if any — see rule 5>.

Strict rules:
1. ALWAYS begin with the literal token "[paw_patrol]"
2. ALWAYS use character names; never "the pup", "the team", or "they"
3. Describe SPECIFIC physical actions with concrete verbs (not "is in the scene")
4. If Ryder is visible, explicitly note that he stands UPRIGHT on two legs
5. If a character speaks, INCLUDE the dialogue verbatim in double quotes, attributed by name. Format: <Name> says, "<exact words heard>". You can hear the audio — transcribe what is actually said. If the speaker is off-screen, say <off-screen voice> says, "...". If multiple characters speak, include each. Skip if there's no clear dialogue (only sound effects, music, or barking).
6. 2-3 sentences total, under 80 words
7. Be factual — no "amazing", "exciting", "adorable", etc.

Output ONLY the single caption — no commentary, no preamble, no bullet points."""


def caption_one(client, video_bytes: bytes, max_retries: int = 4) -> str | None:
    """Call Gemini 2.5 Pro on one video, with exponential-backoff retry."""
    from google.genai import types

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
                    PROMPT,
                ],
            )
            text = (resp.text or "").strip()
            if not text.startswith("[paw_patrol]"):
                # Sometimes Gemini ignores the literal-token instruction; fix it up.
                text = "[paw_patrol] " + text.lstrip("[]paw_patrol] ").lstrip()
            return text
        except Exception as e:
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{max_retries} after error: {type(e).__name__}: {str(e)[:140]}", flush=True)
            time.sleep(wait)
    return None


def process_clip(client, row: dict, idx: int, total: int) -> tuple[dict, str | None]:
    """Read video bytes and caption it. Returns (row, new_caption | None on failure)."""
    media_path = Path(row["media_path"])
    if not media_path.exists():
        print(f"[{idx}/{total}] MISSING: {media_path}", flush=True)
        return row, None
    video_bytes = media_path.read_bytes()
    new_caption = caption_one(client, video_bytes)
    name = media_path.name
    if new_caption:
        print(f"[{idx}/{total}] {name}: {new_caption[:120]}{'...' if len(new_caption) > 120 else ''}", flush=True)
    else:
        print(f"[{idx}/{total}] {name}: FAILED after retries", flush=True)
    return row, new_caption


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel Gemini calls (default 5; Vertex AI handles rate limits)")
    parser.add_argument("--resume", action="store_true",
                        help="If output-csv exists, skip clips already captioned in it")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process first N clips (for testing). 0 = all.")
    args = parser.parse_args()

    from google import genai

    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

    with open(args.input_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.limit > 0:
        rows = rows[:args.limit]
        print(f"Limiting to first {args.limit} clips for testing", flush=True)

    already_done: dict[str, str] = {}
    if args.resume and args.output_csv.exists():
        with open(args.output_csv) as f:
            for r in csv.DictReader(f):
                if r.get("caption", "").startswith("[paw_patrol]"):
                    already_done[r["media_path"]] = r["caption"]
        print(f"Resume: {len(already_done)} clips already captioned, skipping those", flush=True)

    pending = [r for r in rows if r["media_path"] not in already_done]
    print(f"Captioning {len(pending)} clips with {MODEL} via Vertex AI ({args.workers} workers)", flush=True)
    print(f"Skipping {len(rows) - len(pending)} already-done", flush=True)
    print()

    results: dict[str, str] = dict(already_done)  # media_path -> new_caption
    failures: list[str] = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_clip, client, row, i + 1, len(pending)): row
            for i, row in enumerate(pending)
        }
        for fut in as_completed(futures):
            row, new_caption = fut.result()
            if new_caption:
                results[row["media_path"]] = new_caption
            else:
                failures.append(row["media_path"])

    elapsed = time.time() - t0
    print(flush=True)
    print(f"Done in {elapsed:.1f}s. Captioned: {len(results) - len(already_done)}. Failures: {len(failures)}", flush=True)
    if failures:
        print("Failed paths:", flush=True)
        for p in failures:
            print(f"  {p}", flush=True)

    # Write out CSV preserving original row order; keep original caption for any failures
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["caption", "media_path"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "caption": results.get(r["media_path"], r["caption"]),
                "media_path": r["media_path"],
            })

    print(f"Wrote {args.output_csv}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
