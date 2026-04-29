#!/usr/bin/env python3
"""Translate every active clip caption in metadata.json to a target language.

No API key required — uses deep_translator's GoogleTranslator (Google's
public web endpoint). Output is a sidecar JSON `captions_<lang>.json`
mapping clip index → translated caption.

Idempotent: if `captions_<lang>.json` exists, missing indices are filled in
without re-translating ones that are already there.

Usage:
    python translate_captions.py --metadata path/to/metadata.json \\
        --lang he --workers 8 --out path/to/captions_he.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def translate_one(text: str, target: str, retries: int = 3) -> str:
    from deep_translator import GoogleTranslator
    last_err = None
    for i in range(retries):
        try:
            return GoogleTranslator(source="en", target=target).translate(text) or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1 + i * 2)
    raise RuntimeError(f"translate failed after {retries}: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--lang", default="he", help="Target language code (default: he = Hebrew)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="Cap total clips translated (0 = all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    md = json.loads(Path(args.metadata).read_text())
    out_path = Path(args.out)
    existing: dict[str, str] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())

    todo: list[tuple[int, str]] = []
    for c in md.get("clips", []):
        if c.get("status", "active") != "active":
            continue
        if not c.get("download_ok", True) or not c.get("prompt"):
            continue
        idx = str(c["index"])
        if idx in existing and existing[idx]:
            continue
        todo.append((c["index"], c["prompt"]))

    if args.limit and args.limit > 0:
        todo = todo[:args.limit]
    print(f"to translate: {len(todo)} (already cached: {len(existing)})", flush=True)
    if not todo:
        print("nothing to do; output already complete.")
        return 0

    results: dict[str, str] = dict(existing)
    failed: list[tuple[int, str]] = []
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(translate_one, text, args.lang): (idx, text) for idx, text in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            idx, text = futures[fut]
            try:
                results[str(idx)] = fut.result()
            except Exception as e:  # noqa: BLE001
                failed.append((idx, str(e)))
                print(f"  [{idx}] FAIL: {e}", flush=True)
            if i % 25 == 0 or i == len(todo):
                elapsed = time.monotonic() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(todo) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(todo)}] {rate:.1f}/s, ETA {eta:.0f}s", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path} ({len(results)} translations, {len(failed)} failed)")
    if failed:
        print("failed indices:", [i for i, _ in failed])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
