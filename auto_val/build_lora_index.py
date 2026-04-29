#!/usr/bin/env python3
"""
build_lora_index.py — discover every LoRA benchmark dir under lora_results/,
generate per-bench compare.html + compare_config.json (when missing), and
write a master lora_results/index.html that links them all.

A "benchmark dir" is any directory that contains:
  - one or more `step_NNNNN/` subdirs holding `<scene>_lora.mp4` files
  - a sibling or local `_base/` dir holding `<scene>.mp4` files

Each scene = mp4 stem in `_base/` (excluding any `*_lora` files).
Each checkpoint = step_NNNNN dir that has at least one matching `<scene>_lora.mp4`.

Usage:
    cd /Users/efrattaig/projects/sm/LTX-2
    python auto_val/build_lora_index.py
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

REPO  = Path("/Users/efrattaig/projects/sm/LTX-2")
ROOT  = (REPO / "lora_results").resolve()
TEMPLATE = ROOT / "goldenPlus150_version1" / "goldenPLUS150_benchmark" / "compare.html"

STEP_RE = re.compile(r"^step_(\d+)$")

# Manual annotations for known benchmark dirs. Key = relative path under lora_results/.
ANNOTATIONS: dict[str, dict[str, str]] = {
    "goldenPlus150_version1/goldenPLUS150_benchmark": {
        "label": "Phase 3 baseline — original captions, rank 32",
        "scenes": "BM_v1 (7 scenes) · 4s @ 832×960",
    },
    "goldenPlus150_version1/goldenPLUS150_bm_v2": {
        "label": "Phase 3 baseline — original captions, rank 32 (10s rerun of step 25k+40k)",
        "scenes": "BM_v1 (7 scenes) · 9.7s @ per-scene dims (1280×672)",
    },
    "goldenPlus150_version1/goldenPLUS150_v2_phase2_bm": {
        "label": "Phase 4 curriculum — NEW captions, rank 32 (warm-started from Phase 3 step 7k) [partial bm machine — superseded]",
        "scenes": "BM_v1 (7 scenes) · 9.7s @ per-scene dims",
    },
    "goldenPlus150_version1/goldenPLUS150_v2_rank128_bm": {
        "label": "Phase 4 — rank 128, original captions, from-scratch [partial bm machine — superseded]",
        "scenes": "BM_v1 (7 scenes) · 9.7s @ per-scene dims",
    },
    "goldenPlus150_version1/goldenPLUS150_v2_rank32_scratch_bm": {
        "label": "Phase 4 — rank 32, original captions, from-scratch (isolates warm-start) [partial bm machine — superseded]",
        "scenes": "BM_v1 (7 scenes) · 9.7s @ per-scene dims",
    },
    "goldenPlus150_version1/goldenPLUS150_v2_phase2": {
        "label": "Phase 2 curriculum — NEW captions, rank 32 (warm-started, step 10k FINAL)",
        "scenes": "5 Skye scenes · 9.7s @ per-scene dims",
        "scene_filter": ["skye_chase_crosswalk_safety", "skye_chase_crosswalk_safety_chase_talk", "skye_chase_garden_bench_sleep", "skye_chase_trampoline_backflip", "skye_helicopter_birthday_gili"],
    },
    "goldenPlus150_version1/goldenPLUS150_v2_rank128": {
        "label": "Rank-128 ablation — original captions, from-scratch (step 10k FINAL)",
        "scenes": "5 Skye scenes · 9.7s @ per-scene dims",
        "scene_filter": ["skye_chase_crosswalk_safety", "skye_chase_crosswalk_safety_chase_talk", "skye_chase_garden_bench_sleep", "skye_chase_trampoline_backflip", "skye_helicopter_birthday_gili"],
    },
    "goldenPlus150_version1/goldenPLUS150_v2_rank32_scratch": {
        "label": "Rank-32 from-scratch — original captions (isolates warm-start effect, step 10k FINAL)",
        "scenes": "5 Skye scenes · 9.7s @ per-scene dims",
        "scene_filter": ["skye_chase_crosswalk_safety", "skye_chase_crosswalk_safety_chase_talk", "skye_chase_garden_bench_sleep", "skye_chase_trampoline_backflip", "skye_helicopter_birthday_gili"],
    },
    "full_episods/full_cast_exp1_benchmark": {
        "label": "Full-episode LoRA — paw_patrol s01e03a, rank 32",
        "scenes": "BM_v1 (7 scenes) · 4s @ 832×960",
    },
    "full_episods/full_cast_exp1_bm_v2": {
        "label": "Full-episode LoRA — 10s rerun of step 6k+10k",
        "scenes": "BM_v1 (7 scenes) · 9.7s @ per-scene dims",
    },
    "chase_lora/benchmarks/chase_exp1_highcap": {
        "label": "Chase LoRA on Chase-only prompts (rank 32 + FFN)",
        "scenes": "Chase prompts (5 scenes)",
    },
    "chase_lora/benchmarks_xeval/goldenPLUS150_benchmark": {
        "label": "Cross-eval: golden 150 LoRA evaluated on Chase prompts",
        "scenes": "Chase prompts (5 scenes)",
    },
}


def find_base_dir(bench: Path) -> Path | None:
    """Find _base for a bench: same-dir, then parent."""
    for cand in (bench / "_base", bench.parent / "_base"):
        if cand.is_dir() and any(cand.glob("*.mp4")):
            return cand
    return None


def find_benchmarks(root: Path) -> list[Path]:
    """Any dir with at least one step_NNNNN/ subdir (lora videos may not exist yet)
    OR any dir that has a _base/ + a sibling-or-self step_*/ shape. Follows symlinks."""
    import os
    benches: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        d = Path(dirpath)
        if not STEP_RE.match(d.name):
            continue
        benches.add(d.parent.resolve())
    return sorted(benches)


def build_config(bench: Path, scene_filter: list[str] | None = None) -> dict | None:
    base = find_base_dir(bench)
    if base is None:
        return None

    scenes = sorted(
        p.stem for p in base.glob("*.mp4")
        if not p.stem.endswith("_lora")
    )
    if scene_filter is not None:
        scenes = [s for s in scenes if s in set(scene_filter)]
    if not scenes:
        return None

    # base path relative to bench
    base_rel = "_base" if (bench / "_base").is_dir() else "../_base"

    checkpoints = [{"label": "base", "folder": base_rel, "filename": "{scene}.mp4"}]
    step_dirs = sorted(
        (p for p in bench.iterdir() if p.is_dir() and STEP_RE.match(p.name)),
        key=lambda p: int(STEP_RE.match(p.name).group(1)),
    )
    for sd in step_dirs:
        # Include every step_NNNNN dir as a column, even if no lora videos yet.
        # Empty cells render as dashed placeholders in compare.html.
        checkpoints.append({
            "label": sd.name,
            "folder": sd.name,
            "filename": "{scene}_lora.mp4",
        })

    return {
        "title": bench.name,
        "subtitle": f"{len(scenes)} scenes × {len(checkpoints)} checkpoints (incl. base)",
        "scenes": scenes,
        "checkpoints": checkpoints,
    }


def count_videos(bench: Path, cfg: dict) -> tuple[int, int]:
    """Return (present, total) for the present-vs-expected video grid."""
    present = total = 0
    for ckpt in cfg["checkpoints"]:
        folder = bench / ckpt["folder"]
        for scene in cfg["scenes"]:
            total += 1
            if (folder / ckpt["filename"].format(scene=scene)).is_file():
                present += 1
    return present, total


def ensure_viewer(bench: Path, cfg: dict) -> None:
    cfg_path = bench / "compare_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    html_path = bench / "compare.html"
    # Always overwrite so template tweaks (size controls, etc.) propagate.
    if html_path.resolve() != TEMPLATE.resolve():
        shutil.copy2(TEMPLATE, html_path)


def render_index(entries: list[dict]) -> str:
    # group by top-level dir under lora_results/
    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(e["group"], []).append(e)

    cards_html = []
    for group in sorted(groups):
        items = sorted(groups[group], key=lambda x: x["rel"])
        cards = []
        for it in items:
            pct = 0 if it["total"] == 0 else round(100 * it["present"] / it["total"])
            badge_class = "ok" if pct == 100 else ("partial" if pct > 0 else "empty")
            ann = ANNOTATIONS.get(it["rel"], {})
            label_html = f'<div class="card-label">{ann["label"]}</div>' if ann.get("label") else ""
            scenes_str = ann.get("scenes", f"{it['scenes']} scenes")
            cards.append(f"""
        <a class="card" href="{it['rel']}/compare.html">
          <div class="card-title">{it['name']}</div>
          {label_html}
          <div class="card-meta">
            {scenes_str} &middot; {it['ckpts']} checkpoint cols
            <span class="badge {badge_class}">{it['present']}/{it['total']} &middot; {pct}%</span>
          </div>
          <div class="card-rel">{it['rel']}</div>
        </a>""")
        cards_html.append(f"""
      <section>
        <h2>{group}</h2>
        <div class="grid">{''.join(cards)}
        </div>
      </section>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>LoRA benchmarks — index</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 24px 32px 60px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      background: #0e0e11; color: #e6e6e6;
    }}
    h1 {{ font-size: 22px; margin: 0 0 4px; }}
    .top-meta {{ color: #888; font-size: 13px; margin-bottom: 28px; }}
    h2 {{ font-size: 14px; color: #9ec; text-transform: uppercase; letter-spacing: 0.05em; margin: 32px 0 12px; }}
    .grid {{
      display: grid; gap: 12px;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    }}
    .card {{
      display: block;
      background: #1a1a20; border: 1px solid #2a2a32;
      border-radius: 10px; padding: 14px 16px;
      text-decoration: none; color: inherit;
      transition: background 0.1s, border-color 0.1s;
    }}
    .card:hover {{ background: #22222a; border-color: #3a3a48; }}
    .card-title {{ font-size: 14px; font-weight: 600; color: #e6e6e6; margin-bottom: 4px; }}
    .card-label {{ font-size: 12px; color: #cde; margin-bottom: 6px; line-height: 1.4; }}
    .card-meta {{ font-size: 12px; color: #aaa; }}
    .card-rel {{ font-size: 11px; color: #555; margin-top: 6px; font-family: ui-monospace, monospace; }}
    .badge {{
      display: inline-block; margin-left: 8px;
      padding: 2px 6px; border-radius: 4px; font-size: 11px;
      background: #333; color: #ddd;
    }}
    .badge.ok      {{ background: #1f5c2c; color: #c8f0d2; }}
    .badge.partial {{ background: #5c4a1f; color: #f0e0c0; }}
    .badge.empty   {{ background: #5c1f1f; color: #ffc0c0; }}
  </style>
</head>
<body>
  <h1>LoRA benchmark index</h1>
  <div class="top-meta">{len(entries)} benchmarks discovered &middot; click any card to open the per-scene compare viewer</div>
  {''.join(cards_html)}
</body>
</html>
"""


def main() -> int:
    benches = find_benchmarks(ROOT)
    print(f"Discovered {len(benches)} benchmark dirs")
    entries = []
    for bench in benches:
        # Find rel path first so we can look up annotations
        try:
            rel_for_lookup = bench.relative_to(ROOT)
        except ValueError:
            rel_for_lookup = None
            for sym in ROOT.iterdir():
                if sym.is_symlink() and bench.is_relative_to(sym.resolve()):
                    rel_for_lookup = Path(sym.name) / bench.relative_to(sym.resolve())
                    break
        ann = ANNOTATIONS.get(str(rel_for_lookup), {}) if rel_for_lookup else {}
        scene_filter = ann.get("scene_filter")
        cfg = build_config(bench, scene_filter=scene_filter)
        if cfg is None:
            print(f"  SKIP (no _base or no scenes): {rel_for_lookup}")
            continue
        ensure_viewer(bench, cfg)
        present, total = count_videos(bench, cfg)
        try:
            rel = bench.relative_to(ROOT)
        except ValueError:
            # Symlinked-in path resolved outside ROOT; map back via skye_old etc.
            for sym in ROOT.iterdir():
                if sym.is_symlink() and bench.is_relative_to(sym.resolve()):
                    rel = Path(sym.name) / bench.relative_to(sym.resolve())
                    break
            else:
                continue
        parts = rel.parts
        entries.append({
            "rel": str(rel),
            "group": parts[0],
            "name": parts[-1],
            "scenes": len(cfg["scenes"]),
            "ckpts": len(cfg["checkpoints"]),
            "present": present,
            "total": total,
        })
        print(f"  OK  {rel}  ({len(cfg['scenes'])}x{len(cfg['checkpoints'])}, {present}/{total} videos)")

    index_html = render_index(entries)
    index_path = ROOT / "index.html"
    index_path.write_text(index_html)
    print(f"\nWrote {index_path}")
    print(f"  → http://localhost:8800/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
