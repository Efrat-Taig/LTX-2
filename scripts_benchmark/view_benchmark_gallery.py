#!/usr/bin/env python3
"""
view_benchmark_gallery.py

Starts a local HTTP server and opens a comparison gallery in your browser.
Videos are served directly from disk — no base64 embedding, loads instantly.

Usage
-----
    # 1. Sync results from server, then open gallery
    python view_benchmark_gallery.py --sync

    # 2. Open gallery with already-synced results
    python view_benchmark_gallery.py

    # 3. Just sync, don't open browser
    python view_benchmark_gallery.py --sync --no-open

Options
-------
    --sync        rsync benchmarks from server before opening
    --port PORT   local port (default: 8765)
    --no-open     don't open browser automatically
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import unquote

SCRIPT_DIR    = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR / "output" / "benchmarks"

SERVER_BENCHMARKS = "efrattaig@35.238.2.51:/home/efrattaig/LTX-2/output/benchmarks/"

SCENE_ORDER = [
    "bey", "crsms", "holoween", "party", "snow",
    "of_silly_goose", "of_eagle", "of_lifeguard", "of_lift", "of_everest",
]
SCENE_LABELS = {
    "bey":            "Lookout (sunny)",
    "crsms":          "Christmas",
    "holoween":       "Halloween",
    "party":          "Party",
    "snow":           "Snowy lookout",
    "of_silly_goose": "🔬 Silly goose",
    "of_eagle":       "🔬 Eagle warning",
    "of_lifeguard":   "🔬 Lifeguard",
    "of_lift":        "🔬 Lift/harness",
    "of_everest":     "🔬 Everest visit",
}

# One accent color per experiment (cycles if >7 experiments)
EXP_COLORS = ["#7cf", "#f9a", "#9f9", "#fb7", "#cf9", "#9cf", "#fc9"]


# ─────────────────────────────────────────────────────────────────────────────

def rsync_from_server() -> bool:
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync", "-av", "--progress",
        "--include=*.mp4", "--include=*/", "--exclude=*",
        SERVER_BENCHMARKS,
        str(BENCHMARKS_DIR) + "/",
    ]
    print(f"Syncing from server…\n  {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode == 0


def collect_videos(base: Path) -> dict:
    """
    Returns:
        {
          "_base": {0: {scene: rel_path_str}},
          "skye_expN_xxx": {step_int: {scene: rel_path_str}},
          ...
        }
    rel_path_str is relative to `base`, using forward slashes.
    _lora.mp4 intermediates are excluded.
    """
    data: dict = {}
    for mp4 in sorted(base.rglob("*.mp4")):
        if mp4.stem.endswith("_lora"):
            continue
        try:
            rel = mp4.relative_to(base)
            parts = rel.parts
            rel_str = "/".join(parts)   # always forward slashes for JS/URL use

            if len(parts) == 2 and parts[0] == "_base":
                scene = parts[1].removesuffix(".mp4")
                data.setdefault("_base", {}).setdefault(0, {})[scene] = rel_str
                continue

            if len(parts) != 3:
                continue
            exp_name, step_dir, scene_file = parts
            if not step_dir.startswith("step_"):
                continue
            step  = int(step_dir.split("_")[1])
            scene = scene_file.removesuffix(".mp4")
        except (ValueError, IndexError):
            continue

        data.setdefault(exp_name, {}).setdefault(step, {})[scene] = rel_str

    return data


# ─────────────────────────────────────────────────────────────────────────────

def build_html(data: dict, port: int) -> str:
    exp_names = sorted(k for k in data if k != "_base")
    exp_colors = {exp: EXP_COLORS[i % len(EXP_COLORS)] for i, exp in enumerate(exp_names)}

    all_scenes = list(SCENE_ORDER) + sorted(
        s for s in {s for exp in data.values() for step in exp.values() for s in step}
        if s not in SCENE_ORDER
    )

    js_data        = json.dumps(data)
    js_scenes      = json.dumps(all_scenes)
    js_scene_labels = json.dumps(SCENE_LABELS)
    js_exp_names   = json.dumps(exp_names)
    js_exp_colors  = json.dumps(exp_colors)

    total_ckpts = sum(len(steps) for exp, steps in data.items() if exp != "_base")
    total_vids  = sum(
        len(scenes)
        for exp, steps in data.items() if exp != "_base"
        for scenes in steps.values()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Skye LoRA Benchmark Gallery</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: system-ui, -apple-system, sans-serif;
    background: #111; color: #ddd;
    padding: 16px 20px;
  }}

  /* ── Header ── */
  .header {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
  h1 {{ font-size: 18px; color: #fff; font-weight: 700; }}
  .stats {{ font-size: 12px; color: #555; }}

  /* ── Control bar ── */
  .controls {{
    display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
    background: #181818; border: 1px solid #2a2a2a; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 20px;
  }}
  .controls label {{ font-size: 12px; color: #888; display: flex; align-items: center; gap: 5px; }}
  .controls select, .controls input[type=number] {{
    background: #222; color: #eee; border: 1px solid #3a3a3a;
    border-radius: 4px; padding: 3px 8px; font-size: 12px;
  }}
  .controls input[type=number] {{ width: 76px; }}
  .sep {{ width: 1px; height: 22px; background: #2a2a2a; align-self: center; }}

  /* ── Tab buttons ── */
  .tabs {{ display: flex; gap: 4px; }}
  .tab {{
    background: #222; color: #777; border: 1px solid #333;
    border-radius: 5px; padding: 4px 12px; cursor: pointer; font-size: 12px;
    transition: all .15s;
  }}
  .tab:hover {{ color: #ccc; }}
  .tab.active {{ background: #1a3550; color: #7cf; border-color: #4a8fc0; }}

  /* ── Section ── */
  .section {{ margin-bottom: 24px; }}
  .section-hdr {{
    font-size: 13px; font-weight: 600; color: #999;
    padding: 5px 0 8px; border-bottom: 1px solid #222; margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px;
  }}
  .dot {{
    display: inline-block; width: 9px; height: 9px;
    border-radius: 50%; flex-shrink: 0;
  }}

  /* ── Video cards ── */
  .row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .card {{
    background: #161616; border: 1px solid #2a2a2a; border-radius: 8px;
    padding: 8px; flex: 0 0 auto; transition: border-color .15s;
  }}
  .card:hover {{ border-color: #4a8fc0; }}
  .card-title {{ font-size: 11px; font-weight: 600; color: #bbb; margin-bottom: 2px; }}
  .card-sub   {{ font-size: 10px; color: #555; margin-bottom: 6px; }}
  video {{
    display: block; border-radius: 4px; background: #000;
    border: 1px solid #1e1e1e;
  }}
  .empty {{
    width: 240px; height: 135px; background: #141414; border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    color: #2a2a2a; font-size: 28px;
  }}

  /* ── No-results msg ── */
  .no-results {{ color: #444; padding: 40px 0; font-size: 14px; }}
</style>
</head>
<body>

<div class="header">
  <h1>Skye LoRA Benchmark Gallery</h1>
  <span class="stats">{total_vids} videos &nbsp;·&nbsp; {total_ckpts} checkpoints &nbsp;·&nbsp;
    {len(exp_names)} experiment(s) &nbsp;·&nbsp; seed=42, 5 s, 768×1024</span>
</div>

<div class="controls">
  <div class="tabs">
    <button class="tab active" id="tab-scene" onclick="setMode('by-scene')">By scene</button>
    <button class="tab"        id="tab-ckpt"  onclick="setMode('by-ckpt')">By checkpoint</button>
  </div>

  <div class="sep"></div>

  <label>Scene
    <select id="scene-filter" onchange="render()">
      <option value="">All scenes</option>
    </select>
  </label>

  <label>Experiment
    <select id="exp-filter" onchange="render()">
      <option value="">All</option>
    </select>
  </label>

  <label>Steps ≥
    <input type="number" id="step-min" value="0" step="500" onchange="render()">
  </label>
  <label>≤
    <input type="number" id="step-max" value="99999" step="500" onchange="render()">
  </label>

  <div class="sep"></div>

  <label><input type="checkbox" id="show-base" onchange="render()"> Base model</label>
  <label><input type="checkbox" id="muted" checked onchange="toggleMute()"> Muted</label>
</div>

<div id="grid"></div>

<script>
const DATA         = {js_data};
const SCENES       = {js_scenes};
const SCENE_LABELS = {js_scene_labels};
const EXP_NAMES    = {js_exp_names};
const EXP_COLORS   = {js_exp_colors};
const PORT         = {port};

let mode = 'by-scene';

/* ── populate filters ── */
const sceneFilter = document.getElementById('scene-filter');
SCENES.forEach(s => {{
  const o = document.createElement('option');
  o.value = s;
  o.textContent = SCENE_LABELS[s] || s;
  sceneFilter.appendChild(o);
}});

const expFilter = document.getElementById('exp-filter');
EXP_NAMES.forEach(e => {{
  const o = document.createElement('option');
  o.value = e;
  o.textContent = e.replace(/^skye_/, '');
  expFilter.appendChild(o);
}});

/* ── mode toggle ── */
function setMode(m) {{
  mode = m;
  document.getElementById('tab-scene').classList.toggle('active', m === 'by-scene');
  document.getElementById('tab-ckpt').classList.toggle('active',  m === 'by-ckpt');
  render();
}}

/* ── mute toggle (applies to all existing videos) ── */
function toggleMute() {{
  const muted = document.getElementById('muted').checked;
  document.querySelectorAll('video').forEach(v => v.muted = muted);
}}

/* ── card builders ── */
function videoCard(relPath, title, sub, expName) {{
  const color = EXP_COLORS[expName] || '#888';
  const muted = document.getElementById('muted').checked ? 'muted' : '';
  const src   = `http://localhost:${{PORT}}/videos/${{relPath}}`;
  return `<div class="card">
    <div class="card-title"><span class="dot" style="background:${{color}}"></span> ${{title}}</div>
    <div class="card-sub">${{sub}}</div>
    <video width="240" controls loop ${{muted}} src="${{src}}" preload="metadata"></video>
  </div>`;
}}

function emptyCard(label) {{
  return `<div class="card">
    <div class="card-title" style="color:#333">${{label}}</div>
    <div class="empty">—</div>
  </div>`;
}}

/* ── main render ── */
function render() {{
  const sceneF  = document.getElementById('scene-filter').value;
  const expF    = document.getElementById('exp-filter').value;
  const stepMin = parseInt(document.getElementById('step-min').value) || 0;
  const stepMax = parseInt(document.getElementById('step-max').value) || 99999;
  const showBase = document.getElementById('show-base').checked;

  const grid = document.getElementById('grid');
  grid.innerHTML = '';

  const filteredScenes = sceneF ? [sceneF] : SCENES;
  const filteredExps   = expF   ? [expF]   : EXP_NAMES;

  if (mode === 'by-scene') {{
    // Each section = one scene.  Cards = all (exp × step) combos.
    filteredScenes.forEach(scene => {{
      let cards = '';

      if (showBase && DATA['_base']?.[0]?.[scene]) {{
        cards += videoCard(DATA['_base'][0][scene], 'Base model', '(no LoRA)', '_base');
      }}

      filteredExps.forEach(exp => {{
        if (!DATA[exp]) return;
        Object.keys(DATA[exp])
          .map(Number)
          .filter(s => s >= stepMin && s <= stepMax)
          .sort((a, b) => a - b)
          .forEach(step => {{
            const rel = DATA[exp]?.[step]?.[scene];
            if (rel) {{
              const expShort = exp.replace(/^skye_/, '');
              cards += videoCard(rel, expShort, `step ${{step.toLocaleString()}}`, exp);
            }}
          }});
      }});

      if (!cards) return;

      const label = SCENE_LABELS[scene] || scene;
      const section = document.createElement('div');
      section.className = 'section';
      section.innerHTML = `
        <div class="section-hdr">${{label}}</div>
        <div class="row">${{cards}}</div>`;
      grid.appendChild(section);
    }});

  }} else {{
    // By checkpoint: each section = (exp, step). Cards = all scenes.
    filteredExps.forEach(exp => {{
      if (!DATA[exp]) return;
      const color = EXP_COLORS[exp] || '#888';
      const expShort = exp.replace(/^skye_/, '');

      Object.keys(DATA[exp])
        .map(Number)
        .filter(s => s >= stepMin && s <= stepMax)
        .sort((a, b) => a - b)
        .forEach(step => {{
          let cards = '';
          filteredScenes.forEach(scene => {{
            const label = SCENE_LABELS[scene] || scene;
            const rel   = DATA[exp]?.[step]?.[scene];
            if (rel) {{
              cards += videoCard(rel, label, `${{expShort}} · step ${{step.toLocaleString()}}`, exp);
            }} else {{
              cards += emptyCard(label);
            }}
          }});

          const section = document.createElement('div');
          section.className = 'section';
          section.innerHTML = `
            <div class="section-hdr">
              <span class="dot" style="background:${{color}}"></span>
              ${{expShort}} &nbsp;·&nbsp; step ${{step.toLocaleString()}}
            </div>
            <div class="row">${{cards}}</div>`;
          grid.appendChild(section);
        }});
    }});
  }}

  if (!grid.children.length) {{
    grid.innerHTML = '<p class="no-results">No videos match the current filters.</p>';
  }}
}}

render();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────

def make_handler(benchmarks_dir: Path, html: str):
    """Return a request handler class that serves the gallery."""
    benchmarks_root = str(benchmarks_dir.resolve())

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = unquote(self.path)

            if path in ("/", "/gallery"):
                body = html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            elif path.startswith("/videos/"):
                rel = path[len("/videos/"):]
                # Prevent path traversal
                video_path = (benchmarks_dir / rel).resolve()
                if not str(video_path).startswith(benchmarks_root):
                    self.send_response(403); self.end_headers(); return
                if not video_path.exists():
                    self.send_response(404); self.end_headers(); return
                # Stream file in chunks
                size = video_path.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(video_path, "rb") as f:
                    while chunk := f.read(1 << 16):
                        self.wfile.write(chunk)

            else:
                self.send_response(404); self.end_headers()

        def log_message(self, *_):
            pass  # suppress per-request noise

    return Handler


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Skye LoRA benchmark gallery — local HTTP server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sync",    action="store_true", help="rsync from server first")
    parser.add_argument("--port",    type=int, default=8765, help="local port")
    parser.add_argument("--no-open", action="store_true",   help="don't open browser")
    args = parser.parse_args()

    if args.sync:
        ok = rsync_from_server()
        if not ok:
            print("rsync failed — check SSH access to the server.", file=sys.stderr)
            return 1
        print()

    if not BENCHMARKS_DIR.exists():
        print(f"Benchmarks dir not found: {BENCHMARKS_DIR}")
        print("Run with --sync to pull results from the server.")
        return 1

    data = collect_videos(BENCHMARKS_DIR)
    exp_count = len([k for k in data if k != "_base"])
    vid_count = sum(
        len(scenes)
        for exp, steps in data.items() if exp != "_base"
        for scenes in steps.values()
    )

    if not data or (exp_count == 0 and "_base" not in data):
        print("No benchmark videos found yet.")
        return 0

    print(f"Found {vid_count} video(s) across {exp_count} experiment(s).")

    html    = build_html(data, args.port)
    handler = make_handler(BENCHMARKS_DIR, html)
    server  = http.server.HTTPServer(("127.0.0.1", args.port), handler)

    url = f"http://localhost:{args.port}"
    print(f"Gallery: {url}  (Ctrl-C to stop)")

    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
