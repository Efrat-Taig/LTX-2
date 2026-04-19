#!/usr/bin/env python3
"""
view_compare2.py

Focused 2-checkpoint comparison gallery.
Pick any two experiment+step combos; all scene videos play in perfect sync.

Usage
-----
    python view_compare2.py
    python view_compare2.py --port 8768
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import threading
import webbrowser
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "lora_results_FIX" / "benchmarks"

SCENE_ORDER = ["bey", "crsms", "holoween", "party", "snow",
               "of_silly_goose", "of_eagle", "of_lifeguard", "of_lift", "of_everest"]
SCENE_LABELS = {
    "bey":            "Lookout (sunny)",
    "crsms":          "Christmas",
    "holoween":       "Halloween",
    "party":          "Party",
    "snow":           "Snowy lookout",
    "of_silly_goose": "🔬 Silly goose",
    "of_eagle":       "🔬 Eagle warning",
    "of_lifeguard":   "🔬 Lifeguard",
    "of_lift":        "🔬 Lift / harness",
    "of_everest":     "🔬 Everest visit",
}
EXP_LABELS = {
    "skye_exp1_baseline": "EXP-1  Baseline  (rank 16, 1k steps)",
    "skye_exp2_standard": "EXP-2  T2V  (rank 32, 5k steps)",
    "skye_exp2_10k":      "EXP-2-10K  T2V  (rank 32, 10k steps)",
    "skye_exp4_i2v":      "EXP-4  I2V  (rank 32, 2.5k steps)",
    "skye_exp4_10k":      "EXP-4-10K  I2V  (rank 32, 10k steps)",
}


def discover() -> dict:
    """{ exp: { step_int: { scene: rel_path_to_lora_mp4 } } }"""
    data: dict[str, dict[int, dict[str, str]]] = {}
    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
            continue
        exp = exp_dir.name
        data[exp] = {}
        for step_dir in sorted(exp_dir.iterdir()):
            if not step_dir.is_dir():
                continue
            step = int(step_dir.name.replace("step_", ""))
            scenes: dict[str, str] = {}
            for f in step_dir.glob("*_lora.mp4"):
                scene = f.stem.replace("_lora", "")
                scenes[scene] = str(f.relative_to(SCRIPT_DIR))
            if scenes:
                data[exp][step] = scenes
    return data


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Compare two checkpoints</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0a0a0a; color: #ddd; font-family: system-ui, sans-serif; }

/* ── header ── */
.header { padding: 18px 24px 14px; border-bottom: 1px solid #1a1a1a;
          display: flex; align-items: center; gap: 24px; flex-wrap: wrap; }
.header h1 { font-size: 16px; color: #fff; white-space: nowrap; }

.picker { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.picker-block { display: flex; flex-direction: column; gap: 4px; }
.picker-block label { font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: .05em; }
.picker-block select { background: #161616; color: #ccc; border: 1px solid #2a2a2a;
  border-radius: 8px; padding: 7px 10px; font-size: 13px; min-width: 220px; cursor: pointer; }
.vs { font-size: 20px; color: #333; font-weight: 700; padding-top: 18px; }

.transport { display: flex; gap: 8px; align-items: center; margin-left: auto; }
.btn { cursor: pointer; padding: 8px 18px; border-radius: 8px; border: none; font-size: 13px;
       font-weight: 600; transition: background .15s; }
.btn-play   { background: #2563eb; color: #fff; }
.btn-play:hover { background: #1d4ed8; }
.btn-pause  { background: #1e1e1e; color: #aaa; border: 1px solid #2a2a2a; }
.btn-pause:hover { background: #252525; }
.btn-restart{ background: #1e1e1e; color: #aaa; border: 1px solid #2a2a2a; }
.btn-restart:hover { background: #252525; }

/* ── column headers ── */
.col-headers { display: grid; grid-template-columns: 140px 1fr 1fr;
               gap: 0; border-bottom: 2px solid #1a1a1a; }
.col-hdr { padding: 10px 12px; font-size: 13px; font-weight: 600; text-align: center;
           background: #111; }
.col-hdr.left  { color: #60a5fa; border-left: 3px solid #2563eb; }
.col-hdr.right { color: #fb923c; border-left: 3px solid #ea580c; }
.col-hdr.scene-hdr { color: #444; font-size: 11px; text-transform: uppercase;
                     letter-spacing: .06em; text-align: left; }

/* ── rows ── */
.scene-row { display: grid; grid-template-columns: 140px 1fr 1fr;
             border-bottom: 1px solid #111; }
.scene-row:hover { background: #0e0e0e; }
.scene-label { padding: 0 12px; display: flex; align-items: center;
               font-size: 13px; color: #888; font-weight: 500; }
.scene-label .overfit-tag { font-size: 10px; color: #333; display: block; margin-top: 2px; }

.video-cell { padding: 8px 10px; }
video { width: 100%; border-radius: 8px; background: #000; display: block;
        cursor: pointer; outline: none; }
video:hover { box-shadow: 0 0 0 2px #2563eb44; }
.left-cell  video { border-top: 3px solid #2563eb22; }
.right-cell video { border-top: 3px solid #ea580c22; }

.section-divider { grid-column: 1/-1; background: #111;
  font-size: 10px; color: #333; text-transform: uppercase; letter-spacing: .08em;
  padding: 5px 14px; }
</style>
</head>
<body>

<div class="header">
  <h1>Compare two checkpoints</h1>
  <div class="picker">
    <div class="picker-block">
      <label>Left (blue)</label>
      <select id="sel-left"></select>
    </div>
    <span class="vs">vs</span>
    <div class="picker-block">
      <label>Right (orange)</label>
      <select id="sel-right"></select>
    </div>
  </div>
  <div class="transport">
    <button class="btn btn-restart" onclick="restartAll()">↺ Restart</button>
    <button class="btn btn-pause"   onclick="pauseAll()">⏸ Pause</button>
    <button class="btn btn-play"    onclick="playAll()">▶ Play all</button>
  </div>
</div>

<div class="col-headers">
  <div class="col-hdr scene-hdr">Scene</div>
  <div class="col-hdr left"  id="hdr-left">—</div>
  <div class="col-hdr right" id="hdr-right">—</div>
</div>

<div id="rows"></div>

<script>
const DATA        = __DATA__;
const EXP_LABELS  = __EXP_LABELS__;
const SCENE_ORDER = __SCENE_ORDER__;
const SCENE_LABELS= __SCENE_LABELS__;
const MAIN_SCENES = ["bey","crsms","holoween","party","snow"];

// Build flat list of options: [{exp, step, label}]
const OPTIONS = [];
Object.entries(DATA).forEach(([exp, steps]) => {
  const expLabel = EXP_LABELS[exp] || exp;
  Object.keys(steps).map(Number).sort((a,b)=>a-b).forEach(step => {
    OPTIONS.push({ exp, step, label: `${expLabel}  —  step ${step.toLocaleString()}` });
  });
});

function populateSelect(selId, defaultExp, defaultStep) {
  const sel = document.getElementById(selId);
  OPTIONS.forEach((o, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = o.label;
    if (o.exp === defaultExp && o.step === defaultStep) opt.selected = true;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', render);
}

// Default: exp2_10k step 5000 vs step 10000
const defExp = 'skye_exp2_10k';
populateSelect('sel-left',  defExp, 5000);
populateSelect('sel-right', defExp, 10000);

// ── Sync engine ─────────────────────────────────────────────────────────────
let syncing = false;

function allVideos() { return [...document.querySelectorAll('video')]; }

function attachSync() {
  allVideos().forEach(v => {
    v.onplay = () => {
      if (syncing) return; syncing = true;
      allVideos().filter(x=>x!==v).forEach(x => {
        x.currentTime = v.currentTime; x.play().catch(()=>{});
      });
      syncing = false;
    };
    v.onpause = () => {
      if (syncing) return; syncing = true;
      allVideos().filter(x=>x!==v).forEach(x => x.pause());
      syncing = false;
    };
    v.onseeked = () => {
      if (syncing) return; syncing = true;
      allVideos().filter(x=>x!==v).forEach(x => { x.currentTime = v.currentTime; });
      syncing = false;
    };
  });
}

function playAll() {
  const vs = allVideos(); if (!vs.length) return;
  syncing = true;
  vs.forEach(v => { v.currentTime = vs[0].currentTime; v.play().catch(()=>{}); });
  syncing = false;
}
function pauseAll()   { allVideos().forEach(v => v.pause()); }
function restartAll() {
  syncing = true;
  allVideos().forEach(v => { v.currentTime = 0; v.pause(); });
  syncing = false;
  setTimeout(() => allVideos().forEach(v => v.play().catch(()=>{})), 50);
}

// ── Render ───────────────────────────────────────────────────────────────────
function render() {
  const leftOpt  = OPTIONS[document.getElementById('sel-left').value];
  const rightOpt = OPTIONS[document.getElementById('sel-right').value];

  document.getElementById('hdr-left').textContent  = leftOpt.label;
  document.getElementById('hdr-right').textContent = rightOpt.label;

  const leftScenes  = DATA[leftOpt.exp]?.[leftOpt.step]  || {};
  const rightScenes = DATA[rightOpt.exp]?.[rightOpt.step] || {};

  let html = '';
  let inOverfit = false;

  SCENE_ORDER.forEach(scene => {
    const isOverfit = !MAIN_SCENES.includes(scene);
    if (isOverfit && !inOverfit) {
      inOverfit = true;
      html += `<div class="scene-row"><div class="section-divider">Overfit / training-data scenes</div></div>`;
    }

    const leftSrc  = leftScenes[scene];
    const rightSrc = rightScenes[scene];

    html += `<div class="scene-row">`;
    html += `<div class="scene-label">${SCENE_LABELS[scene]||scene}</div>`;

    html += `<div class="video-cell left-cell">`;
    if (leftSrc) html += `<video src="/${leftSrc}" preload="auto" muted playsinline></video>`;
    else         html += `<div style="color:#333;padding:20px;text-align:center">—</div>`;
    html += `</div>`;

    html += `<div class="video-cell right-cell">`;
    if (rightSrc) html += `<video src="/${rightSrc}" preload="auto" muted playsinline></video>`;
    else          html += `<div style="color:#333;padding:20px;text-align:center">—</div>`;
    html += `</div>`;

    html += `</div>`;
  });

  document.getElementById('rows').innerHTML = html;
  attachSync();
}

render();
</script>
</body>
</html>
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, page_html: str, **kwargs):
        self._html = page_html
        super().__init__(*args, directory=str(SCRIPT_DIR), **kwargs)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = self._html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def log_message(self, *_): pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()

    data = discover()
    # Stringify int step keys for JSON
    data_js = {
        exp: {str(step): scenes for step, scenes in steps.items()}
        for exp, steps in data.items()
    }

    page_html = (
        HTML
        .replace("__DATA__",        json.dumps(data_js))
        .replace("__EXP_LABELS__",  json.dumps(EXP_LABELS))
        .replace("__SCENE_ORDER__", json.dumps(SCENE_ORDER))
        .replace("__SCENE_LABELS__",json.dumps(SCENE_LABELS))
    )

    handler = functools.partial(Handler, page_html=page_html)
    server  = http.server.HTTPServer(("127.0.0.1", args.port), handler)
    print(f"Gallery: http://127.0.0.1:{args.port}")
    print("Defaults to EXP-2-10K  step 5,000  vs  step 10,000")
    print("Press Ctrl+C to stop.\n")
    threading.Timer(0.4, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
