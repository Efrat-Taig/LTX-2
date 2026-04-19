#!/usr/bin/env python3
"""
view_lora_results_gallery.py

Browser gallery for lora_results_FIX benchmark videos.

Structure expected:
  lora_results_FIX/benchmarks/
    _base/                        base model (no LoRA), one video per scene
    <exp_name>/step_XXXXX/        LoRA outputs
      <scene>.mp4                 base (copy; prefer _base/ when available)
      <scene>_lora.mp4            LoRA output

Layout
------
  Rows    = scenes  (main 5 + overfit 5)
  Columns = experiments at a chosen step
  Toggle  = show LoRA only  OR  Base | LoRA side-by-side
  Control = per-experiment step selector (defaults to last)

Usage
-----
    python view_lora_results_gallery.py
    python view_lora_results_gallery.py --port 8767
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import threading
import webbrowser
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
RESULTS_DIR  = SCRIPT_DIR / "lora_results_FIX" / "benchmarks"

EXP_ORDER = [
    "skye_exp1_baseline",
    "skye_exp2_standard",
    "skye_exp2_10k",
    "skye_exp4_i2v",
    "skye_exp4_10k",
]
EXP_LABELS = {
    "skye_exp1_baseline": "EXP-1  Baseline",
    "skye_exp2_standard": "EXP-2  T2V 5k",
    "skye_exp2_10k":      "EXP-2-10K  T2V 10k",
    "skye_exp4_i2v":      "EXP-4  I2V 2.5k",
    "skye_exp4_10k":      "EXP-4-10K  I2V 10k",
}

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


def discover() -> dict:
    """
    Returns {
      "base": {scene: rel_path},
      "exps": {exp: {step_int: {scene: rel_path}}}   (lora paths only)
    }
    """
    result = {"base": {}, "exps": {}}

    # Base videos
    base_dir = RESULTS_DIR / "_base"
    if base_dir.is_dir():
        for f in base_dir.glob("*.mp4"):
            result["base"][f.stem] = str(f.relative_to(SCRIPT_DIR))

    # LoRA videos
    for exp_dir in RESULTS_DIR.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
            continue
        exp = exp_dir.name
        result["exps"][exp] = {}
        for step_dir in sorted(exp_dir.iterdir()):
            if not step_dir.is_dir():
                continue
            step = int(step_dir.name.replace("step_", ""))
            result["exps"][exp][step] = {}
            for f in step_dir.glob("*_lora.mp4"):
                scene = f.stem.replace("_lora", "")
                result["exps"][exp][step][scene] = str(f.relative_to(SCRIPT_DIR))

    return result


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Skye LoRA — Benchmark Results</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0d0d; color: #e0e0e0; font-family: system-ui, sans-serif; font-size: 13px; }
h1 { padding: 16px 20px 6px; font-size: 18px; color: #fff; }
.subtitle { padding: 2px 20px 10px; color: #555; font-size: 12px; }

.controls { padding: 8px 20px 14px; display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap;
            border-bottom: 1px solid #1e1e1e; }
.ctrl-group { display: flex; flex-direction: column; gap: 5px; }
.ctrl-label { color: #666; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }

.tab-group { display: flex; gap: 6px; flex-wrap: wrap; }
.tab-group label { cursor: pointer; padding: 5px 12px; border-radius: 20px;
  background: #1c1c1c; border: 1px solid #333; color: #999; user-select: none; display: inline-block; font-size: 12px; }
.tab-group input[type=radio] { display: none; }
.tab-group input[type=radio]:checked + label { background: #2563eb; border-color: #2563eb; color: #fff; }

.step-row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.step-row span { color: #555; font-size: 11px; min-width: 100px; }
select { background: #1c1c1c; color: #ccc; border: 1px solid #333; border-radius: 6px; padding: 4px 8px; font-size: 12px; }

/* Table */
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th { position: sticky; top: 0; z-index: 10; background: #141414; padding: 10px 10px 8px;
     border-bottom: 2px solid #222; font-size: 12px; color: #999; text-align: center; min-width: 200px; }
th.scene-col { min-width: 130px; text-align: left; padding-left: 16px; }
th .step-badge { font-size: 10px; color: #3b82f6; margin-top: 2px; }
td { padding: 8px 8px; border-bottom: 1px solid #161616; vertical-align: top; }
td.scene-cell { color: #aaa; font-size: 12px; padding-left: 16px; vertical-align: middle; font-weight: 500; }
td.scene-cell .scene-group { font-size: 10px; color: #444; margin-top: 2px; }

.video-pair { display: flex; gap: 4px; }
.video-pair.lora-only video { width: 100%; }
.video-pair.side-by-side video { width: 50%; }
video { border-radius: 5px; background: #111; display: block; cursor: pointer; }
video:hover { outline: 2px solid #2563eb; }
.vid-label { font-size: 10px; color: #444; text-align: center; padding: 2px 0; }
.missing { color: #2a2a2a; font-size: 11px; padding: 20px 0; text-align: center; }

/* Section separator */
tr.section-header td { background: #111; color: #444; font-size: 11px;
  text-transform: uppercase; letter-spacing: .08em; padding: 6px 16px; }
</style>
</head>
<body>
<h1>Skye LoRA — Benchmark Results</h1>
<p class="subtitle">lora_results_FIX &nbsp;·&nbsp; base vs LoRA per experiment &amp; step</p>

<div class="controls">
  <div class="ctrl-group">
    <span class="ctrl-label">View mode</span>
    <div class="tab-group">
      <input type="radio" name="mode" id="lora-only" value="lora-only" checked><label for="lora-only">LoRA only</label>
      <input type="radio" name="mode" id="side-by-side" value="side-by-side"><label for="side-by-side">Base | LoRA</label>
    </div>
  </div>
  <div class="ctrl-group">
    <span class="ctrl-label">Scenes</span>
    <div class="tab-group">
      <input type="radio" name="scenes" id="sc-main" value="main" checked><label for="sc-main">Main (5)</label>
      <input type="radio" name="scenes" id="sc-overfit" value="overfit"><label for="sc-overfit">Overfit (5)</label>
      <input type="radio" name="scenes" id="sc-all" value="all"><label for="sc-all">All (10)</label>
    </div>
  </div>
  <div class="ctrl-group" id="step-controls">
    <span class="ctrl-label">Steps</span>
    <div id="step-selectors"></div>
  </div>
</div>

<div class="table-wrap">
  <table id="main-table"></table>
</div>

<script>
const DATA        = __DATA__;
const EXP_ORDER   = __EXP_ORDER__;
const EXP_LABELS  = __EXP_LABELS__;
const SCENE_ORDER = __SCENE_ORDER__;
const SCENE_LABELS= __SCENE_LABELS__;

const MAIN_SCENES   = ["bey","crsms","holoween","party","snow"];
const OVERFIT_SCENES= ["of_silly_goose","of_eagle","of_lifeguard","of_lift","of_everest"];

// Available experiments (in order, filtered to those with data)
const exps = EXP_ORDER.filter(e => DATA.exps[e]);

// State: chosen step per experiment (default = last)
const chosenStep = {};
exps.forEach(e => {
    const steps = Object.keys(DATA.exps[e]).map(Number).sort((a,b)=>a-b);
    chosenStep[e] = steps[steps.length - 1];
});

// Build step selectors
function buildStepSelectors() {
    const container = document.getElementById('step-selectors');
    container.innerHTML = '';
    exps.forEach(e => {
        const steps = Object.keys(DATA.exps[e]).map(Number).sort((a,b)=>a-b);
        const row = document.createElement('div');
        row.className = 'step-row';
        row.innerHTML = `<span>${EXP_LABELS[e] || e}</span>`;
        const sel = document.createElement('select');
        sel.id = 'step-' + e;
        steps.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s;
            opt.textContent = `step ${s.toLocaleString()}`;
            if (s === chosenStep[e]) opt.selected = true;
            sel.appendChild(opt);
        });
        sel.addEventListener('change', () => { chosenStep[e] = Number(sel.value); render(); });
        row.appendChild(sel);
        container.appendChild(row);
    });
}

function getScenes() {
    const v = document.querySelector('input[name=scenes]:checked').value;
    if (v === 'main')    return MAIN_SCENES;
    if (v === 'overfit') return OVERFIT_SCENES;
    return SCENE_ORDER;
}

function render() {
    const mode   = document.querySelector('input[name=mode]:checked').value;
    const scenes = getScenes();
    const showBase = mode === 'side-by-side';

    let html = '<thead><tr><th class="scene-col">Scene</th>';
    exps.forEach(e => {
        html += `<th>${EXP_LABELS[e]||e}<div class="step-badge">step ${chosenStep[e].toLocaleString()}</div></th>`;
    });
    if (showBase) html += `<th>Base (no LoRA)</th>`;
    html += '</tr></thead><tbody>';

    const mainDone = new Set(MAIN_SCENES);
    let inOverfit = false;
    scenes.forEach(scene => {
        if (!mainDone.has(scene) && !inOverfit) {
            inOverfit = true;
            const colspan = exps.length + 1 + (showBase ? 1 : 0);
            html += `<tr class="section-header"><td colspan="${colspan}">Overfit / training-data scenes</td></tr>`;
        }

        html += `<tr>`;
        html += `<td class="scene-cell">${SCENE_LABELS[scene]||scene}</td>`;

        exps.forEach(e => {
            const step   = chosenStep[e];
            const stepData = (DATA.exps[e]||{})[step] || {};
            const loraSrc  = stepData[scene];
            const baseSrc  = DATA.base[scene];

            html += '<td>';
            if (loraSrc) {
                html += `<div class="video-pair ${mode}">`;
                if (showBase && baseSrc) {
                    html += `<div><video src="/${baseSrc}" controls muted playsinline preload="none"></video><div class="vid-label">Base</div></div>`;
                }
                html += `<div><video src="/${loraSrc}" controls muted playsinline preload="none"></video><div class="vid-label">LoRA</div></div>`;
                html += '</div>';
            } else {
                html += '<div class="missing">—</div>';
            }
            html += '</td>';
        });

        if (showBase) {
            const baseSrc = DATA.base[scene];
            html += '<td>';
            if (baseSrc) {
                html += `<video src="/${baseSrc}" controls muted playsinline preload="none" style="width:100%"></video>`;
            } else {
                html += '<div class="missing">—</div>';
            }
            html += '</td>';
        }

        html += '</tr>';
    });

    html += '</tbody>';
    document.getElementById('main-table').innerHTML = html;
}

buildStepSelectors();
document.querySelectorAll('input[name=mode], input[name=scenes]').forEach(r =>
    r.addEventListener('change', render));
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

    def log_message(self, *_):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    data = discover()
    exps_found = [e for e in EXP_ORDER if e in data["exps"]]
    total = sum(
        len(scenes)
        for steps in data["exps"].values()
        for scenes in steps.values()
    )
    print(f"Found {len(exps_found)} experiments, {total} LoRA videos")

    # Stringify step keys for JSON
    data_js = {
        "base": data["base"],
        "exps": {
            exp: {str(step): scenes for step, scenes in steps.items()}
            for exp, steps in data["exps"].items()
        }
    }

    page_html = (
        HTML
        .replace("__DATA__",        json.dumps(data_js))
        .replace("__EXP_ORDER__",   json.dumps(EXP_ORDER))
        .replace("__EXP_LABELS__",  json.dumps(EXP_LABELS))
        .replace("__SCENE_ORDER__", json.dumps(SCENE_ORDER))
        .replace("__SCENE_LABELS__",json.dumps(SCENE_LABELS))
    )

    handler = functools.partial(Handler, page_html=page_html)
    server = http.server.HTTPServer(("127.0.0.1", args.port), handler)
    print(f"Gallery: http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop.\n")
    threading.Timer(0.4, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
