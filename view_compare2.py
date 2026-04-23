#!/usr/bin/env python3
"""
view_compare2.py

Clean 2-checkpoint comparison: one scene per row, two columns, all synced.

Sync model
----------
  Video[0] (top-left) is the MASTER.
  All others are FOLLOWERS — synced to master via timeupdate + drift correction.
  Clicking any video, or the transport buttons, controls the master.
  A shared scrub bar + time display at the top.

Audio
-----
  Videos are served with audio. Only the master plays audio (others are muted)
  so you hear one clean audio track, not a chorus.

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

SCRIPT_DIR   = Path(__file__).resolve().parent
RESULTS_DIRS = [
    SCRIPT_DIR / "lora_results_FIX" / "benchmarks",
    SCRIPT_DIR / "lora_results_FIX" / "benchmarks" / "skye_exp3_highcap" / "benchmarks",
]

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
    "skye_exp3_highcap":  "EXP-3  HighCap  (rank 32 + FFN, 15k steps)",
    "skye_exp4_i2v":      "EXP-4  I2V  (rank 32, 2.5k steps)",
    "skye_exp4_10k":           "EXP-4-10K  I2V  (rank 32, 10k steps)",
    "skye_exp3_highres":       "EXP-3-HiRes  T2V+FFN  (768×1024, 50% I2V)",
    "skye_exp5_clean_highres": "EXP-5  Clean+HiRes  I2V+FFN  (768×1024, 80% I2V)",
}


def discover() -> dict:
    """{ exp: { step_int: { scene: rel_path_to_lora_mp4 } } }"""
    data: dict[str, dict[int, dict[str, str]]] = {}
    for results_dir in RESULTS_DIRS:
        if not results_dir.is_dir():
            continue
        for exp_dir in sorted(results_dir.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
                continue
            exp = exp_dir.name
            if exp not in data:
                data[exp] = {}
            for step_dir in sorted(exp_dir.iterdir()):
                if not step_dir.is_dir() or not step_dir.name.startswith("step_"):
                    continue
                step = int(step_dir.name.replace("step_", ""))
                scenes: dict[str, str] = {}
                for f in step_dir.glob("*_lora.mp4"):
                    scene = f.stem.replace("_lora", "")
                    scenes[scene] = str(f.relative_to(SCRIPT_DIR))
                if scenes:
                    data[exp][step] = scenes
    return data


def discover_base() -> dict[str, str]:
    """{ scene: rel_path } for base (no-LoRA) reference videos."""
    base: dict[str, str] = {}
    for results_dir in RESULTS_DIRS:
        base_dir = results_dir / "_base"
        if not base_dir.is_dir():
            continue
        for f in base_dir.glob("*.mp4"):
            scene = f.stem  # no _lora suffix
            if scene not in base:
                base[scene] = str(f.relative_to(SCRIPT_DIR))
    return base


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Compare two checkpoints</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0a0a0a; color: #ddd; font-family: system-ui, sans-serif; }

/* ── header ── */
.header { padding: 16px 20px 12px; border-bottom: 1px solid #1a1a1a;
          display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
h1 { font-size: 15px; color: #fff; white-space: nowrap; }

.picker { display: flex; align-items: center; gap: 10px; }
.picker-block { display: flex; flex-direction: column; gap: 3px; }
.picker-block label { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: .05em; }
.picker-block select { background: #161616; color: #ccc; border: 1px solid #282828;
  border-radius: 7px; padding: 6px 10px; font-size: 12px; min-width: 200px; cursor: pointer; }
.vs { font-size: 18px; color: #333; font-weight: 700; padding-top: 16px; }

/* ── transport bar ── */
.transport { background: #111; border-bottom: 1px solid #1a1a1a;
             padding: 8px 20px; display: flex; align-items: center; gap: 12px; }
.btn { cursor: pointer; padding: 6px 16px; border-radius: 7px; border: none;
       font-size: 12px; font-weight: 600; letter-spacing: .02em; }
.btn-play    { background: #2563eb; color: #fff; }
.btn-play:hover { background: #1d4ed8; }
.btn-pause   { background: #1e1e1e; color: #aaa; border: 1px solid #2a2a2a; }
.btn-restart { background: #1e1e1e; color: #aaa; border: 1px solid #2a2a2a; }
.btn-pause:hover, .btn-restart:hover { background: #252525; }

.progress-wrap { flex: 1; display: flex; align-items: center; gap: 8px; }
#progress { flex: 1; accent-color: #2563eb; cursor: pointer; height: 4px; }
#time-disp { font-size: 11px; color: #555; min-width: 40px; text-align: right; font-variant-numeric: tabular-nums; }
#status { font-size: 11px; color: #c0392b; min-width: 180px; }

/* ── column headers ── */
.col-headers { display: grid; grid-template-columns: 140px 1fr 1fr 0.55fr; }
.chdr { padding: 9px 12px 7px; font-size: 12px; font-weight: 600; text-align: center;
        background: #0f0f0f; border-bottom: 2px solid #1a1a1a; }
.chdr.left  { color: #60a5fa; border-left: 3px solid #2563eb; }
.chdr.right { color: #fb923c; border-left: 3px solid #ea580c; }
.chdr.base  { color: #4ade80; border-left: 3px solid #16a34a; font-size: 10px; }
.chdr.scene-hdr { color: #333; font-size: 10px; text-transform: uppercase; letter-spacing: .07em;
                  text-align: left; }

/* ── rows ── */
.scene-row { display: grid; grid-template-columns: 140px 1fr 1fr 0.55fr;
             border-bottom: 1px solid #111; }
.scene-row:hover { background: #0d0d0d; }
.scene-lbl { display: flex; align-items: center; gap: 8px; padding: 0 10px;
             font-size: 12px; color: #777; font-weight: 500; }

/* ── per-row controls ── */
.row-ctrl { display: flex; flex-direction: column; align-items: center; gap: 4px; flex-shrink: 0; }

.row-play-btn {
  cursor: pointer;
  width: 26px; height: 26px;
  border-radius: 50%;
  border: 1px solid #2a2a2a;
  background: #161616;
  color: #60a5fa;
  font-size: 9px;
  display: flex; align-items: center; justify-content: center;
  transition: background 0.15s, color 0.15s;
  line-height: 1;
}
.row-play-btn:hover { background: #2563eb; color: #fff; border-color: #2563eb; }
.row-play-btn.active { background: #1e3a5f; color: #93c5fd; border-color: #2563eb44; }

/* phase indicator dots */
.phase-pips { display: flex; gap: 3px; }
.phase-pips span {
  width: 5px; height: 5px; border-radius: 50%;
  background: #222; transition: background 0.2s;
}
/* pip 0 = both muted (gray), pip 1 = left audio (blue), pip 2 = right audio (orange) */
.phase-pips span.p0.lit { background: #555; }
.phase-pips span.p1.lit { background: #2563eb; }
.phase-pips span.p2.lit { background: #ea580c; }

.vcell { padding: 6px 8px; }
video { width: 100%; border-radius: 7px; background: #000; display: block; cursor: pointer; }
video.row-master { border-top: 3px solid #2563eb55; }
.vcell.right video { border-top: 3px solid #ea580c55; }
.vcell.base video  { border-top: 3px solid #16a34a55; opacity: 0.85; }

.section-sep { grid-column: 1/-1; background: #0f0f0f; color: #2a2a2a;
  font-size: 10px; text-transform: uppercase; letter-spacing: .07em; padding: 5px 14px; }
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
</div>

<div class="transport">
  <button class="btn btn-restart" onclick="restart()">↺ Restart</button>
  <button class="btn btn-pause"   onclick="pause()">⏸ Pause</button>
  <button class="btn btn-play"    onclick="play()">▶ Play all</button>
  <div class="progress-wrap">
    <input type="range" id="progress" min="0" max="1000" value="0">
    <span id="time-disp">0.0 s</span>
  </div>
  <span id="status"></span>
</div>

<div class="col-headers">
  <div class="chdr scene-hdr">Scene</div>
  <div class="chdr left"  id="hdr-left">—</div>
  <div class="chdr right" id="hdr-right">—</div>
  <div class="chdr base">Base model<br><span style="font-size:9px;color:#555;font-weight:400">(no LoRA)</span></div>
</div>

<div id="rows"></div>

<script>
const DATA        = __DATA__;
const EXP_LABELS  = __EXP_LABELS__;
const SCENE_ORDER = __SCENE_ORDER__;
const SCENE_LABELS= __SCENE_LABELS__;
const BASE_SCENES = __BASE_SCENES__;
const MAIN_SCENES = ["bey","crsms","holoween","party","snow"];

// ── Options list — focus exps only, 1k-step intervals ────────────────────────
const FOCUS_EXPS = ["skye_exp2_standard", "skye_exp2_10k", "skye_exp3_highcap", "skye_exp4_10k", "skye_exp3_highres", "skye_exp5_clean_highres"];
const OPTIONS = [];
Object.entries(DATA)
    .filter(([exp]) => FOCUS_EXPS.includes(exp))
    .sort(([a], [b]) => FOCUS_EXPS.indexOf(a) - FOCUS_EXPS.indexOf(b))
    .forEach(([exp, steps]) => {
        const label = EXP_LABELS[exp] || exp;
        Object.keys(steps).map(Number)
            .filter(s => s % 1000 === 0)
            .sort((a, b) => a - b)
            .forEach(step => {
                OPTIONS.push({ exp, step, label: `${label}  —  step ${step.toLocaleString()}` });
            });
    });

function populateSelect(id, defExp, defStep) {
    const sel = document.getElementById(id);
    OPTIONS.forEach((o, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = o.label;
        if (o.exp === defExp && o.step === defStep) opt.selected = true;
        sel.appendChild(opt);
    });
    sel.addEventListener('change', render);
}
populateSelect('sel-left',  'skye_exp2_10k', 5000);
populateSelect('sel-right', 'skye_exp2_10k', 10000);

// ── Phase definitions ─────────────────────────────────────────────────────────
// phase 0 = idle  (no playback)
// phase 1 = both muted  (pure visual diff)
// phase 2 = left audio, right muted
// phase 3 = left muted, right audio
const PHASES = [
    null,                                         // 0 idle — unused
    { leftMuted: true,  rightMuted: true  },      // 1 visual
    { leftMuted: false, rightMuted: true  },      // 2 left audio
    { leftMuted: true,  rightMuted: false },      // 3 right audio
];

const status   = document.getElementById('status');
const progress = document.getElementById('progress');
const timeDisp = document.getElementById('time-disp');

let rowData = [];  // [{ left, right, btn, pips, phase }]

function updateRowUI(rd) {
    const idle    = rd.phase === 0;
    const paused  = rd.left?.paused ?? true;
    if (rd.btn) {
        rd.btn.textContent = (idle || paused) ? '▶' : '⏸';
        rd.btn.classList.toggle('active', !idle && !paused);
    }
    if (rd.pips) {
        rd.pips.forEach((pip, i) => {
            pip.classList.toggle('lit', i + 1 === rd.phase);
        });
    }
}

function applyPhase(rd) {
    const cfg = PHASES[rd.phase];
    if (!cfg) return;
    if (rd.left)  rd.left.muted  = cfg.leftMuted;
    if (rd.right) rd.right.muted = cfg.rightMuted;
    if (rd.base)  rd.base.muted  = true;   // base is always muted — visual reference only
}

function startPhase(rd, phase) {
    rd.phase = phase;
    applyPhase(rd);
    if (rd.left)  rd.left.currentTime  = 0;
    if (rd.right) rd.right.currentTime = 0;
    if (rd.base)  rd.base.currentTime  = 0;
    rd.left?.play().catch(err => { status.textContent = '▶ failed: ' + err.message; });
    updateRowUI(rd);
}

function attachRowSync(rd) {
    const { left, right, base } = rd;
    left.classList.add('row-master');
    rd.phase = 0;

    // Sync right + base to left
    left.addEventListener('timeupdate', () => {
        const t = left.currentTime;
        if (right && Math.abs(right.currentTime - t) > 0.15) right.currentTime = t;
        if (base  && Math.abs(base.currentTime  - t) > 0.15) base.currentTime  = t;
        if (rowData[0]?.left === left && left.duration) {
            progress.value = Math.round((t / left.duration) * 1000);
            timeDisp.textContent = t.toFixed(1) + ' s';
        }
    });
    left.addEventListener('play',  () => {
        right?.play().catch(()=>{});
        base?.play().catch(()=>{});
        updateRowUI(rd);
    });
    left.addEventListener('pause', () => { right?.pause(); base?.pause(); updateRowUI(rd); });
    left.addEventListener('error', () => {
        status.textContent = 'Load error: ' + (left.error?.message || '?');
    });

    // Auto-advance phases when a phase finishes
    left.addEventListener('ended', () => {
        right?.pause();
        if (rd.phase > 0 && rd.phase < 3) {
            // Skip phases that have no video for that side
            let next = rd.phase + 1;
            if (next === 2 && !rd.left)  next++;   // no left  → skip left-audio phase
            if (next === 3 && !rd.right) next++;   // no right → skip right-audio phase
            if (next <= 3) { startPhase(rd, next); return; }
        }
        rd.phase = 0;
        updateRowUI(rd);
    });

    // Row play button — starts full 3-phase sequence, or pauses/resumes mid-phase
    rd.btn?.addEventListener('click', e => {
        e.stopPropagation();
        if (rd.phase === 0) {
            startPhase(rd, 1);
        } else if (left.paused) {
            left.play().catch(err => { status.textContent = '▶ failed: ' + err.message; });
        } else {
            left.pause();
        }
    });

    // Click video = same as button
    [left, right].filter(Boolean).forEach(v => {
        v.addEventListener('click', () => {
            if (rd.phase === 0) { startPhase(rd, 1); }
            else if (left.paused) { left.play().catch(()=>{}); }
            else { left.pause(); }
        });
    });
}

// ── Global transport ──────────────────────────────────────────────────────────
function play() {
    if (!rowData.length) { status.textContent = 'No videos loaded yet'; return; }
    rowData.forEach(rd => startPhase(rd, 1));
}
function pause()   { rowData.forEach(rd => { rd.left?.pause(); }); }
function restart() { play(); }

// Global scrub seeks all videos
progress.addEventListener('input', () => {
    const first = rowData[0]?.left;
    if (!first?.duration) return;
    const t = (progress.value / 1000) * first.duration;
    rowData.forEach(rd => {
        if (rd.left)  rd.left.currentTime  = t;
        if (rd.right) rd.right.currentTime = t;
        if (rd.base)  rd.base.currentTime  = t;
    });
    timeDisp.textContent = t.toFixed(1) + ' s';
});

// ── Render ───────────────────────────────────────────────────────────────────
function render() {
    const lo = OPTIONS[document.getElementById('sel-left').value];
    const ro = OPTIONS[document.getElementById('sel-right').value];

    document.getElementById('hdr-left').textContent  = lo.label;
    document.getElementById('hdr-right').textContent = ro.label;

    const lScenes = DATA[lo.exp]?.[String(lo.step)] || {};
    const rScenes = DATA[ro.exp]?.[String(ro.step)] || {};

    let html = '';
    let overfit = false;
    SCENE_ORDER.forEach(scene => {
        if (!MAIN_SCENES.includes(scene) && !overfit) {
            overfit = true;
            html += '<div class="scene-row"><div class="section-sep">Overfit / training-data scenes</div></div>';
        }
        const ls = lScenes[scene], rs = rScenes[scene];
        const hasVideos = ls || rs;
        const ctrl = hasVideos ? `
            <div class="row-ctrl">
              <button class="row-play-btn" title="Play: visual → left audio → right audio">▶</button>
              <div class="phase-pips">
                <span class="p0" title="Visual (both muted)"></span>
                <span class="p1" title="Left audio"></span>
                <span class="p2" title="Right audio"></span>
              </div>
            </div>` : '';
        const bs = BASE_SCENES[scene];
        html += `<div class="scene-row" data-scene="${scene}">`;
        html += `<div class="scene-lbl">${ctrl}<span>${SCENE_LABELS[scene] || scene}</span></div>`;
        html += `<div class="vcell left">`  + (ls ? `<video src="/${ls}" preload="auto" playsinline></video>` : '<div style="color:#222;padding:20px;text-align:center">—</div>') + `</div>`;
        html += `<div class="vcell right">` + (rs ? `<video src="/${rs}" preload="auto" playsinline></video>` : '<div style="color:#222;padding:20px;text-align:center">—</div>') + `</div>`;
        html += `<div class="vcell base">` + (bs ? `<video src="/${bs}" preload="auto" muted playsinline></video>` : '<div style="color:#1a1a1a;padding:20px;text-align:center">—</div>') + `</div>`;
        html += `</div>`;
    });

    document.getElementById('rows').innerHTML = html;
    progress.value = 0;
    timeDisp.textContent = '0.0 s';
    status.textContent = '';

    rowData = [];
    document.querySelectorAll('.scene-row[data-scene]').forEach(rowEl => {
        const left  = rowEl.querySelector('.vcell.left video');
        const right = rowEl.querySelector('.vcell.right video');
        const base  = rowEl.querySelector('.vcell.base video');
        const btn   = rowEl.querySelector('.row-play-btn');
        const pips  = [...rowEl.querySelectorAll('.phase-pips span')];
        if (!left && !right) return;
        // All start muted; phases control unmuting
        if (left)  left.muted  = true;
        if (right) right.muted = true;
        if (base)  base.muted  = true;
        const rd = { left: left || right, right: left ? right : null, base: base || null, btn, pips, phase: 0 };
        rowData.push(rd);
        attachRowSync(rd);
    });
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
    data_js = {
        exp: {str(step): scenes for step, scenes in steps.items()}
        for exp, steps in data.items()
    }
    base_scenes = discover_base()

    page_html = (
        HTML
        .replace("__DATA__",        json.dumps(data_js))
        .replace("__EXP_LABELS__",  json.dumps(EXP_LABELS))
        .replace("__SCENE_ORDER__", json.dumps(SCENE_ORDER))
        .replace("__SCENE_LABELS__",json.dumps(SCENE_LABELS))
        .replace("__BASE_SCENES__", json.dumps(base_scenes))
    )

    handler = functools.partial(Handler, page_html=page_html)
    server  = http.server.HTTPServer(("127.0.0.1", args.port), handler)
    print(f"Gallery: http://127.0.0.1:{args.port}")
    print("Defaults: EXP-2-10K step 5,000 (left, with audio) vs step 10,000 (right, muted)")
    print("Press Ctrl+C to stop.\n")
    threading.Timer(0.4, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
