#!/usr/bin/env python3
"""
view_samples_gallery.py

Syncs validation-sample videos from the training server and opens a browser
gallery for comparing experiments side-by-side.

Layout
------
  Columns = experiments  (exp1_baseline, exp2_standard, exp2_10k, exp4_i2v, exp4_10k …)
  Rows    = checkpoint steps  (aligned across experiments)
  Tabs    = validation prompt  (1 = helicopter, 2 = aerial roll, 3 = Lookout tower)

Usage
-----
    python view_samples_gallery.py           # sync from server, then open gallery
    python view_samples_gallery.py --no-sync # skip sync (use already-downloaded videos)
    python view_samples_gallery.py --sync-only
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

SCRIPT_DIR  = Path(__file__).resolve().parent
SAMPLES_DIR = SCRIPT_DIR / "output" / "samples"

SERVER   = "efrattaig@35.238.2.51"
REMOTE_OUTPUTS = f"{SERVER}:/home/efrattaig/LTX-2/outputs/"

PROMPT_LABELS = {
    "1": "🚁 Helicopter cockpit",
    "2": "🌀 Aerial barrel roll",
    "3": "🏠 Lookout tower",
}

EXP_ORDER = [
    "skye_exp1_baseline",
    "skye_exp2_standard",
    "skye_exp2_10k",
    "skye_exp4_i2v",
    "skye_exp4_10k",
    "skye_exp3_highcap",
    "skye_exp3_highres",
    "skye_exp4_highres",
]
EXP_LABELS = {
    "skye_exp1_baseline":  "EXP-1  Baseline (rank16, 1k steps)",
    "skye_exp2_standard":  "EXP-2  T2V standard (rank32, 5k steps)",
    "skye_exp2_10k":       "EXP-2-10K  T2V extended (rank32, 10k steps)",
    "skye_exp4_i2v":       "EXP-4  I2V focus (rank32, 2.5k steps)",
    "skye_exp4_10k":       "EXP-4-10K  I2V extended (rank32, 10k steps)",
    "skye_exp3_highcap":   "EXP-3  Highcap T2V +FFN (rank32, 15k steps)",
    "skye_exp3_highres":   "EXP-3-Highres  Portrait T2V +FFN",
    "skye_exp4_highres":   "EXP-4-Highres  Portrait I2V +FFN ⭐",
}


# ── Sync ──────────────────────────────────────────────────────────────────────

def sync_from_server() -> None:
    print("Syncing validation samples from server…")
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "rsync", "-avz", "--include=*/", "--include=samples/***",
            "--exclude=*", REMOTE_OUTPUTS, str(SAMPLES_DIR) + "/",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("rsync error:", result.stderr[-500:], file=sys.stderr)
        sys.exit(1)

    # Re-encode any video that might be yuv444p → yuv420p for macOS/browser compat
    videos = list(SAMPLES_DIR.rglob("*.mp4"))
    need_reencode = [v for v in videos if not (v.parent / (v.stem + ".ok")).exists()]
    if need_reencode:
        print(f"Re-encoding {len(need_reencode)} video(s) for macOS compatibility…")
        for v in need_reencode:
            tmp = v.with_suffix(".tmp.mp4")
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", str(v),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                 "-c:a", "aac", "-b:a", "128k", str(tmp)],
                capture_output=True,
            )
            if r.returncode == 0:
                v.unlink()
                tmp.rename(v)
                (v.parent / (v.stem + ".ok")).touch()  # mark as done so we skip next time
            else:
                tmp.unlink(missing_ok=True)
    print(f"Sync complete. {len(videos)} videos in {SAMPLES_DIR}\n")


# ── Discover ──────────────────────────────────────────────────────────────────

def discover() -> dict:
    """Return {exp_name: {step_int: {prompt_idx: rel_path}}}"""
    data: dict[str, dict[int, dict[str, str]]] = {}
    for exp_dir in SAMPLES_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
        samples_dir = exp_dir / "samples"
        if not samples_dir.is_dir():
            continue
        exp = exp_dir.name
        data[exp] = {}
        for f in samples_dir.glob("step_*.mp4"):
            parts = f.stem.split("_")   # step_000250_1 → ['step', '000250', '1']
            if len(parts) != 3:
                continue
            step = int(parts[1])
            prompt = parts[2]
            data[exp].setdefault(step, {})[prompt] = str(f.relative_to(SCRIPT_DIR))
    return data


# ── HTTP server ───────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Skye LoRA — Validation Gallery</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0d0d; color: #e0e0e0; font-family: system-ui, sans-serif; font-size: 13px; }
h1 { padding: 16px 20px 8px; font-size: 18px; color: #fff; }
.controls { padding: 8px 20px 12px; display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
.tab-group label { cursor: pointer; padding: 6px 14px; border-radius: 20px;
  background: #222; border: 1px solid #444; color: #aaa; user-select: none; display: inline-block; }
.tab-group input[type=radio] { display: none; }
.tab-group input[type=radio]:checked + label { background: #3b82f6; border-color: #3b82f6; color: #fff; }
.filter-label { color: #888; font-size: 12px; }
select { background: #222; color: #ccc; border: 1px solid #444; border-radius: 6px; padding: 5px 8px; }
table { width: 100%; border-collapse: collapse; }
th { position: sticky; top: 0; background: #1a1a1a; padding: 10px 8px; border-bottom: 2px solid #333;
     font-size: 12px; color: #bbb; text-align: center; min-width: 220px; }
th.step-col { min-width: 80px; color: #666; }
td { padding: 6px 8px; border-bottom: 1px solid #1a1a1a; vertical-align: top; }
td.step-cell { color: #555; font-size: 12px; text-align: center; padding-top: 12px; }
td.video-cell { }
video { width: 100%; border-radius: 6px; background: #111; display: block; cursor: pointer; }
video:hover { outline: 2px solid #3b82f6; }
.empty { height: 40px; }
.exp-label { font-size: 11px; color: #666; padding: 2px 0 4px; }
</style>
</head>
<body>
<h1>Skye LoRA — Validation Sample Gallery</h1>
<div class="controls">
  <div class="tab-group" id="prompt-tabs">
    <span class="filter-label">Prompt:</span>
    <input type="radio" name="prompt" id="p1" value="1" checked><label for="p1">🚁 Helicopter cockpit</label>
    <input type="radio" name="prompt" id="p2" value="2"><label for="p2">🌀 Aerial barrel roll</label>
    <input type="radio" name="prompt" id="p3" value="3"><label for="p3">🏠 Lookout tower</label>
  </div>
  <div>
    <span class="filter-label">Steps: </span>
    <select id="step-filter">
      <option value="all">All steps</option>
      <option value="last1">Last checkpoint only</option>
      <option value="last3">Last 3 checkpoints</option>
      <option value="last5">Last 5 checkpoints</option>
    </select>
  </div>
</div>

<div id="table-container"></div>

<script>
const DATA = __DATA__;
const EXP_ORDER = __EXP_ORDER__;
const EXP_LABELS = __EXP_LABELS__;

// Filter to experiments that have data
const exps = EXP_ORDER.filter(e => DATA[e] && Object.keys(DATA[e]).length > 0);

// Collect all steps per experiment
function getSteps(exp, filter) {
    const all = Object.keys(DATA[exp] || {}).map(Number).sort((a,b) => a-b);
    if (filter === 'all') return all;
    const n = filter === 'last1' ? 1 : filter === 'last3' ? 3 : 5;
    return all.slice(-n);
}

function render() {
    const prompt = document.querySelector('input[name=prompt]:checked').value;
    const filter = document.getElementById('step-filter').value;

    // Build step→set-of-exps mapping for row alignment
    const stepSet = new Set();
    exps.forEach(e => getSteps(e, filter).forEach(s => stepSet.add(s)));
    const steps = [...stepSet].sort((a,b) => a-b);

    let html = '<table><thead><tr><th class="step-col">Step</th>';
    exps.forEach(e => {
        html += `<th>${EXP_LABELS[e] || e}</th>`;
    });
    html += '</tr></thead><tbody>';

    steps.forEach(step => {
        html += `<tr><td class="step-cell">${step.toLocaleString()}</td>`;
        exps.forEach(e => {
            const expData = DATA[e] || {};
            const stepData = expData[step] || {};
            const src = stepData[prompt];
            if (src) {
                html += `<td class="video-cell"><video src="/${src}" controls muted playsinline preload="none"></video></td>`;
            } else {
                html += `<td><div class="empty"></div></td>`;
            }
        });
        html += '</tr>';
    });

    html += '</tbody></table>';
    document.getElementById('table-container').innerHTML = html;
}

document.querySelectorAll('input[name=prompt]').forEach(r => r.addEventListener('change', render));
document.getElementById('step-filter').addEventListener('change', render);
render();
</script>
</body>
</html>
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, data_json, exp_order_json, exp_labels_json, **kwargs):
        self._data = data_json
        self._exp_order = exp_order_json
        self._exp_labels = exp_labels_json
        super().__init__(*args, directory=str(SCRIPT_DIR), **kwargs)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (
                HTML_TEMPLATE
                .replace("__DATA__", self._data)
                .replace("__EXP_ORDER__", self._exp_order)
                .replace("__EXP_LABELS__", self._exp_labels)
            )
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def log_message(self, *_):
        pass


def serve(port: int) -> None:
    data = discover()
    # Re-key data so steps are ints serializable as JSON object keys (strings)
    data_js: dict[str, dict[str, dict[str, str]]] = {
        exp: {str(step): prompts for step, prompts in steps.items()}
        for exp, steps in data.items()
    }
    data_json     = json.dumps(data_js)
    exp_order_json = json.dumps(EXP_ORDER)
    exp_labels_json = json.dumps(EXP_LABELS)

    handler = functools.partial(
        Handler,
        data_json=data_json,
        exp_order_json=exp_order_json,
        exp_labels_json=exp_labels_json,
    )
    server = http.server.HTTPServer(("127.0.0.1", port), handler)

    exps_found = [e for e in EXP_ORDER if e in data]
    total_videos = sum(
        len(prompts)
        for steps in data.values()
        for prompts in steps.values()
    )
    print(f"Gallery ready:  http://127.0.0.1:{port}")
    print(f"  Experiments : {exps_found}")
    print(f"  Videos      : {total_videos}")
    print("Press Ctrl+C to stop.\n")

    threading.Timer(0.4, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    server.serve_forever()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Gallery for Skye LoRA validation sample videos")
    parser.add_argument("--no-sync",   action="store_true", help="Skip server sync")
    parser.add_argument("--sync-only", action="store_true", help="Sync then exit")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    if not args.no_sync:
        sync_from_server()

    if args.sync_only:
        return

    serve(args.port)


if __name__ == "__main__":
    main()
