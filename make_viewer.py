#!/usr/bin/env python3
"""
make_viewer.py

Generate a single HTML page for comparing all exp_no_lora variants side-by-side.

Features
--------
- One row per scene: baseline + each variant, each cell labeled.
- Synchronized playback per scene: "play all" button plays every cell together.
- Click a cell to solo its audio (others mute).
- Loops each video so you can watch mid-clip artifacts repeatedly.
- All videos are referenced by relative path — copy the HTML + source MP4s
  anywhere and it still works.

Output: output/exp_no_lora_comparison/viewer.html
"""

from __future__ import annotations

from html import escape
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CHASE_BASE = SCRIPT_DIR / "lora_results" / "chase_lora" / "benchmarks" / "_base"
SKYE_BASE  = SCRIPT_DIR / "lora_results_FIX" / "benchmarks" / "_base"
HELI_BASE  = (
    SCRIPT_DIR / "output" / "output2"
    / "run_skye_like_api__skye_helicopter_birthday_gili__768x1024_15steps_8.04s_seed1660213644.mp4"
)
EXP_ROOT   = SCRIPT_DIR / "lora_results" / "exp_no_lora"
OUT_DIR    = SCRIPT_DIR / "output" / "exp_no_lora_comparison"
HTML_PATH  = OUT_DIR / "viewer.html"


def chase(name: str, file: str) -> dict:
    return {
        "name": f"chase_{name}",
        "display": f"Chase — {name}",
        "baseline": CHASE_BASE / file,
        "variants": ["baseline_notext", "t1_prompt", "t2_stg0.5", "t2_stg1.0", "t2_stg2.0"],
        "exp_key": f"chase_{name}",
    }


def skye(name: str, file: str, *, acfg: bool = False) -> dict:
    extra = ["t3_acfg12", "t3_acfg18"] if acfg else []
    return {
        "name": f"skye_{name}",
        "display": f"Skye — {name}",
        "baseline": SKYE_BASE / file,
        "variants": ["baseline_notext", "t1_prompt", "t2_stg0.5", "t2_stg1.0", "t2_stg2.0"] + extra,
        "exp_key": f"skye_{name}",
    }


SCENES = [
    chase("normal",    "normal.mp4"),
    chase("halloween", "halloween.mp4"),
    chase("crms",      "crms.mp4"),
    chase("snow",      "snow.mp4"),
    chase("party",     "party.mp4"),
    skye("bey",        "bey.mp4",     acfg=True),
    skye("crsms",      "crsms.mp4",   acfg=True),
    skye("holoween",   "holoween.mp4"),
    skye("party",      "party.mp4"),
    skye("snow",       "snow.mp4",    acfg=True),
    {
        "name": "skye_helicopter",
        "display": "Skye — helicopter",
        "baseline": HELI_BASE,
        "variants": ["baseline_notext", "t1_prompt", "t2_stg0.5", "t2_stg1.0", "t2_stg2.0", "t3_acfg12", "t3_acfg18"],
        "exp_key": "skye_helicopter",
    },
]


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(OUT_DIR.resolve().parent.parent)).replace("\\", "/")


def collect(scene: dict) -> list[tuple[str, Path]]:
    clips: list[tuple[str, Path]] = []
    if scene["baseline"].exists():
        clips.append(("baseline", scene["baseline"]))
    for v in scene["variants"]:
        p = EXP_ROOT / v / f"{scene['exp_key']}.mp4"
        if p.exists():
            clips.append((v, p))
    return clips


CSS = """
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, sans-serif; background: #0a0a0a; color: #eee;
       margin: 0; padding: 24px; }
h1 { color: #ffdc00; margin: 0 0 4px 0; }
.lead { color: #aaa; margin: 0 0 20px 0; font-size: 14px; }
nav { position: sticky; top: 0; background: #0a0a0a; padding: 10px 0 14px 0;
      border-bottom: 1px solid #222; margin-bottom: 18px; z-index: 10; font-size: 13px; }
nav a { color: #9cf; margin-right: 14px; text-decoration: none; }
nav a:hover { color: #fff; }
.scene { border: 1px solid #222; border-radius: 10px; margin-bottom: 26px; padding: 18px;
         background: #141414; scroll-margin-top: 60px; }
.scene-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.scene-title { font-size: 18px; color: #ffdc00; font-weight: 600; }
.ctrl button { background: #1f1f1f; color: #fff; border: 1px solid #444; padding: 6px 12px;
               cursor: pointer; margin-left: 6px; border-radius: 5px; font-size: 13px; }
.ctrl button:hover { background: #2a2a2a; border-color: #666; }
.row { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
.cell { display: flex; flex-direction: column; border: 3px solid transparent; border-radius: 6px;
        transition: border-color 0.15s, box-shadow 0.15s; }
.cell .label { background: #1f1f1f; color: #ddd; padding: 6px 8px; font-size: 12px;
               text-align: center; font-weight: 600; letter-spacing: 0.02em; border-radius: 3px 3px 0 0; }
.cell video { width: 100%; background: black; cursor: pointer; display: block;
              border-radius: 0 0 3px 3px; }
.cell.playing { border-color: #2ee062; box-shadow: 0 0 12px rgba(46, 224, 98, 0.5); }
.cell.playing .label { background: #2ee062; color: #000; }
details.prompts { margin-top: 14px; border: 1px solid #2a2a2a; border-radius: 6px; background: #101010; }
details.prompts > summary { cursor: pointer; padding: 8px 12px; color: #9cf; font-size: 13px;
                            list-style: none; user-select: none; }
details.prompts > summary::-webkit-details-marker { display: none; }
details.prompts > summary::before { content: "▸ "; color: #666; }
details.prompts[open] > summary::before { content: "▾ "; color: #9cf; }
.prompts-body { padding: 10px 12px 14px 12px; display: grid; grid-template-columns: 1fr 1fr;
                gap: 10px; }
.prompt-block { display: flex; flex-direction: column; gap: 4px; }
.prompt-block .head { font-size: 11px; color: #888; letter-spacing: 0.04em;
                      text-transform: uppercase; }
.prompt-block textarea, .prompt-block .text-ro {
  background: #0a0a0a; color: #ddd; border: 1px solid #333; border-radius: 4px;
  padding: 8px 10px; font-family: inherit; font-size: 12px; line-height: 1.5;
  resize: vertical; min-height: 120px; white-space: pre-wrap;
  font-family: ui-monospace, 'SF Mono', Menlo, monospace;
}
.prompt-block.editable { grid-column: 1 / span 2; }
.prompt-block .actions { display: flex; gap: 8px; align-items: center; margin-top: 4px; }
.prompt-block button { background: #225; color: #fff; border: 1px solid #446; padding: 6px 12px;
                       cursor: pointer; border-radius: 5px; font-size: 12px; }
.prompt-block button:hover:not(:disabled) { background: #334; }
.prompt-block button:disabled { opacity: 0.5; cursor: not-allowed; }
.prompt-block button.danger { background: #522; border-color: #844; }
.prompt-block button.danger:hover:not(:disabled) { background: #733; }
.prompt-block .status { color: #2ee062; font-size: 11px; }
.prompt-block .status.err { color: #f55; }
.prompt-block .meta { color: #888; font-size: 11px; }
.server-warn { background: #3a2a00; color: #ffcc66; border: 1px solid #664a00;
               padding: 8px 12px; border-radius: 6px; margin-bottom: 14px; font-size: 13px; }
.legend { color: #888; font-size: 12px; margin-top: 4px; }
"""


JS = """
// Per-scene sequential player state.
const sceneState = new WeakMap();

function stopScene(scene) {
  scene.querySelectorAll('video').forEach(v => { v.pause(); v.muted = true; });
  scene.querySelectorAll('.cell').forEach(c => c.classList.remove('playing'));
  const st = sceneState.get(scene) || {};
  st.active = false;
  st.parallel = false;
  sceneState.set(scene, st);
  const btn = scene.querySelector('.btn-play');
  if (btn) btn.textContent = '▶ Play sequence';
}

function playCell(scene, idx) {
  const vids = [...scene.querySelectorAll('video')];
  if (idx >= vids.length) {
    stopScene(scene);
    return;
  }
  // Mute and pause all, then play this one.
  vids.forEach((v, i) => {
    v.pause();
    v.muted = true;
    v.closest('.cell').classList.remove('playing');
  });
  const v = vids[idx];
  v.currentTime = 0;
  v.muted = false;
  v.closest('.cell').classList.add('playing');
  const state = sceneState.get(scene) || {};
  state.current = idx;
  state.active = true;
  sceneState.set(scene, state);
  v.play().catch(err => console.warn('play() failed:', err));
}

function playScene(btn) {
  const scene = btn.closest('.scene');
  const st = sceneState.get(scene) || {};
  if (st.active) {
    stopScene(scene);
    return;
  }
  btn.textContent = '⏸ Stop';
  playCell(scene, 0);
}

function stopFromBtn(btn) {
  stopScene(btn.closest('.scene'));
}

function playAllTogether(btn) {
  const scene = btn.closest('.scene');
  const vids = [...scene.querySelectorAll('video')];
  stopScene(scene);
  // Mark as parallel playback so the per-video pause handler doesn't strip frames.
  const state = sceneState.get(scene) || {};
  state.parallel = true;
  state.active = false;
  sceneState.set(scene, state);
  vids.forEach(v => {
    v.muted = true;
    v.currentTime = 0;
    v.closest('.cell').classList.add('playing');
    v.play().catch(err => console.warn('play() failed:', err));
  });
}

function restart(btn) {
  const scene = btn.closest('.scene');
  stopScene(scene);
  const playBtn = scene.querySelector('.btn-play');
  playBtn.textContent = '⏸ Stop';
  playCell(scene, 0);
}

// When a video ends, advance to the next cell in its scene.
document.querySelectorAll('.cell video').forEach(v => {
  v.addEventListener('ended', () => {
    const scene = v.closest('.scene');
    const st = sceneState.get(scene);
    if (!st || !st.active) return;
    playCell(scene, st.current + 1);
  });
  // Click a cell: play only that one (solo), not the full sequence.
  v.addEventListener('click', () => {
    const scene = v.closest('.scene');
    const vids = [...scene.querySelectorAll('video')];
    const idx = vids.indexOf(v);
    const st = sceneState.get(scene) || {};
    st.active = false;   // single-cell mode — no auto-advance
    sceneState.set(scene, st);
    vids.forEach(o => { o.pause(); o.muted = true; o.closest('.cell').classList.remove('playing'); });
    v.currentTime = 0;
    v.muted = false;
    v.closest('.cell').classList.add('playing');
    v.play();
    const btn = scene.querySelector('.btn-play');
    if (btn) btn.textContent = '▶ Play sequence';
  });
  v.addEventListener('pause', () => {
    // If manually paused and not part of an active sequence or parallel run,
    // drop the green frame.
    const scene = v.closest('.scene');
    const st = sceneState.get(scene);
    if (!st || (!st.active && !st.parallel)) v.closest('.cell').classList.remove('playing');
  });
});

// ── Prompt editor add-on ──────────────────────────────────────────────────
const API_AVAILABLE = location.protocol.startsWith('http');

async function loadPrompts() {
  if (!API_AVAILABLE) {
    document.getElementById('server-warn')?.style.setProperty('display', 'block');
    document.querySelectorAll('details.prompts').forEach(d => {
      d.querySelector('.prompts-body').innerHTML =
        "<div style='color:#888;grid-column:1/span 2'>Start the local server to load + edit prompts:" +
        "<br><code style='color:#9cf'>python serve_viewer.py</code></div>";
    });
    return;
  }
  try {
    const resp = await fetch('/api/scenes');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    document.querySelectorAll('details.prompts').forEach(d => {
      const scene = d.dataset.scene;
      const s = data[scene];
      if (!s) {
        d.querySelector('.prompts-body').innerHTML =
          "<div style='color:#888;grid-column:1/span 2'>No prompt data for this scene.</div>";
        return;
      }
      d.querySelector('.baseline-text').textContent = s.baseline_prompt || "(empty)";
      d.querySelector('.baseline-src').textContent  = 'source: ' + (s.baseline_source || '—');
      d.querySelector('.t1-text').textContent       = s.t1_prompt || "(empty)";
      const ta = d.querySelector('textarea.override');
      if (s.override) {
        ta.value = s.override.prompt;
        d.querySelector('.override-meta').textContent = 'saved ' + s.override.updated;
        d.querySelector('.override-del').disabled = false;
      } else {
        ta.placeholder = "Type a new baseline prompt…";
        d.querySelector('.override-meta').textContent = '(no override saved)';
        d.querySelector('.override-del').disabled = true;
      }
    });
  } catch (err) {
    console.error(err);
    document.querySelectorAll('.prompts-body').forEach(b => {
      b.innerHTML = "<div style='color:#f55;grid-column:1/span 2'>Failed to load prompts: " +
                    err.message + "</div>";
    });
  }
}

async function saveOverride(btn) {
  const details = btn.closest('details.prompts');
  const scene = details.dataset.scene;
  const ta = details.querySelector('textarea.override');
  const status = details.querySelector('.override-status');
  const prompt = ta.value.trim();
  if (!prompt) { status.textContent = 'prompt is empty'; status.className = 'status err'; return; }
  btn.disabled = true;
  status.textContent = 'saving…'; status.className = 'status';
  try {
    const resp = await fetch('/api/override/' + encodeURIComponent(scene), {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt}),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'save failed');
    status.textContent = '✓ saved ' + data.override.updated; status.className = 'status';
    details.querySelector('.override-meta').textContent = 'saved ' + data.override.updated;
    details.querySelector('.override-del').disabled = false;
  } catch (err) {
    status.textContent = '✗ ' + err.message; status.className = 'status err';
  } finally {
    btn.disabled = false;
  }
}

async function deleteOverride(btn) {
  const details = btn.closest('details.prompts');
  const scene = details.dataset.scene;
  const status = details.querySelector('.override-status');
  if (!confirm('Delete the saved override for ' + scene + '?')) return;
  btn.disabled = true;
  try {
    const resp = await fetch('/api/override/' + encodeURIComponent(scene), {method: 'DELETE'});
    if (!resp.ok) throw new Error('delete failed');
    details.querySelector('textarea.override').value = '';
    details.querySelector('.override-meta').textContent = '(no override saved)';
    status.textContent = 'deleted'; status.className = 'status';
  } catch (err) {
    status.textContent = '✗ ' + err.message; status.className = 'status err';
    btn.disabled = false;
  }
}

document.addEventListener('DOMContentLoaded', loadPrompts);

// Spacebar plays/stops the scene currently most in view.
document.addEventListener('keydown', e => {
  if (e.code !== 'Space' || e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  e.preventDefault();
  const scenes = [...document.querySelectorAll('.scene')];
  const vh = window.innerHeight;
  const inView = scenes.find(s => {
    const r = s.getBoundingClientRect();
    return r.top < vh * 0.5 && r.bottom > vh * 0.2;
  }) || scenes[0];
  if (inView) playScene(inView.querySelector('.btn-play'));
});
"""


def render() -> str:
    lines: list[str] = []
    lines.append("<!doctype html><html><head>")
    lines.append('<meta charset="utf-8">')
    lines.append("<title>exp_no_lora — variant comparison</title>")
    lines.append(f"<style>{CSS}</style></head><body>")
    lines.append("<h1>exp_no_lora — variant comparison</h1>")
    lines.append("<p class='lead'>▶ Play sequence = each variant one-by-one with its own audio. "
                 "▦ Play all together = all variants simultaneously, muted (for visual side-by-side). "
                 "⏹ Stop = halt the scene. Click any cell to solo one. Current cell(s) outlined in green. "
                 "Space = play/stop the scene in view.</p>")
    lines.append("<div class='server-warn' id='server-warn' style='display:none'>"
                 "Prompt editor needs the local server — start it with "
                 "<code>python serve_viewer.py</code> and re-open the viewer at "
                 "<code>http://127.0.0.1:8765/output/exp_no_lora_comparison/viewer.html</code>."
                 "</div>")

    # Top nav
    nav = " ".join(
        f"<a href='#{escape(s['name'])}'>{escape(s['display'])}</a>" for s in SCENES
    )
    lines.append(f"<nav>{nav}</nav>")

    for scene in SCENES:
        clips = collect(scene)
        if len(clips) < 2:
            lines.append(f"<div class='scene' id='{escape(scene['name'])}'>"
                         f"<div class='scene-head'><div class='scene-title'>"
                         f"{escape(scene['display'])}</div></div>"
                         f"<div class='legend'>Only {len(clips)} clip available — "
                         f"variants may still be rendering.</div></div>")
            continue

        lines.append(f"<div class='scene' id='{escape(scene['name'])}'>")
        lines.append("<div class='scene-head'>")
        lines.append(f"<div class='scene-title'>{escape(scene['display'])} "
                     f"<span class='legend'>· {len(clips)} variants</span></div>")
        lines.append("<div class='ctrl'>")
        lines.append("<button class='btn-play' onclick='playScene(this)'>▶ Play sequence</button>")
        lines.append("<button onclick='playAllTogether(this)'>▦ Play all together (muted)</button>")
        lines.append("<button onclick='stopFromBtn(this)'>⏹ Stop</button>")
        lines.append("<button onclick='restart(this)'>⏮ Restart</button>")
        lines.append("</div></div>")

        lines.append("<div class='row'>")
        for label, path in clips:
            lines.append("<div class='cell'>")
            lines.append(f"<div class='label'>{escape(label)}</div>")
            lines.append(f"<video src='../../{escape(rel(path))}' "
                         f"preload='metadata' playsinline muted></video>")
            lines.append("</div>")
        lines.append("</div>")

        # Prompts add-on: populated by /api/scenes at page load.
        lines.append(f"<details class='prompts' data-scene='{escape(scene['name'])}'>")
        lines.append("<summary>Prompts — baseline · t1 · override</summary>")
        lines.append("<div class='prompts-body'>")
        lines.append("<div class='prompt-block'>"
                     "<div class='head'>Baseline prompt</div>"
                     "<div class='text-ro baseline-text'>loading…</div>"
                     "<div class='meta baseline-src'></div>"
                     "</div>")
        lines.append("<div class='prompt-block'>"
                     "<div class='head'>Prompt 1 (t1_prompt)</div>"
                     "<div class='text-ro t1-text'>loading…</div>"
                     "</div>")
        lines.append("<div class='prompt-block editable'>"
                     "<div class='head'>New baseline prompt (override)</div>"
                     "<textarea class='override' rows='6'></textarea>"
                     "<div class='actions'>"
                     "<button onclick='saveOverride(this)'>💾 Save override</button>"
                     "<button class='danger override-del' onclick='deleteOverride(this)' disabled>"
                     "🗑 Delete</button>"
                     "<span class='meta override-meta'></span>"
                     "<span class='override-status'></span>"
                     "</div></div>")
        lines.append("</div></details>")
        lines.append("</div>")

    lines.append(f"<script>{JS}</script></body></html>")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html = render()
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {HTML_PATH}")
    print(f"Open in browser: file://{HTML_PATH.resolve()}")


if __name__ == "__main__":
    main()
