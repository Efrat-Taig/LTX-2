#!/usr/bin/env python3
"""
serve_viewer.py

Local HTTP server for the exp_no_lora viewer with a prompt-editing add-on.

Serves the repo root so existing video paths resolve. On top of that, exposes
an API:

  GET  /api/scenes              → scene metadata (baseline prompt, t1 prompt, current override)
  POST /api/override/<scene>    → {"prompt": "..."}  — saves to prompt_overrides.json

Persists overrides at repo-root `prompt_overrides.json`. They are NOT yet wired
into any runner — that's a separate step. This server just captures edits.

Usage:
  python serve_viewer.py            # opens browser to viewer
  python serve_viewer.py --no-open  # skips auto-open
  python serve_viewer.py --port 9000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parent
OVERRIDES_PATH = REPO_ROOT / "prompt_overrides.json"
VIEWER_REL     = "output/exp_no_lora_comparison/viewer.html"
HELI_CFG       = REPO_ROOT / "inputs" / "BM_v1" / "benchmark_v1" / "skye_helicopter_birthday_gili" / "config.json"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def read_scene_data() -> dict:
    """Collect baseline + t1 prompts for every scene. Read at request time so
    edits to the source files show up on refresh."""
    chase_bm = _load_module(REPO_ROOT / "run_chase_benchmark.py", "_chase_bm")
    skye_bm  = _load_module(REPO_ROOT / "run_skye_benchmark.py",  "_skye_bm")
    exp_no   = _load_module(REPO_ROOT / "run_exp_no_lora.py",     "_exp_no")

    scenes: dict[str, dict] = {}

    for s in chase_bm.BENCHMARK_SCENES:
        key = f"chase_{s['name']}"
        scenes[key] = {
            "display":         f"Chase — {s['name']}",
            "baseline_prompt": s["prompt"],
            "baseline_source": "run_chase_benchmark.py",
        }

    for s in skye_bm.BENCHMARK_SCENES:
        key = f"skye_{s['name']}"
        scenes[key] = {
            "display":         f"Skye — {s['name']}",
            "baseline_prompt": s["prompt"],
            "baseline_source": "run_skye_benchmark.py",
        }

    # Skye helicopter: baseline prompt is the original JSON config.
    if HELI_CFG.exists():
        cfg = json.loads(HELI_CFG.read_text())
        heli_prompt = cfg["global_prompt"] + "\n\n" + cfg["specific_prompts"][0]
        scenes["skye_helicopter"] = {
            "display":         "Skye — helicopter",
            "baseline_prompt": heli_prompt,
            "baseline_source": str(HELI_CFG.relative_to(REPO_ROOT)),
        }

    # Overlay t1_prompt from the exp_no_lora scene list.
    for s in list(exp_no.CHASE_SCENES) + list(exp_no.SKYE_SCENES):
        key = f"{s['char']}_{s['name']}"
        if key in scenes:
            scenes[key]["t1_prompt"] = s["prompt"]
        else:
            scenes[key] = {
                "display":         f"{s['char'].capitalize()} — {s['name']}",
                "baseline_prompt": "",
                "baseline_source": "(no baseline source located)",
                "t1_prompt":       s["prompt"],
            }

    overrides = read_overrides()
    for key, data in scenes.items():
        if key in overrides:
            data["override"] = overrides[key]

    return scenes


def read_overrides() -> dict:
    if not OVERRIDES_PATH.exists():
        return {}
    try:
        return json.loads(OVERRIDES_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def write_override(scene: str, prompt: str) -> dict:
    data = read_overrides()
    data[scene] = {
        "prompt":  prompt,
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    OVERRIDES_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data[scene]


def delete_override(scene: str) -> None:
    data = read_overrides()
    if scene in data:
        del data[scene]
        OVERRIDES_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {fmt % args}\n")

    # ── API routes ────────────────────────────────────────────────────────

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/scenes":
            try:
                self._send_json(HTTPStatus.OK, read_scene_data())
            except Exception as exc:  # surface any load error in the UI
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path.startswith("/api/override/"):
            scene = self.path[len("/api/override/"):]
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8")
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"bad json: {exc}"})
                return
            prompt = (body.get("prompt") or "").strip()
            if not prompt:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "empty prompt"})
                return
            saved = write_override(scene, prompt)
            self._send_json(HTTPStatus.OK, {"scene": scene, "override": saved})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

    def do_DELETE(self) -> None:  # noqa: N802
        if self.path.startswith("/api/override/"):
            scene = self.path[len("/api/override/"):]
            delete_override(scene)
            self._send_json(HTTPStatus.OK, {"scene": scene, "deleted": True})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    # Fail fast if prompt modules don't import.
    read_scene_data()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/{VIEWER_REL}"
    print(f"Serving repo root {REPO_ROOT}")
    print(f"Overrides file:   {OVERRIDES_PATH}")
    print(f"Viewer:           {url}")
    print("Ctrl-C to stop.")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
