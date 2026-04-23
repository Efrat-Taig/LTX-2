# VLM Video QA Pipeline

A VLM-native evaluator for AI-generated video. Nine specialised **Gemini 2.5 Pro** agents watch the same video in parallel, each answering one tightly-scoped question about what they see and hear. A final aggregator synthesises their reports into a producer-grade verdict with a score, a top-issue callout, and paragraph-level analysis.

It is intentionally **not** a classical computer-vision pipeline: it does not extract motion vectors, track pixel blobs, or compute SSIM. It gives Gemini the raw `.mp4` bytes and lets the VLM do semantic reasoning that no classical detector can match.

Character detection for generating body- and face-crop grids is done with **Gemini 2.5 Flash** — the same vendor, much cheaper — so the pipeline works on arbitrary characters (dogs, humans, robots, aliens) rather than being locked to a hardcoded color palette.

---

## Table of contents

1. [Why this architecture](#why-this-architecture)
2. [High-level pipeline](#high-level-pipeline)
3. [Perception layer](#perception-layer)
4. [The seven agents, in detail](#the-seven-agents-in-detail)
5. [The aggregator](#the-aggregator)
6. [Anti-hallucination design](#anti-hallucination-design)
7. [How to run it](#how-to-run-it)
8. [Cost and wall-clock timing](#cost-and-wall-clock-timing)
9. [File layout](#file-layout)
10. [Known limitations](#known-limitations)
11. [Extending the pipeline](#extending-the-pipeline)

---

## Why this architecture

The original iteration of this project was an eighteen-agent classical-CV pipeline: MediaPipe FaceMesh, DINOv2 embeddings, optical flow, SSIM, Hough circles. It failed badly on real defects the end user cared about:

- A character who says *"look at my backflip"* but never flips. No classical agent watches semantic alignment between spoken action and visual action.
- A street sign in the background that visibly teleports across the scene. An object-permanence check needs *world understanding*, not just patch tracking.
- Uncanny, frozen eyes on stylised dogs. No off-the-shelf face-mesh model detects PAW-Patrol-style cartoon characters.
- The classical pipeline ranked a video that was visually good but semantically broken (wrong character speaks the wrong lines) as the top scorer.

A Vision-Language Model watching the video can answer these questions directly. The architectural bet is that **a few well-scoped VLM queries outperform many brittle classical detectors**. This pipeline validates that bet.

### Why Gemini 2.5 Pro specifically

- **Native video input.** You hand it a `.mp4`; it decodes audio and video internally. No frame sampling required for most agents.
- **Audio-with-video reasoning.** It hears dialogue, SFX, and voice timbre — critical for lip-sync and voice-quality evaluation.
- **Strong semantic comparison.** It can compare a user-supplied prompt against the generated video and describe mismatches.
- **Vertex AI availability.** The target team already has credentials and billing configured for Vertex; no new vendor to onboard.
- **Cost.** At roughly $0.05–0.10 per 10-second video, it's cheap enough to run on every benchmark generation.

---

## High-level pipeline

```
                 video.mp4
                     │
                     ▼
        ┌──────────────────────────┐
        │   Perception layer       │
        │   (local, CPU)           │
        │                          │
        │  Whisper → transcript    │
        │  Whisper → segments w/   │
        │           timestamps     │
        │                          │
        │  cv2 → 9 evenly-spaced   │
        │        keyframes → grid  │
        │                          │
        │  cv2 → face-region       │
        │        speech pairs →    │
        │        grid              │
        └──────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │   7 agents in parallel   │
        │   (Gemini 2.5 Pro)       │
        └──────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │   Aggregator             │
        │   (Gemini 2.5 Pro)       │
        │                          │
        │   score + verdict +      │
        │   headline + what works  │
        │   + what breaks + top    │
        │   issue                  │
        └──────────────────────────┘
                     │
                     ▼
            report.json / report.md
```

Concretely:

1. `run.py::run_video(video_path, prompt)` is the entry point.
2. It invokes `perception.py::transcribe_with_segments()` to get the Whisper transcript plus a list of `SpeechSegment(start, end, text)` objects.
3. It invokes `keyframes.py::extract_evenly_spaced()` + `compose_grid()` to build a 3×3 comic-strip PNG of the video's keyframes.
4. For each of the first four speech segments it extracts **two** frames (one near the start of the line, one near the end), runs each through `keyframes.py::extract_face_region()` to crop to the character's head, and composes them into an 8-panel grid where each row is `[START | END]` of one spoken line.
5. It reads the raw video bytes into memory.
6. It fans out to **7 agents in parallel** via `ThreadPoolExecutor`. Each agent is a single `Gemini 2.5 Pro` call with the raw video and whichever evidence images / text that agent needs.
7. It sends all 7 agent reports plus the original generation prompt to the **aggregator** (also Gemini 2.5 Pro) which produces the final synthesis.
8. It writes `report.json` (machine-readable) and `report.md` (human-readable) into the run folder.

The viewer at `viewer.html` reads `manifest.json` (built by `run_compare.py`) and lets the user compare model outputs side-by-side.

---

## Perception layer

Everything that happens *before* the VLM. Its only purpose is to produce inputs the VLM would otherwise not have, or could not easily compute:

### Whisper transcript with timestamps — [`perception.py`](perception.py)

```python
text, segments = transcribe_with_segments(video_path)
# text: full transcript string
# segments: list of SpeechSegment(start, end, text) with per-line timestamps
```

Whisper `base` model, CPU inference, runs in ~1 second on a 10s clip. We need the segment-level timestamps for the lip-sync grid.

### Evenly-spaced keyframe grid — [`keyframes.py::extract_evenly_spaced()`](keyframes.py)

Nine frames, evenly spaced, trimmed to the middle 90% of the clip to skip fade-in / fade-out artefacts. Composed into a 3×3 grid, each panel labeled `#N t=X.XXs`. Used by Environment Stability and Rendering Defects agents.

**Why keyframes when Gemini can watch the full video?** Because VLMs systematically *average* visual impressions during playback — a body distortion that flickers for one frame gets smoothed out. Forcing the model to inspect still frames side-by-side breaks that averaging.

### Character-crop grid — [`keyframes.py::extract_character_crop()`](keyframes.py) + `extract_evenly_spaced()`

Sixteen frames, evenly spaced, each one **cropped to the character** using HSV warm-fur detection (falling back to center-bottom if no blob is detected). Generous 25% padding keeps heads/feet in-frame even when the fur blob is imperfect. Composed into a 4×4 grid at 512px per panel.

**Why a character-only grid instead of full frames?** When the Character Performance agent sees full-frame panels, the character occupies maybe 30% of each panel — the rest is wasted on background. Identity and anatomy defects are hard to compare at that scale. Cropping to the character gives the VLM 3-5× more pixels on the body and face, which is exactly what it needs to notice "in panel 8 the left hind leg is visibly shorter than in panel 2." On the Seedance trampoline case this flipped the agent from "no defects observed" (pre-grid) to catching *"The character deforms into a smeared, unrecognizable object... which is a significant defect beyond simple motion blur."*

### Speech-timestamped face-crop grid — [`keyframes.py::extract_at_timestamps()`](keyframes.py) + [`extract_face_region()`](keyframes.py)

This is the grid the Lip-Sync agent consumes. For the first four detected speech segments we:

1. Pick two timestamps inside each segment: one at 15% of its span, one at 85%.
2. Extract those two frames.
3. Crop each frame to the character's head region using HSV warm-fur detection (*"cartoon dog fur is always yellow/tan"*). Falls back to a center-bottom crop if no blob is detected.
4. Lay the 8 frames out as four rows of `[START | END]` panels at 640px wide.

The face crop gives the VLM roughly 5× more pixels on the mouth than the uncropped frame. Seeing mouth shape change (or fail to change) between the start and end of a spoken line is the signal the Lip-Sync agent needs.

---

## The seven agents, in detail

Each agent is a single Gemini 2.5 Pro call declared as a function in [`agents.py`](agents.py). All agents share the same response contract (see [Anti-hallucination design](#anti-hallucination-design) below).

### What each agent receives as input

**Every agent receives the full native video** (the `.mp4` bytes). Gemini 2.5 Pro decodes ~240 frames and the audio track internally. **On top of that**, some agents also receive evidence images (the grids) or text context:

| Agent | Full video | + Grid image | + Text context |
|---|---|---|---|
| 1 · Body Consistency | ✅ | body-crop grid (4×4, 16 panels) | — |
| 2 · Face & Uncanny Valley | ✅ | face-crop grid (4×4, 16 panels) | — |
| 3 · Motion & Weight | ✅ | — | — |
| 4 · Environment Stability | ✅ | keyframe grid (4×4, 16 full frames) | — |
| 5 · Rendering Defects | ✅ | keyframe grid (4×4, 16 full frames) | — |
| 6 · Audio Quality | ✅ | — | — |
| 7 · Lip-Sync Correspondence | ✅ | speech-timestamp face-crop grid (4 rows × 2 panels) | segment captions block |
| 8 · Speech & Dialogue Coherence | ✅ | — | Whisper transcript + per-segment timing |
| 9 · Prompt Fidelity | ✅ | — | Whisper transcript + original prompt |

The four grids are saved alongside `report.json` so they can be inspected after the fact. The viewer renders them as clickable thumbnails in the "Inputs fed to the VLM" section — so for any analyzed video you can see exactly what each agent saw.

**Character detection** for the body-crop and face-crop grids is done by **Gemini 2.5 Flash**, called once per keyframe (in parallel). Flash returns JSON bounding boxes `{name, body: [x1,y1,x2,y2], face: [x1,y1,x2,y2]}` for every visible character. The pipeline uses these coordinates to crop consistently across frames — no color-space heuristics, no character-type assumptions.

### Agent 1 — Body Consistency

- **Function:** [`run_body_consistency(video_bytes, body_grid)`](agents.py)
- **Inputs:** native video + 4×4 **body-crop grid** (16 panels, Gemini-detected body bboxes)
- **Scope:** the character body only — identity and anatomy. NOT face quality, lip-sync, or motion.
  1. **Identity consistency** — same character throughout? species, coloration, accessories (hats, vests, badges) all preserved across panels?
  2. **Anatomy & proportions** — panel-by-panel: does the body stay consistent? Any panel with a stretched limb, oversized head, or deformed torso?
- **Why the body-crop grid** (instead of full-frame keyframes): when the character body fills each panel, identity-drift and anatomy deformation become directly comparable across time. The VLM can say *"in panels 9-11 the body deforms into a fluid smear beyond normal motion blur"* because the character is large and central.
- **Why split from face quality:** faces and bodies are different failure modes. A body can be anatomically perfect while the face is uncanny — and vice versa. The face crop is 5-10× tighter than the body crop, which is what face-quality evaluation needs.
- **Catches:** swapped clothing, mid-clip design drift, body-part distortion, "fluid smear during fast action" defects.

### Agent 2 — Face & Uncanny Valley

- **Function:** [`run_face_uncanny(video_bytes, face_grid)`](agents.py)
- **Inputs:** native video + 4×4 **face-crop grid** (16 panels, Gemini-detected face bboxes — tight crops on eyes + mouth + nose)
- **Scope:** facial performance and uncanny valley. NOT body anatomy, NOT lip-sync.
  1. **Eyes alive vs dead** — do the eyes blink across the clip? Gaze shifts? Or glassy and staring?
  2. **Expression change** — does the face shift with emotional beats, or is it frozen in one pose?
  3. **Face anatomy** — are eyes/mouth/nose positioned correctly in every panel?
  4. **Uncanny valley** — subtle wrongness: too-wide smile, rigid / waxy appearance, uneven animation.
- **Why this is its own agent:** uncanny-valley defects are some of the most-noticed issues in AI video, and they require tight face crops to catch. Folded into a broader agent, they get dismissed with "face looks fine."
- **Catches:** frozen faces, dead eyes, melted features, uncanny micro-wrongness.

### Agent 2 — Motion & Weight

- **Function:** `run_motion_weight(video_bytes)`
- **Inputs:** native video only
- **Scope:** how things move. Floating, foot-sliding, phantom momentum, impacts without reaction, interpenetration.
- **Key framing choice:** the prompt explicitly tells the model this is a CGI cartoon and that exaggerated bounce / squash-and-stretch are features, not defects. Without this caveat, VLMs flag stylised animation as physics violations.
- **Out of scope:** identity, rendering, lip-sync.
- **Why no grid:** motion defects are temporal — they need video playback to see. A still frame can't show sliding.

### Agent 3 — Environment Stability

- **Function:** `run_environment_stability(video_bytes, keyframe_grid)`
- **Inputs:** native video + 3×3 keyframe grid
- **Scope:** the background only. Characters are explicitly ignored. Focus on object permanence: signs, benches, mailboxes, buildings, trees, fences, vehicles. Do they teleport, disappear, duplicate, or re-appear in a new world position?
- **Why the grid:** this is THE defect class the end user cared about most — a street sign visibly jumping across the scene. Object-permanence defects are most visible when you compare static panels, because normal camera parallax hides them in motion.
- **Guard:** the prompt explicitly notes that camera-induced parallax is not a defect; only objects whose *world* position changes count.

### Agent 4 — Rendering Defects

- **Function:** `run_rendering_defects(video_bytes, keyframe_grid)`
- **Inputs:** native video + 3×3 keyframe grid
- **Scope:** frame-level rendering QC. Limb corruption, hand errors (extra / fused fingers), morphing geometry, texture swimming, depth / occlusion errors, flash artefacts.
- **Why the grid:** single-frame glitches are invisible at playback speed. The grid exposes them.
- **Out of scope:** character animation quality (Agent 1), motion (Agent 2).

### Agent 5 — Audio Quality

- **Function:** `run_audio_quality(video_bytes)`
- **Inputs:** native video only (Gemini hears the audio track natively)
- **Scope:** technical audio quality — *not* whether the words match a script. That's Agents 6 and 7.
- Specifically evaluates: voice naturalness, clarity, age/character fit, pops/dropouts/cut-offs, ambient mix.
- **Why a dedicated agent:** voice artefacts (robotic TTS, mid-word cut-offs, wrong-age voice) are common failure modes for AI video and they're invisible to any classical video CV.

### Agent 6 — Lip-Sync Correspondence *(the hardest agent to get right)*

- **Function:** `run_lipsync_correspondence(video_bytes, speech_grid, speech_captions)`
- **Inputs:** native video + speech-timestamped face-crop grid + list of captions (kept out of the image)
- **Scope:** does a character's mouth actually *move* while each line is spoken?
- **Why this is hard:**
  - Charity bias — if the VLM hears speech, it tends to assume someone must be speaking even when all visible mouths are closed.
  - Static ambiguity — a cartoon "smile" pose looks identical to an actively-speaking mouth in a single frame.
- **How the prompt defeats both:**
  1. **Paired frames per line.** The grid has START-frame and END-frame columns for each spoken line. The model must compare mouth *shape change* across the pair, not just mouth *state* at one instant.
  2. **Face-region crops.** Cropping to the head gives the model 5× more pixels on the mouth. A closed smile vs. a mid-speech open vowel is unambiguous when you can see the teeth and tongue.
  3. **Observe-first, caption-second protocol.** The prompt requires the model to describe each panel's mouths *before* seeing the caption of what was said. This prevents the caption from priming a charitable answer.
  4. **Explicit bias warning.** "The audio track plays dialogue regardless of whether any mouth is moving. Do not charitably assume a closed mouth is actually speaking."
  5. **Required enumeration.** The model must output one line per row of the grid explicitly stating "IDENTICAL / DIFFERENT" for each character. No pass-by-the-overall-impression answer is accepted.
- **Catches:** the "audio-from-nowhere" defect class (the video's audio track plays a line but no character's mouth actually moves), and wrong-character-speaks cases where character B's mouth is moving during character A's line.

### Agent 8 — Speech & Dialogue Coherence

- **Function:** [`run_speech_coherence(video_bytes, transcript, segments_block)`](agents.py)
- **Inputs:** native video (audio track) + Whisper transcript + per-segment timing
- **Scope:** the linguistic quality of spoken dialogue, independent of any script. NOT voice timbre (Agent 6), NOT script match (Agent 9), NOT lip-sync (Agent 7).
  1. **Real words** — is every spoken word an actual word? AI generators sometimes produce plausible-sounding gibberish like "bluntrip" or "folvery".
  2. **Grammar & sense** — are sentences well-formed? Not "the cat sleeps the pizza."
  3. **Mid-word cut-off** — does the clip end with a dangling word like "You're the b-"?
  4. **Word pacing** — reasonable pauses between words, not machine-gun or fused delivery.
- **Why separate from Audio Quality:** Audio Quality judges the voice (is it robotic / clipped / wrong-age). Speech Coherence judges the CONTENT of what's spoken (are those real words in grammatical sentences?). These are orthogonal — a beautifully-voiced TTS can speak perfect gibberish, and a glitchy voice can say real sentences.
- **Catches:** gibberish dialogue, ungrammatical phrasing, truncated endings, bad TTS pacing.

### Agent 9 — Prompt Fidelity

- **Function:** `run_prompt_fidelity(video_bytes, transcript, prompt_text)`
- **Inputs:** native video + Whisper transcript + original generation prompt
- **Scope:** the one semantic agent. Three axes:
  1. **Scene fidelity** — does the setting and overall action shown match what the prompt described?
  2. **Action vs dialogue** — if a character says "look at my backflip", do they actually do a backflip?
  3. **Speaker assignment** — does the right character speak the right lines (given what the prompt scripted)?
- **Replaces two older agents.** The v1 design had a separate Action-Dialogue agent and a Prompt-Adherence agent. They produced nearly identical findings on every defect case. Merged.
- **Explicit no-op on transcription noise.** The prompt tells the model that tiny ASR mistakes ("yee-ho" vs "yee-haw") are not defects — only meaning mismatches are.
- **Catches:** missing scripted actions, wrong-character-speaks-the-wrong-line, off-prompt scenes.

---

## The aggregator

- **Function:** [`run_aggregator(reports, prompt_text)`](agents.py)
- **Inputs:** the seven agent reports + original generation prompt
- **Output:** a JSON dict with `overall_verdict` (PASS / CONDITIONAL_PASS / FAIL), integer `score` 0-100, `headline`, `what_works`, `what_breaks`, `top_issue`.
- **Design:**
  - Low temperature (0.1).
  - **Explicit weighting rules** in the prompt: Prompt Fidelity and Lip-Sync defects are the most serious; rendering and character performance are serious; audio quality is moderate; motion is moderate unless clearly broken.
  - **"Do not introduce claims not present in the reports."** The aggregator must ground its synthesis in what the specialists already found — it's not another judgment pass.
- **Scoring rubric, from the prompt:**
  - 85–100 PASS: deliverable with polish notes.
  - 70–84 CONDITIONAL_PASS: watchable but has clear flaws.
  - 40–69 FAIL: significant defects — regenerate.
  - 0–39 FAIL: fundamentally broken or off-prompt.

---

## Anti-hallucination design

VLMs are prone to (a) charitable interpretation, (b) inventing findings to fill expected output space, (c) confabulating precise numbers and timestamps. Every agent prompt is designed to counter these failure modes.

### Output contract

Every agent sees this contract at the end of its prompt ([`_OUTPUT_CONTRACT`](agents.py)):

```
1. Respond in this exact JSON schema, nothing else:
   { "verdict": "great" | "minor_issues" | "major_issues",
     "summary": "<one sentence>", "findings": "<evidence-grounded paragraph>" }

2. GROUNDING: every claim in `findings` must reference what you actually saw.
   Do NOT invent defects. Do NOT speculate outside this agent's scope.

3. DEFAULT TO 'great': if nothing is wrong, verdict is 'great' and the summary
   is literally "No defects observed in this scope." Padding is failure.

4. NO TIMESTAMPS, NO NUMERIC METRICS, NO PIXEL COORDINATES. Describe naturally
   (e.g. "as the character lands", "in panel 4 of the grid").

5. When referencing the provided keyframe grid, cite the panel number.
```

### Why each rule matters

- **"Default to great."** Without this, agents invent minor issues to look diligent. With it, the trampoline Character Performance agent correctly said "No defects observed in this scope." on the Veo clip.
- **"No numeric metrics or timestamps."** Asking a VLM for "exact time of the defect" invites it to confabulate "at 3.2 seconds" based on nothing. Panel numbers reference concrete visible evidence.
- **Temperature 0.1.** Reduces speculative language without making the model refuse ambiguous cases.
- **Observe-first-caption-second** (Lip-Sync only): captions about what was spoken come AFTER the prompt asks the model to describe the image. Prevents caption-priming ("if audio says 'hello', the mouth must be open").

### Empirical check

The pipeline was tested on known failure cases that the old classical pipeline missed:

- **Wan 2.6 trampoline** — audio plays but no character's mouth moves. Pre-fix Lip-Sync agent said "great, mouths are moving in sync." Post-fix Lip-Sync agent said: *"The characters' mouths do not move or change shape to match the dialogue, remaining in a static smile throughout the entire clip."* Score: FAIL 45.
- **Veo trampoline** — the pink pup speaks the lines scripted for the blue pup. Prompt Fidelity correctly flags this as a major issue.

---

## How to run it

### Prerequisites

- Python 3.11+
- `pip install google-genai openai-whisper opencv-python pillow loguru`
- `ffmpeg` on your `PATH` (used by Whisper to demux audio)
- A Google Cloud project with Vertex AI enabled
- Environment variables:
  ```bash
  export GOOGLE_CLOUD_PROJECT=<your-project>
  export GOOGLE_CLOUD_LOCATION=global
  export GOOGLE_GENAI_USE_VERTEXAI=True
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
  ```

### Single video

```bash
cd auto_val
python -m vlm_pipeline.run my_video.mp4 \
    --prompt-file my_prompt.txt \
    --out-dir ./outputs_vlm
```

Writes a timestamped run folder with `report.json`, `report.md`, `keyframe_grid.png`, `speech_grid.png`, and a copy of the video.

### Multi-model benchmark

`run_compare.py` runs the pipeline on the same scene across all six generation models found in `outputs_BM_v1/`:

```bash
python -m vlm_pipeline.run_compare \
    --scene skye_chase_trampoline_backflip \
    --out-dir ./outputs_vlm_compare_trampoline
```

Writes one run folder per model plus a `manifest.json` consumed by the viewer.

### Viewer

```bash
cd auto_val
python -m http.server 8765
# open http://localhost:8765/
```

The landing page at `index.html` lists all scene comparisons that exist. Clicking one loads its viewer with:

- Top row of tabs, one per model, sorted by score (highest first). Each tab is color-coded: green = PASS, amber = CONDITIONAL_PASS, red = FAIL.
- Left pane: video autoplays on tab click.
- Right pane: score card, top-issue callout, per-agent verdict cards with expandable findings, transcript, original prompt.

---

## Cost and wall-clock timing

Measured on 10-second 1080p `.mp4` inputs, Vertex AI `gemini-2.5-pro`.

| Stage | Wall-clock | Notes |
|---|---|---|
| Whisper transcription | ~1 s | CPU, base model |
| Keyframe + speech-grid extraction | ~3 s | cv2 + PIL |
| 7 agents in parallel | 60–120 s | Bottleneck is the slowest agent; aggregator cannot start until all seven finish |
| Aggregator | 15–45 s | |
| **Per video total** | **~90–180 s** | |

Cost (Vertex AI Gemini 2.5 Pro pricing, input $1.25/M tok, output $10/M tok):

| Stage | Input tokens | Output tokens | Cost |
|---|---|---|---|
| 7 agents × (video + text + image) | ~3.5 K × 7 = 25 K | ~500 × 7 = 3.5 K | $0.03 + $0.035 |
| Aggregator | ~3 K | ~500 | $0.004 + $0.005 |
| **Per video** | **~28 K** | **~4 K** | **~$0.08** |

Ballpark: **$0.05–0.10 per video**, **6 videos ≈ $0.30–0.60** for a full benchmark.

---

## File layout

```
auto_val/
├── index.html                     # landing page that lists scene comparisons
├── vlm_pipeline/
│   ├── __init__.py
│   ├── agents.py                  # all 7 agent prompts + aggregator + Gemini client
│   ├── perception.py              # Whisper (transcript + timestamps)
│   ├── keyframes.py               # extraction + grid composition + face crop
│   ├── run.py                     # single-video orchestrator
│   ├── run_compare.py             # multi-model benchmark
│   ├── viewer.html                # template — copied into each output dir
│   └── README.md                  # this file
├── outputs_vlm/                   # single-run outputs
└── outputs_vlm_compare_<scene>/   # multi-model benchmark outputs
    ├── viewer.html
    ├── manifest.json              # { model, folder, video, score, verdict, headline }
    └── <video_name>__<timestamp>/
        ├── report.json
        ├── report.md
        ├── keyframe_grid.png
        ├── speech_grid.png
        └── <video_name>.mp4
```

### Who calls what

1. **Entry points:**
   - `run.py::main()` — CLI for single video.
   - `run_compare.py::main()` — multi-model benchmark; shells out to `run.py` via subprocess per video.

2. **Inside `run.py::run_video()`:**
   - `perception.py::transcribe_with_segments()` — Whisper.
   - `keyframes.py::extract_evenly_spaced()` + `compose_grid()` — 3×3 keyframe grid.
   - `keyframes.py::extract_at_timestamps()` + `extract_face_region()` + `compose_grid()` — speech-paired face-crop grid.
   - `agents.py::prewarm_client()` — initialise Gemini client in the main thread before fanning out.
   - `ThreadPoolExecutor.submit()` with one task per entry in `agents.AGENTS`, each calling an agent function.
   - `agents.py::run_aggregator()` — final synthesis.

3. **Inside every agent function in `agents.py`:**
   - `_run(agent_id, title, prompt, video_bytes, extra, image_bytes)` — generic per-agent wrapper.
   - `_call(...)` — builds the `types.Part` list (video + optional grid + text), invokes Gemini, parses JSON.
   - `_parse_json(text)` — tolerant JSON extraction (handles markdown fences, trailing prose).

4. **Inside `agents.py::run_aggregator()`:**
   - Assembles an agent-reports block and a weighted scoring prompt.
   - One Gemini call (no images, pure text reasoning over the reports).

---

## Known limitations

### Things the pipeline explicitly doesn't do

- **No timestamps in findings.** The output contract forbids them. Asking a VLM "at what time?" invites hallucinated numbers. Descriptions reference panels ("in panel 4 of the grid") or moments ("as the character lands"), which are grounded in visible evidence.
- **No pose estimation.** Considered and rejected: pretrained pose estimators are trained on humans. On PAW-Patrol-style cartoon quadrupeds they detect nothing reliable. The keyframe grid + VLM-reasoning combo handles pose issues well enough.
- **No SAM / per-object segmentation.** Considered and rejected as over-engineering: on these tightly-framed single-character shots, cropping to the HSV warm-fur blob is good enough for the lip-sync agent's needs, and the VLM's own visual attention handles the rest.

### Known failure modes the pipeline still has

- **Charity bias remains partial.** Even with the observe-first-caption-second protocol, the VLM occasionally smooths over subtle defects. The `major_issues` threshold on Lip-Sync and the adversarial prompt wording mitigate it but don't eliminate it.
- **HSV fur-detection assumes golden / tan / warm-palette characters.** Works for PAW Patrol. A character in all-black or all-white fur would fall through to the center-bottom crop fallback. For a broader content domain, a different character-detection heuristic would be needed — or use a proper zero-shot detector like OWL-ViT or SAM with text prompts.
- **Run-to-run variance.** Gemini at temperature 0.1 is mostly deterministic but not fully. Character Performance in particular has occasionally said "great" where it earlier caught a defect on the same clip. If repeatability matters, either drop to temp 0.0 or run each agent 2-3 times and take majority.
- **Wall-clock of 90–180 seconds** is not an interactive UX. Acceptable for a nightly benchmark; not for a chat loop.

---

## Extending the pipeline

### Adding a new agent

1. Write a new `run_my_agent(video_bytes, ...)` function in `agents.py`. Follow the existing structure:
   - Clear scope statement at the top of the prompt.
   - Explicit "what's in scope" and "what's out of scope" bullet lists.
   - "Default to great if nothing is wrong" reminder.
   - End with the shared `_OUTPUT_CONTRACT` (appended automatically by `_call`).
2. Add an entry to the `AGENTS` registry at the bottom of `agents.py`. The entry is a 3-tuple `(agent_id, runner_fn, tuple_of_required_kwargs)`. The kwarg names must match keys in `run.py::kwargs_pool`.
3. If the new agent needs data the existing perception layer doesn't produce, add it to `run.py::run_video()` between perception and agent dispatch, and add it to `kwargs_pool`.
4. Update the aggregator's scoring weights in `run_aggregator()`'s prompt if the new agent should be weighted differently.

### Using a different VLM

- The abstraction point is `_call()` in `agents.py`. It takes a system prompt, video bytes, optional extra text, and optional image bytes, and returns `{verdict, summary, findings}`.
- To swap Gemini for, e.g., Claude Sonnet with vision: replace the body of `_call()` to call the Anthropic SDK. Claude does not have native video input, so you'd need to change the perception layer to sample N frames and send them as separate image parts (the old v1 design).
- The output contract is provider-agnostic — it just asks for a JSON response in a fixed shape.

### Changing the aggregator's weighting

Edit the block in `run_aggregator()`'s system prompt that starts with `Weighting:`. The aggregator follows the rubric you give it. Don't change the integer-range definitions in the `Scoring guide:` section without matching the viewer's color thresholds.

---

*Built for the Smiti autonomous-QA pipeline, April 2026. Designed by efrat.t@smiti.ai + Claude Opus 4.7.*
