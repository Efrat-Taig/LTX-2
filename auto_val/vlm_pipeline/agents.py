"""
Agents — nine specialist VLM evaluators. Each is a single Gemini 2.5 Pro call
on the native video plus optional evidence images / text context.

Design:
- Every agent receives the full .mp4 so Gemini hears audio and sees all frames.
- Some agents additionally receive a cropped grid (body, face, or speech)
  because static side-by-side panels defeat the VLM's playback-averaging bias.
- Every agent has a narrow scope, anti-hallucination guards, and defaults to
  `great` when no defect is observed.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional

from google import genai
from google.genai import types

MODEL_ID = "gemini-2.5-pro"


@dataclass
class AgentReport:
    agent_id: str
    title: str
    focus: str
    verdict: str               # 'great' | 'minor_issues' | 'major_issues'
    summary: str
    findings: str
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Gemini client
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "shapeshifter-459611")
_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

_cached_client: Optional[genai.Client] = None
_client_lock = threading.Lock()


def _client() -> genai.Client:
    global _cached_client
    if _cached_client is not None:
        return _cached_client
    with _client_lock:
        if _cached_client is None:
            _cached_client = genai.Client(vertexai=True, project=_PROJECT, location=_LOCATION)
    return _cached_client


def prewarm_client() -> None:
    _client()


# ─────────────────────────────────────────────────────────────────────────────
# Output contract
# ─────────────────────────────────────────────────────────────────────────────

_OUTPUT_CONTRACT = """
CRITICAL OUTPUT RULES:

1. Respond in this exact JSON schema, nothing else:
{
  "verdict": "great" | "minor_issues" | "major_issues",
  "summary": "<one sentence. if no defects, literally say so.>",
  "findings": "<evidence-grounded paragraph. Only state things you directly observed. Quote dialogue verbatim.>"
}

2. GROUNDING: every claim must reference what you actually saw or heard.
   Do NOT invent defects. Do NOT speculate outside this agent's scope.

3. DEFAULT TO 'great': if nothing is wrong in your scope, verdict is 'great'
   and the summary is "No defects observed in this scope." Padding is failure.

4. NO TIMESTAMPS, NO NUMERIC METRICS, NO PIXEL COORDINATES. Describe naturally
   (e.g. "as the character lands", "in panel 4 of the grid").

5. When referencing a keyframe grid, cite the panel number (e.g. "in panel 4
   the character's left leg is shorter than in panel 2").
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Generic runner
# ─────────────────────────────────────────────────────────────────────────────

def _call(
    system_prompt: str,
    video_bytes: bytes,
    extra_text: str = "",
    image_bytes: Optional[bytes] = None,
    image_caption: str = "",
) -> dict:
    parts: list = [types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")]
    if image_bytes is not None:
        if image_caption:
            parts.append(image_caption)
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
    body = system_prompt.strip() + "\n\n" + _OUTPUT_CONTRACT
    if extra_text:
        body += "\n\n=== ADDITIONAL CONTEXT ===\n" + extra_text
    parts.append(body)

    last_exc = None
    for attempt in range(3):
        try:
            resp = _client().models.generate_content(
                model=MODEL_ID,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    http_options=types.HttpOptions(timeout=240_000),
                ),
            )
            return _parse_json((resp.text or "").strip())
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if any(code in msg for code in ("499", "503", "504", "CANCELLED", "UNAVAILABLE", "DEADLINE")):
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
            raise
    raise last_exc


def _parse_json(text: str) -> dict:
    if not text:
        return {"verdict": "great", "summary": "(empty response)", "findings": ""}
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {"verdict": "great",
                "summary": "(unparseable model output — treated as clean)",
                "findings": text[:2000]}
    verdict = str(obj.get("verdict", "great")).lower().strip()
    if verdict not in {"great", "minor_issues", "major_issues"}:
        verdict = "great"
    return {
        "verdict": verdict,
        "summary": str(obj.get("summary", "")).strip(),
        "findings": str(obj.get("findings", "")).strip(),
    }


def _run(agent_id, title, prompt, video_bytes, extra="", image_bytes=None, image_caption=""):
    t0 = time.time()
    data = _call(prompt, video_bytes, extra, image_bytes, image_caption)
    return AgentReport(
        agent_id=agent_id,
        title=title,
        focus=prompt.split("\n")[0].strip(),
        verdict=data["verdict"],
        summary=data["summary"],
        findings=data["findings"],
        elapsed_s=round(time.time() - t0, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 — Body Consistency
# ─────────────────────────────────────────────────────────────────────────────

def run_body_consistency(video_bytes: bytes, body_grid: bytes) -> AgentReport:
    prompt = """
You are a continuity supervisor. Your scope is the CHARACTER BODY ONLY —
NOT faces, NOT motion, NOT lip-sync.

You have the full video AND a 4×4 BODY-CROP GRID (16 panels, each cropped
to the primary character's body). Compare panel-by-panel:

A. IDENTITY CONSISTENCY — Is the SAME character in every panel? Species,
   body shape, fur / skin color, clothing, accessories (vest, badge, hat,
   collar)? Any panel where the character's design morphs or swaps?

B. ANATOMY & PROPORTIONS — In each panel, is the body plausible and
   consistent with the others? Is a limb stretched, missing, duplicated,
   or the torso warped? Any panel where the body dissolves into a smear
   beyond reasonable motion blur?

IMPORTANT: cite panel numbers when flagging defects (e.g. "in panel 9 the
torso stretches unnaturally; in panels 1-8 the body looks consistent").

Out of scope: face quality (separate agent), eyes (separate agent), lip-sync
(separate agent), motion naturalness (separate agent).

If no identity or anatomy defects, verdict 'great'.
""".strip()
    return _run(
        "body_consistency",
        "Body Consistency",
        prompt,
        video_bytes,
        image_bytes=body_grid,
        image_caption="BODY-CROP GRID — 16 panels, each cropped to the primary character's body (panels numbered 1-16):",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — Face & Uncanny Valley
# ─────────────────────────────────────────────────────────────────────────────

def run_face_uncanny(video_bytes: bytes, face_grid: bytes) -> AgentReport:
    prompt = """
You are a character-animation director specialising in FACIAL performance.
Your scope is the FACE only. Use the provided 4×4 FACE-CROP GRID (16 panels,
each tightly zoomed on the character's face across the clip) plus the video.

Evaluate the face on four dimensions:

A. EYES ALIVE vs DEAD — Do the eyes blink across the clip? Do they change
   direction / gaze? Are they glassy, empty, or "staring-through-you"?
   Dead eyes across all panels = major issue.

B. EXPRESSION CHANGE — Do facial expressions shift with the emotional beats
   of the scene? Or is the face frozen in one pose across every panel?
   A face locked in one expression for the whole clip = major issue.

C. FACE ANATOMY — Are the eyes / mouth / nose positioned correctly and
   sized plausibly in every panel? Any panel where features melt, shift
   asymmetrically, or look mis-aligned?

D. UNCANNY VALLEY — Does the face feel subtly 'wrong'? Too-wide smile, eyes
   that don't track, rigid or waxy appearance, uneven mouth animation? Flag
   specific reasons, not just 'it feels off'.

IMPORTANT: cite panel numbers. Do not smooth over subtle face issues —
uncanny valley is exactly the class of defect humans notice but VLMs tend
to miss by saying "looks fine."

If the face is alive, expressive, and anatomically stable, verdict 'great'.
""".strip()
    return _run(
        "face_uncanny",
        "Face & Uncanny Valley",
        prompt,
        video_bytes,
        image_bytes=face_grid,
        image_caption="FACE-CROP GRID — 16 panels, each tightly cropped to the primary character's face (panels numbered 1-16):",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3 — Motion & Weight
# ─────────────────────────────────────────────────────────────────────────────

def run_motion_weight(video_bytes: bytes) -> AgentReport:
    prompt = """
You are a motion-quality reviewer. Your scope is HOW THINGS MOVE.

Flag only these observable defects:
- Floating (character hangs in the air without support)
- Foot-sliding (feet move without stepping)
- Phantom momentum (motion with no apparent cause)
- Impacts without reaction (landings with no squash / recoil)
- Bodies passing through solid objects

This is CGI animation — stylised bounce, exaggerated squash-and-stretch, and
cartoon physics are fine and should NOT be flagged. Only broken-looking,
non-stylised motion glitches count.

If motion looks intentional and coherent, verdict 'great'.
""".strip()
    return _run("motion_weight", "Motion & Weight", prompt, video_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4 — Environment Stability
# ─────────────────────────────────────────────────────────────────────────────

def run_environment_stability(video_bytes: bytes, keyframe_grid: bytes) -> AgentReport:
    prompt = """
You are a world-permanence reviewer. Your scope is the BACKGROUND ONLY —
ignore the characters entirely.

Use the 4×4 keyframe grid to compare background objects across panels.
Flag only:
- Static objects (signs, benches, buildings, trees, fences, vehicles) that
  TELEPORT, DISAPPEAR, DUPLICATE, or re-appear in a different world position
- Flickering or unstable background rendering
- Scene geometry that changes mid-clip

Camera-parallax motion of stable objects is NOT a defect — only changes in
their WORLD position count. Name each affected object and cite panel numbers.

If the background is stable, verdict 'great'.
""".strip()
    return _run(
        "environment_stability",
        "Environment Stability",
        prompt,
        video_bytes,
        image_bytes=keyframe_grid,
        image_caption="KEYFRAME GRID — 16 full-frame panels, evenly spaced, numbered 1-16:",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 5 — Rendering Defects
# ─────────────────────────────────────────────────────────────────────────────

def run_rendering_defects(video_bytes: bytes, keyframe_grid: bytes) -> AgentReport:
    prompt = """
You are a QC reviewer scanning for rendering artefacts. Your scope is
FRAME-LEVEL RENDERING QUALITY. Use the 4×4 keyframe grid to inspect
per-panel, then confirm with the video.

Flag only these pixel-level defects:
- Limb corruption (extra / missing / fused fingers, impossible joints)
- Hand rendering errors
- Morphing / warping geometry
- Textures swimming or crawling inconsistently
- Depth or occlusion errors (pieces passing through solid objects incorrectly)
- Single-frame flash artefacts

Describe each defect concretely and cite the panel number.

If rendering is clean across all panels, verdict 'great'.
""".strip()
    return _run(
        "rendering_defects",
        "Rendering Defects",
        prompt,
        video_bytes,
        image_bytes=keyframe_grid,
        image_caption="KEYFRAME GRID for rendering inspection (panels 1-16):",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 6 — Audio Quality
# ─────────────────────────────────────────────────────────────────────────────

def run_audio_quality(video_bytes: bytes) -> AgentReport:
    prompt = """
You are an audio-quality reviewer. You can hear the video's audio track.
Scope: TECHNICAL audio quality only — NOT whether the words match the script
(that's a different agent) and NOT whether mouths move (another agent).

Evaluate:
- Voice naturalness — robotic / stuttery / pitch-shifted?
- Clarity — intelligible or muffled / clipping / distorted?
- Age / character fit — voice matches character appearance?
- Artefacts — pops, dropouts, silences, mid-word cut-offs
- Background — ambient / SFX mix coherent?

Quote any glitched phrase verbatim.
If audio sounds clean and natural, verdict 'great'.
""".strip()
    return _run("audio_quality", "Audio Quality", prompt, video_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 7 — Lip-Sync Correspondence
# ─────────────────────────────────────────────────────────────────────────────

def run_lipsync_correspondence(
    video_bytes: bytes,
    speech_grid: Optional[bytes],
    speech_captions: list[str],
) -> AgentReport:
    if not speech_grid:
        return AgentReport(
            agent_id="lipsync_correspondence",
            title="Lip-Sync Correspondence",
            focus="Does mouth movement accompany each spoken line?",
            verdict="great",
            summary="No detectable speech in the clip — lip-sync check is not applicable.",
            findings="",
            elapsed_s=0.0,
        )

    n_segs = len(speech_captions)
    segment_block = "\n".join(
        f"  Row {i + 1} (panels {2*i + 1} and {2*i + 2}): \"{c}\""
        for i, c in enumerate(speech_captions)
    )
    prompt = f"""
You are a lip-sync correspondence reviewer. Your scope is narrow and critical.

The grid has {n_segs} ROWS. Each row = ONE spoken line. LEFT panel is the
FIRST frame of the line; RIGHT panel is the LAST. For actual speech to be
happening, the character's mouth must change SHAPE between left and right
(forming different phonemes).

STEP 1 — WITHOUT READING THE CAPTIONS:
For each row, describe the MOUTH of the main visible character in the LEFT
panel, then the SAME character in the RIGHT panel. State plainly:
  - "IDENTICAL" (same pose — a smile or same open angle in both frames)
  - "DIFFERENT" (mouth is clearly forming a different shape)

CRITICAL DISTINCTION: a static cartoon smile (mouth slightly open, teeth
showing, unchanging) is NOT speaking. Speaking means mouth shape CHANGES
between start and end of the line.

STEP 2 — NOW READ THE CAPTIONS:
{segment_block}

STEP 3 — JUDGE:
- Every row IDENTICAL shapes → AUDIO FROM NOWHERE (major defect).
- Wrong speaker (wrong character's mouth moves) → major defect.
- Mouth changes but poorly matches phonemes → minor defect.
- Mouths clearly move in sync with speech → 'great'.

BIAS WARNING: audio plays regardless of whether a mouth moves. DO NOT
charitably assume a closed / smiling mouth is speaking because you hear
audio. Trust your eyes, not the audio.

Your `findings` MUST enumerate each row explicitly: "Row 1: Skye=CLOSED/OPEN,
Chase=CLOSED/OPEN; caption 'Yee-haw!'; assessment: ..."
""".strip()
    return _run(
        "lipsync_correspondence",
        "Lip-Sync Correspondence",
        prompt,
        video_bytes,
        image_bytes=speech_grid,
        image_caption="SPEECH-TIMESTAMP GRID (each row = one spoken line, LEFT=start frame, RIGHT=end frame):",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 8 — Speech & Dialogue Coherence  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def run_speech_coherence(
    video_bytes: bytes,
    transcript: str,
    segments_block: str,
) -> AgentReport:
    """
    Evaluate the LINGUISTIC quality of the spoken dialogue: are the words
    real, grammar sensible, ending complete, word-spacing natural?

    This task is primarily TEXT analysis of the Whisper transcript plus the
    per-segment timestamps (which let us detect long gaps or suspiciously
    fast pacing). It does NOT need Gemini Pro's full video-reasoning stack —
    running it on Flash cuts wall-clock from minutes to seconds. If the
    transcript already shows a dangling hyphen we flag mid-word cut-off
    from text alone; pauses are judged from segment timestamps.
    """
    import time as _time
    t0 = _time.time()

    if not transcript.strip():
        return AgentReport(
            agent_id="speech_coherence",
            title="Speech & Dialogue Coherence",
            focus="Linguistic quality of spoken dialogue",
            verdict="great",
            summary="No speech detected in the clip — speech coherence check skipped.",
            findings="",
            elapsed_s=round(_time.time() - t0, 1),
        )

    prompt = f"""
You are a speech-coherence reviewer. You are given the Whisper transcript of
an AI-generated short video, plus per-segment timestamps. Judge the LINGUISTIC
quality of the spoken content only. No video is provided (this is a
text-reasoning task).

Evaluate four dimensions:

A. REAL WORDS — Is every word in the transcript an actual word in the
   detected language? AI TTS sometimes produces plausible-sounding gibberish
   (e.g. "bluntrip", "folvery"). Quote the phrase if you find any.

B. GRAMMAR & SENSE — Are the sentences grammatically and semantically
   coherent? Short cartoon exclamations are fine; scrambled syntax is not.

C. MID-WORD CUT-OFF — Does the transcript end mid-word or mid-sentence?
   (e.g. "You're the b-", "This is a-"). A dangling hyphen or missing final
   punctuation signals a truncation defect.

D. WORD PACING — Do the SEGMENT TIMESTAMPS below suggest natural pacing?
   - Very fast: words fused / segments < 0.05s per word (machine-gun TTS).
   - Suspicious gaps: multi-second silence between consecutive segments
     mid-sentence is weird.

Quote exact phrases you critique. If the transcript is clean, verdict 'great'.

=== WHISPER TRANSCRIPT ===
{transcript}

=== SEGMENT TIMING ===
{segments_block}
""".strip()

    # Flash-only call, no video — should return in 2-5s.
    parts = [prompt + "\n\n" + _OUTPUT_CONTRACT]
    last_exc = None
    data = None
    for attempt in range(3):
        try:
            resp = _client().models.generate_content(
                model="gemini-2.5-flash",
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    http_options=types.HttpOptions(timeout=60_000),
                ),
            )
            data = _parse_json((resp.text or "").strip())
            break
        except Exception as exc:
            last_exc = exc
            if any(code in str(exc) for code in ("499", "503", "504", "CANCELLED", "UNAVAILABLE", "DEADLINE")):
                if attempt < 2:
                    import time as _t2
                    _t2.sleep(2 ** attempt)
                    continue
            break

    if data is None:
        return AgentReport(
            agent_id="speech_coherence",
            title="Speech & Dialogue Coherence",
            focus="Linguistic quality of spoken dialogue",
            verdict="great",
            summary=f"(agent error: {last_exc})",
            findings=str(last_exc or "unparseable"),
            elapsed_s=round(_time.time() - t0, 1),
        )

    return AgentReport(
        agent_id="speech_coherence",
        title="Speech & Dialogue Coherence",
        focus="Linguistic quality of spoken dialogue",
        verdict=data["verdict"],
        summary=data["summary"],
        findings=data["findings"],
        elapsed_s=round(_time.time() - t0, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 9 — Character Consistency Audit (judge mode, strict rubric)
# ─────────────────────────────────────────────────────────────────────────────

def run_character_consistency_audit(video_bytes: bytes, face_grid: bytes) -> AgentReport:
    """
    Strict Judge-mode audit of character drift across the face-crop grid.

    Uses a fixed four-axis rubric (Rendering / Geometry / Assets / Color) with
    numeric drift scores 0-10. The VLM compares every panel against Panel 1
    (reference) and reports the WORST drift observed across the grid. Output
    format is strict: four metric lines + a final conclusion.

    Verdict is derived from the aggregate score (sum of the four metrics,
    max 40): 0-8 great, 9-20 minor_issues, 21-40 major_issues.
    """
    import time
    import re as _re

    judge_prompt = """
Role: You are an automated Visual Consistency Judge. Your sole purpose is to
detect and quantify "Character Drift" between frames. Do not offer creative
advice or generation tips.

You are given a 4×4 FACE-CROP GRID of 16 panels. Treat Panel 1 as the
REFERENCE character.

IMPORTANT — multi-character handling: the scene may contain more than one
distinct character. A panel showing a DIFFERENT character than the one in
Panel 1 is NOT drift — it is simply a different character. SKIP such panels
entirely. Only compare panels where you believe the SAME character as Panel 1
is visible. (If every panel shows a different character than Panel 1 pick the
most commonly-appearing character as the reference instead.)

For each panel where the SAME character as the reference is visible, compare
the face against the reference on the four metrics below. Report the WORST
drift observed across those same-character comparisons.

Evaluation Metrics (the Delta):

Rendering Delta: Identify changes in shader complexity, lighting models
(e.g. flat to volumetric), and texture resolution.

Geometric Delta: Identify shifts in the underlying mesh, bone structure,
muzzle volume, and ear placement.

Asset Delta: Compare logos, badges, and clothing patterns for any change in
"vector" precision or 3D depth.

Color Delta: Detect shifts in hue, saturation, or luminance in the
character's primary palette.

For each metric, give a Drift Score 0-10 where 0 is identical and 10 is a
complete style collapse.

Your response must be a JSON object matching this shape exactly:
{
  "metric_rendering": { "description": "<short description>", "score": <int 0-10> },
  "metric_geometry":  { "description": "<short description>", "score": <int 0-10> },
  "metric_assets":    { "description": "<short description>", "score": <int 0-10> },
  "metric_color":     { "description": "<short description>", "score": <int 0-10> },
  "pass_fail": "PASS" | "FAIL",
  "aggregate_score": <int 0-40>
}

Use PASS when aggregate_score ≤ 8, FAIL otherwise.
Respond with JSON only, no prose.
""".strip()

    parts: list = [types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")]
    parts.append("FACE-CROP GRID — 16 panels, Panel 1 is the REFERENCE for comparison:")
    parts.append(types.Part.from_bytes(data=face_grid, mime_type="image/png"))
    parts.append(judge_prompt)

    t0 = time.time()
    last_exc = None
    data = None
    for attempt in range(3):
        try:
            resp = _client().models.generate_content(
                model=MODEL_ID,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    http_options=types.HttpOptions(timeout=240_000),
                ),
            )
            text = (resp.text or "").strip()
            m = _re.search(r"\{.*\}", text, _re.DOTALL)
            if m:
                text = m.group(0)
            data = json.loads(text)
            break
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if any(code in msg for code in ("499", "503", "504", "CANCELLED", "UNAVAILABLE", "DEADLINE")):
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
            # Fall through with data=None
            break

    if data is None:
        return AgentReport(
            agent_id="character_consistency_audit",
            title="Character Consistency Audit",
            focus="Strict four-axis drift rubric on face grid",
            verdict="great",
            summary=f"(agent error: {last_exc})",
            findings=str(last_exc or "unparseable"),
            elapsed_s=round(time.time() - t0, 1),
        )

    def _metric_line(label: str, key: str) -> str:
        m = data.get(key, {}) or {}
        desc = str(m.get("description", "")).strip()
        score = m.get("score", 0)
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(10, score))
        return f"[{label}]: {desc} | Score: {score}/10"

    try:
        agg = int(data.get("aggregate_score", 0))
    except (TypeError, ValueError):
        agg = 0
    agg = max(0, min(40, agg))
    pass_fail = str(data.get("pass_fail", "FAIL")).upper().strip()
    if pass_fail not in {"PASS", "FAIL"}:
        pass_fail = "FAIL" if agg > 8 else "PASS"

    if agg <= 8:
        verdict = "great"
    elif agg <= 20:
        verdict = "minor_issues"
    else:
        verdict = "major_issues"

    findings_strict = "\n".join([
        "Metric 1 " + _metric_line("Rendering", "metric_rendering"),
        "Metric 2 " + _metric_line("Geometry",  "metric_geometry"),
        "Metric 3 " + _metric_line("Assets",    "metric_assets"),
        "Metric 4 " + _metric_line("Color",     "metric_color"),
        "",
        f"Final Conclusion: {pass_fail} | Aggregate Drift Score: {agg}/40",
    ])

    summary = f"Aggregate drift {agg}/40 — {pass_fail}"
    return AgentReport(
        agent_id="character_consistency_audit",
        title="Character Consistency Audit",
        focus="Strict four-axis drift rubric on face grid",
        verdict=verdict,
        summary=summary,
        findings=findings_strict,
        elapsed_s=round(time.time() - t0, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 10 — Prompt Fidelity
# ─────────────────────────────────────────────────────────────────────────────

def run_prompt_fidelity(video_bytes: bytes, transcript: str, prompt_text: str) -> AgentReport:
    prompt = """
You are the script supervisor. You have the intended GENERATION PROMPT and
the Whisper TRANSCRIPT. Scope: does the video deliver the prompt?

A. SCENE FIDELITY — Setting, characters, overall action shown match the prompt?

B. ACTION vs DIALOGUE — If dialogue references an action ("look at my backflip"),
   does that action actually happen?

C. SPEAKER ASSIGNMENT — If the prompt scripts character A to say a line, is
   character A the one whose mouth moves during it?

Quote dialogue verbatim; describe specific visual evidence.

DO NOT compare the transcript word-by-word to the prompt — tiny ASR errors
like "yee-ho" vs "yee-haw" are not defects. Care about MEANING.

Wrong speaker or promised-action missing is MAJOR.
If everything lines up, verdict 'great'.
""".strip()
    extra = f"WHISPER TRANSCRIPT:\n{transcript}\n\nORIGINAL GENERATION PROMPT:\n{prompt_text}"
    return _run("prompt_fidelity", "Prompt Fidelity", prompt, video_bytes, extra)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Reasoner (Deduction Engine) — optional downstream agent
# ─────────────────────────────────────────────────────────────────────────────

def run_prompt_reasoner(
    video_bytes: bytes,
    reports: list[AgentReport],
    aggregator: dict,
    prompt_text: str,
) -> dict:
    """
    The "Visual Detective" / Deduction Engine.

    Downstream of the specialist agents and aggregator. Given the original
    prompt, the specialist findings, and the video itself, propose a
    rewritten prompt that removes the SEMANTIC TRIGGERS causing the
    model's observed failures.

    Guiding principle: the model is a next-frame predictor pulled around
    by its training biases. The fix is never "tell the model not to do X"
    — it is to add a new physical anchor that makes X impossible.

    Returns a dict (never raises). If the aggregator verdict is PASS with
    no defects worth addressing, the agent returns a skip marker instead
    of a rewrite.
    """
    if not reports and not aggregator:
        return {
            "skipped_reason": "no specialist reports available",
            "diagnosed_conflicts": [],
            "trigger_attributions": [],
            "counterfactual_edits": [],
            "rewritten_prompt": "",
            "rationale": "",
            "elapsed_s": 0.0,
        }

    try:
        score = int(aggregator.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    overall = str(aggregator.get("overall_verdict", "")).upper()
    has_real_defects = any(
        r.verdict in {"minor_issues", "major_issues"} for r in reports
    )
    if overall == "PASS" and score >= 90 and not has_real_defects:
        return {
            "skipped_reason": "video passed cleanly — no prompt rewrite warranted",
            "diagnosed_conflicts": [],
            "trigger_attributions": [],
            "counterfactual_edits": [],
            "rewritten_prompt": prompt_text,
            "rationale": "",
            "elapsed_s": 0.0,
        }

    specialist_block = "\n\n".join(
        f"### {r.title} — [{r.verdict}]\n"
        f"Summary: {r.summary}\n"
        f"Findings: {r.findings}"
        for r in reports
    )
    agg_block = "\n".join([
        f"Verdict: {aggregator.get('overall_verdict', '')}",
        f"Score: {aggregator.get('score', '')}",
        f"Headline: {aggregator.get('headline', '')}",
        f"What works: {aggregator.get('what_works', '')}",
        f"What breaks: {aggregator.get('what_breaks', '')}",
        f"Top issue: {aggregator.get('top_issue', '')}",
    ])

    system = f"""
You are the PROMPT REASONING AGENT — a "visual detective" whose job is to
understand WHY a text-to-video model failed to deliver the requested video
and propose a SMARTER prompt that removes the hidden triggers behind the
failure.

═══════════════════════════════════════════════════════════════════════════
CORE PHILOSOPHY
═══════════════════════════════════════════════════════════════════════════

Video models are next-frame prediction engines. They do not "read" prompts
— they pattern-match words to visual automations learned from training
data. If a phrase is statistically linked to a motion, a lighting change,
or a composition, the model will produce that automation regardless of
what the rest of the prompt says.

Negative prompts ("the car is NOT moving") are mostly ignored because the
token "moving" still activates motion priors. The correct intervention is
TRIGGER REMOVAL: rewrite the prompt so the physical setup makes the
unwanted behavior logically impossible.

Examples of the pattern:
  • Prompt says "stationary car" but wheels spin + background scrolls.
    Hidden trigger: "hands on the steering wheel" → driving.
    Smart fix: "driver turned sideways to the passenger, hands on knees."
  • Prompt says "silent cliff-top portrait" but hair whips violently.
    Hidden trigger: "cliff" → high winds in training data.
    Smart fix: "hair tied in a tight bun" — removes the wind's canvas.
  • Prompt says "quiet cafe conversation" but characters gesture wildly.
    Hidden trigger: "busy cafe" → commotion as a theme.
    Smart fix: "holding a warm tea cup with both hands" — locks the torso.
  • Prompt says "calm night" but face flickers in changing light.
    Hidden trigger: "TV" or "neon" → attempted dynamic ray-tracing.
    Smart fix: "room lit by a static, warm lamp."

The word "static" applied to lighting, the word "tied" applied to hair,
the word "resting" applied to hands — these are ANCHORS. An anchor is a
physical fact that the model must honor, and that precludes the error.

═══════════════════════════════════════════════════════════════════════════
SCOPE
═══════════════════════════════════════════════════════════════════════════

You fix defects that are plausibly caused by PROMPT-ACTIVATED TRAINING BIAS:
  • Unintended motion (things moving that shouldn't)
  • Hyperactive / unstable composition
  • Lighting or reflection instability
  • Environment flicker from crowded scene descriptions
  • Prompt-vs-outcome mismatches where the outcome is a known bias
    ("cliff" → wind, "wet" → reflections, "busy" → commotion)

You DO NOT attempt to fix defects caused by model capacity limits:
  • Character identity drift / face morphing
  • Anatomy corruption (extra fingers, melting limbs)
  • Lip-sync failures
  • Audio artefacts / gibberish speech
  • Text rendering
If the defect class is capacity-limited, flag it and DO NOT pretend a
prompt rewrite will help — that would mislead the user.

═══════════════════════════════════════════════════════════════════════════
REASONING PROTOCOL — follow all four steps
═══════════════════════════════════════════════════════════════════════════

STEP 1 · GAP ANALYSIS
  Compare three vectors:
    (a) Intent — what the original prompt asked for.
    (b) Outcome — what the specialists observed in the video.
    (c) Bias — which training-data association most likely bridges (a)→(b).
  Only produce conflicts that are TRIGGER-BASED. Skip capacity failures.

STEP 2 · INFLUENTIAL OBJECT MAPPING
  Examine the video directly. For each conflict, identify the concrete
  visible element(s) or phrase(s) in the prompt that most plausibly caused
  the model to activate the unwanted automation. Be specific — "hands on
  the wheel", not "driving-related content".

STEP 3 · COUNTERFACTUAL SIMULATION (mental)
  For each candidate edit, ask: "If I remove or replace this element, does
  the probability of the unwanted automation drop?" Prefer edits that
  introduce a POSITIVE physical anchor (a new, concrete, static fact) over
  edits that just delete words. Avoid negations.

STEP 4 · SUBTLE REDESIGN
  Produce the full rewritten prompt. Constraints:
    • Preserve the creative intent — same scene, same characters, same
      emotional tone. This is minimum-edit, not a new concept.
    • Use concrete, physical language. Prefer "hands resting on knees"
      over "hands are still".
    • Never use "not", "no", "without", "never" to express an absence of
      motion or effect. Always express it as a physical fact that makes
      the absence automatic.
    • Keep the prompt roughly the same length as the original.

═══════════════════════════════════════════════════════════════════════════
INPUTS
═══════════════════════════════════════════════════════════════════════════

ORIGINAL GENERATION PROMPT:
{prompt_text}

AGGREGATOR VERDICT:
{agg_block}

SPECIALIST REPORTS:
{specialist_block}

You also have the full native video.

═══════════════════════════════════════════════════════════════════════════
OUTPUT — JSON only, matching this schema exactly
═══════════════════════════════════════════════════════════════════════════

{{
  "skipped_reason": null,
  "diagnosed_conflicts": [
    {{
      "intent": "<what the prompt asked for, concretely>",
      "outcome": "<what the video delivered, concretely — grounded in a specialist finding>",
      "source_agent": "<agent_id that surfaced this, e.g. motion_weight>",
      "is_capacity_limited": false
    }}
  ],
  "trigger_attributions": [
    {{
      "visible_element": "<concrete element observed in the video, e.g. 'driver's hands gripping the steering wheel'>",
      "prompt_phrase": "<exact substring from the original prompt that likely activated this, or '' if it is an implicit genre bias>",
      "hypothesized_bias": "<the training-data association, e.g. 'steering-wheel grip → driving motion'>",
      "confidence": "high"
    }}
  ],
  "counterfactual_edits": [
    {{
      "operation": "replace",
      "from": "<exact substring from original prompt>",
      "to": "<the new physical anchor>",
      "reasoning": "<one sentence: why this severs the trigger>"
    }}
  ],
  "rewritten_prompt": "<the full revised prompt, edits applied, same creative intent>",
  "rationale": "<one short paragraph: the single strongest anchor added and the bias it defeats>"
}}

If every defect is capacity-limited and no trigger-based fix applies, set
"skipped_reason" to a short explanation, leave edits empty, and return the
original prompt unchanged as "rewritten_prompt".

Confidence values allowed: "high" | "medium" | "low".
Operation values allowed: "replace" | "insert" | "delete".
Respond with JSON only. No prose before or after.
""".strip()

    parts: list = [types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"), system]

    t0 = time.time()
    last_exc = None
    data = None
    for attempt in range(3):
        try:
            resp = _client().models.generate_content(
                model=MODEL_ID,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    http_options=types.HttpOptions(timeout=240_000),
                ),
            )
            text = (resp.text or "").strip()
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                text = m.group(0)
            data = json.loads(text)
            break
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if any(code in msg for code in ("499", "503", "504", "CANCELLED", "UNAVAILABLE", "DEADLINE")):
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
            break

    elapsed = round(time.time() - t0, 1)

    if data is None:
        return {
            "skipped_reason": f"(agent error: {last_exc})",
            "diagnosed_conflicts": [],
            "trigger_attributions": [],
            "counterfactual_edits": [],
            "rewritten_prompt": prompt_text,
            "rationale": "",
            "elapsed_s": elapsed,
        }

    def _as_list(v) -> list:
        return v if isinstance(v, list) else []

    def _as_str(v) -> str:
        return str(v).strip() if v is not None else ""

    conflicts = []
    for c in _as_list(data.get("diagnosed_conflicts")):
        if not isinstance(c, dict):
            continue
        conflicts.append({
            "intent": _as_str(c.get("intent")),
            "outcome": _as_str(c.get("outcome")),
            "source_agent": _as_str(c.get("source_agent")),
            "is_capacity_limited": bool(c.get("is_capacity_limited", False)),
        })

    triggers = []
    for t in _as_list(data.get("trigger_attributions")):
        if not isinstance(t, dict):
            continue
        conf = _as_str(t.get("confidence")).lower() or "medium"
        if conf not in {"high", "medium", "low"}:
            conf = "medium"
        triggers.append({
            "visible_element": _as_str(t.get("visible_element")),
            "prompt_phrase": _as_str(t.get("prompt_phrase")),
            "hypothesized_bias": _as_str(t.get("hypothesized_bias")),
            "confidence": conf,
        })

    edits = []
    for e in _as_list(data.get("counterfactual_edits")):
        if not isinstance(e, dict):
            continue
        op = _as_str(e.get("operation")).lower() or "replace"
        if op not in {"replace", "insert", "delete"}:
            op = "replace"
        edits.append({
            "operation": op,
            "from": _as_str(e.get("from")),
            "to": _as_str(e.get("to")),
            "reasoning": _as_str(e.get("reasoning")),
        })

    rewritten = _as_str(data.get("rewritten_prompt")) or prompt_text
    rationale = _as_str(data.get("rationale"))
    skipped = data.get("skipped_reason")
    skipped = _as_str(skipped) if skipped else None

    return {
        "skipped_reason": skipped,
        "diagnosed_conflicts": conflicts,
        "trigger_attributions": triggers,
        "counterfactual_edits": edits,
        "rewritten_prompt": rewritten,
        "rationale": rationale,
        "elapsed_s": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────────────────────────────────────

def run_aggregator(reports: list[AgentReport], prompt_text: str) -> dict:
    agent_blocks = "\n\n".join(
        f"### {r.title} — [{r.verdict}]\n{r.summary}\n\nDetails: {r.findings}"
        for r in reports
    )

    system = f"""
You are the supervising QA director. Nine specialist agents have each watched
the video and written their analyses. Synthesise into a final verdict.

--- ORIGINAL GENERATION PROMPT ---
{prompt_text}

--- SPECIALIST REPORTS ---
{agent_blocks}

Respond as JSON with exactly these keys:
{{
  "overall_verdict": "PASS" | "CONDITIONAL_PASS" | "FAIL",
  "score": <integer 0-100>,
  "headline": "<one sentence a producer would read>",
  "what_works": "<one paragraph, grounded in the reports>",
  "what_breaks": "<one paragraph, grounded in the reports — real failures only>",
  "top_issue": "<the single most important issue to fix if only one could be>"
}}

Scoring guide:
- 85-100 PASS: deliverable, polish notes only.
- 70-84 CONDITIONAL_PASS: watchable but clear flaws.
- 40-69 FAIL: significant defects — regenerate.
- 0-39 FAIL: fundamentally broken.

Weighting:
- Prompt Fidelity + Lip-Sync + Speech Coherence defects are MOST serious —
  a major issue on any alone drops to CONDITIONAL_PASS or FAIL.
- Face & Uncanny Valley issues are serious (viewers notice uncanny faces).
- Rendering defects are serious.
- Body Consistency issues are serious.
- Audio Quality and Environment Stability are moderate if isolated.
- Motion is moderate unless clearly broken.

Do NOT introduce claims not present in the reports. Trust the specialists.
Respond with JSON only.
""".strip()

    resp = _client().models.generate_content(
        model=MODEL_ID,
        contents=system,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            http_options=types.HttpOptions(timeout=240_000),
        ),
    )
    text = (resp.text or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = {"overall_verdict": "FAIL", "score": 0,
               "headline": "(aggregator unparseable)", "what_works": "",
               "what_breaks": text[:500], "top_issue": ""}
    verdict = str(obj.get("overall_verdict", "FAIL")).upper().replace(" ", "_")
    if verdict not in {"PASS", "CONDITIONAL_PASS", "FAIL"}:
        verdict = "FAIL"
    try:
        score = int(obj.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    return {
        "overall_verdict": verdict,
        "score": max(0, min(100, score)),
        "headline": str(obj.get("headline", "")).strip(),
        "what_works": str(obj.get("what_works", "")).strip(),
        "what_breaks": str(obj.get("what_breaks", "")).strip(),
        "top_issue": str(obj.get("top_issue", "")).strip(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent registry — each entry: (agent_id, runner_fn, tuple-of-extra-kwargs)
# ─────────────────────────────────────────────────────────────────────────────

AGENTS = [
    ("body_consistency",            run_body_consistency,            ("body_grid",)),
    ("face_uncanny",                run_face_uncanny,                ("face_grid",)),
    ("character_consistency_audit", run_character_consistency_audit, ("face_grid",)),
    ("motion_weight",               run_motion_weight,               ()),
    ("environment_stability",       run_environment_stability,       ("keyframe_grid",)),
    ("rendering_defects",           run_rendering_defects,           ("keyframe_grid",)),
    ("audio_quality",               run_audio_quality,               ()),
    ("lipsync_correspondence",      run_lipsync_correspondence,      ("speech_grid", "speech_captions")),
    ("speech_coherence",            run_speech_coherence,            ("transcript", "segments_block")),
    ("prompt_fidelity",             run_prompt_fidelity,             ("transcript", "prompt_text")),
]
