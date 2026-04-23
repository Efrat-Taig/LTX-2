"""
Perception layer — only what VLM agents can't do natively:
Whisper transcription with segment-level timestamps (for the lip-sync agent).
"""
from __future__ import annotations

import ssl
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from loguru import logger

_whisper_model = None


@dataclass
class SpeechSegment:
    start: float
    end: float
    text: str

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    ssl._create_default_https_context = ssl._create_unverified_context
    import whisper
    logger.info("Loading Whisper base model…")
    _whisper_model = whisper.load_model("base")
    return _whisper_model


def transcribe_with_segments(video_path: str) -> Tuple[str, List[SpeechSegment]]:
    """
    Run Whisper on the video's audio.
    Returns (full transcript text, list of SpeechSegment).
    """
    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", wav_tmp.name],
            check=True, capture_output=True,
        )
        model = _load_whisper()
        result = model.transcribe(wav_tmp.name, word_timestamps=False, language=None)
        text = (result.get("text") or "").strip()
        segments = [
            SpeechSegment(
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                text=(s.get("text") or "").strip(),
            )
            for s in result.get("segments", [])
        ]
        return text, segments
    except Exception as exc:
        logger.warning(f"Transcription failed: {exc}")
        return "", []
    finally:
        try:
            Path(wav_tmp.name).unlink()
        except OSError:
            pass


# Backwards-compat helper
def transcribe(video_path: str) -> str:
    text, _ = transcribe_with_segments(video_path)
    return text
