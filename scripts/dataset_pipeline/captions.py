"""Caption resolver — looks up `scene_caption` for a clip URL across all
known parquet sources for the Golden Set.

Tested empirically (2026-04-28) to provide 100% coverage for both
chase_golden.json (605/605) and skye_golden.json (467/467) when all four
parquets below are downloaded.

Returns a single string (the prose `scene_caption`), brand-token-substituted
if requested. The `transcript_text` (spoken dialogue) is intentionally
ignored — it's a Step 2/3 (audio/dialogue, frozen) concern.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# All parquets that together cover Chase + Skye Golden Sets. Order doesn't
# matter; first-seen URL wins on duplicates.
PARQUET_SOURCES = [
    "gs://video_gen_dataset/dataset/labelbox/Project_Lipsync/Chase/chase_all_seasoned_results_filtered.parquet",
    "gs://video_gen_dataset/dataset/labelbox/Project_Lipsync/Other/Golden_dataset/skye_all_seasons_results_filtered_augmented_no_text_train_20260101_100234.parquet",
    "gs://video_gen_dataset/dataset/labelbox/chase_golden_dataset.parquet",
    "gs://video_gen_dataset/dataset/labelbox/Project_Lipsync/Other/Golden_dataset/skye_golden_dataset.parquet",
]

BRAND_TOKEN_MAP = {
    "chase": "CHASE_PP",
    "skye":  "SKYE_PP",
}

_BRAND_TOKENS_YAML = Path(__file__).resolve().parent / "brand_tokens.yaml"


def load_brand_tokens(config_path: Path | None = None) -> dict[str, str]:
    """Load the brand-token substitution map from YAML. Falls back to the
    BRAND_TOKEN_MAP module-level default if the file is missing or unreadable.
    Keys are normalized to lowercase. Values are returned as-is.
    """
    path = config_path or _BRAND_TOKENS_YAML
    try:
        import yaml
        with path.open() as f:
            cfg = yaml.safe_load(f) or {}
        subs = (cfg.get("substitutions") or {})
        return {str(k).lower(): str(v) for k, v in subs.items()}
    except Exception:  # noqa: BLE001
        return {k.lower(): v for k, v in BRAND_TOKEN_MAP.items()}


@dataclass
class CaptionEntry:
    url: str
    caption: str
    speaker: str | None
    source_parquet: str


class CaptionIndex:
    """In-memory dict of url → CaptionEntry, built once per build."""

    def __init__(self) -> None:
        self._by_url: dict[str, CaptionEntry] = {}

    def __len__(self) -> int:
        return len(self._by_url)

    def get(self, url: str) -> CaptionEntry | None:
        return self._by_url.get(url)

    def load_parquet(self, local_path: Path, source_uri: str) -> int:
        import pyarrow.parquet as pq
        t = pq.read_table(str(local_path))
        cols = t.column_names
        if "output_video_path" not in cols or "scene_caption" not in cols:
            return 0
        urls = t.column("output_video_path").to_pylist()
        caps = t.column("scene_caption").to_pylist()
        spks = t.column("speaker").to_pylist() if "speaker" in cols else [None] * len(urls)
        added = 0
        for u, c, s in zip(urls, caps, spks):
            if not u or not c:
                continue
            if u in self._by_url:
                continue
            self._by_url[u] = CaptionEntry(url=u, caption=c, speaker=s, source_parquet=source_uri)
            added += 1
        return added


def fetch_parquets(cache_dir: Path) -> list[tuple[Path, str]]:
    """Download every PARQUET_SOURCES entry to cache_dir, idempotent.
    Returns [(local_path, gs_uri), ...]."""
    import subprocess
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for uri in PARQUET_SOURCES:
        name = uri.rsplit("/", 1)[-1]
        local = cache_dir / name
        if not local.exists():
            r = subprocess.run(
                ["gcloud", "storage", "cp", uri, str(local)],
                capture_output=True, text=True, check=False,
            )
            if r.returncode != 0:
                # parquet may be missing in some buckets; skip but flag
                print(f"WARN: failed to fetch {uri}: {r.stderr.strip().splitlines()[-1] if r.stderr else '?'}")
                continue
        out.append((local, uri))
    return out


def build_caption_index(cache_dir: Path) -> CaptionIndex:
    idx = CaptionIndex()
    for local, uri in fetch_parquets(cache_dir):
        idx.load_parquet(local, uri)
    return idx


# --- Brand token substitution -----------------------------------------------

def _word_boundary_replace(text: str, needle: str, repl: str) -> str:
    # Case-insensitive whole-word substitution. Preserves surrounding punctuation.
    pattern = re.compile(rf"\b{re.escape(needle)}\b", flags=re.IGNORECASE)
    return pattern.sub(repl, text)


def apply_brand_tokens(caption: str, mapping: dict[str, str] | None = None) -> str:
    """Substitute character names → brand tokens (whole-word, case-insensitive).
    `mapping` is {raw_name: token}. If None, loads from brand_tokens.yaml.
    Longer names are processed first to avoid clobbering compound substrings."""
    m = mapping if mapping is not None else load_brand_tokens()
    if not m:
        return caption
    out = caption
    for raw in sorted(m.keys(), key=len, reverse=True):
        out = _word_boundary_replace(out, raw, m[raw])
    return out
