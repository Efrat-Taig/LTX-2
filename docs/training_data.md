# Training Data — Live Map

> **Read first.** This is the canonical map of where Paw Patrol training data lives. Update it whenever you add, move, or retire data. Anything not in this map is presumed not load-bearing.
>
> **Constraint:** no file is to be moved or deleted without explicit user approval. Recommended consolidation lives at the bottom — it's a *target state*, not a to-do list.

Last refreshed: **2026-04-28** from `docs/phase0/data_inventory.csv` (audit script: `scripts/phase0/audit_data.py`).

---

## TL;DR

Training data lives in **three GCS prefixes, all under `gs://video_gen_dataset/`**:

| Location | Role | Size | Files | Trust |
|---|---|---:|---:|---|
| `raw/paw_patrol/` | Source-of-truth raw broadcast episodes (10 seasons + scripts + some audio) | **494 GB** | 958 | high — never edit |
| `TinyStories/data_sets/` | Curated/processed character datasets + preprocessed latents | 13 GB | 3 362 | high — used by trainer today |
| `dataset/labelbox/` | Labelbox export: per-clip mp4+wav pairs + JSON/parquet manifests for the Golden Set | 4 GB | 27 442 | medium — verify which clips made it into TinyStories |

Local working tree contains **only** LoRA inference outputs and benchmark conditioning images. **Zero training material lives locally.**

The audit found **9 966 duplicate groups** by md5 across the three prefixes — same files copied between locations during prior reorganizations. ~150+ are cross-prefix (same clip in `raw/`, `labelbox/`, *and* `TinyStories/`).

---

## 1. `gs://video_gen_dataset/raw/paw_patrol/` — Source of truth

**494 GB, 958 files.** Untouched broadcast episodes. The only place where original, uncut content lives. Everything else in the map is derived from this.

| Season | Files | Size | Notes |
|---|---:|---:|---|
| `season_1/` | 74 | 22.2 GB | |
| `season_2/` | 93 | 23.8 GB | |
| `season_3/` | 98 | 16.6 GB | |
| `season_4/` | 88 | 48.3 GB | |
| `season_5/` | 78 | 48.5 GB | |
| `season_6/` | 77 | 63.6 GB | |
| `season_7/` | 78 | 65.3 GB | |
| `season_8/` | 129 | 75.3 GB | |
| `season_9/` | 78 | 69.0 GB | |
| `season_10/` | 158 | 59.5 GB | larger file count, may include extras |
| `test_episodes/` | 7 | 2.1 GB | held-out for eval |

**Per-episode layout** (typical): `season_N/compressed_videos/PawPatrol_SXX_EYY_A.{mp4,pdf}` — mp4 is the episode video, pdf is the broadcast script. Some seasons also have `.wav` audio side-files (53 wavs total; not yet investigated which seasons).

**File-type breakdown:** 513 mp4 (videos), 388 pdf (scripts), 53 wav (audio).

**Status:** raw, never used directly for training. Source for freeze-frame anchor extraction and clip extraction into the Golden Set.

---

## 2. `gs://video_gen_dataset/TinyStories/data_sets/` — Curated/processed

**13.18 GB, 3 362 files.** The "operational" location for character-specific datasets. Mix of source mp4s, preprocessed latents (.pt), and per-character manifests.

| Subprefix | Files | Content | Status |
|---|---:|---|---|
| `golden_skye/videos/` | 636 mp4 | Original golden Skye clips | **training-ready (videos)** |
| `golden_chase/` | 1 (marker) | Empty placeholder — golden Chase clips live elsewhere (likely `dataset/labelbox/` or were never copied here) | **needs reconciliation** |
| `full_cast_raw/` | 136 mp4 + 4 csv | Source clips for the multi-character "full cast" dataset + dataset CSVs (orig + recaptioned) | training-ready |
| `full_cast_combined_preprocessed/` | 1 563 .pt | Preprocessed latents (~472 clips × multiple latent files) — what the trainer reads directly | **trainer-ready (latents)** |
| `phase4_phase1_chase_skye/` | 1 021 files | Phase 4 curriculum subset (chase + skye golden only) | training-ready |

**Filename convention** in `full_cast_combined_preprocessed/`: per-clip `.pt` files holding latents (probably tuples of video+text+audio embeddings — verify by reading one). The trainer's preprocess script (`packages/ltx-trainer/scripts/process_captions.py`) generates these.

**Per-character manifest CSVs**: `full_cast_raw/*.csv` (4 files — `_orig.csv` and `_recaptioned.csv` per character or per dataset; not opened yet).

---

## 3. `gs://video_gen_dataset/dataset/labelbox/` — Labelbox prep

**4.21 GB, 27 442 files.** The Labelbox export. Where the Golden Set was assembled clip-by-clip with per-clip captions and lipsync data.

| Subprefix | Files | Content |
|---|---:|---|
| `output/` | 22 669 | Small per-clip annotation files (Labelbox label JSONs, etc.). 0.04 GB total — these are tiny metadata files. |
| `videos/` | 4 721 | 2 363 mp4 + 2 361 wav — paired video/audio per clip. The actual Golden Set clip pool. |
| `Project_Lipsync/` | 49 | Per-character subdirs (Chase/, Everest/, …) with lipsync-specific exports |
| `chase_golden.json` | 1 | Top-level Chase Golden Set manifest |
| `skye_golden.json` | 1 | Top-level Skye Golden Set manifest |
| `chase_golden_dataset.parquet` | 1 | Tabular form of the Chase manifest |

**File-type breakdown:** 2 363 mp4 + 2 361 wav (video/audio pairs) + 22 669 small jsons + 18 parquet + 2 csv.

**Why the very high file count vs small total size:** the `output/` subdir has tens of thousands of tiny per-frame or per-annotation JSONs, not training data per se.

**Status:** medium-trust. The Golden Set canonical record but the trainer doesn't read directly from here — it reads from `TinyStories/data_sets/`. There's an implicit ETL between the two that isn't currently documented.

---

## 4. Cross-cutting observations

- **9 966 md5 duplicate groups.** The same physical bytes appear in multiple paths. Some are within-prefix (Labelbox keeping intermediate copies); ≥147 cross at least two of the three top-level prefixes. **Risk:** the Trainer could be loading the same clip twice via different paths. Worth a one-time pass with `duplicates.csv` to confirm `full_cast_raw/*.mp4` doesn't intersect `labelbox/videos/*.mp4` for the same scene.
- **`golden_chase/` is empty** under `TinyStories/data_sets/`. The Chase Golden Set videos must live in `dataset/labelbox/videos/` or `dataset/labelbox/Project_Lipsync/Chase/`. Needs explicit pointer.
- **No anchor stills under these three prefixes.** The 138 character anchor jpgs at `gs://video_gen_dataset/TinyStories/character_images_bank/` are now out-of-scope per current `pre_flight.yaml`, but they're real and load-bearing. Either re-include them in the audit scope or note them here as a fourth canonical location. **Recommend re-including.**
- **494 GB raw vs 13 GB curated** — only ~3% of source video has been turned into training material. Plenty of room to grow.

---

## 5. Proposed consolidation (target state — not yet)

> **Reminder:** no moves until you sign off. This section is the *plan*, not the change.

The current layout is the result of several reorganizations. To get to "all training data in one central, organized place" without breaking the trainer (which currently reads from `TinyStories/data_sets/`), I'd propose:

```
gs://video_gen_dataset/
  raw/paw_patrol/                              ← unchanged (source of truth)
    season_1..10/...
    test_episodes/

  training_data/                               ← NEW canonical home; replaces TinyStories/data_sets/
    chase/
      golden/                                  ← from labelbox + golden_chase reconciliation
        videos/*.mp4
        captions.csv
      anchors/                                 ← from character_images_bank/CHASE + freeze-frame extraction
        *.jpg (named PawPatrol_SXX_EYY_*_fXXXX.jpg)
      latents/                                 ← preprocessed .pt for the trainer (lives or is regenerated here)
      manifest.json                            ← clip count, captions version, source provenance
    skye/                                      ← same structure
      golden/, anchors/, latents/, manifest.json
    full_cast/                                 ← multi-character
      golden/, anchors/, latents/, manifest.json

  training_runs/                               ← already exists per old TRAINING_PROTOCOL.md
    ltx2/{character}/{exp_name}/{checkpoints,run_info.json,config.yaml}

  archive/                                     ← NEW: where pre-consolidation paths go on cutover
    labelbox-2026-04-28/                       ← snapshot of dataset/labelbox/ before migration
    tinystories-2026-04-28/                    ← snapshot of TinyStories/* before migration
```

**Why this shape:**

1. **One canonical home per character** (`training_data/{character}/`) directly mirrors `vision.md`'s philosophy: each brand entity gets its own dedicated drawer.
2. **`golden/` + `anchors/` + `latents/` colocated** means a single ETL pipeline owns the chain `raw → curated → preprocessed`. Today it's split across `dataset/labelbox/` (curation) and `TinyStories/data_sets/full_cast_combined_preprocessed/` (latents) with no documented link.
3. **`manifest.json` is the load-bearing artifact**, not the folder layout — it lists clip count, caption version, source episode/scene, the matching latent files. Trainer reads the manifest, not the folder.
4. **`archive/` keeps a dated snapshot** of the prior layout on cutover. No deletions until we've trained one experiment from the new layout end-to-end and verified parity.

**What this does NOT change:**
- `raw/paw_patrol/` stays put.
- `training_runs/` stays put.
- The trainer continues reading the *same paths it reads today* until cutover. Migration can be staged: copy → re-train one experiment from new paths → verify → archive old paths.

---

## 6. Open questions for you (before I propose any moves)

1. **Where does the Chase Golden Set actually live?** `golden_chase/` is empty; manifests are in `dataset/labelbox/`. Is the canonical Chase video pool in `labelbox/videos/`, `labelbox/Project_Lipsync/Chase/`, or somewhere else?
2. **Should `character_images_bank/` be included** in this map? It has 138 jpgs which look exactly like extracted anchor stills (`PawPatrol_SXX_EYY_*_fXXXX.jpg`).
3. **What's the source of truth for captions** — the per-clip JSONs in `dataset/labelbox/output/`, the parquet manifests, the recaptioned CSVs in `full_cast_raw/`, or something else?
4. **Are `full_cast_combined_preprocessed/*.pt` reproducible** from raw inputs (i.e. can we re-run preprocessing) or are they treated as terminal artifacts?

Answer those and I can refine the consolidation proposal into an actual migration plan.

---

## 7. How to refresh this map

```bash
# Rerun the audit (≈1 min on the 3 narrowed prefixes)
python scripts/phase0/audit_data.py --no-local

# Then update the per-prefix counts/sizes in this file from:
cat docs/phase0/coverage.md
```

The numeric tables above should match `docs/phase0/coverage.md` exactly. If they drift, the audit was rerun without updating this map — fix here.
