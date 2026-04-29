# Training Data — Live Map

> **Read first.** Where Paw Patrol training data lives, what's in each location, and what role it plays.
> Update whenever you add, retire, or restructure a dataset.

Last refreshed: **2026-04-28** (post-rebuild — chase_golden_v1 + skye_golden_v2 produced via the `dataset_pipeline/` rebuild).

---

## TL;DR

| Tier | Location | Role | Status |
|---|---|---|---|
| **Source of truth (raw)** | `gs://video_gen_dataset/raw/paw_patrol/` | Untouched broadcast episodes (10 seasons, 494 GB, 958 files) | Never edited — read-only |
| **Source of truth (curation)** | `gs://video_gen_dataset/dataset/labelbox/{chase,skye}_golden.json` + sibling `_filtered.parquet`s | Which clips to use + their captions | Read-only inputs |
| **Trainer-ready** ⭐ | `gs://video_gen_dataset/TinyStories/training_data/{chase_golden_v1, skye_golden_v2}/` | Self-describing datasets the trainer reads | The new canonical home |
| **Legacy** | `gs://video_gen_dataset/TinyStories/data_sets/{golden_chase, golden_skye, full_cast_*, phase4_*}/` | Old pipeline output (mixed pre-processed mp4s + .pt latents) | Superseded by `training_data/`. Treat as archive. |

The local working tree contains only LoRA inference outputs and benchmark inputs — **zero training material**.

---

## 1. The new canonical home: `gs://video_gen_dataset/TinyStories/training_data/`

Each dataset there is self-contained and follows a single layout (see [`training_data_pipeline.md`](training_data_pipeline.md) §2 for the full schema):

```
training_data/{dataset_id}/
    metadata.json                                          ← canonical record (raw captions, status, md5, layout, provenance)
    source_gallery.mp4                                     ← raw clips, full duration, letterboxed for canvas fit
    training_gallery.mp4                                   ← 49-frame letterboxed clips — what the model trains on
    raw_videos/0NNN.mp4                                    ← untouched manifest downloads
    processed_videos/
        0NNN.mp4                                           ← 960×544 letterboxed + 49-frame trim (the trainer's input)
        processing_summary.txt                             ← exact ffmpeg recipe
    precomputed/
        vae_latents_video/processed_videos/0NNN.pt         ← VAE video latents (128×7×17×30, bf16)
        text_prompt_conditions/processed_videos/0NNN.pt    ← Gemma text embeddings (1024×4096, bf16)
```

**Training-time bridge:** the LTX-2 trainer hardcodes `latents/` and `conditions/` as its read paths. After pulling a dataset to a training machine, run `bash scripts/dataset_pipeline/stage_for_training.sh DATASET_DIR` once — it creates `precomputed/{latents,conditions}` symlinks pointing at the human-readable dirs. Trainer then reads through the symlinks. No fork, no rename.

**Currently shipped:**

| Dataset | Clip count | Source | Status |
|---|---:|---|---|
| `chase_golden_v1` | 605 | `chase_golden.json` (URLs) + `chase_all_seasoned_filtered.parquet` (captions) | built |
| `skye_golden_v2`  | 111 | `skye_golden_dataset_v2.parquet` (curation+captions) joined to augmented parquet (working URLs) | built |

`{character}_golden_v{N}` naming: bump `vN` only on a full rebuild (manifest changed, brand-token policy changed enough to need re-encoding, encoding parameters changed). Per-clip deletions stay in the same version with index gaps.

---

## 2. Source-of-truth bucket contents (unchanged inputs)

### `gs://video_gen_dataset/raw/paw_patrol/` (494 GB, 958 files)

10 seasons of broadcast PAW Patrol — `season_N/compressed_videos/PawPatrol_SXX_EYY_*.{mp4,pdf,wav}` (mp4 = video, pdf = script, wav = audio side-files where present).

Never edited. The Golden Set's per-scene clips trace back here via `scene_filename`.

### `gs://video_gen_dataset/dataset/labelbox/`

The Labelbox export. The pipeline reads only:
- `chase_golden.json` (605 entries) — the chase curation list
- `skye_golden.json` (467 entries; v1 — superseded by v2 parquet for current builds)
- `Project_Lipsync/Chase/chase_all_seasoned_results_filtered.parquet` (chase captions, 2 989 rows)
- `Project_Lipsync/Other/Golden_dataset/skye_all_seasons_results_filtered_augmented_no_text_train_*.parquet` (skye captions + working URLs for v2 join)

**Caveat:** `chase_golden_dataset.parquet` at the labelbox root is mis-named — contains only Skye rows. Don't use it.

The 22 669 small JSONs in `dataset/labelbox/output/` are Labelbox per-frame annotations; the pipeline doesn't read them.

---

## 3. Legacy locations — archive candidates

| Old path | What was there | Replaced by |
|---|---|---|
| `gs://video_gen_dataset/TinyStories/data_sets/golden_chase/` | (empty marker — was meant to hold chase clips, never populated) | `training_data/chase_golden_v1/videos/` |
| `gs://video_gen_dataset/TinyStories/data_sets/golden_skye/videos/` | 304 mp4s with augmentation suffixes; 111 v2-curated clips matched only ~8 by base stem | `training_data/skye_golden_v2/videos/` (rebuilt from manifest) |
| `gs://video_gen_dataset/TinyStories/data_sets/full_cast_combined_preprocessed/` | 1 563 .pt files (preprocessed latents for an older pipeline) | `training_data/{ds}/precomputed/` (regenerated per-dataset) |
| `gs://video_gen_dataset/TinyStories/data_sets/full_cast_raw/` | 136 mp4s + 4 csv | (not in current scope; if needed → build a `full_cast_v1` dataset via the same pipeline) |
| `gs://video_gen_dataset/TinyStories/data_sets/phase4_phase1_chase_skye/` | 1 021 files (Phase 4 curriculum subset) | (build `phase4_v1` later if needed) |
| `gs://video_gen_dataset/TinyStories/character_images_bank/CHASE,SKYE,…/*.jpg` | 138 freeze-frames named `PawPatrol_SXX_EYY_*_fXXXX.jpg` | Still useful as anchor-still inputs; not in scope for the current visual-baseline rebuild |

**Action on legacy:** nothing destructive. Once the new datasets are validated end-to-end (a real training run, not just smoke), we'll move these into `gs://video_gen_dataset/_archive/2026-MM-DD/` with a 30-day GCS lifecycle policy. **Awaiting user approval before any move.**

---

## 4. Sanity checks

- **9 966 md5 duplicate groups** in the legacy `data_sets/` + `dataset/labelbox/` audit (2026-04-28). Most are within Labelbox's intermediate copies. The new `training_data/` layout has zero internal duplication (one `videos/0NNN.mp4` per index).
- **Cross-prefix duplicates** between `raw/`, `labelbox/videos/`, and `data_sets/`: ~150 known. Under the rebuild, the trainer only ever reads from `training_data/{ds}/`, so cross-prefix overlap is no longer a training-time risk.

---

## 5. How to use this map

**To check what's in a dataset:**
```bash
gcloud storage cp gs://.../{ds}/metadata.json /tmp/{ds}_md.json && jq '{stats, captioning, galleries}' /tmp/{ds}_md.json
```

**To pull a dataset for QA review (galleries only — fastest):**
```bash
mkdir -p ~/Desktop/{ds}
for f in source_gallery.mp4 training_gallery.mp4; do
  gcloud storage cp gs://.../{ds}/$f ~/Desktop/{ds}/$f
done
open ~/Desktop/{ds}/*.mp4
```

**To rebuild from scratch:**
```bash
# On the GPU VM:
bash scripts/dataset_pipeline/build_full.sh chase
bash scripts/dataset_pipeline/build_full.sh skye

# Then push to GCS from your local (VM SA can't write to training_data/ prefix):
bash scripts/dataset_pipeline/upload_dataset_to_gcs.sh chase_golden_v1
bash scripts/dataset_pipeline/upload_dataset_to_gcs.sh skye_golden_v2
```

**To prepare for training on a fresh machine:**
```bash
gcloud storage rsync -r gs://video_gen_dataset/TinyStories/training_data/chase_golden_v1 \
    ~/data/chase_golden_v1
bash scripts/dataset_pipeline/stage_for_training.sh ~/data/chase_golden_v1
# Now point a training config's data.preprocessed_data_root at ~/data/chase_golden_v1/precomputed
```

**To delete a bad clip:**
```bash
python scripts/dataset_pipeline/manage.py delete --dataset {dataset_dir} --index 42 --reason "..."
```

---

## 6. What this map does NOT cover

- **Anchor stills (high-res, character × shot type)** — separate pipeline, not yet built. Inputs would be `gs://video_gen_dataset/TinyStories/character_images_bank/` (138 existing freeze-frames) + new extractions from `raw/paw_patrol/`.
- **Audio data prep for Step 2 (frozen)** — pipeline supports `--with-audio` in `process_dataset.py` but Step 2 work is on hold per `active_plan.md`.
- **Multi-character datasets (`full_cast_v1`, etc.)** — pipeline supports it; just hasn't been built. Same `build_full.sh` template applies.
