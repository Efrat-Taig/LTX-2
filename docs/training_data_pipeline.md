# Training Data Pipeline — Implementation Reference

> **Status:** built and validated end-to-end (smoke train passed 2026-04-28).
> **Single source of truth:** the per-character JSON manifest in
> `gs://video_gen_dataset/dataset/labelbox/`. Captions resolved via the
> matching `_filtered.parquet`. Everything else is derived.

---

## 1. Inputs

Two sources of truth, both in `gs://video_gen_dataset/`:

| What | Where |
|---|---|
| Curation (which clips, in what order) | `dataset/labelbox/{chase_golden.json, skye_golden.json}` |
| Captions (`scene_caption` per clip) | `dataset/labelbox/Project_Lipsync/Chase/chase_all_seasoned_results_filtered.parquet` (chase)<br>`dataset/labelbox/Project_Lipsync/Other/Golden_dataset/skye_all_seasons_results_filtered_augmented_no_text_train_*.parquet` (skye) |
| Source clip mp4s | URLs inside the JSON (chase) or joined-via-aug-parquet for skye_v2 |
| Brand-token map | `scripts/dataset_pipeline/brand_tokens.yaml` (configurable, applied just-in-time) |

Everything else is reproducible.

---

## 2. Canonical layout per dataset

```
gs://video_gen_dataset/TinyStories/training_data/{dataset_id}/
    metadata.json                                          ← canonical record (raw caps, status, md5, layout, provenance)
    source_gallery.mp4                                     ← raw clips for visual context (full duration)
    training_gallery.mp4                                   ← what the model trains on (49-frame letterboxed)
    raw_videos/0001.mp4 … 000N.mp4                         ← untouched manifest downloads (variable resolution + duration)
    processed_videos/
        0001.mp4 … 000N.mp4                                ← letterboxed to 960×544, trimmed to 49 frames @ 24fps
        processing_summary.txt                             ← exact ffmpeg recipe for the transform (informational)
    precomputed/
        vae_latents_video/processed_videos/0001.pt … .pt   ← VAE-encoded video latents, shape (128, 7, 17, 30) bf16
        text_prompt_conditions/processed_videos/0001.pt    ← Gemma-encoded prompt embeddings, (1024, 4096) bf16
```

> **Trainer constraint:** the LTX-2 trainer's `process_dataset.py` hardcodes `latents/` and `conditions/` as its read paths. The GCS canonical names above are the human-readable form. On any local training machine, run `scripts/dataset_pipeline/stage_for_training.sh DATASET_DIR` once to create symlinks `precomputed/latents → vae_latents_video` and `precomputed/conditions → text_prompt_conditions`. Trainer then reads through the symlinks — no fork, no rename.

`{dataset_id}` convention: `{character}_golden_v{N}`. Currently:
- `chase_golden_v1` — 605 clips from `chase_golden.json`
- `skye_golden_v2` — 111 clips from `skye_golden.json` joined to the augmented parquet for live URLs (the v2 parquet's own URLs are stale — see §7)

---

## 3. `metadata.json` — what each field is

```jsonc
{
  "dataset_id": "chase_golden_v1",
  "character": "chase",
  "version": 1,
  "created_at": "2026-04-28T...",
  "source": {
    "manifest_label": "chase_golden.json (URLs) + chase_all_seasoned_results_filtered.parquet (captions)",
    "raw_footage_root": "gs://video_gen_dataset/raw/paw_patrol/"
  },
  "captioning": {
    "version": "v1",
    "brand_token_format": "raw character names; substitution applied just-in-time",
    "substitution_at_build_time": false,
    "substitution_config": "scripts/dataset_pipeline/brand_tokens.yaml"
  },
  "stats": { "active_clips": 605, "missing_clips": 0 },
  "clips": [
    {
      "index": 1,
      "status": "active",                           // active | deleted
      "source_url": "gs://...../scene_84_..._output.mp4",
      "source_episode": "PawPatrol_S01_E02_A",
      "source_season": 1,
      "source_scene_number": 84.0,
      "speaker": "CHASE",
      "video": "raw_videos/0001.mp4",               // untouched download, relative to dataset root
      "video_md5": "...",
      "prompt": "Chase, a happy animated dog, ...",  // RAW — no brand token here
      "duration_s": 2.46, "width": 1252, "height": 1080, "fps": 24.0,
      "download_ok": true, "note": ""
    }
  ],
  "galleries": {
    "source":   { "path": "source_gallery.mp4",   "transform": "letterbox to 960×544, full original duration" },
    "training": { "path": "training_gallery.mp4", "transform": "letterbox to 960×544, trimmed to 49 frames @24fps" }
  }
}
```

Mutation lives here. `manage.py delete --index N` flips `status`, moves files to `_trash/<ts>/`, appends `audit/deletions.log`, regenerates galleries.

---

## 4. Brand tokens — just-in-time substitution

Map: `scripts/dataset_pipeline/brand_tokens.yaml`
```yaml
substitutions:
  Chase: CHASE_PP
  Skye:  SKYE_PP
```

**Where it's applied** (only two places — never persists):

1. **`render_gallery_for_dataset.py`** — burns substituted text into the gallery overlay so QA matches what the model sees.
2. **`emit_trainer_csv.py`** — writes `_trainer_input.csv` with substituted `caption` column for the trainer.

`metadata.json[*].prompt` always carries the **raw** character name. Changing the token format → edit the YAML, re-run gallery + CSV emit. Source data never re-extracted.

---

## 5. Gallery videos

Both galleries: 960×784 canvas (544 video + 240 caption band). Index `#NNNN` top-left, caption bottom band, source audio preserved.

| | source_gallery.mp4 | training_gallery.mp4 |
|---|---|---|
| Source files | `raw_videos/` (untouched downloads) | `processed_videos/` (pre-letterboxed) |
| Per-clip transform | `scale-to-fit + black-bar pad` (letterbox) | none — already letterboxed |
| Per-clip duration | full original (variable, 2–6 s) | 49 frames @ 24fps = 2.04s exactly |
| Use | "what the source actually looks like" | "what the model trains on" |

Compare them side-by-side: if `training_gallery` clips a key action that's still ongoing in `source_gallery`, that clip is a candidate for `manage.py delete`.

**Why letterbox not crop:** earlier center-crop policy was clipping characters' ears. Letterbox preserves all original content; black bars are a visible signal to QA, not learned-bad-data.

---

## 6. The build pipeline

For one character at full scale:

```bash
# On the GPU VM:
bash scripts/dataset_pipeline/build_full.sh chase    # → ~/training_data_full/chase_golden_v1
bash scripts/dataset_pipeline/build_full.sh skye     # → ~/training_data_full/skye_golden_v2
```

Internally that runs four stages (each idempotent — re-running skips completed work):

| # | Stage | Script | What it does |
|---|---|---|---|
| 1 | raw | `build_poc.py --limit 0` | Resolves manifest URLs, downloads `raw_videos/*.mp4`, ffprobes each, writes `metadata.json` |
| 2 | letterbox + galleries | `render_gallery_for_dataset.py` | Pre-processes raw → `processed_videos/` (960×544 letterbox + 49-frame trim), renders both galleries with brand tokens, drops `processing_summary.txt` |
| 3 | trainer CSV | `emit_trainer_csv.py` | Writes `_trainer_input.csv` (ephemeral) at dataset root with `caption` (substituted) + `video_path` (→ `processed_videos/0NNN.mp4`) |
| 4 | latents + conditions | `process_dataset.py` (LTX-2 trainer) → `build_full.sh` rename step | Encodes VAE latents + Gemma text embeddings; trainer writes to `precomputed/{latents,conditions}/` then `build_full.sh` renames them to `precomputed/{vae_latents_video,text_prompt_conditions}/` |
| 5 | (training-time only) | `stage_for_training.sh` | On the local training machine, creates `precomputed/{latents,conditions}` symlinks pointing at the human-readable dirs so the trainer's `PrecomputedDataset` Just Works |

Stage 4 is the only GPU-bound stage. On A100 80GB it processes ~1 clip every 5–10s; full chase (605) ≈ 60 min, full skye (111) ≈ 12 min.

---

## 7. Skye-specific quirk: v2 manifest URLs are stale

`skye_golden_dataset_v2.parquet` was published with URLs pointing to a target GCS layout that was never populated (`TinyStories/data_sets/golden_skye/videos/paw_patrol/...`). 0/111 of those URLs resolve.

`build_poc.py:build_skye()` joins each v2 row to `skye_all_seasons_results_filtered_augmented_no_text_train_*.parquet` on `(season_number, episode_number, scene_number, internal_episode, speaker='SKYE')` and uses that row's URL — which IS live. 111/111 join coverage verified. Captions still come from v2 (the curated set the user picked).

If a future v3 publishes valid URLs, the join is replaced by direct `output_video_path` reads.

---

## 8. The "delete clip #42" workflow

```bash
python scripts/dataset_pipeline/manage.py delete \
  --dataset /local/training_data/chase_golden_v1 \
  --index 42 \
  --reason "low quality, character off-model"
```

Atomic per invocation:
1. Move `raw_videos/0042.mp4`, `processed_videos/0042.mp4`, `precomputed/{vae_latents_video,text_prompt_conditions}/processed_videos/0042.pt`, `_qa_render_cache/{source,training}/per_clip/0042.mp4` → `_trash/<ts>/`.
2. Update `metadata.json`: clip 42's `status` → `deleted`, `deletion` block populated.
3. Append `audit/deletions.log`.
4. Re-concat both galleries (per-clip overlays are kept; missing index just yields a gap).
5. Print summary.

Inverse: `manage.py restore --index 42` (recovers from `_trash/` if still there).

Indices never renumber — gaps are intentional. "Clip #42" means index 42 forever.

---

## 9. How to validate a dataset before training

1. Download `qa_gallery` files locally:
   ```bash
   for f in source training; do
     gcloud storage cp gs://.../{dataset}/${f}_gallery.mp4 ~/Desktop/${dataset}_${f}.mp4
   done
   ```
2. Watch `training_gallery.mp4` end-to-end. Note any indices where the action is half-cut or the caption is off.
3. For suspicious indices, open `source_gallery.mp4` to check if the original framing is salvageable.
4. `manage.py delete --indices 5,17,42-44 --reason "..."`.
5. Re-run `render_gallery_for_dataset.py` (rebuilds galleries with gaps) + `emit_trainer_csv.py` (excludes deleted clips).
6. Re-run `process_dataset.py` to re-encode any added/changed clips (existing clip latents are skipped).

---

## 10. Smoke train (pipeline validator, not real training)

`packages/ltx-trainer/configs/_smoke_chase_v1.yaml` — minimal 50-step rank-16 LoRA on the 5-clip chase POC. Run:

```bash
/home/efrattaig/.local/bin/uv run python packages/ltx-trainer/scripts/train.py \
  packages/ltx-trainer/configs/_smoke_chase_v1.yaml
```

Pass criteria, all met on 2026-04-28:
- ✅ Trainer reads `precomputed/{latents,conditions}` without errors.
- ✅ Loss decreases (50/50 steps, final loss 0.346, no NaN).
- ✅ LoRA weights saved at `outputs/_smoke_chase_v1/checkpoints/lora_weights_step_00050.safetensors`.
- ✅ 2.1 min total (3.4 s/step), 50 GB peak GPU memory.

The smoke is the contract: any future dataset rebuild must still pass this before scale-up training.

---

## 11. Pipeline scripts — reference

| Script | Purpose |
|---|---|
| `scripts/dataset_pipeline/build_poc.py` | Stage 1: resolve manifest, download raw clips → `raw_videos/`, write metadata.json. `--limit 0` = full manifest. |
| `scripts/dataset_pipeline/captions.py` | Caption resolver: multi-parquet URL→caption lookup; `apply_brand_tokens(s, mapping)`; `load_brand_tokens()` reads YAML. |
| `scripts/dataset_pipeline/gallery.py` | ffmpeg renderers: `transform_to_training_format()`, `render_per_clip(mode='source'|'training')`, concat helpers. |
| `scripts/dataset_pipeline/render_gallery_for_dataset.py` | Stage 2 driver: `raw_videos/` → `processed_videos/` + dual gallery + metadata update. |
| `scripts/dataset_pipeline/emit_trainer_csv.py` | Stage 3: write `_trainer_input.csv` (brand-token substitution applied). |
| `scripts/dataset_pipeline/build_full.sh` | Stages 1–4 wrapper for one character; renames trainer outputs to `vae_latents_video`/`text_prompt_conditions` + drops `processing_summary.txt`. |
| `scripts/dataset_pipeline/stage_for_training.sh` | Local-only: creates `precomputed/{latents,conditions}` symlinks before training. |
| `scripts/dataset_pipeline/upload_dataset_to_gcs.sh` | Pull built dataset from VM, push to GCS via local creds (with `--partial` retries). |
| `scripts/dataset_pipeline/manage.py` | Mutation: `delete`, `restore`, `list`, `rebuild-gallery`. |
| `scripts/dataset_pipeline/brand_tokens.yaml` | Substitution map (configurable, no rebuild needed to change). |
| `scripts/dataset_pipeline/processing_summary.txt` | Static template copied into each dataset's `processed_videos/` at build time. |
| `packages/ltx-trainer/configs/_smoke_chase_v1.yaml` | 50-step pipeline validator config. (gitignored — scp'd to VM.) |
