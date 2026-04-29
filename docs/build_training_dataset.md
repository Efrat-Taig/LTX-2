# Building a Training Dataset — Operational Runbook

> **Purpose:** the step-by-step for creating, updating, or rebuilding a Paw Patrol training dataset, end-to-end. Self-contained: a new developer should be able to build a dataset by following this doc with zero outside context.
>
> **Related docs (don't duplicate, just link):**
> - `docs/training_data_pipeline.md` — the **spec** (layout, schemas, why each piece exists).
> - `docs/training_data.md` — the **live map** of which datasets currently exist in GCS.
>
> If those two and this one ever drift, **fix this one first** — it's what people will actually run.

---

## 1. The three execution contexts

Every command in this doc runs in exactly one of three places. Know which before you start.

| Context | Where | Auth used | Runs… | Why there |
|---|---|---|---|---|
| **Local laptop** | macOS workstation | `gcloud auth login` (your user account) | code edits, git, GCS uploads, translation | Has fresh creds for the `training_data/` GCS prefix. Local ffmpeg lacks `drawtext`, so no rendering here. |
| **GPU VM** | `ltx-2-development` (GCE, `us-central1-c`) | instance SA — limited GCS access | clip downloads, ffmpeg preprocessing, gallery render, latent encoding | Has the LTX-2 + Gemma weights and a full ffmpeg with `libfribidi`/`drawtext`. |
| **Training machine** | wherever `train.py` runs (often the same VM) | n/a (reads from local disk) | the actual LoRA training | Reads `precomputed/{latents,conditions}/` via symlinks created by `stage_for_training.sh`. |

**Critical asymmetry: the VM SA *cannot* write to `gs://video_gen_dataset/TinyStories/training_data/`.** All uploads go through your laptop. `upload_dataset_to_gcs.sh` automates the VM→local→GCS dance.

---

## 2. Input contract — what you need before building

A dataset build needs three things, in this order:

### 2.1 A curation manifest in GCS

Two manifest formats are supported (both currently live in `gs://video_gen_dataset/dataset/labelbox/`):

- **JSON list** (chase pattern). Each entry: `{"data": {"video": "gs://...mp4", "meta": {"speaker": "CHASE", "scene_filename": "...", "season_number": 1, ...}}}`. Captions come from a sibling parquet, joined by URL.
- **Parquet** (skye-v2 pattern) with `output_video_path` + `scene_caption` columns. If the URLs are stale, join to a sister parquet that has live URLs by `(season_number, episode_number, scene_number, internal_episode, speaker)`.

**Verify URLs resolve before building.** Pick 5 random URLs from the manifest and run `gcloud storage ls` on each. If they 404, you need to find a join parquet (see `training_data_pipeline.md` §7 for the skye-v2 example). Don't start a build with broken URLs — you'll find out at clip 312 of 605.

### 2.2 An entry in `brand_tokens.yaml` for any new character

`scripts/dataset_pipeline/brand_tokens.yaml`:
```yaml
substitutions:
  Chase:    CHASE_PP
  Skye:     SKYE_PP
  Marshall: MARSHALL_PP    # ← add this line for a new character
```

This map is **read just-in-time** — at gallery render and at trainer CSV emit. Editing it does **not** invalidate any existing latents or videos. To roll out a token change to an already-built dataset, see §6.4.

### 2.3 (For a brand-new character) a manifest-specific build function

`scripts/dataset_pipeline/build_poc.py` has one function per source manifest pattern (`build_chase`, `build_skye`). For a new character whose manifest follows one of those patterns, just add a case in `build_full.sh` (see §5). For a *new* manifest pattern, you write a 30-line `build_marshall(...)` that mirrors the existing two — see existing functions as templates.

---

## 3. Quickstart — rebuild an existing dataset

You're rebuilding `chase_golden_v1` because someone re-curated the manifest. End-to-end:

### Step A — find the GPU VM's current external IP (it changes on restart)
```bash
# Local laptop
gcloud compute instances describe ltx-2-development --zone=us-central1-c \
  --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
# → e.g. 35.184.4.0
```

If the VM is `TERMINATED`, start it with `gcloud compute instances start ltx-2-development --zone=us-central1-c` and wait for `RUNNING` plus an SSH-able port.

### Step B — run the build on the VM
```bash
# Local laptop
ssh efrattaig@<VM_IP> 'cd ~/LTX-2 && git pull'

# Then run the build via SSH or while logged in:
ssh efrattaig@<VM_IP> 'nohup bash ~/LTX-2/scripts/dataset_pipeline/build_full.sh chase \
  > /tmp/build_full_chase.log 2>&1 < /dev/null & echo "PID $!"'
```

The build runs four stages serially (`build_poc → render_gallery_for_dataset → emit_trainer_csv → process_dataset → rename`). For full chase (605 clips) on an A100: ~85 min wall.

Tail progress:
```bash
ssh efrattaig@<VM_IP> 'tail -f /tmp/build_full_chase.log'
```

### Step C — upload to GCS from your laptop
```bash
# Local laptop — uses your gcloud user creds (refresh first if expired):
gcloud auth login   # only if `gcloud auth print-access-token` fails

bash scripts/dataset_pipeline/upload_dataset_to_gcs.sh chase_golden_v1
```

This `rsync`s VM → `/tmp/training_data_full/chase_golden_v1/` → GCS. ~10 min for chase, ~3 min for skye. Has retry-with-resume built in — drops are common, no big deal.

### Step D — verify
```bash
# Local laptop
gcloud storage ls --recursive gs://video_gen_dataset/TinyStories/training_data/chase_golden_v1 \
  | grep -v "/$" | wc -l
# → ~2 447 objects for chase

gcloud storage cat gs://video_gen_dataset/TinyStories/training_data/chase_golden_v1/metadata.json \
  | jq '{stats, layout_version}'
```

Pull both galleries and watch:
```bash
mkdir -p ~/Desktop/chase
for f in source_gallery.mp4 training_gallery.mp4; do
  gcloud storage cp gs://video_gen_dataset/TinyStories/training_data/chase_golden_v1/$f ~/Desktop/chase/$f
done
open ~/Desktop/chase/*.mp4
```

---

## 4. Recipe — start training on a fresh machine

```bash
# Wherever train.py will run (often the same GPU VM):

# 1. Pull the dataset.
mkdir -p ~/data
gcloud storage rsync -r \
  gs://video_gen_dataset/TinyStories/training_data/chase_golden_v1 \
  ~/data/chase_golden_v1

# 2. Bridge the trainer's hardcoded path expectations.
bash ~/LTX-2/scripts/dataset_pipeline/stage_for_training.sh \
  ~/data/chase_golden_v1
# → creates precomputed/latents → vae_latents_video
# →         precomputed/conditions → text_prompt_conditions

# 3. Point a training config at it.
#    In your *.yaml:
#      data:
#        preprocessed_data_root: "/home/<user>/data/chase_golden_v1/precomputed"

# 4. Launch.
cd ~/LTX-2/packages/ltx-trainer
uv run python scripts/train.py path/to/your_config.yaml
```

The smoke-train config at `packages/ltx-trainer/configs/_smoke_chase_v1.yaml` is the minimal validator (50 steps, no audio, no W&B). Use it as a sanity check after any pipeline change.

---

## 5. Recipe — add a new character (e.g. Marshall)

You have a `marshall_golden.json` manifest and `marshall_*.parquet` for captions, both already in `gs://.../dataset/labelbox/`.

```bash
# Local laptop
cd ~/projects/sm/LTX-2
```

### 5.1 Add the brand token
Edit `scripts/dataset_pipeline/brand_tokens.yaml`:
```yaml
substitutions:
  Chase:    CHASE_PP
  Skye:     SKYE_PP
  Marshall: MARSHALL_PP    # ← new line
```

### 5.2 Add a build function (only if the manifest format differs)
If `marshall_golden.json` has the same shape as `chase_golden.json` (it should, if it came out of Labelbox), you can reuse `build_chase` logic. If not, add `build_marshall` to `scripts/dataset_pipeline/build_poc.py` mirroring one of the existing two functions.

### 5.3 Add a case in `build_full.sh`
```bash
case "$CHAR" in
  chase)    DS_NAME="chase_golden_v1";    MANIFEST="gs://.../chase_golden.json" ;;
  skye)     DS_NAME="skye_golden_v2";     MANIFEST="gs://.../skye_golden.json"  ;;
  marshall) DS_NAME="marshall_golden_v1"; MANIFEST="gs://.../marshall_golden.json" ;;   # ← new
  *) echo "usage: $0 {chase|skye|marshall}" >&2; exit 2 ;;
esac
```

If `build_marshall` is needed (5.2), also add it to the dispatcher block in `build_poc.py`'s `main()`.

### 5.4 Commit and run
```bash
git add scripts/dataset_pipeline/{brand_tokens.yaml,build_poc.py,build_full.sh}
git commit -m "marshall: add brand token + build dispatch"
git push

# On the VM:
ssh efrattaig@<VM_IP> 'cd ~/LTX-2 && git pull'
ssh efrattaig@<VM_IP> 'nohup bash ~/LTX-2/scripts/dataset_pipeline/build_full.sh marshall \
  > /tmp/build_full_marshall.log 2>&1 < /dev/null &'
```

Wait, then upload as in §3.C. Add an entry to `docs/training_data.md` under "Currently shipped".

---

## 6. Other recipes

### 6.1 Bump a dataset version (e.g. `chase_golden_v1 → v2` because re-captioning)

1. Update `build_full.sh`: change `DS_NAME="chase_golden_v1"` → `chase_golden_v2`.
2. Run §3.B and §3.C with `chase`.
3. Add the new dataset to `docs/training_data.md`.
4. **Don't delete `chase_golden_v1` from GCS** until a real training run on `v2` validates. Keep both during transition.

### 6.2 Delete bad clips by index (after watching the gallery)

```bash
# On the VM (or wherever the dataset directory lives locally):
python ~/LTX-2/scripts/dataset_pipeline/manage.py delete \
  --dataset ~/training_data_full/chase_golden_v1 \
  --indices 17,42,89-92 \
  --reason "characters off-frame in 49-frame window"

# Then re-render galleries with the gaps:
python ~/LTX-2/scripts/dataset_pipeline/render_gallery_for_dataset.py \
  --dataset ~/training_data_full/chase_golden_v1

# Re-emit the trainer CSV (excludes deleted indices):
python ~/LTX-2/scripts/dataset_pipeline/emit_trainer_csv.py \
  --dataset ~/training_data_full/chase_golden_v1

# Re-upload from your laptop:
bash scripts/dataset_pipeline/upload_dataset_to_gcs.sh chase_golden_v1
```

Indices never renumber — gaps are intentional. "Clip #42" means index 42 forever, even after deletion.

### 6.3 Regenerate just the galleries (no encoding)

E.g. after editing fonts, layout, or wrap width in `gallery.py`:

```bash
# On the VM:
rm -rf ~/training_data_full/chase_golden_v1/_qa_render_cache
rm -f  ~/training_data_full/chase_golden_v1/{source,training}_gallery.mp4
python ~/LTX-2/scripts/dataset_pipeline/render_gallery_for_dataset.py \
  --dataset ~/training_data_full/chase_golden_v1
```

Then upload as in §3.C — `gcloud storage rsync` will only push the changed mp4s.

### 6.4 Regenerate captions/embeddings after a `brand_tokens.yaml` change

Video latents don't depend on the caption text, so they don't need re-encoding. Only the conditions:

```bash
# On the VM:
rm -rf ~/training_data_full/chase_golden_v1/precomputed/text_prompt_conditions
python ~/LTX-2/scripts/dataset_pipeline/emit_trainer_csv.py \
  --dataset ~/training_data_full/chase_golden_v1
cd ~/LTX-2/packages/ltx-trainer
/home/efrattaig/.local/bin/uv run python scripts/process_dataset.py \
  ~/training_data_full/chase_golden_v1/_trainer_input.csv \
  --resolution-buckets 960x544x49 \
  --model-path /home/efrattaig/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path /home/efrattaig/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --output-dir ~/training_data_full/chase_golden_v1/precomputed \
  --batch-size 1
mv ~/training_data_full/chase_golden_v1/precomputed/conditions \
   ~/training_data_full/chase_golden_v1/precomputed/text_prompt_conditions
```

The video latents at `precomputed/vae_latents_video/` are untouched. Re-render galleries (§6.3) so the burned-in captions match. Re-upload.

### 6.5 Add a Hebrew (or other-language) gallery

```bash
# Local laptop — translate captions:
gcloud storage cp gs://video_gen_dataset/TinyStories/training_data/chase_golden_v1/metadata.json \
  /tmp/chase_metadata.json
python scripts/dataset_pipeline/translate_captions.py \
  --metadata /tmp/chase_metadata.json --lang he --workers 12 \
  --out /tmp/chase_captions_he.json

# Push to VM:
scp /tmp/chase_captions_he.json efrattaig@<VM_IP>:~/training_data_full/chase_golden_v1/captions_he.json

# On VM — render Hebrew gallery:
ssh efrattaig@<VM_IP> 'python ~/LTX-2/scripts/dataset_pipeline/render_hebrew_gallery.py \
  --dataset ~/training_data_full/chase_golden_v1 \
  --captions ~/training_data_full/chase_golden_v1/captions_he.json'

# Upload from laptop:
scp efrattaig@<VM_IP>:~/training_data_full/chase_golden_v1/other/gallery.mp4 \
  /tmp/chase_he_gallery.mp4
gcloud storage cp /tmp/chase_he_gallery.mp4 \
  gs://video_gen_dataset/TinyStories/training_data/chase_golden_v1/other/gallery.mp4
```

`translate_captions.py` uses `deep-translator` (Google Translate web endpoint, no API key). Other languages: pass `--lang fr`, `--lang es`, etc.

---

## 7. Authentication & environment

### 7.1 Local gcloud refresh (when uploads start failing with "Reauthentication failed")
```bash
gcloud auth login                          # for `gcloud storage` commands
gcloud auth application-default login      # only if you're using Vertex/Gemini SDKs (not used in current pipeline)
```

### 7.2 Finding the GPU VM IP
```bash
gcloud compute instances describe ltx-2-development --zone=us-central1-c \
  --format="value(status,networkInterfaces[0].accessConfigs[0].natIP)"
```

If the IP changed since last session, clear stale SSH host keys:
```bash
ssh-keygen -R <OLD_IP>
ssh -o StrictHostKeyChecking=accept-new efrattaig@<NEW_IP> 'echo ok'
```

### 7.3 VM SA permissions
The VM's instance SA (`942580369887-compute@developer.gserviceaccount.com`) can READ `video_gen_dataset/` (so `build_poc.py`'s clip downloads work) but **cannot WRITE to `training_data/`**. Don't try to bypass this — use the laptop-side upload script.

---

## 8. Common failures & fixes

| Symptom | Cause | Fix |
|---|---|---|
| `process_videos.py: Skipping … has 44 frames, less than 49` | Source clip is shorter than 49 frames after trim | Acceptable. The clip stays in `metadata.json` but no latent is encoded for it. Document expected count. |
| `gcloud storage rsync: Reauthentication failed.` | Local user creds expired | `gcloud auth login` and rerun |
| VM SSH timeouts | VM was restarted, IP changed | `gcloud compute instances describe ltx-2-development …` to get the new IP |
| `gcloud.storage.cp … objects.list denied` (from VM) | VM SA doesn't have write on `training_data/` | Don't push from VM. Use `upload_dataset_to_gcs.sh` from your laptop. |
| Trainer crashes with `Found N invalid video paths` | `_trainer_input.csv` `video_path` is relative to wrong dir | CSV must be at the **dataset root**, not inside `precomputed/`. `emit_trainer_csv.py` already handles this; if you wrote it manually, check the path. |
| Trainer can't find `precomputed/latents/` | Forgot `stage_for_training.sh` | `bash scripts/dataset_pipeline/stage_for_training.sh DATASET_DIR` (idempotent — safe to re-run). |
| `rsync: Connection reset by peer` mid-upload | Network blip + SSH session drop | `--partial` is on. Just rerun the upload script — it resumes from where it broke. |
| Hebrew text renders as boxes | Font lacks Hebrew glyphs | `render_hebrew_gallery.py` uses DejaVu Sans Bold which has Hebrew. Check the font path on the VM. |
| Latent count is N-7 (or similar) instead of N | Some clips < 49 source frames; auto-skipped | Expected; see first row above. |

---

## 9. File reference

Everything in `scripts/dataset_pipeline/`:

| File | Role |
|---|---|
| `build_full.sh` | **The orchestrator.** Runs stages 1–4 + the post-rename. One per character per run. |
| `build_poc.py` | Stage 1: resolve manifest URLs, download `raw_videos/`, write `metadata.json`. `--limit 0` = full manifest. |
| `captions.py` | URL→caption resolver (multi-parquet) + `apply_brand_tokens()` + `load_brand_tokens()`. |
| `gallery.py` | Pure ffmpeg helpers: `transform_to_training_format`, `render_per_clip(mode='source'|'training')`, concat. |
| `render_gallery_for_dataset.py` | Stage 2 driver: preprocess → `processed_videos/`, render both galleries, drop `processing_summary.txt`. |
| `emit_trainer_csv.py` | Stage 3: write `_trainer_input.csv` at dataset root with brand-token-substituted captions. |
| `manage.py` | Mutation: `delete --indices 5,17,42-44`, `restore`, `list`, `rebuild-gallery`. Atomic per call. |
| `stage_for_training.sh` | Bridge for trainer: creates `precomputed/{latents,conditions}` symlinks. Run once after pulling a dataset to a training machine. |
| `upload_dataset_to_gcs.sh` | VM → local rsync (with `--partial` + retries) → GCS push. Run from laptop. |
| `translate_captions.py` | Translate `metadata.json[*].prompt` to a target language. No API key (uses `deep-translator`). |
| `render_hebrew_gallery.py` | RTL-aware gallery render using a translation file. Output → `<dataset>/other/gallery.mp4`. |
| `brand_tokens.yaml` | Substitution map. Edit and re-run §6.4 to roll out. |
| `processing_summary.txt` | Static template copied into each dataset's `processed_videos/` by `build_full.sh`. |

Trainer-side:
| File | Role |
|---|---|
| `packages/ltx-trainer/scripts/process_dataset.py` | Stage 4 (LTX-2 trainer): encodes VAE latents + Gemma text embeddings. **Hardcodes `latents/` and `conditions/`** — that's why we have the rename + symlink dance. |
| `packages/ltx-trainer/configs/_smoke_chase_v1.yaml` | Pipeline validator (50 steps, rank 16, no audio). Use after any pipeline change. **Note:** configs are gitignored per repo convention; deployed via `scp`. |

---

## 10. The pipeline in one diagram

```
                  ┌──────────────────────────────────────────┐
                  │ INPUT: gs://.../labelbox/{char}_golden.* │
                  │ (manifest + captions parquet)            │
                  └──────────────────────────────────────────┘
                                      │
                                      ▼
        ┌────────────────────────  GPU VM  ────────────────────────────┐
        │  build_full.sh CHAR                                          │
        │    ├─ build_poc.py            → raw_videos/0NNN.mp4          │
        │    ├─ render_gallery_for…     → processed_videos/0NNN.mp4    │
        │    │                          → source_gallery.mp4           │
        │    │                          → training_gallery.mp4         │
        │    │                          → processing_summary.txt       │
        │    ├─ emit_trainer_csv.py     → _trainer_input.csv (root)    │
        │    └─ process_dataset.py +    → precomputed/                 │
        │       rename                       vae_latents_video/        │
        │                                    text_prompt_conditions/   │
        └──────────────────────────────────────────────────────────────┘
                                      │
                                      ▼  (upload_dataset_to_gcs.sh on laptop)
                                      │
        ┌──────  GCS canonical home  ──────────────────────────────────┐
        │  gs://video_gen_dataset/TinyStories/training_data/{ds}/      │
        │      metadata.json                                           │
        │      source_gallery.mp4    training_gallery.mp4              │
        │      raw_videos/  processed_videos/  precomputed/{…}         │
        └──────────────────────────────────────────────────────────────┘
                                      │
                                      ▼  (gcloud storage rsync to training machine)
                                      │
        ┌──────  Training machine  ────────────────────────────────────┐
        │  stage_for_training.sh DATASET_DIR                           │
        │    creates precomputed/{latents,conditions} symlinks         │
        │  uv run python scripts/train.py your_config.yaml             │
        └──────────────────────────────────────────────────────────────┘
```

That's the entire workflow. Anything not on this diagram is out of scope for the dataset pipeline.

---

**Last reviewed:** 2026-04-29 (after the `raw_videos/`, `vae_latents_video/`, `text_prompt_conditions/` rename). If the structure changes again, update §1, §3, the diagram, and the file table — in that order.
