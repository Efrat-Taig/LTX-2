# Training & Artifact Protocol

> **How to invoke:** Tell any Claude session: *"Read TRAINING_PROTOCOL.md before doing anything."*
> File lives at: `packages/ltx-trainer/../../../TRAINING_PROTOCOL.md` (repo root of LTX-2).

---

## Core Rule

**Every training run is self-contained in a GCS folder.** Anyone with bucket access can reproduce inference from that folder alone — no hunting across servers.

---

## GCS Folder Structure

```
gs://video_gen_dataset/
  training_runs/
    {model}/              # e.g. ltx2
      {character}/        # e.g. skye, chase, marshall
        {exp_name}/       # e.g. exp1_highcap, exp3_rank64
          run_info.json       ← W&B link, server, git commit, status
          config.yaml         ← exact copy of the training config used
          checkpoints/        ← all .safetensors files
          validation_samples/ ← validation videos saved during training (optional)
  datasets/
    {model}/
      {character}/
        {dataset_name}/   # e.g. golden_chase_960x832
          manifest.json       ← clip count, resolution, caption format, server path
```

### Naming conventions
- `{model}`: `ltx2`
- `{character}`: lowercase — `skye`, `chase`, `marshall`, etc.
- `{exp_name}`: `exp{N}_{short_description}` — e.g. `exp1_highcap`, `exp3_rank64`, `exp5_clean_highres`

---

## run_info.json Schema

Create this file **before training starts** and update it when training completes.

```json
{
  "exp_name": "chase_exp1_highcap",
  "character": "chase",
  "model": "ltx2",
  "model_version": "ltx-2.3-22b",

  "server": "34.56.137.11",
  "server_user": "efrat_t_smiti_ai",
  "git_commit": "<sha>",

  "started_at": "2026-04-20T19:21:00Z",
  "finished_at": null,
  "total_steps": 10000,
  "status": "running",

  "wandb": {
    "project": "ltx-2-chase-lora",
    "run_id": "w3j8it80",
    "run_url": "https://wandb.ai/shayzweig-smiti-ai/ltx-2-chase-lora/runs/w3j8it80"
  },

  "lora": {
    "rank": 32,
    "alpha": 32,
    "target_modules": ["attn", "ffn"],
    "resolution": "960x832",
    "steps": 10000,
    "i2v_conditioning_p": 0.5
  },

  "dataset": {
    "name": "golden_chase_960x832",
    "clip_count": 231,
    "server_path": "/home/efrat_t_smiti_ai/data/golden_chase_preprocessed",
    "gcs_manifest": "gs://video_gen_dataset/datasets/ltx2/chase/golden_chase_960x832/manifest.json"
  }
}
```

---

## Claude Protocol — Step by Step

### 1. Starting a new training run

Before launching training on any server:

```bash
# a. Get current git commit
GIT_COMMIT=$(ssh $SERVER "cd $REPO && git rev parse HEAD")

# b. Build run_info.json (status: "running", finished_at: null)
# c. Upload config + initial run_info to GCS
gsutil cp config.yaml gs://video_gen_dataset/training_runs/{model}/{character}/{exp_name}/config.yaml
gsutil cp run_info.json gs://video_gen_dataset/training_runs/{model}/{character}/{exp_name}/run_info.json
```

Do this **before** `nohup ... train.py`.

### 2. After training completes

```bash
# Upload all checkpoints
gsutil -m cp -r {server_outputs}/checkpoints/ \
  gs://video_gen_dataset/training_runs/{model}/{character}/{exp_name}/checkpoints/

# Update run_info.json: set status="complete", finished_at=<timestamp>
gsutil cp run_info.json \
  gs://video_gen_dataset/training_runs/{model}/{character}/{exp_name}/run_info.json
```

### 3. New dataset / preprocessing run

After `process_dataset.py` completes, upload a manifest (not the latents — too large):

```json
{
  "dataset_name": "golden_chase_960x832",
  "character": "chase",
  "source_csv": "gs://video_gen_dataset/dataset/labelbox/.../chase_golden_dataset.csv",
  "clip_count": 231,
  "filtered_count": 5,
  "resolution": "960x832",
  "frame_buckets": ["960x832x49", "960x832x97"],
  "caption_format": "CHASE says [transcript]. [scene_caption]",
  "with_audio": true,
  "server_path": "/home/efrat_t_smiti_ai/data/golden_chase_preprocessed",
  "preprocessed_at": "2026-04-20T18:00:00Z"
}
```

```bash
gsutil cp manifest.json \
  gs://video_gen_dataset/datasets/ltx2/chase/golden_chase_960x832/manifest.json
```

### 4. Running inference from any machine

```bash
# Pull what you need from GCS
gsutil cp gs://video_gen_dataset/training_runs/ltx2/chase/exp1_highcap/config.yaml .
gsutil cp gs://video_gen_dataset/training_runs/ltx2/chase/exp1_highcap/checkpoints/lora_weights_step_10000.safetensors .
gsutil cat gs://video_gen_dataset/training_runs/ltx2/chase/exp1_highcap/run_info.json  # W&B link etc.
```

---

## W&B Integration

- Always include the W&B run URL in `run_info.json` (copy from training log)
- W&B stores: loss curves, validation video GIFs, hyperparams, system metrics
- `run_info.json` is the permanent record; W&B is the interactive dashboard
- If W&B is disabled for a run, note `"wandb": null` in run_info.json

---

## Existing runs to backfill (as of 2026-04-21)

These runs exist but predate this protocol. Backfill their GCS folders when convenient:

| Run | GCS path (current) | Status |
|-----|---------------------|--------|
| skye/exp1–exp6 | `gs://video_gen_dataset/TinyStories/lora_ckpt/` | needs reorganizing |
| chase/exp1_highcap | `gs://video_gen_dataset/TinyStories/lora_ckpt/golden_chase/chase_exp1_highcap/` | checkpoints uploaded, needs run_info.json |

---

## Quick Reference — GCS paths

| What | Path |
|------|------|
| Chase EXP-1 checkpoints | `gs://video_gen_dataset/TinyStories/lora_ckpt/golden_chase/chase_exp1_highcap/` |
| Raw dataset CSVs | `gs://video_gen_dataset/dataset/labelbox/Project_Lipsync/Other/Golden_dataset/` |
| New organized runs root | `gs://video_gen_dataset/training_runs/` |
