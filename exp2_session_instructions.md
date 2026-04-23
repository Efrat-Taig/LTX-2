# EXP-2 Session Instructions: Full-Cast Combined LoRA

Read TRAINING_PROTOCOL.md before doing anything.

---

## Context

EXP-1 is currently training on machine 34.56.137.11 (10k steps, 150 PAW Patrol clips).
EXP-2 will train on the **combined dataset** on a **new machine** (to be opened by the user).

---

## What is already done

- Combined preprocessed dataset is ready on **34.56.137.11**:
  - `/home/efrat_t_smiti_ai/data/full_cast_combined_preprocessed/`
  - 472 clips total: Chase golden (231) + Skye standard (106) + PAW Patrol 150-clips (135)
  - Symlinked from three source directories — do not delete the sources
  - latents: 472, audio_latents: 472, conditions: 483 (extras ignored by trainer)

- Training config is ready on 34.56.137.11:
  - `/home/efrat_t_smiti_ai/projects/LTX-2/packages/ltx-trainer/configs/full_cast_exp2_combined.yaml`
  - Rank 32 LoRA, attn + FFN, LR 1e-4, 10k steps, checkpoint every 500

- Config is also at (local):
  - `/Users/efrattaig/projects/sm/LTX-2/packages/ltx-trainer/configs/full_cast_exp2_combined.yaml`

---

## What the new session must do

### 1. Transfer preprocessed data to the new machine

The new machine will NOT have the preprocessed data. Two options:

**Option A — Copy from 34.56.137.11 (recommended):**
```bash
# On new machine: pull from 34.56.137.11
rsync -avz efrat_t_smiti_ai@34.56.137.11:/home/efrat_t_smiti_ai/data/full_cast_combined_preprocessed/ \
  /home/efrat_t_smiti_ai/data/full_cast_combined_preprocessed/
```
Note: this copies the resolved files (not symlinks), so ~3-4GB.

**Option B — Copy all three source dirs separately and re-symlink:**
```bash
rsync -avz efrat_t_smiti_ai@34.56.137.11:/home/efrat_t_smiti_ai/data/golden_chase_preprocessed/ ...
rsync -avz efrat_t_smiti_ai@34.56.137.11:/home/efrat_t_smiti_ai/data/golden_skye_preprocessed/ ...
rsync -avz efrat_t_smiti_ai@34.56.137.11:/home/efrat_t_smiti_ai/data/full_cast_exp1_preprocessed/ ...
# then re-create symlinks as done before
```

### 2. Verify models exist on the new machine

Training requires:
- LTX-2.3 model: `/home/efrat_t_smiti_ai/models/LTX-2.3/ltx-2.3-22b-dev.safetensors`
- Gemma text encoder: `/home/efrat_t_smiti_ai/models/gemma-3-12b-it-qat-q4_0-unquantized/`

If not present, they must be downloaded from GCS or copied from 34.56.137.11 (~100GB total).

### 3. Pull the latest code on the new machine

```bash
cd /home/efrat_t_smiti_ai/projects/LTX-2
git pull
```

### 4. Copy config to new machine (if not already via git pull)

```bash
scp efrat_t_smiti_ai@34.56.137.11:/home/efrat_t_smiti_ai/projects/LTX-2/packages/ltx-trainer/configs/full_cast_exp2_combined.yaml \
  /home/efrat_t_smiti_ai/projects/LTX-2/packages/ltx-trainer/configs/
```

### 5. Create run_info.json and upload to GCS (before starting training)

```bash
# Get git commit on new machine
GIT_COMMIT=$(cd /home/efrat_t_smiti_ai/projects/LTX-2 && git rev-parse HEAD)
```

Create `/tmp/run_info_exp2.json`:
```json
{
  "exp_name": "full_cast_exp2_combined",
  "character": "full_cast",
  "model": "ltx2",
  "model_version": "ltx-2.3-22b",
  "server": "<NEW_MACHINE_IP>",
  "server_user": "efrat_t_smiti_ai",
  "git_commit": "<GIT_COMMIT>",
  "started_at": "<ISO_TIMESTAMP>",
  "finished_at": null,
  "total_steps": 10000,
  "status": "running",
  "wandb": {
    "project": "ltx-2-full-cast-lora",
    "run_id": null,
    "run_url": null
  },
  "lora": {
    "rank": 32,
    "alpha": 32,
    "target_modules": ["attn", "ffn"],
    "resolution": "mixed_960x832_960x544_1024x576",
    "frame_bucket": "mixed_49f_113f",
    "steps": 10000,
    "i2v_conditioning_p": 0.5
  },
  "dataset": {
    "name": "full_cast_combined_472clips",
    "clip_count": 472,
    "server_path": "/home/efrat_t_smiti_ai/data/full_cast_combined_preprocessed",
    "sources": [
      {"name": "golden_chase", "clips": 231, "resolution": "960x832"},
      {"name": "golden_skye_std", "clips": 106, "resolution": "960x544"},
      {"name": "paw_patrol_150clips", "clips": 135, "resolution": "1024x576"}
    ]
  }
}
```

Upload to GCS:
```bash
gsutil cp /tmp/run_info_exp2.json gs://video_gen_dataset/training_runs/ltx2/full_cast/exp2_combined_472clips/run_info.json
gsutil cp packages/ltx-trainer/configs/full_cast_exp2_combined.yaml gs://video_gen_dataset/training_runs/ltx2/full_cast/exp2_combined_472clips/config.yaml
```

### 6. Start training

```bash
cd /home/efrat_t_smiti_ai/projects/LTX-2
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /usr/local/bin/uv run python packages/ltx-trainer/scripts/train.py \
  packages/ltx-trainer/configs/full_cast_exp2_combined.yaml \
  2>&1 | tee /tmp/ltx_current.log &
```

Wait ~2 minutes, then check W&B run URL from the log and update run_info.json.

### 7. After training completes (step 10000)

Kill any zombie processes, then run BM_v1 benchmark:
```bash
cd /home/efrat_t_smiti_ai/projects/LTX-2
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /usr/local/bin/uv run python run_bm_v1_lora.py \
  --exp-dir outputs/full_cast_exp2_combined \
  --steps 5000 10000 \
  --out-base lora_results/combined_472clips \
  2>&1 | tee /tmp/ltx_current.log
```

Copy results locally:
```bash
# From local machine:
mkdir -p /Users/efrattaig/projects/sm/LTX-2/lora_results/combined_472clips
scp -r efrat_t_smiti_ai@<NEW_IP>:/home/efrat_t_smiti_ai/projects/LTX-2/lora_results/combined_472clips/full_cast_exp2_combined \
  /Users/efrattaig/projects/sm/LTX-2/lora_results/combined_472clips/
```

Update run_info.json status to "complete" in GCS.

---

## Key notes

- Always check for zombie processes before starting training (`ps aux | grep python | grep -v grep`)
- Always tee to `/tmp/ltx_current.log` so the user can watch with `tail -f`
- If training crashes: resume from last checkpoint by setting `load_checkpoint` in the YAML
- The config is gitignored — always scp it directly, never commit it
- 472 clips × mixed resolution × batch_size=1 means variable latent shapes per step — this is expected and fine
