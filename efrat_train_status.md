# Skye LoRA Training Status
*Last updated: 2026-04-20 (live check)*

---

## Active Machines — Right Now

| Machine | GPUs | User | Status |
|---------|------|------|--------|
| `35.238.2.51` | 1x A100 80GB | `efrattaig` | **EXP-5 training — step 6000/15000** |
| `34.173.240.7` | 2x H100 80GB | `efrat_t_smiti_ai` | **EXP-3-highres BENCHMARK running — steps 6000 + 8000** |
| `34.56.137.11` | 1x H100 80GB | `efrat_t_smiti_ai` | **IDLE** — nothing running |

### ⚠️ EXP-3-highres restart warning
The old EXP-3-highres run was **killed** (SIGTERM) at step 8500. A new run auto-started at ~11:07 today from **step 0** (not resumed — LR scheduler starts fresh). The old checkpoints (steps 6500–8500) still exist in `outputs/skye_exp3_highres/checkpoints/` and are benchmarkable. The new run will generate steps 500, 1000, … alongside them in the same folder.

**Answer to your question: NO, you do not have 2 separate skye_exp3_highres experiments.** There is one experiment dir with checkpoints from two sequential runs:
- Run 1 (killed): steps 6500–8500 (partially complete, benchmarkable)
- Run 2 (running now): step 0 → 15000 (fresh restart, ~11h to finish at 2.7s/step)

---

## SSH Commands

```bash
# A100 (EXP-5)
ssh efrattaig@35.238.2.51
tail -f /tmp/ltx_current.log

# H100 2x (EXP-3-highres, restarted)
ssh -i ~/.ssh/google_compute_engine efrat_t_smiti_ai@34.173.240.7
tail -f /tmp/ltx_current.log

# H100 1x (IDLE — available for new job)
ssh -i ~/.ssh/google_compute_engine efrat_t_smiti_ai@34.56.137.11
```

---

## Experiment Status

| Experiment | Config | Resolution | Steps | I2V | Layers | Status |
|-----------|--------|------------|-------|-----|--------|--------|
| EXP-1 | `skye_exp1_baseline.yaml` | 960×544 | 1000 | 50% | Attn | ✅ Trained + benchmarked |
| EXP-2 | `skye_exp2_standard.yaml` | 960×544 | 5000 | 50% | Attn | ✅ Trained + benchmarked |
| EXP-2-10k | `skye_exp2_10k.yaml` | 960×544 | 10000 | 50% | Attn | ✅ Trained + benchmarked |
| EXP-3-highcap | `skye_exp3_highcap.yaml` | 960×544 | 15000 | 50% | Attn+FFN | ✅ Trained + benchmarked, results local |
| EXP-3-highres | `skye_exp3_highres.yaml` | 768×1024 | 10000 | 50% | Attn+FFN | 🔬 BENCHMARK running on 34.173.240.7 (steps 6000 + 8000, ~35min). Training STOPPED. |
| EXP-4 | `skye_exp4_i2v.yaml` | 960×544 | 2500 | 80% | Attn | ✅ Trained + benchmarked |
| EXP-4-10k | `skye_exp4_10k.yaml` | 960×544 | 10000 | 80% | Attn | ✅ Trained + benchmarked |
| EXP-4-highres | `skye_exp4_highres.yaml` | 768×1024 | 15000 | 80% | Attn+FFN | ✅ Trained + benchmarked (600 videos on H100, NOT YET downloaded) |
| EXP-5 | `skye_exp5_clean_highres.yaml` | 768×1024 | 15000 | 80% | Attn+FFN | 🔄 Training on 35.238.2.51 (step 6000/15000, ETA ~tomorrow morning) |

---

## Pending Tasks (in order)

### 1. Decide what to do with EXP-3-highres Run 1 checkpoints (steps 6500–8500)
They exist on 34.173.240.7, are **valid and benchmarkable**, but are from a partial run (not step 15000). Options:
- Benchmark them now as "partial" run (useful data point)
- Wait for Run 2 to finish and only benchmark that

### 2. Use idle 34.56.137.11 H100 for something
Machine is sitting idle. Candidates:
- Benchmark EXP-4-highres (checkpoints on 34.173.240.7 — would need to copy or run there)
- Start a new experiment

### 3. Download EXP-4-highres benchmark from H100 (GCS relay, ~2GB)
```bash
# On H100:
ssh -i ~/.ssh/google_compute_engine efrat_t_smiti_ai@34.173.240.7 \
  "gsutil -m cp -r /home/efrat_t_smiti_ai/projects/LTX-2/output/benchmarks/skye_exp4_highres/ gs://video_gen_dataset/benchmarks/"
# Locally:
gsutil -m cp -r gs://video_gen_dataset/benchmarks/skye_exp4_highres/ \
  /Users/efrattaig/projects/sm/LTX-2/lora_results_FIX/benchmarks/
```

### 4. Benchmark EXP-5 on 35.238.2.51 (after training completes ~tomorrow)
```bash
ssh efrattaig@35.238.2.51 "cd /home/efrattaig/LTX-2 && \
  nohup /home/efrattaig/.local/bin/uv run python run_skye_benchmark.py \
    --exp-dirs outputs/skye_exp5_clean_highres \
    --output-dir /home/efrattaig/LTX-2/lora_results_FIX \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 \
    2>&1 | tee /tmp/ltx_current.log &"
```

### 5. Benchmark EXP-3-highres Run 2 on 34.173.240.7 (after ~11h)
```bash
ssh -i ~/.ssh/google_compute_engine efrat_t_smiti_ai@34.173.240.7 \
  "cd /home/efrat_t_smiti_ai/projects/LTX-2 && \
   nohup /usr/local/bin/uv run python run_skye_benchmark.py \
     --exp-dirs outputs/skye_exp3_highres \
     --output-dir /home/efrat_t_smiti_ai/projects/LTX-2/lora_results_FIX \
     --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 \
     2>&1 | tee /tmp/ltx_current.log &"
```

### 6. Rebuild comparison videos (after all benchmarks done)
Add to `ALL_EXPERIMENTS` in `create_comparison.py`:
```python
("skye_exp3_highcap",       15000, "EXP-3-highcap\n+FFN 15k"),
("skye_exp4_10k",           10000, "EXP-4-10k\ni2v 10k"),
("skye_exp4_highres",       15000, "EXP-4-highres\nI2V 768x1024"),
("skye_exp5_clean_highres", 15000, "EXP-5\nclean 768x1024 15k"),
("skye_exp3_highres",       15000, "EXP-3-highres\nT2V 768x1024"),
```

---

## Portrait Resolution Matrix

|  | I2V (80%) | T2V (50%) |
|--|-----------|-----------|
| **960×544** | EXP-4-10k ✅ | EXP-2-10k ✅ |
| **768×1024** | EXP-4-highres ✅ (benchmark not downloaded) | EXP-3-highres 🔄 (restarted) |

---

## Key Technical Notes

**LoRA sd_ops fix** — both inference scripts have `sd_ops=LTXV_LORA_COMFY_RENAMING_MAP`. Do not revert.

**Do not resume from mid-run checkpoints** — linear LR scheduler decays to 0.

**BigVGAN crash fix** — `torch.backends.cudnn.enabled = False` in `run_skye_like_api.py`. Do not remove.

**DDP 2-GPU on H100** — use `accelerate launch --config_file packages/ltx-trainer/configs/accelerate/ddp_2gpu.yaml`

**uv path:**
- 35.238.2.51: `/home/efrattaig/.local/bin/uv`
- 34.173.240.7: `/usr/local/bin/uv`
- 34.56.137.11: unknown (machine is fresh/idle)

**W&B:** [ltx-2-skye-lora](https://wandb.ai/shayzweig-smiti-ai/ltx-2-skye-lora)