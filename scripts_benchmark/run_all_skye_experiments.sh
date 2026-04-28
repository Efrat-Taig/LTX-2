#!/bin/bash
# Run all 12 PROMPT_EXPERIMENTS: 3 prompt types × 4 video lengths
# Outputs to LTX2_output/
#
# Usage: from anywhere, run:
#   /home/smiti-user/LTX-2/run_all_skye_experiments.sh
#   or: cd /home/smiti-user/LTX-2 && ./run_all_skye_experiments.sh
#
# Override any option by exporting before running, e.g.:
#   NUM_INFERENCE_STEPS=30 ./run_all_skye_experiments.sh
#
# Quick reference - all options (defaults are optimal):
#   Model & paths:    CHECKPOINT, GEMMA_ROOT, LORA_PATH, IMAGE_PATH
#   Output:           OUTPUT_DIR
#   Resolution:       WIDTH (1280), HEIGHT (704)  # must be divisible by 32
#   Duration:         DURATIONS (4 6 8 10), FRAME_RATE (25.0)  # seconds per clip
#   Quality vs speed: NUM_INFERENCE_STEPS (40)    # 30=fast, 50=quality
#   Prompt strength:  GUIDANCE_SCALE (4.0)       # 3.5-5.0
#   Temporal:         STG_SCALE (1.0), STG_BLOCKS (29), STG_MODE (stg_av)
#   Reproducibility:  SEED (42)
#   Audio/device:     SKIP_AUDIO (false), DEVICE (cuda)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# CONFIGURABLE OPTIONS (override via export before running)
# =============================================================================
# Model & paths
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/models/ltx-2.3-22b-dev.safetensors}"
GEMMA_ROOT="${GEMMA_ROOT:-$SCRIPT_DIR/models/gemma-3-12b-it-qat-q4_0-unquantized}"
LORA_PATH="${LORA_PATH:-}"

# Output
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/LTX2_output}"
IMAGE_PATH="${IMAGE_PATH:-$SCRIPT_DIR/input/sky_in_heli.jpg}"

# Video size & length (width/height must be divisible by 32)
# Matches input/sky_in_heli.jpg aspect ratio (880x1123 ≈ 11:14): 704x896
# 720p landscape: 1280x704 | 540p: 960x544 (faster, less VRAM)
WIDTH="${WIDTH:-704}"
HEIGHT="${HEIGHT:-896}"
FRAME_RATE="${FRAME_RATE:-25.0}"

# Video duration(s) in seconds - which lengths to run (e.g. "15 20 35" or "4 8")
# num_frames auto-computed (must be k*8+1)
DURATIONS="${DURATIONS:-15 20 35}"

# Generation quality & guidance (optimal defaults)
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"   # 30=fast, 40=balanced, 50=quality
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"            # 3.5-5.0: higher = stronger prompt adherence
STG_SCALE="${STG_SCALE:-1.0}"                     # 0.5-1.5: temporal coherence (0 disables)
STG_BLOCKS="${STG_BLOCKS:-29}"                    # Transformer blocks for STG
STG_MODE="${STG_MODE:-stg_av}"                    # stg_av (audio+video) or stg_v (video only)

# Reproducibility
SEED="${SEED:-42}"

# Audio & device
SKIP_AUDIO="${SKIP_AUDIO:-false}"
DEVICE="${DEVICE:-cuda}"
# =============================================================================

mkdir -p "$OUTPUT_DIR"

# duration_sec -> num_frames (must be k*8+1)
# Formula: round((sec * fps - 1) / 8) * 8 + 1
get_frames() {
  local sec="$1"
  local fps="${FRAME_RATE:-25.0}"
  local raw
  raw=$(awk "BEGIN {printf \"%.0f\", $sec * $fps}")
  awk "BEGIN {k=int(($raw - 1) / 8 + 0.5); n=8*k+1; if(n<17)n=17; print n}"
}

export OUTPUT_DIR CHECKPOINT GEMMA_ROOT LORA_PATH IMAGE_PATH
export WIDTH HEIGHT FRAME_RATE NUM_INFERENCE_STEPS GUIDANCE_SCALE STG_SCALE STG_BLOCKS STG_MODE
export SEED SKIP_AUDIO DEVICE

# Total runs = 3 prompt types × number of durations
num_durations=$(echo $DURATIONS | wc -w)
total=$((3 * num_durations))
current=0

echo "=============================================="
echo "LTX-2 Skye Experiments: $total runs (prompt types 1-3 × durations: $DURATIONS s)"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

for pt in 1 2 3; do
  for len in $DURATIONS; do
    current=$((current + 1))
    export PROMPT_TYPE=$pt
    export NUM_FRAMES=$(get_frames $len)
    echo ""
    echo "[$current/$total] PROMPT_TYPE=$pt, ${len}s, ${WIDTH}x${HEIGHT}, ${NUM_INFERENCE_STEPS} steps, seed $SEED"
    if ! "$SCRIPT_DIR/run_skye_video.sh"; then
      echo "Experiment failed. Stopping."
      exit 1
    fi
  done
done

echo ""
echo "=============================================="
echo "All $total experiments completed!"
echo "Outputs in: $OUTPUT_DIR"
echo "=============================================="
