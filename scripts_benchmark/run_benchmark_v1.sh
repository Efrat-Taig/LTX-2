#!/bin/bash
# Run benchmark entries under input/benchmark_v1/ using an LTX image-to-video pipeline.
#
# Each benchmark folder must contain config.json plus images (see input/benchmark_v1/*/).
# Reads: global_prompt, specific_prompts[], negative_prompt, duration, fps, dimensions,
#         seed, num_inference_steps (optional), enhance_prompt (optional), images[] or auto-discover
#         (all *.png/*.jpg/*.jpeg in the benchmark folder, sorted).
# Prints description to the log only (not sent to the model).
# Output videos: {name}_{1stage|hq}_p{index}[_seed{N}]_{duration}s_{width}x{height}.mp4
#
# Pipeline selection (BENCHMARK_PIPELINE):
#   one_stage  — TI2VidOneStagePipeline: only checkpoint + Gemma (default). No distilled LoRA / upsampler.
#   hq         — TI2VidTwoStagesHQPipeline: needs distilled LoRA + spatial upsampler (stage 2 is built around
#                that adapter; it is not an optional “creative” LoRA — the two-stage code requires it).
#   auto       — Use hq if both distilled LoRA and spatial upsampler files exist; otherwise one_stage.
#
# Usage:
#   ./run_benchmark_v1.sh
#   ./run_benchmark_v1.sh skye_chase_trampoline_backflip
#   BENCHMARK_PIPELINE=hq ./run_benchmark_v1.sh   # after downloading HF assets
#
# Fal-aligned run (defaults to OUTPUT_DIR=output/benchmark_v1_fal):
#   cd /path/to/LTX-2 && BENCHMARK_PROFILE=fal ./run_benchmark_v1.sh
#   BENCHMARK_PROFILE=fal ./run_benchmark_v1.sh skye_chase_crosswalk_safety
#   OUTPUT_DIR="$PWD/output/my_run" BENCHMARK_PROFILE=fal ./run_benchmark_v1.sh   # custom folder
#
# Fal-style alignment (optional):
#   BENCHMARK_PROFILE=fal   — snap settings toward fal-ai/ltx-2.3/image-to-video (Pro) defaults;
#       default output folder is output/benchmark_v1_fal (unless OUTPUT_DIR is set).
#       duration → 6, 8, or 10 s; fps → 25; size → 1920×1088 (1080p-class, grid-safe);
#       num_inference_steps → 30 (LTX_2_3_PARAMS default) unless you set NUM_INFERENCE_STEPS.
#       Set BENCHMARK_USE_CONFIG_DIMENSIONS=1 to keep config.json width/height.
#       Set BENCHMARK_FAL_WIDTH / BENCHMARK_FAL_HEIGHT to override the fal profile resolution.
#       Set BENCHMARK_FAL_MINIMAL_NEGATIVE=1 to use an empty negative prompt (API has no user negative).
#   BENCHMARK_PIPELINE=hq may be closer to hosted “Pro” quality if distilled LoRA + upsampler exist.
#
# Env overrides:
#   BENCHMARK_DIR, OUTPUT_DIR, CHECKPOINT, GEMMA_ROOT, DISTILLED_LORA, SPATIAL_UPSAMPLER,
#   SEEDS, PROMPT_INDEX, NUM_INFERENCE_STEPS, BENCHMARK_PIPELINE, BENCHMARK_PROFILE, etc.

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------------------------------------------
# Parameters — edit defaults here (export VAR before running to override)
#
#   BENCHMARK_DIR          = $SCRIPT_DIR/input/benchmark_v1
#   OUTPUT_DIR             = output/benchmark_v1, or output/benchmark_v1_fal if BENCHMARK_PROFILE=fal|fal_pro
#   CHECKPOINT             = $SCRIPT_DIR/models/ltx-2.3-22b-dev.safetensors
#   GEMMA_ROOT             = $SCRIPT_DIR/models/gemma-3-12b-it-qat-q4_0-unquantized
#   DISTILLED_LORA         = $SCRIPT_DIR/models/ltx-2.3-22b-distilled-lora-384.safetensors
#   SPATIAL_UPSAMPLER      = $SCRIPT_DIR/models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors
#   BENCHMARK_PIPELINE     = one_stage
#   SEEDS                  = 42
#   PROMPT_INDEX           = 0
#   NUM_INFERENCE_STEPS    = 50 (30 when BENCHMARK_PROFILE=fal or fal_pro)
#
# When NUM_INFERENCE_STEPS is non-empty (including the default below), it overrides each
# config.json "num_inference_steps". Use :-\"\" instead of :-50 on the assignment line
# to prefer config.json / pipeline defaults.
# ------------------------------------------------------------------------------
BENCHMARK_DIR="${BENCHMARK_DIR:-$SCRIPT_DIR/input/benchmark_v1}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-}"
if [ "$BENCHMARK_PROFILE" = "fal" ] || [ "$BENCHMARK_PROFILE" = "fal_pro" ]; then
  _DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/output/benchmark_v1_fal"
else
  _DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/output/benchmark_v1"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$_DEFAULT_OUTPUT_DIR}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/models/ltx-2.3-22b-dev.safetensors}"
GEMMA_ROOT="${GEMMA_ROOT:-$SCRIPT_DIR/models/gemma-3-12b-it-qat-q4_0-unquantized}"
DISTILLED_LORA="${DISTILLED_LORA:-$SCRIPT_DIR/models/ltx-2.3-22b-distilled-lora-384.safetensors}"
SPATIAL_UPSAMPLER="${SPATIAL_UPSAMPLER:-$SCRIPT_DIR/models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors}"
BENCHMARK_PIPELINE="${BENCHMARK_PIPELINE:-one_stage}"
SEEDS="${SEEDS:-42}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
if [ "$BENCHMARK_PROFILE" = "fal" ] || [ "$BENCHMARK_PROFILE" = "fal_pro" ]; then
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
else
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
fi

require_file() {
  local f="$1"
  local label="$2"
  if [ ! -f "$f" ]; then
    echo "Error: missing $label: $f" >&2
    exit 1
  fi
}

require_file "$CHECKPOINT" "CHECKPOINT"
[ -d "$GEMMA_ROOT" ] || { echo "Error: GEMMA_ROOT must be a directory: $GEMMA_ROOT" >&2; exit 1; }

case "$BENCHMARK_PIPELINE" in
  hq|two_stages_hq)
    EFFECTIVE_PIPELINE=hq
    ;;
  one_stage|1stage|one)
    EFFECTIVE_PIPELINE=one_stage
    ;;
  auto)
    if [ -f "$DISTILLED_LORA" ] && [ -f "$SPATIAL_UPSAMPLER" ]; then
      EFFECTIVE_PIPELINE=hq
    else
      EFFECTIVE_PIPELINE=one_stage
    fi
    ;;
  *)
    echo "Error: unknown BENCHMARK_PIPELINE='$BENCHMARK_PIPELINE' (use one_stage, hq, or auto)" >&2
    exit 1
    ;;
esac

if [ "$EFFECTIVE_PIPELINE" = "hq" ]; then
  require_file "$DISTILLED_LORA" "DISTILLED_LORA (required for two-stage HQ; download from README)"
  require_file "$SPATIAL_UPSAMPLER" "SPATIAL_UPSAMPLER (required for two-stage HQ)"
fi

if [ ! -d "$BENCHMARK_DIR" ]; then
  echo "Error: BENCHMARK_DIR not found: $BENCHMARK_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/benchmark_v1_${EFFECTIVE_PIPELINE}_$(date +%Y%m%d_%H%M%S).log"
echo "Log: $LOG_FILE"

{
  echo "Run configuration (resolved):"
  echo "BENCHMARK_DIR = $BENCHMARK_DIR"
  echo "OUTPUT_DIR = $OUTPUT_DIR"
  echo "CHECKPOINT = $CHECKPOINT"
  echo "GEMMA_ROOT = $GEMMA_ROOT"
  echo "DISTILLED_LORA = $DISTILLED_LORA"
  echo "SPATIAL_UPSAMPLER = $SPATIAL_UPSAMPLER"
  echo "BENCHMARK_PIPELINE = $BENCHMARK_PIPELINE"
  echo "BENCHMARK_PROFILE = ${BENCHMARK_PROFILE:-(unset)}"
  echo "EFFECTIVE_PIPELINE = $EFFECTIVE_PIPELINE"
  if [ "$EFFECTIVE_PIPELINE" = "one_stage" ]; then
    echo "RESOLUTION_SNAP = 32"
    echo "PYTHON_MODULE = ltx_pipelines.ti2vid_one_stage"
  else
    echo "RESOLUTION_SNAP = 64"
    echo "PYTHON_MODULE = ltx_pipelines.ti2vid_two_stages_hq"
  fi
  echo "SEEDS = $SEEDS"
  echo "PROMPT_INDEX = $PROMPT_INDEX"
  if [ -n "$NUM_INFERENCE_STEPS" ]; then
    echo "NUM_INFERENCE_STEPS = $NUM_INFERENCE_STEPS"
  else
    echo "NUM_INFERENCE_STEPS = (unset — per config.json or pipeline default)"
  fi
  echo ""
} | tee -a "$LOG_FILE"

if [ -n "$1" ]; then
  BENCH_NAMES=("$1")
else
  BENCH_NAMES=()
  for d in "$BENCHMARK_DIR"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    [ -f "$d/config.json" ] || continue
    BENCH_NAMES+=("$name")
  done
fi

if [ ${#BENCH_NAMES[@]} -eq 0 ]; then
  echo "Error: No benchmark entries (folders with config.json) in $BENCHMARK_DIR" >&2
  exit 1
fi

run_one() {
  local bench_name="$1"
  local scene_dir="$BENCHMARK_DIR/$bench_name"
  local config="$scene_dir/config.json"

  if [ ! -f "$config" ]; then
    echo "[SKIP] $bench_name: no config.json"
    return 0
  fi

  local skip
  skip=$(jq -r '.skip // false' "$config" 2>/dev/null || echo "false")
  if [ "$skip" = "true" ]; then
    echo "[SKIP] $bench_name: skip=true"
    return 0
  fi

  local desc
  desc=$(jq -r '.description // ""' "$config" 2>/dev/null || echo "")
  [ -n "$desc" ] && echo "[INFO] $bench_name — $desc"

  local global_prompt specific_prompt
  global_prompt=$(jq -r '.global_prompt // ""' "$config" 2>/dev/null || echo "")
  specific_prompt=$(jq -r --argjson idx "$PROMPT_INDEX" '.specific_prompts[$idx] // .specific_prompts[0] // ""' "$config" 2>/dev/null || echo "")

  local prompt
  if [ -n "$global_prompt" ] && [ "$global_prompt" != "null" ]; then
    if [ -n "$specific_prompt" ] && [ "$specific_prompt" != "null" ]; then
      prompt="${global_prompt}

${specific_prompt}"
    else
      prompt="$global_prompt"
    fi
  else
    prompt="${specific_prompt:-High-quality 3D animation}"
  fi

  local negative_prompt duration fps width height seed cfg_steps enhance
  negative_prompt=$(jq -r '.negative_prompt // "blurry, low quality, distorted"' "$config" 2>/dev/null || echo "blurry, low quality, distorted")
  duration=$(jq -r '.duration // 5' "$config" 2>/dev/null || echo "5")
  fps=$(jq -r '.fps // 24' "$config" 2>/dev/null || echo "24")
  width=$(jq -r '.dimensions.width // 768' "$config" 2>/dev/null || echo "768")
  height=$(jq -r '.dimensions.height // 512' "$config" 2>/dev/null || echo "512")
  seed=$(jq -r '.seed // 42' "$config" 2>/dev/null || echo "42")
  cfg_steps=$(jq -r '.num_inference_steps // empty' "$config" 2>/dev/null || echo "")
  enhance=$(jq -r '.enhance_prompt // false' "$config" 2>/dev/null || echo "false")

  if [ "$BENCHMARK_PROFILE" = "fal" ] || [ "$BENCHMARK_PROFILE" = "fal_pro" ]; then
    duration=$(awk -v d="$duration" 'BEGIN {
      d = d + 0
      if (d <= 6) print 6
      else if (d <= 8) print 8
      else if (d <= 10) print 10
      else print 10
    }')
    fps=25
    if [ "${BENCHMARK_USE_CONFIG_DIMENSIONS:-0}" != "1" ]; then
      width="${BENCHMARK_FAL_WIDTH:-1920}"
      height="${BENCHMARK_FAL_HEIGHT:-1088}"
    fi
    if [ "${BENCHMARK_FAL_MINIMAL_NEGATIVE:-0}" = "1" ]; then
      negative_prompt=""
    fi
  fi

  local steps_arg=()
  local effective_steps="${NUM_INFERENCE_STEPS:-$cfg_steps}"
  if [ -n "$effective_steps" ]; then
    steps_arg=(--num-inference-steps "$effective_steps")
  fi

  local enhance_arg=()
  if [ "$enhance" = "true" ]; then
    enhance_arg=(--enhance-prompt)
  fi

  local divisor=32
  [ "$EFFECTIVE_PIPELINE" = "hq" ] && divisor=64
  height=$(( (height / divisor) * divisor ))
  [ "$height" -lt "$divisor" ] && height=$divisor
  width=$(( (width / divisor) * divisor ))
  [ "$width" -lt "$divisor" ] && width=$divisor

  local raw_frames num_frames
  raw_frames=$(awk "BEGIN {printf \"%.0f\", $duration * $fps}")
  num_frames=$(awk "BEGIN {k=int(($raw_frames - 1) / 8 + 0.5); n=8*k+1; if(n<17)n=17; print n}")

  local image_args=()
  if jq -e '.images | length > 0' "$config" >/dev/null 2>&1; then
    local len i
    len=$(jq '.images | length' "$config")
    for i in $(seq 0 $((len - 1))); do
      local img_path frame_idx strength
      img_path=$(jq -r ".images[$i].path" "$config")
      frame_idx=$(jq -r ".images[$i].frame_idx // 0" "$config")
      strength=$(jq -r ".images[$i].strength // 0.9" "$config")
      img_path="$scene_dir/$img_path"
      if [ -f "$img_path" ]; then
        image_args+=(--image "$img_path" "$frame_idx" "$strength")
      fi
    done
  fi

  if [ ${#image_args[@]} -eq 0 ]; then
    local img
    while IFS= read -r img; do
      [ -n "$img" ] && [ -f "$img" ] || continue
      image_args+=(--image "$img" 0 0.9)
    done < <(find "$scene_dir" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | LC_ALL=C sort)
    if [ ${#image_args[@]} -eq 0 ]; then
      echo "[SKIP] $bench_name: no images found"
      return 0
    fi
  fi

  local out_tag
  [ "$EFFECTIVE_PIPELINE" = "hq" ] && out_tag="hq" || out_tag="1stage"

  for s in $SEEDS; do
    local suffix out_path
    [ "$s" = "42" ] && [ "$(echo "$SEEDS" | wc -w)" -eq 1 ] && suffix="" || suffix="_seed${s}"
    out_path="$OUTPUT_DIR/${bench_name}_${out_tag}_p${PROMPT_INDEX}${suffix}_${duration}s_${width}x${height}.mp4"

    echo "[RUN] $bench_name (seed=$s) -> $out_path" | tee -a "$LOG_FILE"
    echo "[RUN] $bench_name (seed=$s) -> $out_path" >> "$LOG_FILE"

    local start end elapsed
    start=$(date +%s.%N)
    cd "$SCRIPT_DIR"
    if [ "$EFFECTIVE_PIPELINE" = "hq" ]; then
      if ! uv run python -m ltx_pipelines.ti2vid_two_stages_hq \
        --checkpoint-path "$CHECKPOINT" \
        --gemma-root "$GEMMA_ROOT" \
        --distilled-lora "$DISTILLED_LORA" 1.0 \
        --spatial-upsampler-path "$SPATIAL_UPSAMPLER" \
        --prompt "$prompt" \
        --negative-prompt "$negative_prompt" \
        "${image_args[@]}" \
        --width "$width" \
        --height "$height" \
        --num-frames "$num_frames" \
        --frame-rate "$fps" \
        --seed "$s" \
        "${steps_arg[@]}" \
        "${enhance_arg[@]}" \
        --output-path "$out_path" 2>&1 | tee -a "$LOG_FILE"; then
        echo "[FAIL] $bench_name (seed=$s)"
        echo "[FAIL] $bench_name (seed=$s)" >> "$LOG_FILE"
        return 1
      fi
    else
      if ! uv run python -m ltx_pipelines.ti2vid_one_stage \
        --checkpoint-path "$CHECKPOINT" \
        --gemma-root "$GEMMA_ROOT" \
        --prompt "$prompt" \
        --negative-prompt "$negative_prompt" \
        "${image_args[@]}" \
        --width "$width" \
        --height "$height" \
        --num-frames "$num_frames" \
        --frame-rate "$fps" \
        --seed "$s" \
        "${steps_arg[@]}" \
        "${enhance_arg[@]}" \
        --output-path "$out_path" 2>&1 | tee -a "$LOG_FILE"; then
        echo "[FAIL] $bench_name (seed=$s)"
        echo "[FAIL] $bench_name (seed=$s)" >> "$LOG_FILE"
        return 1
      fi
    fi

    end=$(date +%s.%N)
    elapsed=$(awk "BEGIN {printf \"%.1f\", $end - $start}")
    echo "[OK] $bench_name (seed=$s) ${elapsed}s"
    echo "[OK] $bench_name (seed=$s) ${elapsed}s" >> "$LOG_FILE"
  done
  return 0
}

failed=0
for name in "${BENCH_NAMES[@]}"; do
  run_one "$name" || failed=$((failed + 1))
done

echo ""
echo "Done. Output: $OUTPUT_DIR  Log: $LOG_FILE"
[ "$failed" -gt 0 ] && exit 1
exit 0
