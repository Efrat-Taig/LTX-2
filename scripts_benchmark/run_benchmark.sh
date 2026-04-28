#!/bin/bash
# Run benchmark over all scenes in input/scenes
#
# Each scene folder contains:
#   - config.json: global_prompt, specific_prompts[], negative_prompt, dimensions, duration, fps, seed, etc.
#   - start_frame.png (and optionally more images for keyframes)
#
# Usage:
#   ./run_benchmark.sh                    # Run all scenes
#   ./run_benchmark.sh chase_dizzy_skye   # Run single scene
#   SEEDS="42 123 456" ./run_benchmark.sh # Multiple seeds per scene
#   PROMPT_INDEX=1 ./run_benchmark.sh     # Use specific_prompts[1] instead of [0]
#
# Environment overrides:
#   SCENES_DIR, OUTPUT_DIR, CHECKPOINT, GEMMA_ROOT
#   SEEDS, PROMPT_INDEX, NUM_INFERENCE_STEPS, ENHANCE_PROMPT

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENES_DIR="${SCENES_DIR:-$SCRIPT_DIR/input/scenes}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output/benchmark}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/models/ltx-2.3-22b-dev.safetensors}"
GEMMA_ROOT="${GEMMA_ROOT:-$SCRIPT_DIR/models/gemma-3-12b-it-qat-q4_0-unquantized}"

# Benchmark options
SEEDS="${SEEDS:-42}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
ENHANCE_PROMPT="${ENHANCE_PROMPT:-}"

# Get list of scenes to run
if [ -n "$1" ]; then
  SCENE_NAMES=("$1")
else
  SCENE_NAMES=()
  for d in "$SCENES_DIR"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    [ -f "$d/config.json" ] || continue
    SCENE_NAMES+=("$name")
  done
fi

if [ ${#SCENE_NAMES[@]} -eq 0 ]; then
  echo "Error: No scenes found in $SCENES_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/benchmark_$(date +%Y%m%d_%H%M%S).log"
echo "Benchmark log: $LOG_FILE"
echo "Scenes: ${SCENE_NAMES[*]}"
echo "Seeds: $SEEDS"
echo ""

run_scene() {
  local scene_name="$1"
  local scene_dir="$SCENES_DIR/$scene_name"
  local config="$scene_dir/config.json"

  if [ ! -f "$config" ]; then
    echo "[SKIP] $scene_name: no config.json"
    return 0
  fi

  # Read config (with jq)
  local prompt global_prompt negative_prompt duration fps width height seed
  local num_inference_steps enhance_prompt_arg
  local image_args_str

  global_prompt=$(jq -r '.global_prompt // ""' "$config" 2>/dev/null || echo "")
  specific_prompt=$(jq -r --argjson idx "$PROMPT_INDEX" '.specific_prompts[$idx] // .specific_prompts[0] // ""' "$config" 2>/dev/null || echo "")
  if [ -z "$specific_prompt" ] || [ "$specific_prompt" = "null" ]; then
    specific_prompt=""
  fi
  # Always prepend global_prompt before specific_prompt
  if [ -n "$global_prompt" ] && [ "$global_prompt" != "null" ]; then
    if [ -n "$specific_prompt" ]; then
      prompt="${global_prompt}

${specific_prompt}"
    else
      prompt="$global_prompt"
    fi
  else
    prompt="${specific_prompt:-High-quality 3D animation}"
  fi

  negative_prompt=$(jq -r '.negative_prompt // "blurry, low quality, distorted"' "$config" 2>/dev/null || echo "blurry, low quality, distorted")
  duration=$(jq -r '.duration // 5' "$config" 2>/dev/null || echo "5")
  fps=$(jq -r '.fps // 24' "$config" 2>/dev/null || echo "24")
  width=$(jq -r '.dimensions.width // 480' "$config" 2>/dev/null || echo "480")
  height=$(jq -r '.dimensions.height // 854' "$config" 2>/dev/null || echo "854")
  seed=$(jq -r '.seed // 42' "$config" 2>/dev/null || echo "42")

  # Optional overrides from config
  local cfg_num_steps
  cfg_num_steps=$(jq -r '.num_inference_steps // empty' "$config" 2>/dev/null || echo "")
  num_inference_steps="${NUM_INFERENCE_STEPS:-$cfg_num_steps}"

  local cfg_enhance
  cfg_enhance=$(jq -r '.enhance_prompt // false' "$config" 2>/dev/null || echo "false")
  if [ -n "$ENHANCE_PROMPT" ] && [ "$ENHANCE_PROMPT" != "0" ]; then
    enhance_prompt_arg="--enhance-prompt"
  elif [ "$cfg_enhance" = "true" ]; then
    enhance_prompt_arg="--enhance-prompt"
  else
    enhance_prompt_arg=""
  fi

  # Height/width must be divisible by 32
  height=$(( (height / 32) * 32 ))
  [ "$height" -lt 32 ] && height=32
  width=$(( (width / 32) * 32 ))
  [ "$width" -lt 32 ] && width=32

  # num_frames = k*8+1
  raw_frames=$(awk "BEGIN {printf \"%.0f\", $duration * $fps}")
  num_frames=$(awk "BEGIN {k=int(($raw_frames - 1) / 8 + 0.5); n=8*k+1; if(n<17)n=17; print n}")

  # Build image args: either from config.images or auto-discover
  image_args=()
  if jq -e '.images | length > 0' "$config" >/dev/null 2>&1; then
    # Explicit images array: [ { "path": "start_frame.png", "frame_idx": 0, "strength": 0.9 }, ... ]
    local len
    len=$(jq '.images | length' "$config")
    for i in $(seq 0 $((len - 1))); do
      local img_path frame_idx strength
      img_path=$(jq -r ".images[$i].path" "$config")
      frame_idx=$(jq -r ".images[$i].frame_idx // 0" "$config")
      strength=$(jq -r ".images[$i].strength // 0.9" "$config")
      img_path="$scene_dir/$img_path"
      [ -f "$img_path" ] && image_args+=(--image "$img_path" "$frame_idx" "$strength")
    done
  fi

  if [ ${#image_args[@]} -eq 0 ]; then
    # Auto-discover: start_frame.png, shot_01_*.png
    if [ -f "$scene_dir/start_frame.png" ]; then
      image_args+=(--image "$scene_dir/start_frame.png" 0 0.9)
    fi
    last_idx=$((num_frames - 1))
    for shot in "$scene_dir"/shot_01_*.png; do
      [ -f "$shot" ] && image_args+=(--image "$shot" "$last_idx" 0.9) && break
    done
  fi

  if [ ${#image_args[@]} -eq 0 ]; then
    first_img=$(find "$scene_dir" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | head -1)
    if [ -n "$first_img" ]; then
      image_args+=(--image "$first_img" 0 0.9)
    else
      echo "[SKIP] $scene_name: no images found"
      return 0
    fi
  fi

  # Run for each seed
  for s in $SEEDS; do
    local suffix
    [ "$s" = "42" ] && [ $(echo "$SEEDS" | wc -w) -eq 1 ] && suffix="" || suffix="_seed${s}"
    local out_path
    out_path="$OUTPUT_DIR/${scene_name}_p${PROMPT_INDEX}${suffix}.mp4"

    echo "[RUN] $scene_name (seed=$s) -> $out_path"
    echo "[RUN] $scene_name (seed=$s) -> $out_path" >> "$LOG_FILE"

    local start end elapsed
    start=$(date +%s.%N)

    cd "$SCRIPT_DIR"
    if uv run python -m ltx_pipelines.ti2vid_one_stage \
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
      ${num_inference_steps:+--num-inference-steps "$num_inference_steps"} \
      $enhance_prompt_arg \
      --output-path "$out_path" 2>&1 | tee -a "$LOG_FILE"; then
      end=$(date +%s.%N)
      elapsed=$(awk "BEGIN {printf \"%.1f\", $end - $start}")
      echo "[OK] $scene_name (seed=$s) completed in ${elapsed}s"
      echo "[OK] $scene_name (seed=$s) completed in ${elapsed}s" >> "$LOG_FILE"
    else
      echo "[FAIL] $scene_name (seed=$s)"
      echo "[FAIL] $scene_name (seed=$s)" >> "$LOG_FILE"
      return 1
    fi
  done
  return 0
}

failed=0
for scene in "${SCENE_NAMES[@]}"; do
  run_scene "$scene" || failed=$((failed + 1))
done

echo ""
echo "Benchmark complete. Log: $LOG_FILE"
[ $failed -gt 0 ] && echo "Failed: $failed scene(s)" && exit 1
exit 0
