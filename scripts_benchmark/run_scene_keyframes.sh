#!/bin/bash
# Run TI2VidOneStagePipeline on images from input/scenes
# Uses multiple keyframe images to generate a video (interpolates between them)
#
# Usage:
#   ./run_scene_keyframes.sh sample_scene
#   ./run_scene_keyframes.sh skye_concerned_lookout   # single image - uses as first frame only
#
# For scenes with multiple images, specify frame indices. By default:
#   - start_frame.png -> frame 0
#   - shot_01_*.png   -> frame 120 (end of 5s @ 24fps)
#
# Requires: checkpoint, Gemma (no distilled LoRA or upsampler needed for one-stage)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENES_DIR="${SCENES_DIR:-$SCRIPT_DIR/input/scenes}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/models/ltx-2.3-22b-dev.safetensors}"
GEMMA_ROOT="${GEMMA_ROOT:-$SCRIPT_DIR/models/gemma-3-12b-it-qat-q4_0-unquantized}"

SCENE_NAME="${1:-sample_scene}"
SCENE_DIR="$SCENES_DIR/$SCENE_NAME"
CONFIG="$SCENE_DIR/config.json"

if [ ! -d "$SCENE_DIR" ]; then
  echo "Error: Scene directory not found: $SCENE_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Read config
if [ -f "$CONFIG" ]; then
  PROMPT=$(jq -r '.specific_prompts[0] // .global_prompt // "High-quality 3D animation"' "$CONFIG" 2>/dev/null || echo "High-quality 3D animation")
  NEGATIVE=$(jq -r '.negative_prompt // "blurry, low quality, distorted"' "$CONFIG" 2>/dev/null || echo "blurry, low quality, distorted")
  DURATION=$(jq -r '.duration // 5' "$CONFIG" 2>/dev/null || echo "5")
  FPS=$(jq -r '.fps // 24' "$CONFIG" 2>/dev/null || echo "24")
  WIDTH=$(jq -r '.dimensions.width // 480' "$CONFIG" 2>/dev/null || echo "480")
  HEIGHT=$(jq -r '.dimensions.height // 854' "$CONFIG" 2>/dev/null || echo "854")
  SEED=$(jq -r '.seed // 42' "$CONFIG" 2>/dev/null || echo "42")
else
  PROMPT="High-quality 3D animation"
  NEGATIVE="blurry, low quality, distorted"
  DURATION=5
  FPS=24
  WIDTH=480
  HEIGHT=854
  SEED=42
fi

# Height must be divisible by 32
HEIGHT=$(( (HEIGHT / 32) * 32 ))
[ "$HEIGHT" -lt 32 ] && HEIGHT=32

# num_frames = k*8+1 for 5 sec @ 24fps -> 121
RAW_FRAMES=$(awk "BEGIN {printf \"%.0f\", $DURATION * $FPS}")
NUM_FRAMES=$(awk "BEGIN {k=int(($RAW_FRAMES - 1) / 8 + 0.5); n=8*k+1; if(n<17)n=17; print n}")

# Build --image args from available images
IMAGE_ARGS=()
if [ -f "$SCENE_DIR/start_frame.png" ]; then
  IMAGE_ARGS+=(--image "$SCENE_DIR/start_frame.png" 0 0.9)
fi
if [ -f "$SCENE_DIR/shot_01_chase_enters_lookout.png" ]; then
  LAST_IDX=$((NUM_FRAMES - 1))
  IMAGE_ARGS+=(--image "$SCENE_DIR/shot_01_chase_enters_lookout.png" "$LAST_IDX" 0.9)
fi

# Fallback: if only start_frame, we still need at least one image
if [ ${#IMAGE_ARGS[@]} -eq 0 ]; then
  # Try any png in the scene
  FIRST_IMG=$(find "$SCENE_DIR" -maxdepth 1 -name "*.png" | head -1)
  if [ -n "$FIRST_IMG" ]; then
    IMAGE_ARGS+=(--image "$FIRST_IMG" 0 0.9)
  else
    echo "Error: No images found in $SCENE_DIR"
    exit 1
  fi
fi

OUTPUT="$OUTPUT_DIR/${SCENE_NAME}_keyframes.mp4"

echo "=============================================="
echo "Scene: $SCENE_NAME"
echo "Images: ${#IMAGE_ARGS[@]} keyframes"
echo "Output: $OUTPUT"
echo "=============================================="

cd "$SCRIPT_DIR"
uv run python -m ltx_pipelines.ti2vid_one_stage \
  --checkpoint-path "$CHECKPOINT" \
  --gemma-root "$GEMMA_ROOT" \
  --prompt "$PROMPT" \
  --negative-prompt "$NEGATIVE" \
  "${IMAGE_ARGS[@]}" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --num-frames "$NUM_FRAMES" \
  --frame-rate "$FPS" \
  --seed "$SEED" \
  --output-path "$OUTPUT"

echo ""
echo "Video saved to: $OUTPUT"
