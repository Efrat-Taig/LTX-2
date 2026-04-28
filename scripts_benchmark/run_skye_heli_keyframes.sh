#!/bin/bash
# Generate video from ALL images in input/skye_heli using Skye helicopter prompts
# Like run_skye_video.sh but with multiple keyframe images (TI2VidOneStagePipeline)
#
# Uses the same Skye prompts and settings as run_skye_video.sh.
# Images are distributed evenly across the video as keyframes.
#
# Usage:
#   ./run_skye_heli_keyframes.sh              # 12s, 512x384 (fits 80GB GPU)
#   DURATION=10 ./run_skye_heli_keyframes.sh  # 10 second video
#   WIDTH=1280 HEIGHT=704 ./run_skye_heli_keyframes.sh  # Higher res (needs tiling/OOM risk)
#   PROMPT_TYPE=2 ./run_skye_heli_keyframes.sh         # Use prompt type 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="${IMAGE_DIR:-$SCRIPT_DIR/input/skye_heli}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/models/ltx-2.3-22b-dev.safetensors}"
GEMMA_ROOT="${GEMMA_ROOT:-$SCRIPT_DIR/models/gemma-3-12b-it-qat-q4_0-unquantized}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/LTX2_kf_res}"
OUTPUT_BASE_NAME="${OUTPUT_BASE_NAME:-skye_helicopter_keyframes}"
SEED="${SEED:-42}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-384}"
FRAME_RATE="${FRAME_RATE:-25.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
DURATION="${DURATION:-10}"
PROMPT_TYPE="${PROMPT_TYPE:-1}"

# duration_sec -> num_frames (must be k*8+1)
get_frames() {
  local sec="$1"
  local fps="${FRAME_RATE:-25.0}"
  local raw
  raw=$(awk "BEGIN {printf \"%.0f\", $sec * $fps}")
  awk "BEGIN {k=int(($raw - 1) / 8 + 0.5); n=8*k+1; if(n<17)n=17; print n}"
}

NUM_FRAMES=$(get_frames "$DURATION")
DURATION_S=$(awk "BEGIN {printf \"%.0f\", $NUM_FRAMES / $FRAME_RATE}")

# Skye prompts (same as run_skye_video.sh)
PROMPT_SUFFIX=" static camera composition maintained for the entire clip. Opening: brief quiet moment at the start — Skye is visible in frame for a beat before she begins speaking. She then says 'hey AYA and Gili!' clearly. Do not start with her speaking at the very first frame. Complete natural ending — no abrupt cutoff, video plays smoothly through to the full duration."

PROMPT_TYPE_1='High-quality 3D animation in the style of Paw Patrol, vibrant colors, clean CGI, cinematic lighting, bright and cheerful atmosphere, smooth character animation, Nick Jr aesthetic, 4k render, detailed textures, expressive facial expressions.

Skye from Paw Patrol, the cute female cockapoo rescue puppy pilot, wearing her pink pilot goggles and pink vest, flying her small pink helicopter high in a bright blue sky with soft fluffy clouds. The helicopter gently hovers in the air while she sits in the cockpit.

Skye looks directly into the camera and talks to the viewer like a cheerful kids TV host. She speaks clearly in FIRST PERSON about her own day. She is happy, excited, and very expressive.

Dialogue (first person, speaking directly to camera):
"hey AYA and Gili! I had such a happy day today!"
"I ate soooo many candies and sweets!"
"It was so yummy and so much fun!"
"But now I need to brush my teeth really well so they stay clean and healthy!"

She laughs, smiles, and talks energetically while sitting in the helicopter.

Camera: medium close-up inside the helicopter cockpit, Skye facing the camera.
Motion: helicopter gently hovering in the sky with soft movement.
Mood: joyful, playful, energetic children'\''s show.
Environment: bright daylight sky, soft clouds.'"$PROMPT_SUFFIX"

PROMPT_TYPE_2='High-quality 3D animation in the style of Paw Patrol, vibrant colors, clean CGI, cinematic lighting, bright cheerful atmosphere, smooth character animation, Nick Jr aesthetic, 4k render, detailed textures, expressive facial expressions.

Skye from Paw Patrol, the female cockapoo rescue puppy pilot wearing pink pilot goggles and a pink vest, sitting inside her small pink helicopter. The helicopter is hovering in a bright blue sky with soft fluffy clouds.

IMPORTANT: The entire video is ONE continuous shot. The camera angle does NOT change. No cuts, no scene transitions, no jump cuts, no camera switches.

The camera stays fixed in a stable medium close-up inside the helicopter cockpit. Skye remains centered in the frame and faces the camera the whole time. The helicopter gently hovers but the framing stays consistent.

Skye looks directly into the camera and talks to the viewer like a cheerful kids TV host. She speaks in FIRST PERSON about her own day. She is happy, excited and expressive.

Dialogue (first person, speaking to camera):
"hey AYA and Gili! I had such a happy day today!"
"I ate soooo many candies and sweets!"
"It was so yummy and so much fun!"
"But now I need to brush my teeth really well so they stay clean and healthy!"

Her mouth movements match the speech. She smiles, laughs and talks energetically.

Motion: only subtle animation — slight helicopter hover, small head movement, blinking, natural mouth movement.
Camera: locked camera, stable framing, no zoom, no rotation, no angle change.
Shot: single continuous shot for the entire clip.
Mood: joyful, playful children'\''s show.
Environment: bright daylight sky with soft clouds.'"$PROMPT_SUFFIX"

PROMPT_TYPE_3='Stable single-shot 3D animated scene of Skye from Paw Patrol inside her pink rescue helicopter cockpit. High-quality children'\''s TV animation, Paw Patrol CGI style, vibrant colors, clean rendering, cinematic but simple lighting, smooth animation, detailed textures.

Scene setup:
Skye, the female cockapoo rescue puppy wearing pink pilot goggles and a pink vest, sits in the pilot seat of her pink helicopter. The helicopter is calmly hovering in a bright blue sky with soft clouds visible through the cockpit window.

Camera and framing:
Locked camera.
Single continuous shot.
Medium close-up framing from inside the cockpit.
Skye centered in frame and facing forward.
No camera movement, no cuts, no transitions, no angle change.

Action:
Skye speaks directly to the viewer like a cheerful children'\''s show host. She talks in first person about her day. Her mouth moves naturally while speaking, with small head movements, blinking, and expressive facial animation.

Dialogue:
"hey AYA and Gili! Guess what? I had such a fun day today!"
"I ate lots and lots of yummy candy!"
"It was super sweet and really fun!"
"But now I need to brush my teeth really well!"

Motion:
Very gentle helicopter hover motion only.
Subtle character animation: blinking, small head movement, natural speaking mouth motion.

Mood:
Bright, friendly, playful children'\''s show atmosphere.'"$PROMPT_SUFFIX"

NEGATIVE_PROMPT_TYPE_1="blurry, overexposed, static, low quality, distorted, ugly, worst quality, JPEG artifacts, extra fingers, deformed, abrupt cutoff, cut off mid-sentence"
NEGATIVE_PROMPT_TYPE_2="scene change, shot change, jump cut, camera switch, camera movement, zoom, pan, rotation, changing angle, changing perspective, multiple scenes, montage, cinematic cuts, dynamic camera, blurry, low quality, compression artifacts, distorted face, deformed body, extra limbs, extra fingers, abrupt cutoff, cut off mid-sentence"

if [ "${PROMPT_TYPE}" = "1" ]; then
  PROMPT="$PROMPT_TYPE_1"
  NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-$NEGATIVE_PROMPT_TYPE_1}"
elif [ "${PROMPT_TYPE}" = "2" ]; then
  PROMPT="$PROMPT_TYPE_2"
  NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-$NEGATIVE_PROMPT_TYPE_2}"
elif [ "${PROMPT_TYPE}" = "3" ]; then
  PROMPT="$PROMPT_TYPE_3"
  NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-$NEGATIVE_PROMPT_TYPE_2}"
else
  PROMPT="$PROMPT_TYPE_1"
  NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-$NEGATIVE_PROMPT_TYPE_1}"
fi

# Collect images (sorted by name for consistent order)
IMAGES=()
while IFS= read -r -d '' f; do
  IMAGES+=("$f")
done < <(find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) -print0 | sort -z)

if [ ${#IMAGES[@]} -eq 0 ]; then
  echo "Error: No images found in $IMAGE_DIR"
  exit 1
fi

# Build --image args: distribute images evenly across frames
# Frame 0 = first image, frame N-1 = last image, others evenly spaced
IMAGE_ARGS=()
LAST_IDX=$((NUM_FRAMES - 1))
NUM_IMAGES=${#IMAGES[@]}

for i in "${!IMAGES[@]}"; do
  if [ "$NUM_IMAGES" -eq 1 ]; then
    FRAME_IDX=0
  else
    FRAME_IDX=$(awk "BEGIN {printf \"%.0f\", $i * $LAST_IDX / ($NUM_IMAGES - 1)}")
  fi
  IMAGE_ARGS+=(--image "${IMAGES[$i]}" "$FRAME_IDX" 0.9)
done

OUTPUT="${OUTPUT_DIR}/${OUTPUT_BASE_NAME}_pt${PROMPT_TYPE}_${DURATION_S}s_${WIDTH}x${HEIGHT}_seed${SEED}.mp4"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Skye Helicopter Keyframes"
echo "=============================================="
echo "Images: ${#IMAGES[@]} from $IMAGE_DIR"
echo "Frames: $NUM_FRAMES (~${DURATION_S}s @ ${FRAME_RATE} fps)"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Prompt type: $PROMPT_TYPE"
echo "Output: $OUTPUT"
echo "=============================================="

cd "$SCRIPT_DIR"
uv run python -m ltx_pipelines.ti2vid_one_stage \
  --checkpoint-path "$CHECKPOINT" \
  --gemma-root "$GEMMA_ROOT" \
  --prompt "$PROMPT" \
  --negative-prompt "$NEGATIVE_PROMPT" \
  "${IMAGE_ARGS[@]}" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --num-frames "$NUM_FRAMES" \
  --frame-rate "$FRAME_RATE" \
  --num-inference-steps "$NUM_INFERENCE_STEPS" \
  --seed "$SEED" \
  --output-path "$OUTPUT"

echo ""
echo "Video saved to: $OUTPUT"
