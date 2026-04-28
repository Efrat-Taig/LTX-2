#!/bin/bash
# Image-to-video generation: Skye from Paw Patrol in her helicopter
# Uses your image and PROMPT_TYPE_1 to generate a video with LTX-2
#
# PREREQUISITES: Download models from https://huggingface.co/Lightricks/LTX-2.3
# - LTX-2.3 checkpoint (ltx-2.3-22b-dev.safetensors)
# - Gemma 3 text encoder (from google/gemma-3-12b-it-qat-q4_0-unquantized)
#
# EXAMPLE: skye_and_chase_bench scene (Chase & Skye racing near Lookout Tower)
#   cd /home/smiti-user/LTX-2 && \
#   IMAGE_PATH="/home/smiti-user/LTX-2/input/scenes/skye_and_chase_bench/Gemini_Generated_Image_x6vk1nx6vk1nx6vk.png" \
#   OUTPUT_BASE_NAME="skye_and_chase_bench" \
#   WIDTH=480 HEIGHT=832 FRAME_RATE=24 DURATIONS="5" SEED=42 \
#   PROMPT='High-quality 3D animation in the style of Paw Patrol, vibrant colors, clean CGI, cinematic lighting, bright and cheerful atmosphere, smooth character animation, Nick Jr aesthetic, 4k render, detailed textures, expressive facial expressions. High-end 3D animation in the PAW Patrol style. The camera is framed in a wide shot near the Lookout Tower, looking down a racing path. Scene: Chase is standing in his police gear on the ground. Skye is hovering just above him (for 2 seconds). Dialogue: Chase: "Okay Skye, the mission is to get to the Lookout as fast as possible. NO shortcuts!" Skye (already moving forward, ascending higher, her voice fading): "Did you say something? Because I am already halfway there!" Action: As Chase finishes his sentence, Skye activates her pink booster and blasts forward. Chase looks up, his head following her ascent, with a stunned, wide-eyed expression of disbelief. The camera pans up with Skye flight path, leaving Chase in the distance, then cuts back to a funny close-up of Chase confused face. Cinematic lighting, fast-paced energy static camera composition maintained for the entire clip' \
#   NEGATIVE_PROMPT="blurry, low quality, distorted" \
#   ./run_skye_video.sh
#
# Configurable (set before running or edit defaults; run_all_skye_experiments.sh exports these):
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/models/ltx-2.3-22b-dev.safetensors}"
GEMMA_ROOT="${GEMMA_ROOT:-$SCRIPT_DIR/models/gemma-3-12b-it-qat-q4_0-unquantized}"
LORA_PATH="${LORA_PATH:-}"
IMAGE_PATH="${IMAGE_PATH:-$SCRIPT_DIR/input/scenes/skye_in_hlicopter/sky_in_heli.jpg}"

# Output naming (MOVA-style)
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR}"
OUTPUT_BASE_NAME="${OUTPUT_BASE_NAME:-skye_helicopter}"
SEED="${SEED:-42}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-704}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"

# Video length: num_frames must be k*8+1 (e.g. 17, 25, 33, 49, 65, 81, 97, 121)
# Duration = num_frames / frame_rate (e.g. 97 frames @ 25fps = 3.88s)
# DURATIONS: when set, generates one video per duration (e.g. "15 20 35"); otherwise single run with NUM_FRAMES
DURATIONS="${DURATIONS:-15 20 35}"
NUM_FRAMES="${NUM_FRAMES:-}"
FRAME_RATE="${FRAME_RATE:-25.0}"

# duration_sec -> num_frames (must be k*8+1)
get_frames() {
  local sec="$1"
  local fps="${FRAME_RATE:-25.0}"
  local raw
  raw=$(awk "BEGIN {printf \"%.0f\", $sec * $fps}")
  awk "BEGIN {k=int(($raw - 1) / 8 + 0.5); n=8*k+1; if(n<17)n=17; print n}"
}

# Generation quality & guidance (optimal defaults)
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
STG_SCALE="${STG_SCALE:-1.0}"
STG_BLOCKS="${STG_BLOCKS:-29}"
STG_MODE="${STG_MODE:-stg_av}"

# Audio & device
SKIP_AUDIO="${SKIP_AUDIO:-false}"
DEVICE="${DEVICE:-cuda}"

# Suffix added to all prompts
PROMPT_SUFFIX=" static camera composition maintained for the entire clip"

# PROMPT_TYPE 1: Full descriptive, cinematic
PROMPT_TYPE_1='High-quality 3D animation in the style of Paw Patrol, vibrant colors, clean CGI, cinematic lighting, bright and cheerful atmosphere, smooth character animation, Nick Jr aesthetic, 4k render, detailed textures, expressive facial expressions.

Skye from Paw Patrol, the cute female cockapoo rescue puppy pilot, wearing her pink pilot goggles and pink vest, flying her small pink helicopter high in a bright blue sky with soft fluffy clouds. The helicopter gently hovers in the air while she sits in the cockpit.

Skye looks directly into the camera and talks to the viewer like a cheerful kids TV host. She speaks clearly in FIRST PERSON about her own day. She is happy, excited, and very expressive.

Dialogue (first person, speaking directly to camera):
"Hi everyone! I had such a happy day today!"
"I ate soooo many candies and sweets!"
"It was so yummy and so much fun!"
"But now I need to brush my teeth really well so they stay clean and healthy!"

She laughs, smiles, and talks energetically while sitting in the helicopter.

Camera: medium close-up inside the helicopter cockpit, Skye facing the camera.
Motion: helicopter gently hovering in the sky with soft movement.
Mood: joyful, playful, energetic children'\''s show.
Environment: bright daylight sky, soft clouds.'"$PROMPT_SUFFIX"

# PROMPT_TYPE 2: Emphasizes single continuous shot, locked camera
PROMPT_TYPE_2='High-quality 3D animation in the style of Paw Patrol, vibrant colors, clean CGI, cinematic lighting, bright cheerful atmosphere, smooth character animation, Nick Jr aesthetic, 4k render, detailed textures, expressive facial expressions.

Skye from Paw Patrol, the female cockapoo rescue puppy pilot wearing pink pilot goggles and a pink vest, sitting inside her small pink helicopter. The helicopter is hovering in a bright blue sky with soft fluffy clouds.

IMPORTANT: The entire video is ONE continuous shot. The camera angle does NOT change. No cuts, no scene transitions, no jump cuts, no camera switches.

The camera stays fixed in a stable medium close-up inside the helicopter cockpit. Skye remains centered in the frame and faces the camera the whole time. The helicopter gently hovers but the framing stays consistent.

Skye looks directly into the camera and talks to the viewer like a cheerful kids TV host. She speaks in FIRST PERSON about her own day. She is happy, excited and expressive.

Dialogue (first person, speaking to camera):
"Hi everyone! I had such a happy day today!"
"I ate soooo many candies and sweets!"
"It was so yummy and so much fun!"
"But now I need to brush my teeth really well so they stay clean and healthy!"

Her mouth movements match the speech. She smiles, laughs and talks energetically.

Motion: only subtle animation — slight helicopter hover, small head movement, blinking, natural mouth movement.
Camera: locked camera, stable framing, no zoom, no rotation, no angle change.
Shot: single continuous shot for the entire clip.
Mood: joyful, playful children'\''s show.
Environment: bright daylight sky with soft clouds.'"$PROMPT_SUFFIX"

# PROMPT_TYPE 3: Minimal, structured, locked camera
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
"Hi friends! Guess what? I had such a fun day today!"
"I ate lots and lots of yummy candy!"
"It was super sweet and really fun!"
"But now I need to brush my teeth really well!"

Motion:
Very gentle helicopter hover motion only.
Subtle character animation: blinking, small head movement, natural speaking mouth motion.

Mood:
Bright, friendly, playful children'\''s show atmosphere.'"$PROMPT_SUFFIX"

# Negative prompts (type 2/3 use camera-stability negative)
NEGATIVE_PROMPT_TYPE_1="blurry, overexposed, static, low quality, distorted, ugly, worst quality, JPEG artifacts, extra fingers, deformed"
NEGATIVE_PROMPT_TYPE_2="scene change, shot change, jump cut, camera switch, camera movement, zoom, pan, rotation, changing angle, changing perspective, multiple scenes, montage, cinematic cuts, dynamic camera, blurry, low quality, compression artifacts, distorted face, deformed body, extra limbs, extra fingers"

# Select prompt and negative prompt by PROMPT_TYPE (1, 2, or 3)
# If PROMPT is already set in environment, use it (e.g. when running for a specific scene)
if [ -z "${PROMPT}" ]; then
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
  # Default: PROMPT_TYPE 1 (allow NEGATIVE_PROMPT override)
  PROMPT="$PROMPT_TYPE_1"
  NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-$NEGATIVE_PROMPT_TYPE_1}"
fi
fi

# Option A: ltx-trainer inference (simpler - only needs checkpoint + Gemma)
# Generates video + audio by default
#
# Option B (production quality): ltx-pipelines TI2VidTwoStagesPipeline
# Requires: checkpoint, Gemma, distilled LoRA, spatial upsampler
#   cd /home/smiti-user/LTX-2 && uv run python -m ltx_pipelines.ti2vid_two_stages \
#     --checkpoint-path "$CHECKPOINT" --gemma-root "$GEMMA_ROOT" \
#     --distilled-lora /path/to/ltx-2.3-22b-distilled-lora-384.safetensors \
#     --spatial-upsampler-path /path/to/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
#     --prompt "$PROMPT" --image "$IMAGE_PATH" 0 1.0 --output-path "$OUTPUT"
#
# Build inference command (avoid arrays for sh/dash compatibility)
cd "$SCRIPT_DIR/packages/ltx-trainer"

run_one() {
  local nf="$1"
  local out="$2"
  uv run python scripts/inference.py \
      --checkpoint "$CHECKPOINT" \
      --text-encoder-path "$GEMMA_ROOT" \
      --prompt "$PROMPT" \
      --negative-prompt "$NEGATIVE_PROMPT" \
      --condition-image "$IMAGE_PATH" \
      --output "$out" \
      --width "$WIDTH" \
      --height "$HEIGHT" \
      --num-frames "$nf" \
      --frame-rate "$FRAME_RATE" \
      --num-inference-steps "$NUM_INFERENCE_STEPS" \
      --guidance-scale "$GUIDANCE_SCALE" \
      --stg-scale "$STG_SCALE" \
      --stg-blocks $STG_BLOCKS \
      --stg-mode "$STG_MODE" \
      --seed "$SEED" \
      --device "$DEVICE" \
      ${LORA_PATH:+--lora-path "$LORA_PATH"} \
      $([ "$SKIP_AUDIO" = "true" ] && echo --skip-audio)
  echo "Video saved to: $out"
}

if [ -n "$NUM_FRAMES" ]; then
  # Single run (e.g. when called from run_all_skye_experiments.sh)
  if [ -z "${OUTPUT}" ]; then
    DURATION_S=$(awk "BEGIN {printf \"%.0f\", $NUM_FRAMES / $FRAME_RATE}")
    if [ -n "${PROMPT_TYPE}" ]; then
      OUTPUT="${OUTPUT_DIR}/${OUTPUT_BASE_NAME}_pt${PROMPT_TYPE}_${DURATION_S}s_${WIDTH}x${HEIGHT}_${NUM_INFERENCE_STEPS}steps_seed${SEED}.mp4"
    else
      OUTPUT="${OUTPUT_DIR}/${OUTPUT_BASE_NAME}_${DURATION_S}s_${WIDTH}x${HEIGHT}_${NUM_INFERENCE_STEPS}steps_seed${SEED}.mp4"
    fi
  fi
  run_one "$NUM_FRAMES" "$OUTPUT"
else
  # Multi-run: loop over DURATIONS (15, 20, 35 seconds)
  for len in $DURATIONS; do
    NUM_FRAMES=$(get_frames "$len")
    DURATION_S=$(awk "BEGIN {printf \"%.0f\", $NUM_FRAMES / $FRAME_RATE}")
    if [ -n "${PROMPT_TYPE}" ]; then
      OUTPUT="${OUTPUT_DIR}/${OUTPUT_BASE_NAME}_pt${PROMPT_TYPE}_${DURATION_S}s_${WIDTH}x${HEIGHT}_${NUM_INFERENCE_STEPS}steps_seed${SEED}.mp4"
    else
      OUTPUT="${OUTPUT_DIR}/${OUTPUT_BASE_NAME}_${DURATION_S}s_${WIDTH}x${HEIGHT}_${NUM_INFERENCE_STEPS}steps_seed${SEED}.mp4"
    fi
    echo ""
    echo "Generating ${len}s video (${NUM_FRAMES} frames)..."
    run_one "$NUM_FRAMES" "$OUTPUT" || exit 1
  done
fi


