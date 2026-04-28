#!/bin/bash
# Generate a video from each of the 3 skye_and_chase_bench prompts
# Uses run_skye_video.sh with different PROMPT and OUTPUT_BASE_NAME per run
#
# Parameters (override via environment):
#   DURATION       Video length in seconds (default: 5)
#   NUM_INFERENCE_STEPS  Inference steps per frame (default: 40)
#
# Examples:
#   DURATION=10 NUM_INFERENCE_STEPS=50 ./run_skye_chase_bench_all_prompts.sh
#   DURATION=15 ./run_skye_chase_bench_all_prompts.sh
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configurable parameters
DURATION="${DURATION:-11}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"

IMAGE_PATH="$SCRIPT_DIR/input/scenes/skye_and_chase_bench/Gemini_Generated_Image_x6vk1nx6vk1nx6vk.png"
export IMAGE_PATH
export WIDTH=480 HEIGHT=832 FRAME_RATE=24 DURATIONS="$DURATION" NUM_INFERENCE_STEPS="$NUM_INFERENCE_STEPS" SEED=42
export NEGATIVE_PROMPT="blurry, low quality, distorted"

echo "Settings: duration=${DURATION}s, steps=${NUM_INFERENCE_STEPS}"
echo ""

# Prompt 1: Long racing scene (Chase & Skye at Lookout Tower)
export OUTPUT_BASE_NAME="skye_and_chase_bench_p1"
export PROMPT='High-quality 3D animation in the style of Paw Patrol, vibrant colors, clean CGI, cinematic lighting, bright and cheerful atmosphere, smooth character animation, Nick Jr aesthetic, 4k render, detailed textures, expressive facial expressions. High-end 3D animation in the PAW Patrol style. The camera is framed in a wide shot near the Lookout Tower, looking down a racing path. Scene: Chase is standing in his police gear on the ground. Skye is hovering just above him (for 2 seconds). Dialogue: Chase: "Okay Skye, the mission is to get to the Lookout as fast as possible. NO shortcuts!" Skye (already moving forward, ascending higher, her voice fading): "Did you say something? Because I am already halfway there!" Action: As Chase finishes his sentence, Skye activates her pink booster and blasts forward. Chase looks up, his head following her ascent, with a stunned, wide-eyed expression of disbelief. The camera pans up with Skye flight path, leaving Chase in the distance, then cuts back to a funny close-up of Chase confused face. Cinematic lighting, fast-paced energy static camera composition maintained for the entire clip'
echo "=== [1/3] Generating prompt 1: racing scene ==="
./run_skye_video.sh

# Prompt 2: Short "No shortcuts!"
export OUTPUT_BASE_NAME="skye_and_chase_bench_p2"
export PROMPT='High-end 3D animation, PAW Patrol style, Nick Jr aesthetic. Wide shot at the Lookout Tower. Chase (German Shepherd police pup) stands on the ground looking up. Skye (Cockapoo pilot pup) hovers above him. Dialogue: Chase says "No shortcuts!" while Skye shouts "I am already there!" Action: Skye activates her pink jet boosters and blasts forward into the sky. Chase head tilts up, following her, with a funny wide-eyed shocked expression. Cinematic lighting, vibrant colors, 4k, smooth CGI animation. static camera composition maintained for the entire clip'
echo "=== [2/3] Generating prompt 2: No shortcuts! ==="
./run_skye_video.sh

# Prompt 3: Bench dialogue
export OUTPUT_BASE_NAME="skye_chase_bench_dialogue"
export PROMPT='High-quality 3D animation, PAW Patrol style, Nick Jr aesthetic, clean CGI, 4k render. Medium static shot of Chase (German Shepherd pup) and Skye (Cockapoo pup) sitting side-by-side on a park bench. Cinematic lighting, vibrant colors. First, Chase turns his head to Skye and clearly says: "Great job today, Skye!", his mouth moving in sync. Then, Skye smiles back, looks at Chase and responds: "We sure did!", her mouth moving clearly. Expressive facial animations, minimal body movement, focused on the dialogue between the two characters. static camera composition maintained for the entire clip'
echo "=== [3/3] Generating prompt 3: bench dialogue ==="
./run_skye_video.sh

echo ""
echo "=== All 3 videos generated ==="
echo "  - skye_and_chase_bench_p1_${DURATION}s_480x832_${NUM_INFERENCE_STEPS}steps_seed42.mp4"
echo "  - skye_and_chase_bench_p2_${DURATION}s_480x832_${NUM_INFERENCE_STEPS}steps_seed42.mp4"
echo "  - skye_chase_bench_dialogue_${DURATION}s_480x832_${NUM_INFERENCE_STEPS}steps_seed42.mp4"
