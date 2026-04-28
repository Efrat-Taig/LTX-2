#!/bin/bash
# Download LTX-2 models for image-to-video generation (video + audio)
#
# For image-to-video with ltx-trainer, you need:
#   1. LTX-2.3 checkpoint (~44GB)
#   2. Gemma 3 text encoder (~24GB)
#
# Gemma requires:
#   1. Accept the license at https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
#   2. Log in: huggingface-cli login (or set HUGGING_FACE_HUB_TOKEN)
#   If you get 403 GatedRepoError, you haven't completed step 1 or 2.

set -e
MODELS_DIR="${MODELS_DIR:-/home/smiti-user/LTX-2/models}"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

echo "=== Downloading LTX-2 models to $MODELS_DIR ==="

# Option 1: Using huggingface-cli (recommended - handles large files, resume, auth)
if command -v huggingface-cli &>/dev/null; then
    echo ""
    echo "Using huggingface-cli..."
    
    # LTX-2.3 checkpoint (required for ltx-trainer inference)
    echo "Downloading LTX-2.3 checkpoint (ltx-2.3-22b-dev.safetensors, ~44GB)..."
    huggingface-cli download Lightricks/LTX-2.3 \
        ltx-2.3-22b-dev.safetensors \
        --local-dir . \
        --local-dir-use-symlinks False
    
    # Gemma 3 text encoder (required)
    echo ""
    echo "Downloading Gemma 3 text encoder (~24GB)..."
    echo "NOTE: If you get 403 GatedRepoError:"
    echo "  1. Visit https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized and accept the license"
    echo "  2. Run: huggingface-cli login"
    echo ""
    huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
        --local-dir gemma-3-12b-it-qat-q4_0-unquantized \
        --local-dir-use-symlinks False
else
    echo ""
    echo "huggingface-cli not found. Install with: pip install huggingface-hub"
    echo "Or use wget (see below)..."
    echo ""
    
    # Option 2: wget (no auth needed for LTX, but Gemma needs HF token)
    echo "Downloading LTX-2.3 checkpoint via wget..."
    wget -c "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors" \
        -O ltx-2.3-22b-dev.safetensors
    
    echo ""
    echo "For Gemma, use huggingface-cli (requires login):"
    echo "  pip install huggingface-hub"
    echo "  huggingface-cli login"
    echo "  huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir gemma-3-12b-it-qat-q4_0-unquantized"
fi

echo ""
echo "=== Done! ==="
echo "Set these paths in run_skye_video.sh:"
echo "  CHECKPOINT=$MODELS_DIR/ltx-2.3-22b-dev.safetensors"
echo "  GEMMA_ROOT=$MODELS_DIR/gemma-3-12b-it-qat-q4_0-unquantized"
echo ""
echo "Then run: bash run_skye_video.sh"
