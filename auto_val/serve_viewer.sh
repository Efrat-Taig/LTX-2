#!/bin/bash
# Serves the VLM comparison viewer at http://localhost:8765
cd "$(dirname "$0")/outputs_vlm_compare"
echo "Serving viewer at http://localhost:8765/viewer.html"
python -m http.server 8765
