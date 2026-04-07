#!/bin/bash

# Start script for Pointing Gemini SAM2 Server
# This server provides object detection and segmentation using Gemini and SAM2

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY environment variable is not set!"
    echo "Please set it with: export GEMINI_API_KEY='your-api-key-here'"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Set PYTHONPATH to include the predicators package
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "WARNING: nvidia-smi failed. GPU may not be available."
    echo "This server requires a CUDA-capable GPU."
    exit 1
fi

# Default parameters
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7100}"
USE_GPU="${USE_GPU:-true}"

echo "Starting Pointing Gemini SAM2 Server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Using GPU: $USE_GPU"
echo ""

# Start the server
if [ "$USE_GPU" = "false" ]; then
    python predicators/spot_utils/perception/server/pointing_gemini_sam2_server_standalone.py \
        --host "$HOST" \
        --port "$PORT" \
        --no-use-gpu
else
    python predicators/spot_utils/perception/server/pointing_gemini_sam2_server_standalone.py \
        --host "$HOST" \
        --port "$PORT" \
        --use-gpu
fi
