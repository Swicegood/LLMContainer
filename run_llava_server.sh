#!/bin/bash

# Set the model directory
MODEL_DIR="/path/to/model/directory"

# Create the model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Download the model files if they don't exist
if [ ! -f "$MODEL_DIR/llava-v1.6-mistral-7b.Q8_0.gguf" ]; then
    echo "Downloading LLaVA model..."
    curl -L https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q8_0.gguf -o "$MODEL_DIR/llava-v1.6-mistral-7b.Q8_0.gguf"
fi

if [ ! -f "$MODEL_DIR/mmproj-model-f16.gguf" ]; then
    echo "Downloading MMPROJ model..."
    curl -L https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf -o "$MODEL_DIR/mmproj-model-f16.gguf"
fi

# Run the Docker container
echo "Starting LLaVA server..."
docker run -p 8080:8080 -v "$MODEL_DIR:/models" llava-server