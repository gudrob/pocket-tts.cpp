#!/bin/bash
# Download Pocket TTS ONNX models from HuggingFace
# Usage: ./download_models.sh [output_dir]

set -e

MODEL_ID="KevinAHM/pocket-tts-onnx"
OUTPUT_DIR="${1:-models}"

echo "=== Pocket TTS Model Downloader ==="
echo "Model: $MODEL_ID"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Create output directory
mkdir -p "$OUTPUT_DIR/onnx"

echo "Downloading models..."

# Download INT8 ONNX models (faster, smaller)
ONNX_FILES=(
    "onnx/mimi_encoder.onnx"
    "onnx/text_conditioner.onnx"
    "onnx/flow_lm_main_int8.onnx"
    "onnx/flow_lm_flow_int8.onnx"
    "onnx/mimi_decoder_int8.onnx"
)

for file in "${ONNX_FILES[@]}"; do
    echo "Downloading $file..."
    huggingface-cli download "$MODEL_ID" "$file" --local-dir "$OUTPUT_DIR" --local-dir-use-symlinks False
done

# Download tokenizer
echo "Downloading tokenizer.model..."
huggingface-cli download "$MODEL_ID" "tokenizer.model" --local-dir "$OUTPUT_DIR" --local-dir-use-symlinks False

# Download reference sample
echo "Downloading reference_sample.wav..."
huggingface-cli download "$MODEL_ID" "reference_sample.wav" --local-dir "$OUTPUT_DIR" --local-dir-use-symlinks False

echo ""
echo "=== Download Complete ==="
echo ""
echo "Directory structure:"
find "$OUTPUT_DIR" -type f | head -20
