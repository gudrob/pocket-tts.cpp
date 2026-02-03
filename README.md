# Pocket TTS C++

A pure C++ port of [Pocket TTS ONNX](https://huggingface.co/KevinAHM/pocket-tts-onnx) - lightweight text-to-speech with zero-shot voice cloning.

## Features

- **No Python dependency** for inference
- **ONNX Runtime** backend  
- **Voice cloning** from any reference audio
- **INT8 quantized models** for fast CPU inference
- **~200MB total model size** (INT8)
- **3x real-time** on Apple Silicon

## Quick Start

```bash
# Clone repository
git clone https://github.com/gudrob/pocket-tts.cpp.git
cd pocket-tts.cpp

# Download models (~200MB)
./download_models.sh

# Build
mkdir build && cd build
cmake ..
make -j4

# Run
./pocket_tts "Hello, this is a test." ../models/reference_sample.wav output.wav
```

## Prerequisites

### macOS (Homebrew)

```bash
brew install onnxruntime sentencepiece libsndfile libsamplerate cmake
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y cmake build-essential pkg-config \
    libsndfile1-dev libsamplerate0-dev

# Install SentencePiece
git clone https://github.com/google/sentencepiece.git
cd sentencepiece && mkdir build && cd build
cmake .. && make -j4 && sudo make install
sudo ldconfig

# Install ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz
tar -xzf onnxruntime-linux-x64-1.17.0.tgz
sudo cp -r onnxruntime-linux-x64-1.17.0/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.17.0/lib/* /usr/local/lib/
sudo ldconfig
```

### Windows (vcpkg)

```powershell
# Install vcpkg if not already installed
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Install dependencies
.\vcpkg install onnxruntime:x64-windows
.\vcpkg install sentencepiece:x64-windows
.\vcpkg install libsndfile:x64-windows
.\vcpkg install libsamplerate:x64-windows

# Build with CMake
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg_root]/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

## Usage

```
pocket_tts [options] <text> <voice_file> <output_file>

Arguments:
  text         Text to synthesize
  voice_file   Reference voice audio file (WAV)
  output_file  Output audio file (WAV)

Options:
  --models-dir <path>   Path to models directory (default: models/onnx)
  --tokenizer <path>    Path to tokenizer.model (default: models/tokenizer.model)
  --precision <p>       Model precision: int8 or fp32 (default: int8)
  --temperature <t>     Sampling temperature 0.0-1.0 (default: 0.7)
  --lsd-steps <n>       Flow matching steps (default: 10)
  --max-frames <n>      Maximum frames to generate (default: 500)
  -h, --help            Show this help
```

## Examples

```bash
# Basic usage
./pocket_tts "Hello world!" voice.wav output.wav

# Use FP32 precision
./pocket_tts --precision fp32 "Hello!" voice.wav output.wav

# Lower temperature for more deterministic output
./pocket_tts --temperature 0.3 "Hello!" voice.wav output.wav
```

## Required Files

After running `download_models.sh`:

```
models/
├── onnx/
│   ├── mimi_encoder.onnx          # 73 MB
│   ├── text_conditioner.onnx      # 16 MB
│   ├── flow_lm_main_int8.onnx     # 76 MB
│   ├── flow_lm_flow_int8.onnx     # 10 MB
│   └── mimi_decoder_int8.onnx     # 23 MB
├── tokenizer.model                 # 58 KB
└── reference_sample.wav            # Example voice
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│  Voice Audio    │────▶│   MIMI Encoder   │────▶ Voice Embeddings
└─────────────────┘     └──────────────────┘
                                                        │
┌─────────────────┐     ┌──────────────────┐            ▼
│      Text       │────▶│   Tokenizer      │────▶ ┌──────────────┐
│                 │     │  (SentencePiece) │      │ Flow LM Main │
└─────────────────┘     └──────────────────┘      └──────────────┘
                                                        │
                        ┌──────────────────┐            ▼
                        │ Text Conditioner │────▶ ┌──────────────┐
                        └──────────────────┘      │ Flow LM Flow │
                                                  │   (Euler)    │
                                                  └──────────────┘
                                                        │
                                                        ▼
                              ┌──────────────────┐
                              │   MIMI Decoder   │────▶ Audio (24kHz)
                              └──────────────────┘
```

## Performance

| Platform | Precision | RTFx |
|----------|-----------|------|
| Apple M1 Pro | INT8 | ~3x |
| Apple M1 Pro | FP32 | ~2x |

RTFx = Real-time factor (>1.0 means faster than real-time)

## Building from Source

### CMake Options

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native"
```

## License

- **Models**: CC BY 4.0 (from [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts))
- **Code**: MIT

## Acknowledgements

- [Kyutai](https://kyutai.org/) - Original Pocket TTS model
- [KevinAHM](https://huggingface.co/KevinAHM) - ONNX export
- [Google SentencePiece](https://github.com/google/sentencepiece) - Tokenizer
