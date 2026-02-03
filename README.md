# Pocket TTS C++

A pure C++ port of [Pocket TTS ONNX](https://huggingface.co/KevinAHM/pocket-tts-onnx) - lightweight text-to-speech with zero-shot voice cloning.

## Features

- 🚀 **No Python dependency** for inference
- 🔊 **Voice cloning** from any reference audio
- ⚡ **~3x real-time** on Apple Silicon (INT8)
- 📦 **~200MB total model size**
- 🔗 **C API** for FFI bindings (Python, C#, etc.)

## Quick Start

```bash
# Clone & download models
git clone https://github.com/gudrob/pocket-tts.cpp.git
cd pocket-tts.cpp
./download_models.sh

# Build
mkdir build && cd build
cmake .. && make -j4

# Run
./pocket_tts "Hello, this is a test." ../models/reference_sample.wav output.wav
```

## Prerequisites

<details>
<summary><b>macOS (Homebrew)</b></summary>

```bash
brew install onnxruntime sentencepiece libsndfile libsamplerate cmake
```
</details>

<details>
<summary><b>Linux (Ubuntu/Debian)</b></summary>

```bash
sudo apt install -y cmake build-essential pkg-config libsndfile1-dev libsamplerate0-dev

# SentencePiece
git clone https://github.com/google/sentencepiece.git
cd sentencepiece && mkdir build && cd build
cmake .. && make -j4 && sudo make install && sudo ldconfig

# ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz
tar -xzf onnxruntime-linux-x64-1.17.0.tgz
sudo cp -r onnxruntime-linux-x64-1.17.0/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.17.0/lib/* /usr/local/lib/
sudo ldconfig
```
</details>

<details>
<summary><b>Windows (vcpkg)</b></summary>

```powershell
git clone https://github.com/Microsoft/vcpkg.git && cd vcpkg
.\bootstrap-vcpkg.bat && .\vcpkg integrate install

.\vcpkg install onnxruntime:x64-windows sentencepiece:x64-windows libsndfile:x64-windows libsamplerate:x64-windows

mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg_root]/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```
</details>

## CLI Usage

```
pocket_tts [options] <text> <voice_file> <output_file>

Options:
  --models-dir <path>   Models directory (default: models/onnx)
  --precision <p>       int8 or fp32 (default: int8)
  --temperature <t>     0.0-1.0 (default: 0.7)
  -h, --help            Show help
```

## C API (for FFI)

The shared library exports a C API for Python, C#, and other languages:

```c
#include "pocket_tts/pocket_tts_c.h"

PocketTTSHandle tts = pocket_tts_create(NULL);
VoiceHandle voice = pocket_tts_encode_voice(tts, "voice.wav");

AudioResult result;
pocket_tts_generate(tts, "Hello!", voice, &result);
// result.data = float[], result.sample_count, result.sample_rate = 24000

pocket_tts_free_audio(&result);
pocket_tts_free_voice(voice);
pocket_tts_destroy(tts);
```

## Language Bindings

| Language | File | Run |
|----------|------|-----|
| **Python** | [test/test_api.py](test/test_api.py) | `python3 test/test_api.py` |
| **C#** | [test/PocketTTS.cs](test/PocketTTS.cs) | See file for usage |
| **C** | [test/test_api.c](test/test_api.c) | `clang -o test test/test_api.c -I include -L build -lpocket_tts` |

## Build Options

```bash
cmake .. -DBUILD_CLI=ON -DBUILD_SHARED=ON
```

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_CLI` | ON | Command-line tool |
| `BUILD_SHARED` | ON | Shared library |

## Performance

| Platform | Precision | RTFx |
|----------|-----------|------|
| Apple M1 Pro | INT8 | ~3x |
| Apple M1 Pro | FP32 | ~2x |

## License

- **Code**: MIT
- **Models**: CC BY 4.0 ([kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts))

## Credits

[Kyutai](https://kyutai.org/) • [KevinAHM](https://huggingface.co/KevinAHM) • [SentencePiece](https://github.com/google/sentencepiece)
