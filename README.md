# Pocket TTS C++

A pure C++ port of [Pocket TTS ONNX](https://huggingface.co/KevinAHM/pocket-tts-onnx) - lightweight text-to-speech with zero-shot voice cloning.

## Features

- **No Python dependency** for inference
- **Voice cloning** from any reference audio
- **~3x real-time** performance on Apple Silicon (INT8)
- **~200MB total model size**
- **C API** for FFI bindings (Python, C#, etc.)
- **Audio streaming** for low-latency playback

## Quick Start

### macOS/Linux

```bash
# 1) Install platform prerequisites first (see section below)
#
# 2) Clone & download models
git clone https://github.com/gudrob/pocket-tts.cpp.git
cd pocket-tts.cpp
./download_models.sh

# 3) Build (macOS/Linux example)
mkdir build && cd build
cmake .. && make -j4

# 4) Run
./pocket_tts "Hello, this is a test." ../models/reference_sample.wav output.wav
```

### Windows (PowerShell)

```powershell
# 1) Install Windows prerequisites first (see section below)
#
# 2) Clone
git clone https://github.com/gudrob/pocket-tts.cpp.git
cd pocket-tts.cpp

# 3) Download models (uses temporary Python venv)
.\download_models.bat
```

## Prerequisites

<details>
<summary><b>macOS (Homebrew)</b></summary>

```bash
brew install onnxruntime sentencepiece cmake
```
</details>

<details>
<summary><b>Linux (Ubuntu/Debian)</b></summary>

```bash
sudo apt install -y cmake build-essential pkg-config

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
# If your project path contains spaces, create a drive alias without spaces
subst P: "C:\path\to\pocket-tts.cpp"
P:

# Bootstrap local vcpkg (repo already contains a vcpkg folder)
cd vcpkg
.\bootstrap-vcpkg.bat
cd ..

# Install required dependencies
.\vcpkg\vcpkg.exe install onnxruntime:x64-windows sentencepiece:x64-windows

# Download models (uses temporary Python venv)
.\download_models.bat

mkdir build && cd build
cmake .. `
  -DBUILD_SHARED=OFF `
  -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DONNXRUNTIME_INCLUDE_DIRS=../vcpkg/installed/x64-windows/include/onnxruntime `
  -DONNXRUNTIME_LIBRARIES=../vcpkg/installed/x64-windows/lib/onnxruntime.lib `
  -DSENTENCEPIECE_INCLUDE_DIRS=../vcpkg/installed/x64-windows/include `
  -DSENTENCEPIECE_LIBRARIES=../vcpkg/installed/x64-windows/lib/sentencepiece.lib

cmake --build . --config Release --target pocket_tts_cli
.\Release\pocket_tts.exe --help
```
</details>

## CLI Usage

```
pocket_tts [options] <text> <voice_file> <output_file>

Options:
  --models-dir <path>   Models directory (default: models/onnx)
  --tokenizer <path>    Tokenizer path (default: models/tokenizer.model)
  --precision <p>       int8 or fp32 (default: int8)
  --temperature <t>     0.0-1.0 (default: 0.7)
  --lsd-steps <n>       Flow matching steps (default: 10)
  --max-frames <n>      Max frames to generate (default: 500)
  -h, --help            Show help
```

## C API (for FFI)

The shared library exports a C API for Python, C#, and other languages.
**Note**: `std::cout` logging is disabled by default when using the C API.

### Basic Generation

```c
#include "pocket_tts/pocket_tts_c.h"

// Configuration (optional, pass NULL for defaults)
PocketTTSConfig config = {
    .models_dir = "models/onnx",
    // Defaults: precision="int8", temperature=0.7, etc.
};

PocketTTSHandle tts = pocket_tts_create(&config);
VoiceHandle voice = pocket_tts_encode_voice(tts, "voice.wav");

AudioResult result;
pocket_tts_generate(tts, "Hello!", voice, &result);
// result.data = float[], result.sample_count, result.sample_rate = 24000

pocket_tts_free_audio(&result);
pocket_tts_free_voice(voice);
pocket_tts_destroy(tts);
```

### Streaming Generation

```c
void audio_callback(const float* samples, int count, int is_final, void* user_data) {
    printf("Received %d samples%s\n", count, is_final ? " [FINAL]" : "");
    // Process audio chunk (e.g., play, save, etc.)
}

StreamingConfig stream_cfg = {
    .chunk_size_frames = 5,  // ~400ms chunks
    .user_data = NULL
};

int total_samples = pocket_tts_generate_streaming(
    tts, "Hello from streaming!", voice, audio_callback, &stream_cfg
);
```

## Language Bindings

| Language | File | Run |
|----------|------|-----|
| **Python** | [test/test_api.py](test/test_api.py) | `python3 test/test_api.py` |
| **C#** | [test/PocketTTS.cs](test/PocketTTS.cs) | `dotnet run --project test/StreamingExample.csproj` |
| **C** | [test/test_api.c](test/test_api.c) | `clang -o test test/test_api.c -I include -L build -lpocket_tts` |

## Node.js Addon (Node 24+)

The Node.js addon lives in [`bindings/node`](bindings/node) and is implemented with N-API.

```bash
# macOS
brew install onnxruntime sentencepiece

cd bindings/node
npm install
npm run build
```

There is also a GitHub Actions matrix workflow for Linux, Windows and macOS prebuilds:
`.github/workflows/node-addon-prebuilds.yml`

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
