# Pocket TTS Node Addon

Node.js N-API bindings for `pocket-tts-cpp`.

## Requirements

- Node.js 24+
- Native dependencies from the C++ project:
  - ONNX Runtime
  - SentencePiece

## Local Build (macOS)

```bash
brew install onnxruntime sentencepiece

cd bindings/node
npm install
npm run build
```

## Environment Variables (optional)

If your dependencies are in non-standard locations, set one or more:

- `POCKET_TTS_PREFIX` (path list, separated by `:` on macOS/Linux, `;` on Windows)
- `ONNXRUNTIME_ROOT`
- `SENTENCEPIECE_ROOT`

## JavaScript Usage

```js
const { PocketTTS, version } = require("@pocket-tts/pocket-tts");

console.log("PocketTTS version:", version());

const tts = new PocketTTS({
  modelsDir: "../../models/onnx",
  tokenizerPath: "../../models/tokenizer.model",
  precision: "int8",
  temperature: 0.7
});

const voice = tts.encodeVoice("../../models/reference_sample.wav");
const { sampleRate, samples } = tts.generate("Hello from Node.js", voice);

console.log({ sampleRate, sampleCount: samples.length });

voice.free();
tts.close();
```
