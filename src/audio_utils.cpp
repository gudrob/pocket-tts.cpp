#include "pocket_tts/audio_utils.hpp"

#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstdint>

namespace pocket_tts {

// ── WAV format structures ──────────────────────────────────────────────

#pragma pack(push, 1)
struct WavHeader {
    char     riffId[4];       // "RIFF"
    uint32_t fileSize;        // file size - 8
    char     waveId[4];       // "WAVE"
};

struct WavChunkHeader {
    char     id[4];
    uint32_t size;
};

struct WavFmtChunk {
    uint16_t audioFormat;     // 1 = PCM, 3 = IEEE float
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
};
#pragma pack(pop)

static bool chunkIdEquals(const char id[4], const char* tag) {
    return id[0] == tag[0] && id[1] == tag[1] && id[2] == tag[2] && id[3] == tag[3];
}

// ── loadWav ────────────────────────────────────────────────────────────

std::vector<float> AudioUtils::loadWav(const std::string& filepath, int targetSampleRate) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open audio file: " + filepath);
    }

    // Read RIFF header
    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file || !chunkIdEquals(header.riffId, "RIFF") || !chunkIdEquals(header.waveId, "WAVE")) {
        throw std::runtime_error("Not a valid WAV file: " + filepath);
    }

    // Scan chunks for fmt and data
    WavFmtChunk fmt{};
    bool fmtFound = false;
    std::vector<float> monoSamples;

    while (file) {
        WavChunkHeader chunk;
        file.read(reinterpret_cast<char*>(&chunk), sizeof(chunk));
        if (!file) break;

        auto chunkStart = file.tellg();

        if (chunkIdEquals(chunk.id, "fmt ")) {
            if (chunk.size < sizeof(WavFmtChunk)) {
                throw std::runtime_error("Invalid fmt chunk in: " + filepath);
            }
            file.read(reinterpret_cast<char*>(&fmt), sizeof(WavFmtChunk));
            if (!file) throw std::runtime_error("Truncated fmt chunk in: " + filepath);

            if (fmt.audioFormat != 1 && fmt.audioFormat != 3) {
                throw std::runtime_error("Unsupported WAV format (only PCM/float supported): " + filepath);
            }
            if (fmt.numChannels < 1 || fmt.numChannels > 2) {
                throw std::runtime_error("Unsupported channel count (only mono/stereo): " + filepath);
            }
            fmtFound = true;

        } else if (chunkIdEquals(chunk.id, "data")) {
            if (!fmtFound) {
                throw std::runtime_error("data chunk before fmt chunk in: " + filepath);
            }

            const size_t bytesPerSample = fmt.bitsPerSample / 8;
            const size_t totalSamples = chunk.size / bytesPerSample;

            if (fmt.audioFormat == 3 && fmt.bitsPerSample == 32) {
                // IEEE float32
                std::vector<float> raw(totalSamples);
                file.read(reinterpret_cast<char*>(raw.data()), chunk.size);
                if (fmt.numChannels == 1) {
                    monoSamples = std::move(raw);
                } else {
                    monoSamples = stereoToMono(raw);
                }
            } else if (fmt.audioFormat == 1 && fmt.bitsPerSample == 16) {
                // PCM int16
                std::vector<int16_t> raw(totalSamples);
                file.read(reinterpret_cast<char*>(raw.data()), chunk.size);
                std::vector<float> floatSamples(totalSamples);
                constexpr float scale = 1.0f / 32768.0f;
                for (size_t i = 0; i < totalSamples; ++i) {
                    floatSamples[i] = static_cast<float>(raw[i]) * scale;
                }
                if (fmt.numChannels == 1) {
                    monoSamples = std::move(floatSamples);
                } else {
                    monoSamples = stereoToMono(floatSamples);
                }
            } else if (fmt.audioFormat == 1 && fmt.bitsPerSample == 24) {
                // PCM int24 (packed 3 bytes)
                std::vector<uint8_t> raw(chunk.size);
                file.read(reinterpret_cast<char*>(raw.data()), chunk.size);
                std::vector<float> floatSamples(totalSamples);
                constexpr float scale = 1.0f / 8388608.0f; // 2^23
                for (size_t i = 0; i < totalSamples; ++i) {
                    size_t offset = i * 3;
                    int32_t val = raw[offset] | (raw[offset + 1] << 8) | (raw[offset + 2] << 16);
                    if (val & 0x800000) val |= 0xFF000000; // sign extend
                    floatSamples[i] = static_cast<float>(val) * scale;
                }
                if (fmt.numChannels == 1) {
                    monoSamples = std::move(floatSamples);
                } else {
                    monoSamples = stereoToMono(floatSamples);
                }
            } else {
                throw std::runtime_error(
                    "Unsupported bit depth " + std::to_string(fmt.bitsPerSample) +
                    " for format " + std::to_string(fmt.audioFormat) + " in: " + filepath);
            }
            break; // done, we have the data
        }

        // Skip to next chunk (chunks are word-aligned)
        uint32_t skipSize = chunk.size + (chunk.size & 1);
        file.seekg(static_cast<std::streamoff>(chunkStart) + skipSize);
    }

    if (monoSamples.empty()) {
        throw std::runtime_error("No audio data found in: " + filepath);
    }

    // Resample if needed
    if (fmtFound && static_cast<int>(fmt.sampleRate) != targetSampleRate) {
        monoSamples = resample(monoSamples, static_cast<int>(fmt.sampleRate), targetSampleRate);
    }

    // Normalize
    monoSamples = normalize(monoSamples);

    return monoSamples;
}

// ── saveWav ────────────────────────────────────────────────────────────

void AudioUtils::saveWav(const std::string& filepath, const std::vector<float>& audioData, int sampleRate) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to create audio file: " + filepath);
    }

    const uint32_t numSamples = static_cast<uint32_t>(audioData.size());
    const uint32_t dataSize = numSamples * sizeof(float);
    const uint32_t fmtChunkSize = 16;

    // RIFF header
    WavHeader header;
    std::memcpy(header.riffId, "RIFF", 4);
    header.fileSize = 4 + (8 + fmtChunkSize) + (8 + dataSize);
    std::memcpy(header.waveId, "WAVE", 4);
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // fmt chunk
    WavChunkHeader fmtHeader;
    std::memcpy(fmtHeader.id, "fmt ", 4);
    fmtHeader.size = fmtChunkSize;
    file.write(reinterpret_cast<const char*>(&fmtHeader), sizeof(fmtHeader));

    WavFmtChunk fmt{};
    fmt.audioFormat = 3; // IEEE float
    fmt.numChannels = 1;
    fmt.sampleRate = static_cast<uint32_t>(sampleRate);
    fmt.bitsPerSample = 32;
    fmt.blockAlign = fmt.numChannels * (fmt.bitsPerSample / 8);
    fmt.byteRate = fmt.sampleRate * fmt.blockAlign;
    file.write(reinterpret_cast<const char*>(&fmt), sizeof(fmt));

    // data chunk
    WavChunkHeader dataHeader;
    std::memcpy(dataHeader.id, "data", 4);
    dataHeader.size = dataSize;
    file.write(reinterpret_cast<const char*>(&dataHeader), sizeof(dataHeader));
    file.write(reinterpret_cast<const char*>(audioData.data()), dataSize);
}

// ── resample (windowed sinc / Lanczos-8) ───────────────────────────────

std::vector<float> AudioUtils::resample(const std::vector<float>& input,
                                        int inputSampleRate,
                                        int outputSampleRate) {
    if (inputSampleRate == outputSampleRate || input.empty()) {
        return input;
    }

    constexpr int LANCZOS_A = 8;  // Lanczos kernel half-width (lobes) — maximum quality

    const double ratio = static_cast<double>(outputSampleRate) / inputSampleRate;
    const size_t outputSize = static_cast<size_t>(std::ceil(input.size() * ratio));
    std::vector<float> output(outputSize);

    const double step = static_cast<double>(inputSampleRate) / outputSampleRate;

    // When downsampling, widen the sinc kernel to act as a low-pass filter
    const double filterScale = (ratio < 1.0) ? ratio : 1.0;
    const double windowRadius = LANCZOS_A / filterScale;

    const auto lanczosKernel = [](double x, int a) -> double {
        if (x == 0.0) return 1.0;
        if (x < -a || x > a) return 0.0;
        const double pi_x = M_PI * x;
        const double pi_x_over_a = M_PI * x / a;
        return (std::sin(pi_x) / pi_x) * (std::sin(pi_x_over_a) / pi_x_over_a);
    };

    const int inputLen = static_cast<int>(input.size());

    for (size_t i = 0; i < outputSize; ++i) {
        const double srcPos = i * step;
        const int center = static_cast<int>(std::floor(srcPos));
        const double fracOffset = srcPos - center;

        double sample = 0.0;
        double weightSum = 0.0;

        const int jMin = static_cast<int>(std::ceil(-windowRadius + fracOffset));
        const int jMax = static_cast<int>(std::floor(windowRadius + fracOffset));

        for (int j = jMin; j <= jMax; ++j) {
            const int srcIdx = center + j;
            if (srcIdx < 0 || srcIdx >= inputLen) continue;

            const double x = (j - fracOffset) * filterScale;
            const double w = lanczosKernel(x, LANCZOS_A);
            sample += input[srcIdx] * w;
            weightSum += w;
        }

        output[i] = (weightSum > 0.0) ? static_cast<float>(sample / weightSum) : 0.0f;
    }

    return output;
}

// ── stereoToMono ───────────────────────────────────────────────────────

std::vector<float> AudioUtils::stereoToMono(const std::vector<float>& stereoData) {
    if (stereoData.size() % 2 != 0) {
        throw std::runtime_error("Stereo data must have even number of samples");
    }

    size_t numFrames = stereoData.size() / 2;
    std::vector<float> mono(numFrames);

    for (size_t i = 0; i < numFrames; ++i) {
        mono[i] = (stereoData[i * 2] + stereoData[i * 2 + 1]) * 0.5f;
    }

    return mono;
}

// ── normalize ──────────────────────────────────────────────────────────

std::vector<float> AudioUtils::normalize(const std::vector<float>& audio) {
    if (audio.empty()) return audio;

    constexpr float TARGET_PEAK = 0.85f;

    float maxVal = 0.0f;
    for (const auto& sample : audio) {
        maxVal = std::max(maxVal, std::abs(sample));
    }

    // Only attenuate if the audio exceeds the target peak; don't amplify quiet audio
    if (maxVal <= TARGET_PEAK) {
        return audio;
    }

    const float gain = TARGET_PEAK / maxVal;

    std::vector<float> normalized(audio.size());
    for (size_t i = 0; i < audio.size(); ++i) {
        normalized[i] = audio[i] * gain;
    }

    return normalized;
}

} // namespace pocket_tts
