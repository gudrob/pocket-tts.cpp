#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>

namespace pocket_tts {

/**
 * @brief Configuration for PocketTTS inference
 */
struct PocketTTSConfig {
    std::string modelsDir = "models/onnx";
    std::string tokenizerPath = "models/tokenizer.model";
    std::string precision = "int8";  // "int8" or "fp32"
    float temperature = 0.7f;
    int lsdSteps = 10;               // Flow matching steps
    int maxFrames = 500;
    int framesAfterEos = 3;
    bool verbose = true;
};

/**
 * @brief Pure C++ ONNX inference engine for Pocket TTS
 * 
 * Supports:
 * - Offline (batch) generation
 * - Voice cloning from audio files
 * - Temperature control for generation diversity
 */
class PocketTTS {
public:
    static constexpr int SAMPLE_RATE = 24000;
    static constexpr int SAMPLES_PER_FRAME = 1920;
    static constexpr float FRAME_DURATION = static_cast<float>(SAMPLES_PER_FRAME) / SAMPLE_RATE;
    
    /**
     * @brief Construct PocketTTS with configuration
     * @param config Configuration options
     */
    explicit PocketTTS(const PocketTTSConfig& config = PocketTTSConfig{});
    ~PocketTTS();
    
    // Disable copy
    PocketTTS(const PocketTTS&) = delete;
    PocketTTS& operator=(const PocketTTS&) = delete;
    
    // Enable move
    PocketTTS(PocketTTS&&) noexcept;
    PocketTTS& operator=(PocketTTS&&) noexcept;
    
    /**
     * @brief Generate audio from text with voice cloning
     * @param text Text to synthesize
     * @param voicePath Path to reference voice audio
     * @return Generated audio samples (float32, 24kHz)
     */
    std::vector<float> generate(
        const std::string& text,
        const std::string& voicePath
    );
    
    /**
     * @brief Encode a voice file to embeddings (for caching)
     * @param audioPath Path to voice audio file
     * @return Voice embeddings
     */
    std::vector<float> encodeVoice(const std::string& audioPath);
    
    /**
     * @brief Generate with pre-computed voice embeddings
     * @param text Text to synthesize
     * @param voiceEmbeddings Pre-computed voice embeddings
     * @param voiceEmbeddingShape Shape [batch, seq, dim]
     * @return Generated audio samples
     */
    std::vector<float> generateWithEmbeddings(
        const std::string& text,
        const std::vector<float>& voiceEmbeddings,
        const std::vector<int64_t>& voiceEmbeddingShape
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pocket_tts
