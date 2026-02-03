#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace pocket_tts {

/**
 * @brief Audio utilities for loading and saving WAV files
 */
class AudioUtils {
public:
    /// Target sample rate for Pocket TTS (24kHz)
    static constexpr int TARGET_SAMPLE_RATE = 24000;
    
    /**
     * @brief Load a WAV file and resample to target sample rate
     * @param filepath Path to the WAV file
     * @param targetSampleRate Target sample rate (default: 24000 Hz)
     * @return Vector of audio samples (mono, float32)
     * @throws std::runtime_error if file cannot be loaded
     */
    static std::vector<float> loadWav(
        const std::string& filepath,
        int targetSampleRate = TARGET_SAMPLE_RATE
    );
    
    /**
     * @brief Save audio data to a WAV file
     * @param filepath Output file path
     * @param audioData Audio samples (mono, float32)
     * @param sampleRate Sample rate of the audio data
     * @throws std::runtime_error if file cannot be saved
     */
    static void saveWav(
        const std::string& filepath,
        const std::vector<float>& audioData,
        int sampleRate = TARGET_SAMPLE_RATE
    );
    
    /**
     * @brief Resample audio data to a different sample rate
     * @param input Input audio samples
     * @param inputSampleRate Sample rate of input
     * @param outputSampleRate Desired output sample rate
     * @return Resampled audio data
     */
    static std::vector<float> resample(
        const std::vector<float>& input,
        int inputSampleRate,
        int outputSampleRate
    );
    
    /**
     * @brief Convert stereo audio to mono by averaging channels
     * @param stereoData Interleaved stereo samples [L, R, L, R, ...]
     * @return Mono audio samples
     */
    static std::vector<float> stereoToMono(const std::vector<float>& stereoData);
    
    /**
     * @brief Normalize audio to [-1, 1] range
     * @param audio Audio samples
     * @return Normalized audio samples
     */
    static std::vector<float> normalize(const std::vector<float>& audio);
};

} // namespace pocket_tts
