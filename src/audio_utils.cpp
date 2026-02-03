#include "pocket_tts/audio_utils.hpp"

#include <sndfile.h>
#include <samplerate.h>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace pocket_tts {

std::vector<float> AudioUtils::loadWav(const std::string& filepath, int targetSampleRate) {
    SF_INFO sfInfo;
    std::memset(&sfInfo, 0, sizeof(sfInfo));
    
    SNDFILE* file = sf_open(filepath.c_str(), SFM_READ, &sfInfo);
    if (!file) {
        throw std::runtime_error("Failed to open audio file: " + filepath + 
                                 " - " + sf_strerror(nullptr));
    }
    
    // Read all samples
    std::vector<float> samples(sfInfo.frames * sfInfo.channels);
    sf_count_t framesRead = sf_readf_float(file, samples.data(), sfInfo.frames);
    sf_close(file);
    
    if (framesRead != sfInfo.frames) {
        throw std::runtime_error("Failed to read all frames from audio file");
    }
    
    // Convert to mono if stereo
    std::vector<float> monoSamples;
    if (sfInfo.channels == 1) {
        monoSamples = std::move(samples);
    } else {
        monoSamples = stereoToMono(samples);
    }
    
    // Resample if needed
    if (sfInfo.samplerate != targetSampleRate) {
        monoSamples = resample(monoSamples, sfInfo.samplerate, targetSampleRate);
    }
    
    // Normalize
    monoSamples = normalize(monoSamples);
    
    return monoSamples;
}

void AudioUtils::saveWav(const std::string& filepath, const std::vector<float>& audioData, int sampleRate) {
    SF_INFO sfInfo;
    std::memset(&sfInfo, 0, sizeof(sfInfo));
    
    sfInfo.samplerate = sampleRate;
    sfInfo.channels = 1;
    sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    
    SNDFILE* file = sf_open(filepath.c_str(), SFM_WRITE, &sfInfo);
    if (!file) {
        throw std::runtime_error("Failed to create audio file: " + filepath +
                                 " - " + sf_strerror(nullptr));
    }
    
    sf_count_t framesWritten = sf_writef_float(file, audioData.data(), 
                                                static_cast<sf_count_t>(audioData.size()));
    sf_close(file);
    
    if (framesWritten != static_cast<sf_count_t>(audioData.size())) {
        throw std::runtime_error("Failed to write all frames to audio file");
    }
}

std::vector<float> AudioUtils::resample(const std::vector<float>& input,
                                        int inputSampleRate,
                                        int outputSampleRate) {
    if (inputSampleRate == outputSampleRate) {
        return input;
    }
    
    double ratio = static_cast<double>(outputSampleRate) / inputSampleRate;
    size_t outputSize = static_cast<size_t>(input.size() * ratio) + 1;
    std::vector<float> output(outputSize);
    
    SRC_DATA srcData;
    srcData.data_in = input.data();
    srcData.input_frames = static_cast<long>(input.size());
    srcData.data_out = output.data();
    srcData.output_frames = static_cast<long>(outputSize);
    srcData.src_ratio = ratio;
    
    // Use high-quality sinc converter
    int error = src_simple(&srcData, SRC_SINC_BEST_QUALITY, 1);
    if (error) {
        throw std::runtime_error("Resampling failed: " + std::string(src_strerror(error)));
    }
    
    output.resize(srcData.output_frames_gen);
    return output;
}

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

std::vector<float> AudioUtils::normalize(const std::vector<float>& audio) {
    if (audio.empty()) return audio;
    
    float maxVal = 0.0f;
    for (const auto& sample : audio) {
        maxVal = std::max(maxVal, std::abs(sample));
    }
    
    if (maxVal <= 1.0f) {
        return audio;  // Already normalized
    }
    
    std::vector<float> normalized(audio.size());
    for (size_t i = 0; i < audio.size(); ++i) {
        normalized[i] = audio[i] / maxVal;
    }
    
    return normalized;
}

} // namespace pocket_tts
