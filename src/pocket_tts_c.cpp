#include "pocket_tts/pocket_tts_c.h"
#include "pocket_tts/pocket_tts.hpp"
#include "pocket_tts/audio_utils.hpp"

#include <string>
#include <vector>
#include <memory>
#include <cstring>

// Thread-local error message
static thread_local std::string g_lastError;

// Version string
static const char* VERSION = "1.0.0";

// Internal voice structure
struct VoiceData {
    std::vector<float> embeddings;
    std::vector<int64_t> shape;
};

// Set error message
static void setError(const std::string& msg) {
    g_lastError = msg;
}

extern "C" {

POCKET_TTS_API PocketTTSHandle pocket_tts_create(const PocketTTSConfig* config) {
    try {
        pocket_tts::PocketTTSConfig cfg;
        
        if (config) {
            if (config->models_dir) cfg.modelsDir = config->models_dir;
            if (config->tokenizer_path) cfg.tokenizerPath = config->tokenizer_path;
            if (config->precision) cfg.precision = config->precision;
            if (config->temperature > 0) cfg.temperature = config->temperature;
            if (config->lsd_steps > 0) cfg.lsdSteps = config->lsd_steps;
            if (config->max_frames > 0) cfg.maxFrames = config->max_frames;
        }
        
        // Disable stdout logging for C API
        cfg.verbose = false;
        
        auto* tts = new pocket_tts::PocketTTS(cfg);
        return static_cast<PocketTTSHandle>(tts);
    } catch (const std::exception& e) {
        setError(std::string("Failed to create PocketTTS: ") + e.what());
        return nullptr;
    }
}

POCKET_TTS_API void pocket_tts_destroy(PocketTTSHandle handle) {
    if (handle) {
        delete static_cast<pocket_tts::PocketTTS*>(handle);
    }
}

POCKET_TTS_API VoiceHandle pocket_tts_encode_voice(PocketTTSHandle handle, const char* audio_path) {
    if (!handle || !audio_path) {
        setError("Invalid handle or audio path");
        return nullptr;
    }
    
    try {
        auto* tts = static_cast<pocket_tts::PocketTTS*>(handle);
        auto embeddings = tts->encodeVoice(audio_path);
        
        // Create voice data structure
        auto* voice = new VoiceData();
        voice->embeddings = std::move(embeddings);
        
        // Estimate shape from size (embeddings are [1, N, 1024])
        int64_t frames = static_cast<int64_t>(voice->embeddings.size()) / 1024;
        voice->shape = {1, frames, 1024};
        
        return static_cast<VoiceHandle>(voice);
    } catch (const std::exception& e) {
        setError(std::string("Failed to encode voice: ") + e.what());
        return nullptr;
    }
}

POCKET_TTS_API VoiceHandle pocket_tts_encode_voice_from_samples(
    PocketTTSHandle handle,
    const float* audio_data,
    int sample_count,
    int sample_rate
) {
    if (!handle || !audio_data || sample_count <= 0) {
        setError("Invalid parameters");
        return nullptr;
    }
    
    try {
        // Resample if needed
        std::vector<float> audio(audio_data, audio_data + sample_count);
        if (sample_rate != pocket_tts::AudioUtils::TARGET_SAMPLE_RATE) {
            audio = pocket_tts::AudioUtils::resample(audio, sample_rate, pocket_tts::AudioUtils::TARGET_SAMPLE_RATE);
        }
        
        // Save to temporary file (PocketTTS currently requires file path)
        // TODO: Add direct buffer encoding to PocketTTS class
        std::string tempPath = "/tmp/pocket_tts_temp_voice.wav";
        pocket_tts::AudioUtils::saveWav(tempPath, audio);
        
        return pocket_tts_encode_voice(handle, tempPath.c_str());
    } catch (const std::exception& e) {
        setError(std::string("Failed to encode voice from samples: ") + e.what());
        return nullptr;
    }
}

POCKET_TTS_API void pocket_tts_free_voice(VoiceHandle voice) {
    if (voice) {
        delete static_cast<VoiceData*>(voice);
    }
}

POCKET_TTS_API int pocket_tts_generate(
    PocketTTSHandle handle,
    const char* text,
    VoiceHandle voice,
    AudioResult* result
) {
    if (!handle || !text || !voice || !result) {
        setError("Invalid parameters");
        return -1;
    }
    
    try {
        auto* tts = static_cast<pocket_tts::PocketTTS*>(handle);
        auto* voiceData = static_cast<VoiceData*>(voice);
        
        // Generate using embeddings
        auto audio = tts->generateWithEmbeddings(text, voiceData->embeddings, voiceData->shape);
        
        // Allocate and copy result
        result->data = new float[audio.size()];
        std::memcpy(result->data, audio.data(), audio.size() * sizeof(float));
        result->sample_count = static_cast<int>(audio.size());
        result->sample_rate = 24000;  // PocketTTS sample rate
        
        return 0;
    } catch (const std::exception& e) {
        setError(std::string("Failed to generate: ") + e.what());
        return -1;
    }
}

POCKET_TTS_API void pocket_tts_free_audio(AudioResult* result) {
    if (result && result->data) {
        delete[] result->data;
        result->data = nullptr;
        result->sample_count = 0;
    }
}

POCKET_TTS_API const char* pocket_tts_get_last_error(void) {
    return g_lastError.c_str();
}

POCKET_TTS_API int pocket_tts_generate_streaming(
    PocketTTSHandle handle,
    const char* text,
    VoiceHandle voice,
    AudioChunkCallbackC callback,
    const StreamingConfig* config
) {
    if (!handle || !text || !voice || !callback) {
        setError("Invalid parameters");
        return -1;
    }
    
    try {
        auto* tts = static_cast<pocket_tts::PocketTTS*>(handle);
        auto* voiceData = static_cast<VoiceData*>(voice);
        
        // Capture user_data for the callback
        void* userData = config ? config->user_data : nullptr;
        
        // Create C++ callback wrapper that calls the C callback
        auto cppCallback = [callback, userData](const float* samples, int count, bool isFinal) {
            callback(samples, count, isFinal ? 1 : 0, userData);
        };
        
        // Build streaming config
        pocket_tts::StreamingConfig streamCfg;
        if (config && config->chunk_size_frames > 0) {
            streamCfg.chunkSizeFrames = config->chunk_size_frames;
        }
        streamCfg.enableCancellation = true;  // Always enable for C API
        
        // Call C++ streaming implementation
        int totalSamples = tts->generateStreaming(
            text,
            voiceData->embeddings,
            voiceData->shape,
            cppCallback,
            streamCfg
        );
        
        return totalSamples;
    } catch (const std::exception& e) {
        setError(std::string("Streaming failed: ") + e.what());
        return -1;
    }
}

POCKET_TTS_API void pocket_tts_cancel_streaming(PocketTTSHandle handle) {
    if (!handle) {
        return;
    }
    
    try {
        auto* tts = static_cast<pocket_tts::PocketTTS*>(handle);
        tts->cancelStreaming();
    } catch (...) {
        // Ignore errors during cancellation
    }
}

POCKET_TTS_API const char* pocket_tts_version(void) {
    return VERSION;
}

} // extern "C"
