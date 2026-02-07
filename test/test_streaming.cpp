#include "pocket_tts/pocket_tts.hpp"
#include "pocket_tts/audio_utils.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Pocket TTS Streaming Test ===" << std::endl;
    
    try {
        // Initialize TTS
        pocket_tts::PocketTTSConfig config;
        config.modelsDir = "models/onnx";
        config.tokenizerPath = "models/tokenizer.model";
        config.precision = "int8";
        
        pocket_tts::PocketTTS tts(config);
        
        // Encode voice
        std::cout << "Encoding voice..." << std::endl;
        auto voiceEmb = tts.encodeVoice("models/reference_sample.wav");
        std::vector<int64_t> voiceShape = {1, static_cast<int64_t>(voiceEmb.size()) / 1024, 1024};
        
        // Test streaming generation
        std::cout << "\n=== Testing Streaming Generation ===" << std::endl;
        
        std::vector<float> allAudio;
        int chunkCount = 0;
        
        auto callback = [&](const float* samples, int count, bool isFinal) {
            chunkCount++;
            std::cout << "Chunk " << chunkCount << ": " << count << " samples"
                      << (isFinal ? " [FINAL]" : "") << std::endl;
            
            // Collect all audio
            allAudio.insert(allAudio.end(), samples, samples + count);
        };
        
        pocket_tts::StreamingConfig streamCfg;
        streamCfg.chunkSizeFrames = 5;  // ~400ms chunks
        streamCfg.enableCancellation = false;
        
        int totalSamples = tts.generateStreaming(
            "Hello! This is a test of the streaming audio generation feature.",
            voiceEmb,
            voiceShape,
            callback,
            streamCfg
        );
        
        std::cout << "\nStreaming complete!" << std::endl;
        std::cout << "Total chunks: " << chunkCount << std::endl;
        std::cout << "Total samples: " << totalSamples << std::endl;
        std::cout << "Collected samples: " << allAudio.size() << std::endl;
        std::cout << "Duration: " << (totalSamples / 24000.0f) << "s" << std::endl;
        
        // Optionally save to file
        if (!allAudio.empty()) {
            pocket_tts::AudioUtils::saveWav("streaming_output.wav", allAudio, 24000);
            std::cout << "Saved to streaming_output.wav" << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
