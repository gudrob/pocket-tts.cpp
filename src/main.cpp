#include "pocket_tts/pocket_tts.hpp"
#include "pocket_tts/audio_utils.hpp"

#include <iostream>
#include <string>
#include <chrono>
#include <cstring>

void printUsage(const char* progName) {
    std::cout << "Pocket TTS - C++ Text-to-Speech with Voice Cloning\n\n";
    std::cout << "Usage: " << progName << " [options] <text> <voice_file> <output_file>\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  text         Text to synthesize\n";
    std::cout << "  voice_file   Reference voice audio file (WAV)\n";
    std::cout << "  output_file  Output audio file (WAV)\n\n";
    std::cout << "Options:\n";
    std::cout << "  --models-dir <path>   Path to models directory (default: models/onnx)\n";
    std::cout << "  --tokenizer <path>    Path to tokenizer.model (default: models/tokenizer.model)\n";
    std::cout << "  --precision <p>       Model precision: int8 or fp32 (default: int8)\n";
    std::cout << "  --temperature <t>     Sampling temperature (default: 0.7)\n";
    std::cout << "  --lsd-steps <n>       Flow matching steps (default: 10)\n";
    std::cout << "  --max-frames <n>      Maximum frames to generate (default: 500)\n";
    std::cout << "  -h, --help            Show this help message\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << progName << " \"Hello, world!\" models/reference_sample.wav output.wav\n";
}

int main(int argc, char* argv[]) {
    pocket_tts::PocketTTSConfig config;
    
    std::string text;
    std::string voiceFile;
    std::string outputFile;
    
    // Parse arguments
    int positionalCount = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--models-dir" && i + 1 < argc) {
            config.modelsDir = argv[++i];
        } else if (arg == "--tokenizer" && i + 1 < argc) {
            config.tokenizerPath = argv[++i];
        } else if (arg == "--precision" && i + 1 < argc) {
            config.precision = argv[++i];
        } else if (arg == "--temperature" && i + 1 < argc) {
            config.temperature = std::stof(argv[++i]);
        } else if (arg == "--lsd-steps" && i + 1 < argc) {
            config.lsdSteps = std::stoi(argv[++i]);
        } else if (arg == "--max-frames" && i + 1 < argc) {
            config.maxFrames = std::stoi(argv[++i]);
        } else if (arg[0] != '-') {
            // Positional argument
            switch (positionalCount) {
                case 0: text = arg; break;
                case 1: voiceFile = arg; break;
                case 2: outputFile = arg; break;
            }
            ++positionalCount;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        }
    }
    
    // Validate arguments
    if (text.empty() || voiceFile.empty() || outputFile.empty()) {
        std::cerr << "Error: Missing required arguments\n\n";
        printUsage(argv[0]);
        return 1;
    }
    
    std::cout << "=== Pocket TTS ===" << std::endl;
    std::cout << "Text: " << text << std::endl;
    std::cout << "Voice: " << voiceFile << std::endl;
    std::cout << "Output: " << outputFile << std::endl;
    std::cout << std::endl;
    
    try {
        // Initialize TTS engine
        std::cout << "Initializing..." << std::endl;
        auto startLoad = std::chrono::high_resolution_clock::now();
        pocket_tts::PocketTTS tts(config);
        auto endLoad = std::chrono::high_resolution_clock::now();
        auto loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endLoad - startLoad).count();
        std::cout << "Loaded in " << (loadTime / 1000.0f) << "s" << std::endl;
        std::cout << std::endl;
        
        // Generate audio
        auto audio = tts.generate(text, voiceFile);
        
        // Save output
        pocket_tts::AudioUtils::saveWav(outputFile, audio);
        std::cout << "Saved to: " << outputFile << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
