#include "pocket_tts/pocket_tts.hpp"
#include "pocket_tts/audio_utils.hpp"
#include "pocket_tts/tokenizer.hpp"

#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <numeric>

namespace pocket_tts {

// Helper to create tensor from vector
template<typename T>
Ort::Value createTensor(Ort::MemoryInfo& memInfo, 
                        std::vector<T>& data,
                        const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<T>(
        memInfo, data.data(), data.size(),
        shape.data(), shape.size()
    );
}

// State entry that can hold different types
struct StateEntry {
    std::vector<float> floatData;
    std::vector<int64_t> int64Data;
    std::vector<uint8_t> boolData;  // Using uint8_t because vector<bool> has no .data()
    std::vector<int64_t> shape;
    ONNXTensorElementDataType dtype;
    
    StateEntry() : dtype(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {}
};

struct PocketTTS::Impl {
    PocketTTSConfig config;
    
    // ONNX Runtime
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "PocketTTS"};
    Ort::SessionOptions sessionOptions;
    Ort::MemoryInfo memoryInfo{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    
    // Models
    std::unique_ptr<Ort::Session> mimiEncoder;
    std::unique_ptr<Ort::Session> textConditioner;
    std::unique_ptr<Ort::Session> flowLmMain;
    std::unique_ptr<Ort::Session> flowLmFlow;
    std::unique_ptr<Ort::Session> mimiDecoder;
    
    // Tokenizer
    std::unique_ptr<Tokenizer> tokenizer;
    
    // Pre-computed flow buffers
    std::vector<std::pair<float, float>> stBuffers;
    
    // RNG for temperature
    std::mt19937 rng{std::random_device{}()};
    
    // Voice cache
    std::map<std::string, std::pair<std::vector<float>, std::vector<int64_t>>> voiceCache;
    
    // Cancellation flag for streaming
    std::atomic<bool> cancelRequested{false};
    
    Impl(const PocketTTSConfig& cfg) : config(cfg) {
        sessionOptions.SetIntraOpNumThreads(3);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        loadModels();
        loadTokenizer();
        precomputeFlowBuffers();
    }
    
    void loadModels() {
        std::string suffix = (config.precision == "int8") ? "_int8" : "";
        
        std::string mimiEncoderPath = config.modelsDir + "/mimi_encoder.onnx";
        std::string textConditionerPath = config.modelsDir + "/text_conditioner.onnx";
        std::string flowLmMainPath = config.modelsDir + "/flow_lm_main" + suffix + ".onnx";
        std::string flowLmFlowPath = config.modelsDir + "/flow_lm_flow" + suffix + ".onnx";
        std::string mimiDecoderPath = config.modelsDir + "/mimi_decoder" + suffix + ".onnx";
        
        if (config.verbose) {
            std::cout << "Loading models from " << config.modelsDir << " (precision: " << config.precision << ")..." << std::endl;
        }
        
        if (config.loadVoiceEncoder) {
            mimiEncoder = std::make_unique<Ort::Session>(env, mimiEncoderPath.c_str(), sessionOptions);
        }
        textConditioner = std::make_unique<Ort::Session>(env, textConditionerPath.c_str(), sessionOptions);
        flowLmMain = std::make_unique<Ort::Session>(env, flowLmMainPath.c_str(), sessionOptions);
        flowLmFlow = std::make_unique<Ort::Session>(env, flowLmFlowPath.c_str(), sessionOptions);
        mimiDecoder = std::make_unique<Ort::Session>(env, mimiDecoderPath.c_str(), sessionOptions);
        
        if (config.verbose) {
            std::cout << "Models loaded successfully." << std::endl;
        }
    }
    
    void loadTokenizer() {
        tokenizer = std::make_unique<Tokenizer>(config.tokenizerPath);
        if (config.verbose) {
            std::cout << "Tokenizer loaded (vocab size: " << tokenizer->vocabSize() << ")." << std::endl;
        }
    }
    
    void precomputeFlowBuffers() {
        float dt = 1.0f / config.lsdSteps;
        stBuffers.clear();
        for (int j = 0; j < config.lsdSteps; ++j) {
            float s = static_cast<float>(j) / config.lsdSteps;
            float t = s + dt;
            stBuffers.emplace_back(s, t);
        }
    }
    
    // Initialize state tensors for stateful model with proper types
    std::map<std::string, StateEntry> initState(Ort::Session& session) {
        std::map<std::string, StateEntry> state;
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t numInputs = session.GetInputCount();
        for (size_t i = 0; i < numInputs; ++i) {
            auto namePtr = session.GetInputNameAllocated(i, allocator);
            std::string name = namePtr.get();
            
            if (name.find("state_") == 0) {
                auto typeInfo = session.GetInputTypeInfo(i);
                auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
                auto shape = tensorInfo.GetShape();
                auto dtype = tensorInfo.GetElementType();
                
                // Replace dynamic dims with their default values from shape
                for (auto& dim : shape) {
                    if (dim < 0) dim = 0;
                }
                
                // Calculate total size
                int64_t totalSize = 1;
                for (auto dim : shape) {
                    if (dim > 0) totalSize *= dim;
                }
                if (totalSize <= 0) totalSize = 1;
                
                StateEntry entry;
                entry.shape = shape;
                entry.dtype = dtype;
                
                switch (dtype) {
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                        entry.floatData.resize(totalSize, 0.0f);
                        break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                        entry.int64Data.resize(totalSize, 0);
                        break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                        entry.boolData.resize(totalSize, 0);
                        break;
                    default:
                        // Default to float
                        entry.floatData.resize(totalSize, 0.0f);
                        entry.dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
                        break;
                }
                
                state[name] = std::move(entry);
            }
        }
        
        return state;
    }
    
    // Create Ort::Value from StateEntry
    Ort::Value createStateValue(StateEntry& entry) {
        switch (entry.dtype) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                return Ort::Value::CreateTensor<float>(
                    memoryInfo, entry.floatData.data(), entry.floatData.size(),
                    entry.shape.data(), entry.shape.size());
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                return Ort::Value::CreateTensor<int64_t>(
                    memoryInfo, entry.int64Data.data(), entry.int64Data.size(),
                    entry.shape.data(), entry.shape.size());
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                // Cast uint8_t* to bool* - they're compatible in memory
                return Ort::Value::CreateTensor<bool>(
                    memoryInfo, reinterpret_cast<bool*>(entry.boolData.data()), entry.boolData.size(),
                    entry.shape.data(), entry.shape.size());
            default:
                return Ort::Value::CreateTensor<float>(
                    memoryInfo, entry.floatData.data(), entry.floatData.size(),
                    entry.shape.data(), entry.shape.size());
        }
    }
    
    // Update state from output tensor
    void updateStateFromOutput(StateEntry& entry, Ort::Value& outputTensor) {
        auto info = outputTensor.GetTensorTypeAndShapeInfo();
        entry.shape = info.GetShape();
        size_t count = info.GetElementCount();
        
        switch (entry.dtype) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                float* data = outputTensor.GetTensorMutableData<float>();
                entry.floatData.assign(data, data + count);
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                int64_t* data = outputTensor.GetTensorMutableData<int64_t>();
                entry.int64Data.assign(data, data + count);
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
                bool* data = outputTensor.GetTensorMutableData<bool>();
                entry.boolData.resize(count);
                for (size_t i = 0; i < count; ++i) {
                    entry.boolData[i] = data[i] ? 1 : 0;
                }
                break;
            }
            default:
                break;
        }
    }
    
    std::vector<float> encodeVoice(const std::string& audioPath) {
        // Check cache
        auto it = voiceCache.find(audioPath);
        if (it != voiceCache.end()) {
            return it->second.first;
        }

        if (!mimiEncoder) {
            throw std::runtime_error("Voice encoder is disabled (loadVoiceEncoder=false).");
        }

        // Load audio
        auto audio = AudioUtils::loadWav(audioPath, AudioUtils::TARGET_SAMPLE_RATE);
        // Long reference clips explode KV-cache memory in the autoregressive pass.
        // A short reference (few seconds) is enough for stable voice conditioning.
        constexpr size_t MAX_REFERENCE_SAMPLES = static_cast<size_t>(AudioUtils::TARGET_SAMPLE_RATE) * 5;
        if (audio.size() > MAX_REFERENCE_SAMPLES) {
            audio.resize(MAX_REFERENCE_SAMPLES);
        }
        
        // Prepare input: [1, 1, samples]
        std::vector<int64_t> audioShape = {1, 1, static_cast<int64_t>(audio.size())};
        auto audioTensor = createTensor(memoryInfo, audio, audioShape);
        
        // Run MIMI encoder
        const char* inputNames[] = {"audio"};
        const char* outputNames[] = {"latents"};
        
        auto outputs = mimiEncoder->Run(
            Ort::RunOptions{nullptr},
            inputNames, &audioTensor, 1,
            outputNames, 1
        );
        
        // Get embeddings
        auto& embTensor = outputs[0];
        auto embInfo = embTensor.GetTensorTypeAndShapeInfo();
        auto embShape = embInfo.GetShape();
        size_t embSize = embInfo.GetElementCount();
        
        float* embData = embTensor.GetTensorMutableData<float>();
        std::vector<float> embeddings(embData, embData + embSize);
        
        // Ensure shape is [1, N, 1024]
        std::vector<int64_t> finalShape = embShape;
        while (finalShape.size() > 3) {
            finalShape.erase(finalShape.begin());
        }
        if (finalShape.size() < 3) {
            finalShape.insert(finalShape.begin(), 1);
        }
        
        // Cache
        voiceCache[audioPath] = {embeddings, finalShape};
        
        return embeddings;
    }
    
    std::vector<float> runTextConditioner(const std::vector<int64_t>& tokenIds) {
        // Prepare input: [1, seq_len]
        std::vector<int64_t> ids = tokenIds;  // Copy for non-const tensor
        std::vector<int64_t> idsShape = {1, static_cast<int64_t>(ids.size())};
        auto idsTensor = createTensor(memoryInfo, ids, idsShape);
        
        const char* inputNames[] = {"token_ids"};
        const char* outputNames[] = {"embeddings"};
        
        auto outputs = textConditioner->Run(
            Ort::RunOptions{nullptr},
            inputNames, &idsTensor, 1,
            outputNames, 1
        );
        
        auto& embTensor = outputs[0];
        auto embInfo = embTensor.GetTensorTypeAndShapeInfo();
        size_t embSize = embInfo.GetElementCount();
        
        float* embData = embTensor.GetTensorMutableData<float>();
        return std::vector<float>(embData, embData + embSize);
    }
    
    // Run flow LM main model - returns conditioning and EOS logit
    std::pair<std::vector<float>, float> runFlowLmMainStep(
        const std::vector<float>& sequence,
        const std::vector<int64_t>& seqShape,
        const std::vector<float>& textEmb,
        const std::vector<int64_t>& textShape,
        std::map<std::string, StateEntry>& state
    ) {
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Build inputs
        std::vector<Ort::Value> inputTensors;
        std::vector<const char*> inputNames;
        std::vector<std::string> inputNameStrs;
        
        // Sequence input
        std::vector<float> seqCopy = sequence;
        inputTensors.push_back(createTensor(memoryInfo, seqCopy, seqShape));
        inputNameStrs.push_back("sequence");
        
        // Text embeddings input
        std::vector<float> textCopy = textEmb;
        inputTensors.push_back(createTensor(memoryInfo, textCopy, textShape));
        inputNameStrs.push_back("text_embeddings");
        
        // State inputs - in sorted order for consistency
        std::vector<std::string> stateNames;
        for (auto& [name, entry] : state) {
            stateNames.push_back(name);
        }
        std::sort(stateNames.begin(), stateNames.end(), [](const std::string& a, const std::string& b) {
            int idxA = std::stoi(a.substr(6));
            int idxB = std::stoi(b.substr(6));
            return idxA < idxB;
        });
        
        for (const auto& name : stateNames) {
            inputTensors.push_back(createStateValue(state[name]));
            inputNameStrs.push_back(name);
        }
        
        // Convert to const char*
        for (const auto& name : inputNameStrs) {
            inputNames.push_back(name.c_str());
        }
        
        // Get output names
        size_t numOutputs = flowLmMain->GetOutputCount();
        std::vector<std::string> outputNameStrs;
        std::vector<const char*> outputNames;
        for (size_t i = 0; i < numOutputs; ++i) {
            auto namePtr = flowLmMain->GetOutputNameAllocated(i, allocator);
            outputNameStrs.push_back(namePtr.get());
        }
        for (const auto& name : outputNameStrs) {
            outputNames.push_back(name.c_str());
        }
        
        // Run
        auto outputs = flowLmMain->Run(
            Ort::RunOptions{nullptr},
            inputNames.data(), inputTensors.data(), inputTensors.size(),
            outputNames.data(), outputNames.size()
        );
        
        // Get conditioning (output 0)
        auto& condTensor = outputs[0];
        auto condInfo = condTensor.GetTensorTypeAndShapeInfo();
        size_t condSize = condInfo.GetElementCount();
        float* condData = condTensor.GetTensorMutableData<float>();
        std::vector<float> conditioning(condData, condData + condSize);
        
        // Get EOS logit (output 1)
        float eosLogit = outputs[1].GetTensorMutableData<float>()[0];
        
        // Update state from outputs
        for (size_t i = 2; i < outputs.size(); ++i) {
            const std::string& outName = outputNameStrs[i];
            if (outName.find("out_state_") == 0) {
                int idx = std::stoi(outName.substr(10));
                std::string stateName = "state_" + std::to_string(idx);
                
                if (state.find(stateName) != state.end()) {
                    updateStateFromOutput(state[stateName], outputs[i]);
                }
            }
        }
        
        return {conditioning, eosLogit};
    }
    
    // Run flow network for Euler integration
    std::vector<float> runFlowLmFlow(
        const std::vector<float>& conditioning,
        float s, float t,
        const std::vector<float>& x
    ) {
        std::vector<float> cCopy = conditioning;
        std::vector<float> sCopy = {s};
        std::vector<float> tCopy = {t};
        std::vector<float> xCopy = x;
        
        std::vector<int64_t> cShape = {1, static_cast<int64_t>(conditioning.size())};
        std::vector<int64_t> stShape = {1, 1};
        std::vector<int64_t> xShape = {1, 32};
        
        auto cTensor = createTensor(memoryInfo, cCopy, cShape);
        auto sTensor = createTensor(memoryInfo, sCopy, stShape);
        auto tTensor = createTensor(memoryInfo, tCopy, stShape);
        auto xTensor = createTensor(memoryInfo, xCopy, xShape);
        
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(cTensor));
        inputs.push_back(std::move(sTensor));
        inputs.push_back(std::move(tTensor));
        inputs.push_back(std::move(xTensor));
        
        const char* inputNames[] = {"c", "s", "t", "x"};
        const char* outputNames[] = {"flow_dir"};
        
        auto outputs = flowLmFlow->Run(
            Ort::RunOptions{nullptr},
            inputNames, inputs.data(), 4,
            outputNames, 1
        );
        
        auto& outTensor = outputs[0];
        float* outData = outTensor.GetTensorMutableData<float>();
        return std::vector<float>(outData, outData + 32);
    }
    
    // Decode latents to audio
    std::vector<float> decodeLatents(const std::vector<std::vector<float>>& latents) {
        if (latents.empty()) return {};
        
        // Initialize decoder state
        auto state = initState(*mimiDecoder);
        
        std::vector<float> audioChunks;
        const int chunkSize = 15;  // Frames per chunk
        
        for (size_t i = 0; i < latents.size(); i += chunkSize) {
            size_t end = std::min(i + chunkSize, latents.size());
            size_t numFrames = end - i;
            
            // Combine frames: [1, numFrames, 32]
            std::vector<float> chunk(numFrames * 32);
            for (size_t j = 0; j < numFrames; ++j) {
                std::copy(latents[i + j].begin(), latents[i + j].end(), 
                          chunk.begin() + j * 32);
            }
            
            std::vector<int64_t> chunkShape = {1, static_cast<int64_t>(numFrames), 32};
            
            // Build inputs
            Ort::AllocatorWithDefaultOptions allocator;
            std::vector<Ort::Value> inputTensors;
            std::vector<const char*> inputNames;
            std::vector<std::string> inputNameStrs;
            
            inputTensors.push_back(createTensor(memoryInfo, chunk, chunkShape));
            inputNameStrs.push_back("latent");
            
            // State inputs in sorted order
            std::vector<std::string> stateNames;
            for (auto& [name, entry] : state) {
                stateNames.push_back(name);
            }
            std::sort(stateNames.begin(), stateNames.end(), [](const std::string& a, const std::string& b) {
                int idxA = std::stoi(a.substr(6));
                int idxB = std::stoi(b.substr(6));
                return idxA < idxB;
            });
            
            for (const auto& name : stateNames) {
                inputTensors.push_back(createStateValue(state[name]));
                inputNameStrs.push_back(name);
            }
            
            for (const auto& name : inputNameStrs) {
                inputNames.push_back(name.c_str());
            }
            
            // Get output names
            size_t numOutputs = mimiDecoder->GetOutputCount();
            std::vector<std::string> outputNameStrs;
            std::vector<const char*> outputNames;
            for (size_t o = 0; o < numOutputs; ++o) {
                auto namePtr = mimiDecoder->GetOutputNameAllocated(o, allocator);
                outputNameStrs.push_back(namePtr.get());
            }
            for (const auto& name : outputNameStrs) {
                outputNames.push_back(name.c_str());
            }
            
            // Run decoder
            auto outputs = mimiDecoder->Run(
                Ort::RunOptions{nullptr},
                inputNames.data(), inputTensors.data(), inputTensors.size(),
                outputNames.data(), outputNames.size()
            );
            
            // Get audio output
            auto& audioTensor = outputs[0];
            auto audioInfo = audioTensor.GetTensorTypeAndShapeInfo();
            size_t audioSize = audioInfo.GetElementCount();
            float* audioData = audioTensor.GetTensorMutableData<float>();
            audioChunks.insert(audioChunks.end(), audioData, audioData + audioSize);
            
            // Update state
            for (size_t k = 1; k < outputs.size(); ++k) {
                const std::string& outName = outputNameStrs[k];
                if (outName.find("out_state_") == 0) {
                    int idx = std::stoi(outName.substr(10));
                    std::string stateName = "state_" + std::to_string(idx);
                    
                    if (state.find(stateName) != state.end()) {
                        updateStateFromOutput(state[stateName], outputs[k]);
                    }
                }
            }
        }
        
        return audioChunks;
    }
    
    std::vector<float> generate(const std::string& text, const std::string& voicePath) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Get voice embeddings
        auto voiceEmb = encodeVoice(voicePath);
        auto voiceShape = voiceCache[voicePath].second;
        
        // Tokenize text
        auto tokenIds = tokenizer->encode(text);
        
        // Get text embeddings
        auto textEmb = runTextConditioner(tokenIds);
        std::vector<int64_t> textShape = {1, static_cast<int64_t>(tokenIds.size()), 1024};
        
        // Initialize flow LM state
        auto lmState = initState(*flowLmMain);
        
        // Empty tensors for conditioning passes
        std::vector<float> emptySeq;
        std::vector<int64_t> emptySeqShape = {1, 0, 32};
        std::vector<float> emptyText;
        std::vector<int64_t> emptyTextShape = {1, 0, 1024};
        
        // Voice conditioning pass
        auto [_, __] = runFlowLmMainStep(emptySeq, emptySeqShape, voiceEmb, voiceShape, lmState);
        
        // Text conditioning pass
        std::tie(_, __) = runFlowLmMainStep(emptySeq, emptySeqShape, textEmb, textShape, lmState);
        
        // Autoregressive generation
        std::vector<std::vector<float>> allLatents;
        std::vector<float> current(32, std::nanf(""));
        std::vector<int64_t> currentShape = {1, 1, 32};
        
        float dt = 1.0f / config.lsdSteps;
        int eosStep = -1;
        
        if (config.verbose) {
            std::cout << "Generating latents..." << std::flush;
        }
        
        for (int step = 0; step < config.maxFrames; ++step) {
            // Run main model
            auto [conditioning, eosLogit] = runFlowLmMainStep(
                current, currentShape, emptyText, emptyTextShape, lmState
            );
            
            // Check EOS
            if (eosLogit > -4.0f && eosStep < 0) {
                eosStep = step;
            }
            
            // Stop after frames_after_eos
            if (eosStep >= 0 && step >= eosStep + config.framesAfterEos) {
                break;
            }
            
            // Flow matching with Euler integration
            std::vector<float> x(32);
            if (config.temperature > 0) {
                float stddev = std::sqrt(config.temperature);
                std::normal_distribution<float> dist(0.0f, stddev);
                for (auto& val : x) {
                    val = dist(rng);
                }
            }
            
            for (int j = 0; j < config.lsdSteps; ++j) {
                float s = stBuffers[j].first;
                float t = stBuffers[j].second;
                auto flowOut = runFlowLmFlow(conditioning, s, t, x);
                for (size_t k = 0; k < 32; ++k) {
                    x[k] += flowOut[k] * dt;
                }
            }
            
            allLatents.push_back(x);
            current = x;
            
            if ((step + 1) % 10 == 0 && config.verbose) {
                std::cout << "." << std::flush;
            }
        }
        
        if (config.verbose) {
            std::cout << " " << allLatents.size() << " frames" << std::endl;
        }
        
        // Decode to audio
        if (config.verbose) {
            std::cout << "Decoding audio..." << std::flush;
        }
        auto audio = decodeLatents(allLatents);
        if (config.verbose) {
            std::cout << " done" << std::endl;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        float audioDuration = static_cast<float>(audio.size()) / SAMPLE_RATE;
        float rtfx = audioDuration / (duration / 1000.0f);
        
        if (config.verbose) {
            std::cout << "Generated " << audioDuration << "s audio in " 
                      << (duration / 1000.0f) << "s (RTFx: " << rtfx << "x)" << std::endl;
        }
        
        return audio;
    }
};

PocketTTS::PocketTTS(const PocketTTSConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

PocketTTS::~PocketTTS() = default;

PocketTTS::PocketTTS(PocketTTS&&) noexcept = default;
PocketTTS& PocketTTS::operator=(PocketTTS&&) noexcept = default;

std::vector<float> PocketTTS::generate(const std::string& text, const std::string& voicePath) {
    return impl_->generate(text, voicePath);
}

std::vector<float> PocketTTS::encodeVoice(const std::string& audioPath) {
    return impl_->encodeVoice(audioPath);
}

std::vector<float> PocketTTS::generateWithEmbeddings(
    const std::string& text,
    const std::vector<float>& voiceEmbeddings,
    const std::vector<int64_t>& voiceEmbeddingShape
) {
    // Store in cache temporarily
    impl_->voiceCache["__embeddings__"] = {voiceEmbeddings, voiceEmbeddingShape};
    return impl_->generate(text, "__embeddings__");
}

int PocketTTS::generateStreaming(
    const std::string& text,
    const std::vector<float>& voiceEmbeddings,
    const std::vector<int64_t>& voiceEmbeddingShape,
    AudioChunkCallback callback,
    const StreamingConfig& streamConfig
) {
    if (!callback) {
        throw std::invalid_argument("Callback must be provided");
    }
    
    // Reset cancellation flag
    impl_->cancelRequested = false;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Tokenize text
    auto tokenIds = impl_->tokenizer->encode(text);
    
    // Get text embeddings
    auto textEmb = impl_->runTextConditioner(tokenIds);
    std::vector<int64_t> textShape = {1, static_cast<int64_t>(tokenIds.size()), 1024};
    
    // Initialize flow LM state
    auto lmState = impl_->initState(*impl_->flowLmMain);
    
    // Empty tensors for conditioning passes
    std::vector<float> emptySeq;
    std::vector<int64_t> emptySeqShape = {1, 0, 32};
    std::vector<float> emptyText;
    std::vector<int64_t> emptyTextShape = {1, 0, 1024};
    
    // Voice conditioning pass
    auto [_, __] = impl_->runFlowLmMainStep(emptySeq, emptySeqShape, 
                                            const_cast<std::vector<float>&>(voiceEmbeddings), 
                                            const_cast<std::vector<int64_t>&>(voiceEmbeddingShape), 
                                            lmState);
    
    // Text conditioning pass
    std::tie(_, __) = impl_->runFlowLmMainStep(emptySeq, emptySeqShape, textEmb, textShape, lmState);
    
    // Autoregressive generation with streaming
    std::vector<std::vector<float>> pendingLatents;
    std::vector<float> current(32, std::nanf(""));
    std::vector<int64_t> currentShape = {1, 1, 32};
    
    float dt = 1.0f / impl_->config.lsdSteps;
    int eosStep = -1;
    int totalSamples = 0;
    
    // Initialize decoder state for streaming
    auto decoderState = impl_->initState(*impl_->mimiDecoder);
    
    if (impl_->config.verbose) {
        std::cout << "Streaming latent generation..." << std::flush;
    }
    
    for (int step = 0; step < impl_->config.maxFrames; ++step) {
        // Check cancellation
        if (streamConfig.enableCancellation && impl_->cancelRequested) {
            if (impl_->config.verbose) {
                std::cout << " cancelled" << std::endl;
            }
            break;
        }
        
        // Run main model
        auto [conditioning, eosLogit] = impl_->runFlowLmMainStep(
            current, currentShape, emptyText, emptyTextShape, lmState
        );
        
        // Check EOS
        if (eosLogit > -4.0f && eosStep < 0) {
            eosStep = step;
        }
        
        // Stop after frames_after_eos
        if (eosStep >= 0 && step >= eosStep + impl_->config.framesAfterEos) {
            break;
        }
        
        // Flow matching with Euler integration
        std::vector<float> x(32);
        if (impl_->config.temperature > 0) {
            float stddev = std::sqrt(impl_->config.temperature);
            std::normal_distribution<float> dist(0.0f, stddev);
            for (auto& val : x) {
                val = dist(impl_->rng);
            }
        }
        
        for (int j = 0; j < impl_->config.lsdSteps; ++j) {
            float s = impl_->stBuffers[j].first;
            float t = impl_->stBuffers[j].second;
            auto flowOut = impl_->runFlowLmFlow(conditioning, s, t, x);
            for (size_t k = 0; k < 32; ++k) {
                x[k] += flowOut[k] * dt;
            }
        }
        
        pendingLatents.push_back(x);
        current = x;
        
        // Decode and stream when we have enough frames
        if (static_cast<int>(pendingLatents.size()) >= streamConfig.chunkSizeFrames || 
            (eosStep >= 0 && step >= eosStep + impl_->config.framesAfterEos)) {
            
            // Decode pending latents
            std::vector<float> chunkAudio;
            
            for (size_t i = 0; i < pendingLatents.size(); i += 15) {
                size_t end = std::min(i + 15, pendingLatents.size());
                size_t numFrames = end - i;
                
                // Combine frames: [1, numFrames, 32]
                std::vector<float> chunk(numFrames * 32);
                for (size_t j = 0; j < numFrames; ++j) {
                    std::copy(pendingLatents[i + j].begin(), pendingLatents[i + j].end(), 
                              chunk.begin() + j * 32);
                }
                
                std::vector<int64_t> chunkShape = {1, static_cast<int64_t>(numFrames), 32};
                
                // Build decoder inputs
                Ort::AllocatorWithDefaultOptions allocator;
                std::vector<Ort::Value> inputTensors;
                std::vector<const char*> inputNames;
                std::vector<std::string> inputNameStrs;
                
                inputTensors.push_back(createTensor(impl_->memoryInfo, chunk, chunkShape));
                inputNameStrs.push_back("latent");
                
                // State inputs in sorted order
                std::vector<std::string> stateNames;
                for (auto& [name, entry] : decoderState) {
                    stateNames.push_back(name);
                }
                std::sort(stateNames.begin(), stateNames.end(), [](const std::string& a, const std::string& b) {
                    int idxA = std::stoi(a.substr(6));
                    int idxB = std::stoi(b.substr(6));
                    return idxA < idxB;
                });
                
                for (const auto& name : stateNames) {
                    inputTensors.push_back(impl_->createStateValue(decoderState[name]));
                    inputNameStrs.push_back(name);
                }
                
                for (const auto& name : inputNameStrs) {
                    inputNames.push_back(name.c_str());
                }
                
                // Get output names
                size_t numOutputs = impl_->mimiDecoder->GetOutputCount();
                std::vector<std::string> outputNameStrs;
                std::vector<const char*> outputNames;
                for (size_t o = 0; o < numOutputs; ++o) {
                    auto namePtr = impl_->mimiDecoder->GetOutputNameAllocated(o, allocator);
                    outputNameStrs.push_back(namePtr.get());
                }
                for (const auto& name : outputNameStrs) {
                    outputNames.push_back(name.c_str());
                }
                
                // Run decoder
                auto outputs = impl_->mimiDecoder->Run(
                    Ort::RunOptions{nullptr},
                    inputNames.data(), inputTensors.data(), inputTensors.size(),
                    outputNames.data(), outputNames.size()
                );
                
                // Get audio output
                auto& audioTensor = outputs[0];
                auto audioInfo = audioTensor.GetTensorTypeAndShapeInfo();
                size_t audioSize = audioInfo.GetElementCount();
                float* audioData = audioTensor.GetTensorMutableData<float>();
                chunkAudio.insert(chunkAudio.end(), audioData, audioData + audioSize);
                
                // Update decoder state
                for (size_t k = 1; k < outputs.size(); ++k) {
                    const std::string& outName = outputNameStrs[k];
                    if (outName.find("out_state_") == 0) {
                        int idx = std::stoi(outName.substr(10));
                        std::string stateName = "state_" + std::to_string(idx);
                        
                        if (decoderState.find(stateName) != decoderState.end()) {
                            impl_->updateStateFromOutput(decoderState[stateName], outputs[k]);
                        }
                    }
                }
            }
            
            // Call user callback with audio chunk
            bool isFinal = (eosStep >= 0 && step >= eosStep + impl_->config.framesAfterEos);
            callback(chunkAudio.data(), static_cast<int>(chunkAudio.size()), isFinal);
            
            totalSamples += static_cast<int>(chunkAudio.size());
            pendingLatents.clear();
            
            if (impl_->config.verbose && !isFinal) {
                std::cout << "." << std::flush;
            }
        }
        
        // Progress callback
        if (streamConfig.onProgress) {
            streamConfig.onProgress(step + 1, 0);  // 0 = unknown total
        }
    }
    
    // Decode and send any remaining latents
    if (!pendingLatents.empty() && !impl_->cancelRequested) {
        std::vector<float> finalAudio;
        
        for (size_t i = 0; i < pendingLatents.size(); i += 15) {
            size_t end = std::min(i + 15, pendingLatents.size());
            size_t numFrames = end - i;
            
            std::vector<float> chunk(numFrames * 32);
            for (size_t j = 0; j < numFrames; ++j) {
                std::copy(pendingLatents[i + j].begin(), pendingLatents[i + j].end(), 
                          chunk.begin() + j * 32);
            }
            
            std::vector<int64_t> chunkShape = {1, static_cast<int64_t>(numFrames), 32};
            
            Ort::AllocatorWithDefaultOptions allocator;
            std::vector<Ort::Value> inputTensors;
            std::vector<const char*> inputNames;
            std::vector<std::string> inputNameStrs;
            
            inputTensors.push_back(createTensor(impl_->memoryInfo, chunk, chunkShape));
            inputNameStrs.push_back("latent");
            
            std::vector<std::string> stateNames;
            for (auto& [name, entry] : decoderState) {
                stateNames.push_back(name);
            }
            std::sort(stateNames.begin(), stateNames.end(), [](const std::string& a, const std::string& b) {
                int idxA = std::stoi(a.substr(6));
                int idxB = std::stoi(b.substr(6));
                return idxA < idxB;
            });
            
            for (const auto& name : stateNames) {
                inputTensors.push_back(impl_->createStateValue(decoderState[name]));
                inputNameStrs.push_back(name);
            }
            
            for (const auto& name : inputNameStrs) {
                inputNames.push_back(name.c_str());
            }
            
            size_t numOutputs = impl_->mimiDecoder->GetOutputCount();
            std::vector<std::string> outputNameStrs;
            std::vector<const char*> outputNames;
            for (size_t o = 0; o < numOutputs; ++o) {
                auto namePtr = impl_->mimiDecoder->GetOutputNameAllocated(o, allocator);
                outputNameStrs.push_back(namePtr.get());
            }
            for (const auto& name : outputNameStrs) {
                outputNames.push_back(name.c_str());
            }
            
            auto outputs = impl_->mimiDecoder->Run(
                Ort::RunOptions{nullptr},
                inputNames.data(), inputTensors.data(), inputTensors.size(),
                outputNames.data(), outputNames.size()
            );
            
            auto& audioTensor = outputs[0];
            auto audioInfo = audioTensor.GetTensorTypeAndShapeInfo();
            size_t audioSize = audioInfo.GetElementCount();
            float* audioData = audioTensor.GetTensorMutableData<float>();
            finalAudio.insert(finalAudio.end(), audioData, audioData + audioSize);
        }
        
        callback(finalAudio.data(), static_cast<int>(finalAudio.size()), true);
        totalSamples += static_cast<int>(finalAudio.size());
    }
    
    if (impl_->config.verbose) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float audioDuration = static_cast<float>(totalSamples) / SAMPLE_RATE;
        float rtfx = audioDuration / (duration / 1000.0f);
        
        std::cout << " done" << std::endl;
        std::cout << "Streamed " << audioDuration << "s audio in " 
                  << (duration / 1000.0f) << "s (RTFx: " << rtfx << "x)" << std::endl;
    }
    
    return totalSamples;
}

void PocketTTS::cancelStreaming() {
    impl_->cancelRequested = true;
}

} // namespace pocket_tts
