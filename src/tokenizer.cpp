#include "pocket_tts/tokenizer.hpp"

#include <sentencepiece_processor.h>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace pocket_tts {

struct Tokenizer::Impl {
    sentencepiece::SentencePieceProcessor processor;
};

Tokenizer::Tokenizer(const std::string& modelPath) 
    : impl_(std::make_unique<Impl>()) {
    auto status = impl_->processor.Load(modelPath);
    if (!status.ok()) {
        throw std::runtime_error("Failed to load tokenizer: " + status.ToString());
    }
}

Tokenizer::~Tokenizer() = default;

Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

std::vector<int64_t> Tokenizer::encode(const std::string& text) const {
    // Preprocess text like Python version
    std::string processedText = text;
    
    // Trim whitespace
    auto start = processedText.find_first_not_of(" \t\n\r");
    auto end = processedText.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) {
        throw std::runtime_error("Text cannot be empty");
    }
    processedText = processedText.substr(start, end - start + 1);
    
    // Ensure proper punctuation at end
    char lastChar = processedText.back();
    if (std::isalnum(static_cast<unsigned char>(lastChar))) {
        processedText += ".";
    }
    
    // Capitalize first letter
    if (!processedText.empty() && std::islower(static_cast<unsigned char>(processedText[0]))) {
        processedText[0] = std::toupper(static_cast<unsigned char>(processedText[0]));
    }
    
    // Encode with SentencePiece
    std::vector<int> ids;
    auto status = impl_->processor.Encode(processedText, &ids);
    if (!status.ok()) {
        throw std::runtime_error("Tokenization failed: " + status.ToString());
    }
    
    // Convert to int64_t for ONNX
    std::vector<int64_t> ids64(ids.begin(), ids.end());
    return ids64;
}

int Tokenizer::vocabSize() const {
    return impl_->processor.GetPieceSize();
}

} // namespace pocket_tts
