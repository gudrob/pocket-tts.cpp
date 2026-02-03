#pragma once

#include <string>
#include <vector>
#include <memory>

namespace pocket_tts {

/**
 * @brief SentencePiece tokenizer wrapper
 */
class Tokenizer {
public:
    /**
     * @brief Construct tokenizer from model file
     * @param modelPath Path to tokenizer.model file
     */
    explicit Tokenizer(const std::string& modelPath);
    ~Tokenizer();
    
    // Disable copy
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    
    // Enable move
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    
    /**
     * @brief Encode text to token IDs
     * @param text Input text
     * @return Vector of token IDs
     */
    std::vector<int64_t> encode(const std::string& text) const;
    
    /**
     * @brief Get vocabulary size
     */
    int vocabSize() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pocket_tts
