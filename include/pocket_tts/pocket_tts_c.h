#ifndef POCKET_TTS_C_H
#define POCKET_TTS_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Platform-specific export macros */
#ifdef _WIN32
    #ifdef POCKET_TTS_BUILDING_DLL
        #define POCKET_TTS_API __declspec(dllexport)
    #else
        #define POCKET_TTS_API __declspec(dllimport)
    #endif
#else
    #define POCKET_TTS_API __attribute__((visibility("default")))
#endif

/* Opaque handles */
typedef void* PocketTTSHandle;
typedef void* VoiceHandle;

/* Configuration */
typedef struct {
    const char* models_dir;      /* Default: "models/onnx" */
    const char* tokenizer_path;  /* Default: "models/tokenizer.model" */
    const char* precision;       /* "int8" or "fp32", default: "int8" */
    float temperature;           /* 0.0-1.0, default: 0.7 */
    int lsd_steps;              /* Flow matching steps, default: 10 */
    int max_frames;             /* Max frames to generate, default: 500 */
} PocketTTSConfig;

/* Result structure for audio */
typedef struct {
    float* data;                /* Audio samples (24kHz mono) */
    int sample_count;           /* Number of samples */
    int sample_rate;            /* Always 24000 */
} AudioResult;

/*
 * Create a new PocketTTS instance.
 * 
 * @param config Configuration options. Pass NULL for defaults.
 * @return Handle to the instance, or NULL on error.
 */
POCKET_TTS_API PocketTTSHandle pocket_tts_create(const PocketTTSConfig* config);

/*
 * Destroy a PocketTTS instance and free resources.
 */
POCKET_TTS_API void pocket_tts_destroy(PocketTTSHandle handle);

/*
 * Encode a voice from an audio file.
 * The returned handle can be reused for multiple generations.
 *
 * @param handle PocketTTS instance
 * @param audio_path Path to WAV file
 * @return Voice handle, or NULL on error
 */
POCKET_TTS_API VoiceHandle pocket_tts_encode_voice(PocketTTSHandle handle, const char* audio_path);

/*
 * Encode a voice from raw audio samples.
 *
 * @param handle PocketTTS instance
 * @param audio_data Audio samples (mono float)
 * @param sample_count Number of samples
 * @param sample_rate Sample rate of input (will be resampled to 24kHz)
 * @return Voice handle, or NULL on error
 */
POCKET_TTS_API VoiceHandle pocket_tts_encode_voice_from_samples(
    PocketTTSHandle handle,
    const float* audio_data,
    int sample_count,
    int sample_rate
);

/*
 * Free a voice handle.
 */
POCKET_TTS_API void pocket_tts_free_voice(VoiceHandle voice);

/*
 * Generate speech from text using a voice.
 *
 * @param handle PocketTTS instance
 * @param text Text to synthesize
 * @param voice Voice handle from pocket_tts_encode_voice
 * @param result Output audio result (caller must call pocket_tts_free_audio)
 * @return 0 on success, non-zero on error
 */
POCKET_TTS_API int pocket_tts_generate(
    PocketTTSHandle handle,
    const char* text,
    VoiceHandle voice,
    AudioResult* result
);

/*
 * Free audio data from AudioResult.
 */
POCKET_TTS_API void pocket_tts_free_audio(AudioResult* result);

/**
 * Callback for audio chunks during streaming generation.
 * 
 * @param samples Audio samples (24kHz mono float)
 * @param sample_count Number of samples in this chunk
 * @param is_final 1 if this is the final chunk, 0 otherwise
 * @param user_data User-provided context pointer
 */
typedef void (*AudioChunkCallbackC)(
    const float* samples,
    int sample_count,
    int is_final,
    void* user_data
);

/*
 * Configuration for streaming generation.
 */
typedef struct {
    int chunk_size_frames;      /* Decode every N frames (default: 5) */
    void* user_data;            /* User context passed to callback */
} StreamingConfig;

/*
 * Generate speech with streaming callback.
 * Audio chunks are delivered progressively as they become available.
 *
 * @param handle PocketTTS instance
 * @param text Text to synthesize
 * @param voice Voice handle from pocket_tts_encode_voice
 * @param callback Called for each audio chunk
 * @param config Streaming configuration (pass NULL for defaults)
 * @return Total number of samples generated, or negative on error
 */
POCKET_TTS_API int pocket_tts_generate_streaming(
    PocketTTSHandle handle,
    const char* text,
    VoiceHandle voice,
    AudioChunkCallbackC callback,
    const StreamingConfig* config
);

/*
 * Cancel ongoing streaming generation.
 * Only works if streaming was started with cancellation enabled.
 *
 * @param handle PocketTTS instance
 */
POCKET_TTS_API void pocket_tts_cancel_streaming(PocketTTSHandle handle);

/*
 * Get the last error message.
 * The returned string is valid until the next API call.
 */
POCKET_TTS_API const char* pocket_tts_get_last_error(void);

/*
 * Get library version.
 */
POCKET_TTS_API const char* pocket_tts_version(void);

#ifdef __cplusplus
}
#endif

#endif /* POCKET_TTS_C_H */
