#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <unistd.h>
#include "pocket_tts/pocket_tts_c.h"

int main(int argc, char* argv[]) {
    printf("Testing Pocket TTS C-API...\n");
    
    // Change to parent directory so model paths work
    char* exe_path = strdup(argv[0]);
    char* dir = dirname(exe_path);
    char parent_dir[1024];
    snprintf(parent_dir, sizeof(parent_dir), "%s/..", dir);
    if (chdir(parent_dir) != 0) {
        // Try from current dir
        chdir("..");
    }
    free(exe_path);
    
    // Create with defaults
    printf("Creating instance...\n");
    PocketTTSHandle handle = pocket_tts_create(NULL);
    if (!handle) {
        printf("Failed to create: %s\n", pocket_tts_get_last_error());
        return 1;
    }
    printf("Instance created\n");
    
    // Encode voice
    printf("Encoding voice...\n");
    VoiceHandle voice = pocket_tts_encode_voice(handle, "models/reference_sample.wav");
    if (!voice) {
        printf("Failed to encode voice: %s\n", pocket_tts_get_last_error());
        pocket_tts_destroy(handle);
        return 1;
    }
    printf("Voice encoded\n");
    
    // Generate
    printf("Generating...\n");
    AudioResult result = {0};
    int ret = pocket_tts_generate(handle, "Hello from C! This is a test.", voice, &result);
    if (ret != 0) {
        printf("Failed to generate: %s\n", pocket_tts_get_last_error());
        pocket_tts_free_voice(voice);
        pocket_tts_destroy(handle);
        return 1;
    }
    printf("Generated %d samples at %dHz (%.2fs)\n", 
           result.sample_count, result.sample_rate,
           (float)result.sample_count / result.sample_rate);
    
    // Cleanup
    pocket_tts_free_audio(&result);
    pocket_tts_free_voice(voice);
    pocket_tts_destroy(handle);
    
    printf("Done!\n");
    return 0;
}
