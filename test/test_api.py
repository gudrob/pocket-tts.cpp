#!/usr/bin/env python3
"""
Pocket TTS Python Example
Demonstrates how to use the Pocket TTS shared library from Python.
"""

import ctypes
import os
import sys
import wave
import struct

# Find library
if sys.platform == 'darwin':
    LIB_NAME = 'libpocket_tts.dylib'
elif sys.platform == 'win32':
    LIB_NAME = 'pocket_tts.dll'
else:
    LIB_NAME = 'libpocket_tts.so'

# Try to find library in common locations
lib_paths = [
    os.path.join(os.path.dirname(__file__), '..', 'build', LIB_NAME),
    os.path.join(os.path.dirname(__file__), '..', LIB_NAME),
    LIB_NAME
]

lib = None
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Change to parent directory so model paths work
os.chdir(parent_dir)

for path in lib_paths:
    try:
        lib = ctypes.CDLL(path)
        break
    except OSError:
        continue

if lib is None:
    print(f"Error: Could not find {LIB_NAME}")
    print("Make sure to build the library first: mkdir build && cd build && cmake .. && make")
    sys.exit(1)


# Define function signatures
lib.pocket_tts_version.restype = ctypes.c_char_p
lib.pocket_tts_version.argtypes = []

lib.pocket_tts_get_last_error.restype = ctypes.c_char_p
lib.pocket_tts_get_last_error.argtypes = []

lib.pocket_tts_create.restype = ctypes.c_void_p
lib.pocket_tts_create.argtypes = [ctypes.c_void_p]

lib.pocket_tts_destroy.restype = None
lib.pocket_tts_destroy.argtypes = [ctypes.c_void_p]

lib.pocket_tts_encode_voice.restype = ctypes.c_void_p
lib.pocket_tts_encode_voice.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

lib.pocket_tts_free_voice.restype = None
lib.pocket_tts_free_voice.argtypes = [ctypes.c_void_p]


class AudioResult(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('sample_count', ctypes.c_int),
        ('sample_rate', ctypes.c_int)
    ]


lib.pocket_tts_generate.restype = ctypes.c_int
lib.pocket_tts_generate.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
    ctypes.POINTER(AudioResult)
]

lib.pocket_tts_free_audio.restype = None
lib.pocket_tts_free_audio.argtypes = [ctypes.POINTER(AudioResult)]


def save_wav(filename: str, samples: list, sample_rate: int = 24000):
    """Save audio samples to a WAV file."""
    with wave.open(filename, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        
        # Convert float to 16-bit PCM
        max_val = max(abs(s) for s in samples) or 1.0
        pcm_data = b''.join(
            struct.pack('<h', int(s / max_val * 32767))
            for s in samples
        )
        wav.writeframes(pcm_data)


def main():
    print(f"Pocket TTS Python Example")
    print(f"Library version: {lib.pocket_tts_version().decode()}")
    print()
    
    # Create instance
    print("Creating PocketTTS instance...")
    handle = lib.pocket_tts_create(None)  # NULL = use defaults
    if not handle:
        print(f"Error: {lib.pocket_tts_get_last_error().decode()}")
        return 1
    
    try:
        # Encode voice
        voice_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'reference_sample.wav')
        print(f"Encoding voice from: {voice_path}")
        voice = lib.pocket_tts_encode_voice(handle, voice_path.encode())
        if not voice:
            print(f"Error: {lib.pocket_tts_get_last_error().decode()}")
            return 1
        
        try:
            # Generate speech
            text = "Hello from Python! This is a test of the Pocket TTS library."
            print(f"Generating: \"{text}\"")
            
            result = AudioResult()
            ret = lib.pocket_tts_generate(handle, text.encode(), voice, ctypes.byref(result))
            
            if ret != 0:
                print(f"Error: {lib.pocket_tts_get_last_error().decode()}")
                return 1
            
            print(f"Generated {result.sample_count} samples at {result.sample_rate}Hz")
            print(f"Duration: {result.sample_count / result.sample_rate:.2f}s")
            
            # Copy audio data to Python list
            audio = [result.data[i] for i in range(result.sample_count)]
            
            # Free C memory
            lib.pocket_tts_free_audio(ctypes.byref(result))
            
            # Save to file
            output_path = os.path.join(os.path.dirname(__file__), 'output_python.wav')
            save_wav(output_path, audio)
            print(f"Saved to: {output_path}")
            
        finally:
            lib.pocket_tts_free_voice(voice)
    finally:
        lib.pocket_tts_destroy(handle)
    
    print("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
