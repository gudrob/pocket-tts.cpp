cmake_minimum_required(VERSION 3.16)
project(pocket_tts VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Platform detection
if(WIN32)
    set(PLATFORM_WINDOWS TRUE)
elseif(APPLE)
    set(PLATFORM_MACOS TRUE)
else()
    set(PLATFORM_LINUX TRUE)
endif()

# Find required packages
if(NOT WIN32)
    find_package(PkgConfig REQUIRED)
endif()

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIRS
    NAMES onnxruntime_cxx_api.h
    PATHS 
        /opt/homebrew/include/onnxruntime
        /usr/local/include/onnxruntime
        /opt/homebrew/include
        /usr/local/include
        /usr/include
        ${ONNXRUNTIME_ROOT}/include
    PATH_SUFFIXES core/session
)

find_library(ONNXRUNTIME_LIBRARIES
    NAMES onnxruntime
    PATHS 
        /opt/homebrew/lib
        /usr/local/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        ${ONNXRUNTIME_ROOT}/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIRS OR NOT ONNXRUNTIME_LIBRARIES)
    message(FATAL_ERROR "ONNX Runtime not found. See README.md for installation instructions.")
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARIES}")

# SentencePiece
find_path(SENTENCEPIECE_INCLUDE_DIRS
    NAMES sentencepiece_processor.h
    PATHS 
        /opt/homebrew/include
        /usr/local/include
        /usr/include
        ${SENTENCEPIECE_ROOT}/include
)

find_library(SENTENCEPIECE_LIBRARIES
    NAMES sentencepiece
    PATHS 
        /opt/homebrew/lib
        /usr/local/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        ${SENTENCEPIECE_ROOT}/lib
)

if(NOT SENTENCEPIECE_INCLUDE_DIRS OR NOT SENTENCEPIECE_LIBRARIES)
    message(FATAL_ERROR "SentencePiece not found. See README.md for installation instructions.")
endif()

message(STATUS "SentencePiece include: ${SENTENCEPIECE_INCLUDE_DIRS}")
message(STATUS "SentencePiece library: ${SENTENCEPIECE_LIBRARIES}")

# libsndfile and libsamplerate
if(WIN32)
    # Windows: Use vcpkg or manual paths
    find_path(SNDFILE_INCLUDE_DIRS NAMES sndfile.h)
    find_library(SNDFILE_LIBRARIES NAMES sndfile libsndfile-1)
    find_path(SAMPLERATE_INCLUDE_DIRS NAMES samplerate.h)
    find_library(SAMPLERATE_LIBRARIES NAMES samplerate libsamplerate-0)
else()
    pkg_check_modules(SNDFILE REQUIRED sndfile)
    pkg_check_modules(SAMPLERATE REQUIRED samplerate)
endif()

# Apple Frameworks
if(APPLE)
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
endif()

# Define source files
set(SOURCES
    src/audio_utils.cpp
    src/tokenizer.cpp
    src/pocket_tts.cpp
    src/main.cpp
)

# Create executable
add_executable(pocket_tts ${SOURCES})

# Include directories
target_include_directories(pocket_tts PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}/..
    ${SENTENCEPIECE_INCLUDE_DIRS}
    ${SNDFILE_INCLUDE_DIRS}
    ${SAMPLERATE_INCLUDE_DIRS}
)

# Link directories for pkg-config libraries
if(NOT WIN32)
    target_link_directories(pocket_tts PRIVATE
        ${SNDFILE_LIBRARY_DIRS}
        ${SAMPLERATE_LIBRARY_DIRS}
    )
endif()

# Link libraries
target_link_libraries(pocket_tts PRIVATE
    ${ONNXRUNTIME_LIBRARIES}
    ${SENTENCEPIECE_LIBRARIES}
    ${SNDFILE_LIBRARIES}
    ${SAMPLERATE_LIBRARIES}
)

if(APPLE)
    target_link_libraries(pocket_tts PRIVATE 
        ${COREML_FRAMEWORK} 
        ${FOUNDATION_FRAMEWORK}
        ${ACCELERATE_FRAMEWORK}
    )
endif()

# Compiler options
if(MSVC)
    target_compile_options(pocket_tts PRIVATE /W4 /O2)
else()
    target_compile_options(pocket_tts PRIVATE -Wall -Wextra -O2)
endif()

# Windows-specific settings
if(WIN32)
    # Copy DLLs on Windows
    add_custom_command(TARGET pocket_tts POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:pocket_tts>
            $<TARGET_FILE_DIR:pocket_tts>
        COMMAND_EXPAND_LISTS
    )
endif()

# Installation
install(TARGETS pocket_tts RUNTIME DESTINATION bin)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Pocket TTS Configuration ===")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
