cmake_minimum_required(VERSION 3.16)
project(pocket_tts VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(BUILD_CLI "Build command-line tool" ON)
option(BUILD_SHARED "Build shared library" ON)

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

# SentencePiece (prefer static)
find_path(SENTENCEPIECE_INCLUDE_DIRS
    NAMES sentencepiece_processor.h
    PATHS 
        /opt/homebrew/include
        /usr/local/include
        /usr/include
        ${SENTENCEPIECE_ROOT}/include
)

# Try static library first, fall back to shared
find_library(SENTENCEPIECE_LIBRARIES
    NAMES libsentencepiece.a sentencepiece
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



# Apple Frameworks
if(APPLE)
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
endif()

# Core library sources (shared between CLI and library)
set(LIB_SOURCES
    src/audio_utils.cpp
    src/tokenizer.cpp
    src/pocket_tts.cpp
)

# Common include directories
set(COMMON_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}/..
    ${SENTENCEPIECE_INCLUDE_DIRS}
)

# Common link libraries
set(COMMON_LIBS
    ${ONNXRUNTIME_LIBRARIES}
    ${SENTENCEPIECE_LIBRARIES}
)

# vcpkg sentencepiece on Windows is often static and needs transitive deps explicitly.
if(WIN32)
    get_filename_component(_SP_LIB_DIR "${SENTENCEPIECE_LIBRARIES}" DIRECTORY)
    if(EXISTS "${_SP_LIB_DIR}")
        file(GLOB _ABSL_LIBS "${_SP_LIB_DIR}/absl*.lib")
        foreach(_extra_lib IN ITEMS
            "${_SP_LIB_DIR}/abseil_dll.lib"
            "${_SP_LIB_DIR}/libprotobuf.lib"
            "${_SP_LIB_DIR}/libprotobuf-lite.lib"
            "${_SP_LIB_DIR}/re2.lib"
            "${_SP_LIB_DIR}/utf8_range.lib"
            "${_SP_LIB_DIR}/utf8_validity.lib"
        )
            if(EXISTS "${_extra_lib}")
                list(APPEND _ABSL_LIBS "${_extra_lib}")
            endif()
        endforeach()
        list(APPEND COMMON_LIBS ${_ABSL_LIBS})
    endif()
endif()

if(APPLE)
    list(APPEND COMMON_LIBS
        ${COREML_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${ACCELERATE_FRAMEWORK}
    )
endif()

# ================================
# Shared Library Target
# ================================
if(BUILD_SHARED)
    add_library(pocket_tts_lib SHARED
        ${LIB_SOURCES}
        src/pocket_tts_c.cpp
    )
    
    set_target_properties(pocket_tts_lib PROPERTIES
        OUTPUT_NAME "pocket_tts"
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        PUBLIC_HEADER "include/pocket_tts/pocket_tts_c.h"
    )
    
    target_include_directories(pocket_tts_lib PRIVATE ${COMMON_INCLUDES})
    

    
    target_link_libraries(pocket_tts_lib PRIVATE ${COMMON_LIBS})
    
    # Export macros
    target_compile_definitions(pocket_tts_lib PRIVATE POCKET_TTS_BUILDING_DLL)
    
    # Symbol visibility
    if(NOT WIN32)
        target_compile_options(pocket_tts_lib PRIVATE -fvisibility=hidden)
    endif()
    
    # Compiler options
    if(MSVC)
        target_compile_options(pocket_tts_lib PRIVATE /W4 /O2)
    else()
        target_compile_options(pocket_tts_lib PRIVATE -Wall -Wextra -O2)
    endif()
    
    # Installation
    install(TARGETS pocket_tts_lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include/pocket_tts
    )
    
    # Also install all headers
    install(DIRECTORY include/pocket_tts
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
endif()

# ================================
# CLI Executable Target
# ================================
if(BUILD_CLI)
    add_executable(pocket_tts_cli
        ${LIB_SOURCES}
        src/main.cpp
    )
    
    set_target_properties(pocket_tts_cli PROPERTIES
        OUTPUT_NAME "pocket_tts"
    )
    
    target_include_directories(pocket_tts_cli PRIVATE ${COMMON_INCLUDES})
    

    
    target_link_libraries(pocket_tts_cli PRIVATE ${COMMON_LIBS})
    
    # Compiler options
    if(MSVC)
        target_compile_options(pocket_tts_cli PRIVATE /W4 /O2)
    else()
        target_compile_options(pocket_tts_cli PRIVATE -Wall -Wextra -O2)
    endif()
    
    if(WIN32 AND NOT BUILD_SHARED)
        target_compile_definitions(pocket_tts_cli PRIVATE POCKET_TTS_STATIC)
    endif()
    
    install(TARGETS pocket_tts_cli RUNTIME DESTINATION bin)
endif()

# ================================
# Test Streaming Executable
# ================================
add_executable(test_streaming
    ${LIB_SOURCES}
    test/test_streaming.cpp
)

target_include_directories(test_streaming PRIVATE ${COMMON_INCLUDES})



target_link_libraries(test_streaming PRIVATE ${COMMON_LIBS})

# Compiler options
if(MSVC)
    target_compile_options(test_streaming PRIVATE /W4 /O2)
else()
    target_compile_options(test_streaming PRIVATE -Wall -Wextra -O2)
endif()

if(WIN32 AND NOT BUILD_SHARED)
    target_compile_definitions(test_streaming PRIVATE POCKET_TTS_STATIC)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "=== Pocket TTS Configuration ===")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build CLI: ${BUILD_CLI}")
message(STATUS "Build Shared Library: ${BUILD_SHARED}")
message(STATUS "")
