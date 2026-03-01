{
  "targets": [
    {
      "target_name": "pocket_tts",
      "sources": [
        "src/addon.cpp",
        "../../src/audio_utils.cpp",
        "../../src/tokenizer.cpp",
        "../../src/pocket_tts.cpp",
        "../../src/pocket_tts_c.cpp"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../../include",
        "<!@(node -p \"require('./scripts/gyp-paths').includeDirs.join(' ')\")"
      ],
      "libraries": [
        "<!@(node -p \"require('./scripts/gyp-paths').libraries.join(' ')\")"
      ],
      "defines": [
        "NAPI_CPP_EXCEPTIONS"
      ],
      "cflags_cc!": [
        "-fno-exceptions"
      ],
      "conditions": [
        [
          "OS==\"linux\"",
          {
            "cflags_cc": [
              "-std=c++17",
              "-Wall",
              "-Wextra"
            ]
          }
        ],
        [
          "OS==\"mac\"",
          {
            "xcode_settings": {
              "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
              "CLANG_CXX_LIBRARY": "libc++",
              "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
              "OTHER_CPLUSPLUSFLAGS": [
                "-Wall",
                "-Wextra"
              ]
            }
          }
        ],
        [
          "OS==\"win\"",
          {
            "defines": [
              "POCKET_TTS_STATIC"
            ],
            "configurations": {
              "Release": {
                "msvs_settings": {
                  "VCCLCompilerTool": {
                    "RuntimeLibrary": "2"
                  }
                }
              },
              "Debug": {
                "msvs_settings": {
                  "VCCLCompilerTool": {
                    "RuntimeLibrary": "3"
                  }
                }
              }
            },
            "msvs_settings": {
              "VCCLCompilerTool": {
                "AdditionalOptions": [
                  "/std:c++17",
                  "/MD"
                ],
                "ExceptionHandling": 1,
                "RuntimeLibrary": "2",
                "WarningLevel": 4
              }
            }
          }
        ]
      ]
    }
  ]
}
