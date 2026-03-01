#include <napi.h>

#include "pocket_tts/pocket_tts_c.h"

#include <cstring>
#include <string>

namespace {

void throwLastError(const Napi::Env& env, const std::string& prefix) {
    const char* lastError = pocket_tts_get_last_error();
    if (lastError && std::strlen(lastError) > 0) {
        Napi::Error::New(env, prefix + ": " + lastError).ThrowAsJavaScriptException();
    } else {
        Napi::Error::New(env, prefix).ThrowAsJavaScriptException();
    }
}

class VoiceWrap final : public Napi::ObjectWrap<VoiceWrap> {
public:
    static Napi::FunctionReference constructor;

    static void init(Napi::Env env, Napi::Object exports) {
        Napi::Function ctor = DefineClass(
            env,
            "Voice",
            {
                InstanceMethod("free", &VoiceWrap::free)
            });

        constructor = Napi::Persistent(ctor);
        constructor.SuppressDestruct();
        exports.Set("Voice", ctor);
    }

    static Napi::Object newInstance(Napi::Env env, VoiceHandle handle) {
        return constructor.New({Napi::External<void>::New(env, handle)});
    }

    explicit VoiceWrap(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<VoiceWrap>(info),
          handle_(nullptr),
          ownsHandle_(false) {
        if (info.Length() != 1 || !info[0].IsExternal()) {
            Napi::TypeError::New(info.Env(), "Voice cannot be constructed directly")
                .ThrowAsJavaScriptException();
            return;
        }

        handle_ = info[0].As<Napi::External<void>>().Data();
        ownsHandle_ = true;
    }

    ~VoiceWrap() override {
        reset();
    }

    VoiceHandle handle() const {
        return handle_;
    }

private:
    Napi::Value free(const Napi::CallbackInfo& info) {
        reset();
        return info.Env().Undefined();
    }

    void reset() {
        if (ownsHandle_ && handle_) {
            pocket_tts_free_voice(handle_);
        }
        handle_ = nullptr;
        ownsHandle_ = false;
    }

    VoiceHandle handle_;
    bool ownsHandle_;
};

Napi::FunctionReference VoiceWrap::constructor;

class PocketTTSWrap final : public Napi::ObjectWrap<PocketTTSWrap> {
public:
    static Napi::FunctionReference constructor;

    static void init(Napi::Env env, Napi::Object exports) {
        Napi::Function ctor = DefineClass(
            env,
            "PocketTTS",
            {
                InstanceMethod("encodeVoice", &PocketTTSWrap::encodeVoice),
                InstanceMethod("encodeVoiceFromSamples", &PocketTTSWrap::encodeVoiceFromSamples),
                InstanceMethod("generate", &PocketTTSWrap::generate),
                InstanceMethod("close", &PocketTTSWrap::close),
                InstanceMethod("version", &PocketTTSWrap::version)
            });

        constructor = Napi::Persistent(ctor);
        constructor.SuppressDestruct();
        exports.Set("PocketTTS", ctor);
    }

    explicit PocketTTSWrap(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<PocketTTSWrap>(info),
          handle_(nullptr) {
        Napi::Env env = info.Env();

        PocketTTSConfig config {};
        bool useConfig = false;

        std::string modelsDir;
        std::string tokenizerPath;
        std::string precision;

        if (info.Length() > 0 && !info[0].IsUndefined() && !info[0].IsNull()) {
            if (!info[0].IsObject()) {
                Napi::TypeError::New(env, "config must be an object").ThrowAsJavaScriptException();
                return;
            }

            Napi::Object cfg = info[0].As<Napi::Object>();

            if (cfg.Has("modelsDir")) {
                modelsDir = cfg.Get("modelsDir").As<Napi::String>().Utf8Value();
                config.models_dir = modelsDir.c_str();
                useConfig = true;
            }

            if (cfg.Has("tokenizerPath")) {
                tokenizerPath = cfg.Get("tokenizerPath").As<Napi::String>().Utf8Value();
                config.tokenizer_path = tokenizerPath.c_str();
                useConfig = true;
            }

            if (cfg.Has("precision")) {
                precision = cfg.Get("precision").As<Napi::String>().Utf8Value();
                config.precision = precision.c_str();
                useConfig = true;
            }

            if (cfg.Has("temperature")) {
                config.temperature = cfg.Get("temperature").As<Napi::Number>().FloatValue();
                useConfig = true;
            }

            if (cfg.Has("lsdSteps")) {
                config.lsd_steps = cfg.Get("lsdSteps").As<Napi::Number>().Int32Value();
                useConfig = true;
            }

            if (cfg.Has("maxFrames")) {
                config.max_frames = cfg.Get("maxFrames").As<Napi::Number>().Int32Value();
                useConfig = true;
            }
        }

        handle_ = pocket_tts_create(useConfig ? &config : nullptr);
        if (!handle_) {
            throwLastError(env, "Failed to create PocketTTS instance");
        }
    }

    ~PocketTTSWrap() override {
        reset();
    }

private:
    Napi::Value encodeVoice(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        if (!handle_) {
            Napi::Error::New(env, "PocketTTS instance already closed").ThrowAsJavaScriptException();
            return env.Null();
        }

        if (info.Length() != 1 || !info[0].IsString()) {
            Napi::TypeError::New(env, "encodeVoice(audioPath) expects a string path")
                .ThrowAsJavaScriptException();
            return env.Null();
        }

        const std::string audioPath = info[0].As<Napi::String>().Utf8Value();
        VoiceHandle voice = pocket_tts_encode_voice(handle_, audioPath.c_str());
        if (!voice) {
            throwLastError(env, "Failed to encode voice");
            return env.Null();
        }

        return VoiceWrap::newInstance(env, voice);
    }

    Napi::Value encodeVoiceFromSamples(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        if (!handle_) {
            Napi::Error::New(env, "PocketTTS instance already closed").ThrowAsJavaScriptException();
            return env.Null();
        }

        if (info.Length() != 2 || !info[0].IsTypedArray() || !info[1].IsNumber()) {
            Napi::TypeError::New(
                env,
                "encodeVoiceFromSamples(samples, sampleRate) expects (Float32Array, number)")
                .ThrowAsJavaScriptException();
            return env.Null();
        }

        Napi::TypedArray typedArray = info[0].As<Napi::TypedArray>();
        if (typedArray.TypedArrayType() != napi_float32_array) {
            Napi::TypeError::New(env, "samples must be a Float32Array").ThrowAsJavaScriptException();
            return env.Null();
        }

        Napi::Float32Array samples = info[0].As<Napi::Float32Array>();
        int sampleRate = info[1].As<Napi::Number>().Int32Value();

        VoiceHandle voice = pocket_tts_encode_voice_from_samples(
            handle_,
            samples.Data(),
            static_cast<int>(samples.ElementLength()),
            sampleRate);

        if (!voice) {
            throwLastError(env, "Failed to encode voice from samples");
            return env.Null();
        }

        return VoiceWrap::newInstance(env, voice);
    }

    Napi::Value generate(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        if (!handle_) {
            Napi::Error::New(env, "PocketTTS instance already closed").ThrowAsJavaScriptException();
            return env.Null();
        }

        if (info.Length() != 2 || !info[0].IsString() || !info[1].IsObject()) {
            Napi::TypeError::New(env, "generate(text, voice) expects (string, Voice)")
                .ThrowAsJavaScriptException();
            return env.Null();
        }

        Napi::Object voiceObject = info[1].As<Napi::Object>();
        if (!voiceObject.InstanceOf(VoiceWrap::constructor.Value())) {
            Napi::TypeError::New(env, "voice must be a Voice instance")
                .ThrowAsJavaScriptException();
            return env.Null();
        }

        VoiceWrap* voice = Napi::ObjectWrap<VoiceWrap>::Unwrap(voiceObject);
        if (!voice || !voice->handle()) {
            Napi::TypeError::New(env, "voice must be a live Voice handle")
                .ThrowAsJavaScriptException();
            return env.Null();
        }

        const std::string text = info[0].As<Napi::String>().Utf8Value();

        AudioResult result {};
        int rc = pocket_tts_generate(handle_, text.c_str(), voice->handle(), &result);
        if (rc != 0) {
            throwLastError(env, "Failed to generate audio");
            return env.Null();
        }

        const int sampleCount = result.sample_count;
        const int sampleRate = result.sample_rate;

        Napi::ArrayBuffer buffer = Napi::ArrayBuffer::New(
            env,
            static_cast<size_t>(sampleCount) * sizeof(float));

        if (sampleCount > 0) {
            std::memcpy(
                buffer.Data(),
                result.data,
                static_cast<size_t>(sampleCount) * sizeof(float));
        }
        pocket_tts_free_audio(&result);

        Napi::Float32Array samples = Napi::Float32Array::New(
            env,
            static_cast<size_t>(sampleCount),
            buffer,
            0);

        Napi::Object output = Napi::Object::New(env);
        output.Set("sampleRate", Napi::Number::New(env, sampleRate));
        output.Set("samples", samples);

        return output;
    }

    Napi::Value close(const Napi::CallbackInfo& info) {
        reset();
        return info.Env().Undefined();
    }

    Napi::Value version(const Napi::CallbackInfo& info) {
        return Napi::String::New(info.Env(), pocket_tts_version());
    }

    void reset() {
        if (handle_) {
            pocket_tts_destroy(handle_);
            handle_ = nullptr;
        }
    }

    PocketTTSHandle handle_;
};

Napi::FunctionReference PocketTTSWrap::constructor;

Napi::Value addonVersion(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), pocket_tts_version());
}

Napi::Object initAddon(Napi::Env env, Napi::Object exports) {
    VoiceWrap::init(env, exports);
    PocketTTSWrap::init(env, exports);
    exports.Set("version", Napi::Function::New(env, addonVersion));
    return exports;
}

} // namespace

NODE_API_MODULE(pocket_tts, initAddon)
