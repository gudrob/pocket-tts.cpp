using System;
using System.IO;
using System.Runtime.InteropServices;

namespace PocketTTS
{
    /// <summary>
    /// Audio result from generation.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct AudioResult
    {
        public IntPtr Data;
        public int SampleCount;
        public int SampleRate;
    }

    /// <summary>
    /// Configuration for PocketTTS.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PocketTTSConfig
    {
        [MarshalAs(UnmanagedType.LPStr)]
        public string ModelsDir;
        
        [MarshalAs(UnmanagedType.LPStr)]
        public string TokenizerPath;
        
        [MarshalAs(UnmanagedType.LPStr)]
        public string Precision;
        
        public float Temperature;
        public int LsdSteps;
        public int MaxFrames;
    }

    /// <summary>
    /// P/Invoke bindings for Pocket TTS native library.
    /// </summary>
    public static class PocketTTSNative
    {
        private const string LibName = "pocket_tts";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr pocket_tts_create(IntPtr config);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pocket_tts_destroy(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr pocket_tts_encode_voice(IntPtr handle, 
            [MarshalAs(UnmanagedType.LPStr)] string audioPath);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pocket_tts_free_voice(IntPtr voice);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pocket_tts_generate(IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)] string text,
            IntPtr voice,
            ref AudioResult result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pocket_tts_free_audio(ref AudioResult result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr pocket_tts_get_last_error();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr pocket_tts_version();
    }

    /// <summary>
    /// Voice handle for generation.
    /// </summary>
    public class Voice : IDisposable
    {
        internal IntPtr Handle { get; private set; }
        
        internal Voice(IntPtr handle)
        {
            Handle = handle;
        }
        
        public void Dispose()
        {
            if (Handle != IntPtr.Zero)
            {
                PocketTTSNative.pocket_tts_free_voice(Handle);
                Handle = IntPtr.Zero;
            }
        }
    }

    /// <summary>
    /// Pocket TTS text-to-speech engine.
    /// </summary>
    public class PocketTTS : IDisposable
    {
        private IntPtr _handle;

        public PocketTTS(string modelsDir = "models/onnx", string tokenizerPath = "models/tokenizer.model", string precision = "int8")
        {
            // For now, use defaults (config struct passing is complex)
            _handle = PocketTTSNative.pocket_tts_create(IntPtr.Zero);
            if (_handle == IntPtr.Zero)
            {
                throw new Exception($"Failed to create PocketTTS: {GetLastError()}");
            }
        }

        public static string Version => Marshal.PtrToStringAnsi(PocketTTSNative.pocket_tts_version()) ?? "";

        public static string GetLastError() => Marshal.PtrToStringAnsi(PocketTTSNative.pocket_tts_get_last_error()) ?? "";

        public Voice EncodeVoice(string audioPath)
        {
            var handle = PocketTTSNative.pocket_tts_encode_voice(_handle, audioPath);
            if (handle == IntPtr.Zero)
            {
                throw new Exception($"Failed to encode voice: {GetLastError()}");
            }
            return new Voice(handle);
        }

        public float[] Generate(string text, Voice voice)
        {
            var result = new AudioResult();
            int ret = PocketTTSNative.pocket_tts_generate(_handle, text, voice.Handle, ref result);
            
            if (ret != 0)
            {
                throw new Exception($"Failed to generate: {GetLastError()}");
            }
            
            try
            {
                var audio = new float[result.SampleCount];
                Marshal.Copy(result.Data, audio, 0, result.SampleCount);
                return audio;
            }
            finally
            {
                PocketTTSNative.pocket_tts_free_audio(ref result);
            }
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                PocketTTSNative.pocket_tts_destroy(_handle);
                _handle = IntPtr.Zero;
            }
        }
    }

    /// <summary>
    /// Example usage.
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine($"Pocket TTS C# Example");
            Console.WriteLine($"Library version: {PocketTTS.Version}");
            Console.WriteLine();
            
            // Change to parent directory so model paths work
            var scriptDir = AppContext.BaseDirectory;
            var projectRoot = Path.GetFullPath(Path.Combine(scriptDir, "..", "..", "..", "..", ".."));
            Environment.CurrentDirectory = projectRoot;
            Console.WriteLine($"Working directory: {Environment.CurrentDirectory}");
            
            using var tts = new PocketTTS();
            
            Console.WriteLine("Encoding voice...");
            using var voice = tts.EncodeVoice("models/reference_sample.wav");
            
            Console.WriteLine("Generating speech...");
            var audio = tts.Generate("Hello from C Sharp! This is a test.", voice);
            
            Console.WriteLine($"Generated {audio.Length} samples");
            Console.WriteLine($"Duration: {audio.Length / 24000.0:F2}s");
            
            // TODO: Save to WAV file
            Console.WriteLine("Done!");
        }
    }
}
