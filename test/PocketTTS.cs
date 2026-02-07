using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Collections.Generic;

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
    /// Callback delegate for streaming audio chunks.
    /// </summary>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void AudioChunkCallback(IntPtr samples, int sampleCount, int isFinal, IntPtr userData);

    /// <summary>
    /// Streaming configuration.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct StreamingConfig
    {
        public int ChunkSizeFrames;
        public IntPtr UserData;
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
        public static extern int pocket_tts_generate_streaming(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)] string text,
            IntPtr voice,
            AudioChunkCallback callback,
            ref StreamingConfig config);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pocket_tts_generate_streaming(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)] string text,
            IntPtr voice,
            AudioChunkCallback callback,
            IntPtr config);  // Overload for null config

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pocket_tts_cancel_streaming(IntPtr handle);

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

        /// <summary>
        /// Generate speech with streaming callback.
        /// </summary>
        /// <param name="text">Text to synthesize</param>
        /// <param name="voice">Voice to use</param>
        /// <param name="onChunk">Callback for each audio chunk (samples, isFinal)</param>
        /// <param name="chunkSizeFrames">Number of frames per chunk (default: 5, ~400ms)</param>
        /// <returns>Total number of samples generated</returns>
        public int GenerateStreaming(
            string text,
            Voice voice,
            Action<float[], bool> onChunk,
            int chunkSizeFrames = 5)
        {
            if (onChunk == null)
                throw new ArgumentNullException(nameof(onChunk));

            // Create callback that marshals data and calls user callback
            AudioChunkCallback nativeCallback = (samples, sampleCount, isFinal, userData) =>
            {
                var audioChunk = new float[sampleCount];
                Marshal.Copy(samples, audioChunk, 0, sampleCount);
                onChunk(audioChunk, isFinal != 0);
            };

            var config = new StreamingConfig
            {
                ChunkSizeFrames = chunkSizeFrames,
                UserData = IntPtr.Zero
            };

            int totalSamples = PocketTTSNative.pocket_tts_generate_streaming(
                _handle, text, voice.Handle, nativeCallback, ref config);

            if (totalSamples < 0)
            {
                throw new Exception($"Streaming failed: {GetLastError()}");
            }

            return totalSamples;
        }

        /// <summary>
        /// Generate speech as an async stream.
        /// Use with: await foreach (var chunk in tts.GenerateStreamingAsync(...))
        /// </summary>
        public async IAsyncEnumerable<float[]> GenerateStreamingAsync(
            string text,
            Voice voice,
            int chunkSizeFrames = 5,
            [System.Runtime.CompilerServices.EnumeratorCancellation] System.Threading.CancellationToken ct = default)
        {
            var chunks = new System.Collections.Concurrent.BlockingCollection<(float[] data, bool isFinal)>();
            Exception error = null;

            // Start generation in background thread
            var task = System.Threading.Tasks.Task.Run(() =>
            {
                try
                {
                    GenerateStreaming(text, voice, (data, isFinal) =>
                    {
                        if (ct.IsCancellationRequested)
                        {
                            CancelStreaming();
                            chunks.CompleteAdding();
                            return;
                        }

                        chunks.Add((data, isFinal));
                        if (isFinal)
                            chunks.CompleteAdding();
                    }, chunkSizeFrames);
                }
                catch (Exception ex)
                {
                    error = ex;
                    chunks.CompleteAdding();
                }
            }, ct);

            // Yield chunks as they become available
            foreach (var (data, isFinal) in chunks.GetConsumingEnumerable())
            {
                ct.ThrowIfCancellationRequested();
                yield return data;
            }

            // Wait for background task to complete
            await task;

            if (error != null)
                throw error;
        }

        /// <summary>
        /// Cancel ongoing streaming generation.
        /// </summary>
        public void CancelStreaming()
        {
            PocketTTSNative.pocket_tts_cancel_streaming(_handle);
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

    /*
    /// <summary>
    /// Example usage (commented out - see StreamingExample.cs for demos).
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
    */
}
