using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace PocketTTS.Examples
{
    /// <summary>
    /// Example demonstrating audio streaming in C#.
    /// </summary>
    class StreamingExample
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("=== Pocket TTS C# Streaming Example ===\n");

            // Change to project root (go up from bin/Debug/net7.0/)
            var scriptDir = AppContext.BaseDirectory;
            var projectRoot = Path.GetFullPath(Path.Combine(scriptDir, "..", "..", "..", ".."));
            Environment.CurrentDirectory = projectRoot;
            Console.WriteLine($"Working directory: {Environment.CurrentDirectory}\n");

            using var tts = new PocketTTS();
            using var voice = tts.EncodeVoice("models/reference_sample.wav");

            string text = "Hello! This is a demonstration of streaming audio generation in C Sharp.";

            // Example 1: Event-based streaming
            Console.WriteLine("Example 1: Event-based streaming");
            Console.WriteLine("----------------------------------");

            var chunks = new List<float[]>();
            int chunkCount = 0;

            int totalSamples = tts.GenerateStreaming(text, voice, (chunk, isFinal) =>
            {
                chunkCount++;
                Console.WriteLine($"Chunk {chunkCount}: {chunk.Length} samples{(isFinal ? " [FINAL]" : "")}");
                chunks.Add(chunk);

                // Here you could play the chunk immediately with an audio library
                // e.g., using NAudio, CSCore, or similar
            }, chunkSizeFrames: 5);

            Console.WriteLine($"\nReceived {chunkCount} chunks, {totalSamples} total samples");
            Console.WriteLine($"Duration: {(totalSamples / 24000.0):F2}s\n");

            // Example 2: Async streaming
            Console.WriteLine("Example 2: Async streaming");
            Console.WriteLine("----------------------------------");

            chunkCount = 0;
            var allAudio = new List<float>();

            await foreach (var chunk in tts.GenerateStreamingAsync(text, voice, chunkSizeFrames: 5))
            {
                chunkCount++;
                Console.WriteLine($"Async chunk {chunkCount}: {chunk.Length} samples");
                allAudio.AddRange(chunk);

                // Simulate async processing (e.g., playing audio)
                await Task.Delay(10);
            }

            Console.WriteLine($"\nReceived {chunkCount} async chunks, {allAudio.Count} total samples");

            // Example 3: Cancellable streaming
            Console.WriteLine("\nExample 3: Cancellable streaming");
            Console.WriteLine("----------------------------------");
            Console.WriteLine("Generating and cancelling after 2 chunks...");

            using var cts = new CancellationTokenSource();
            chunkCount = 0;

            try
            {
                await foreach (var chunk in tts.GenerateStreamingAsync(text, voice, 5, cts.Token))
                {
                    chunkCount++;
                    Console.WriteLine($"Chunk {chunkCount}: {chunk.Length} samples");

                    if (chunkCount >= 2)
                    {
                        Console.WriteLine("Cancelling...");
                        cts.Cancel();
                    }
                }
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("Generation cancelled successfully!");
            }

            Console.WriteLine("\nDone!");
        }
    }
}
