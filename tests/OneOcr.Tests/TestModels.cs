using System;
using System.Collections.Generic;
using System.IO;
using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

namespace OneOcr.Tests;

/// <summary>One tensor to write: values are rounded to fp16, as the exporter does.</summary>
internal sealed class TestTensor
{
    public required string Name { get; init; }
    public required int[] Shape { get; init; }
    public required float[] Values { get; init; }

    /// <summary>Explicit offset relative to the data region; null places it after the previous tensor, aligned to 64.</summary>
    public long? Offset { get; init; }
}

/// <summary>A small container writer used only by tests. The product has no writer in this version; the exporter lives in the training project.</summary>
internal static class TestContainerWriter
{
    public static byte[] Write(
        string architecture, int height, string[] charset, string configJson, IReadOnlyList<TestTensor> tensors,
        float mean = 0.5f, float std = 0.5f, bool invert = false, string? calibrationJson = null)
    {
        // Place tensors.
        var offsets = new long[tensors.Count];
        long cursor = 0;
        for (int i = 0; i < tensors.Count; i++)
        {
            long off = tensors[i].Offset ?? ((cursor + 63) / 64 * 64);
            offsets[i] = off;
            cursor = Math.Max(cursor, off + (tensors[i].Values.Length * 2L));
        }

        var manifest = new MemoryStream();
        using (var w = new Utf8JsonWriter(manifest))
        {
            w.WriteStartObject();
            w.WriteString("architecture", architecture);
            w.WriteString("name", "test-recognizer");
            w.WriteString("version", "0.0.1");
            w.WriteString("license", "MIT OR Apache-2.0");
            w.WriteStartObject("input");
            w.WriteNumber("height", height);
            w.WriteNumber("channels", 1);
            w.WriteNumber("mean", mean);
            w.WriteNumber("std", std);
            w.WriteBoolean("invert", invert);
            w.WriteEndObject();
            w.WriteStartArray("charset");
            foreach (string s in charset)
            {
                w.WriteStringValue(s);
            }

            w.WriteEndArray();
            w.WritePropertyName("config");
            using (JsonDocument doc = JsonDocument.Parse(configJson))
            {
                doc.RootElement.WriteTo(w);
            }

            w.WriteStartArray("tensors");
            for (int i = 0; i < tensors.Count; i++)
            {
                w.WriteStartObject();
                w.WriteString("name", tensors[i].Name);
                w.WriteString("dtype", "f16");
                w.WriteStartArray("shape");
                foreach (int d in tensors[i].Shape)
                {
                    w.WriteNumberValue(d);
                }

                w.WriteEndArray();
                w.WriteNumber("offset", offsets[i]);
                w.WriteNumber("length", tensors[i].Values.Length * 2L);
                w.WriteEndObject();
            }

            w.WriteEndArray();
            if (calibrationJson is not null)
            {
                w.WritePropertyName("calibration");
                using JsonDocument cal = JsonDocument.Parse(calibrationJson);
                cal.RootElement.WriteTo(w);
            }

            w.WriteStartObject("provenance");
            w.WriteString("note", "test");
            w.WriteEndObject();
            w.WriteEndObject();
        }

        byte[] json = manifest.ToArray();
        // Version 1 is the format that carries calibration; a test that passes calibration JSON gets a version 1 file.
        return Assemble(json, tensors, offsets, cursor, calibrationJson is null ? 0u : 1u);
    }

    public static byte[] Assemble(byte[] manifestJson, IReadOnlyList<TestTensor> tensors, long[] offsets, long dataLength, uint version = 0)
    {
        long dataStart = (16 + manifestJson.Length + 63) / 64 * 64;
        var file = new byte[dataStart + dataLength];
        "ONEOCRM"u8.CopyTo(file);
        BinaryPrimitives.WriteUInt32LittleEndian(file.AsSpan(8), version);
        BinaryPrimitives.WriteUInt32LittleEndian(file.AsSpan(12), (uint)manifestJson.Length);
        manifestJson.CopyTo(file, 16);
        for (int i = 0; i < tensors.Count; i++)
        {
            Span<byte> dst = file.AsSpan((int)(dataStart + offsets[i]));
            for (int j = 0; j < tensors[i].Values.Length; j++)
            {
                BinaryPrimitives.WriteHalfLittleEndian(dst[(j * 2)..], (Half)tensors[i].Values[j]);
            }
        }

        return file;
    }
}

/// <summary>Builds random <c>crnn-v0</c> containers of a given shape.</summary>
internal static class TestModels
{
    public sealed record Shape(int Height, (int Out, int PoolH, int PoolW)[] Conv, int Hidden, int LstmLayers, int Classes);

    /// <summary>Same layer pattern as the spec's example, heights 32 to 1 through five pooling stages, narrower channels.</summary>
    public static Shape Small { get; } = new(32, [(8, 2, 2), (16, 2, 2), (16, 2, 1), (32, 2, 1), (32, 2, 1)], 24, 2, 41);

    /// <summary>The layer sizes of the first trained recognizer (hidden 192, 622 classes), whose widths are all multiples of 16.</summary>
    public static Shape SmokeLike { get; } = new(32, [(32, 2, 2), (64, 2, 2), (128, 1, 1), (128, 2, 1), (192, 2, 1), (192, 2, 1)], 192, 2, 622);

    /// <summary>Roughly the size of the intended model.</summary>
    public static Shape Full { get; } = new(32, [(64, 2, 2), (128, 2, 2), (128, 2, 1), (256, 2, 1), (256, 2, 1)], 256, 2, 700);

    public static string[] Charset(int classes)
    {
        var cs = new string[classes];
        cs[0] = "<blank>";
        for (int i = 1; i < classes; i++)
        {
            cs[i] = char.ConvertFromUtf32(0x100 + i);
        }

        return cs;
    }

    public static string ConfigJson(Shape s)
    {
        var sb = new StringBuilder("{\"conv\":[");
        for (int i = 0; i < s.Conv.Length; i++)
        {
            if (i > 0)
            {
                sb.Append(',');
            }

            var (o, ph, pw) = s.Conv[i];
            sb.Append($"{{\"out\":{o},\"kernel\":[3,3],\"padding\":[1,1],\"pool\":[{ph},{pw}]}}");
        }

        sb.Append($"],\"lstm\":{{\"hidden\":{s.Hidden},\"layers\":{s.LstmLayers}}},\"classes\":{s.Classes}}}");
        return sb.ToString();
    }

    public static List<TestTensor> RandomTensors(Shape s, int seed)
    {
        var rng = new Random(seed);
        var list = new List<TestTensor>();
        void Add(string name, double scale, params int[] shape)
        {
            int n = 1;
            foreach (int d in shape)
            {
                n *= d;
            }

            var v = new float[n];
            for (int i = 0; i < n; i++)
            {
                v[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
            }

            list.Add(new TestTensor { Name = name, Shape = shape, Values = v });
        }

        int ch = 1;
        for (int i = 0; i < s.Conv.Length; i++)
        {
            int o = s.Conv[i].Out;
            // Uniform with the variance a He initialization would give, so activations neither vanish nor blow up.
            Add($"conv.{i}.weight", Math.Sqrt(6.0 / (ch * 9)), o, ch, 3, 3);
            Add($"conv.{i}.bias", 0.1, o);
            ch = o;
        }

        double k = 1.0 / Math.Sqrt(s.Hidden);
        int inSize = ch;
        for (int l = 0; l < s.LstmLayers; l++)
        {
            foreach (string suffix in new[] { string.Empty, "_reverse" })
            {
                Add($"lstm.weight_ih_l{l}{suffix}", k * 2, 4 * s.Hidden, inSize);
                Add($"lstm.weight_hh_l{l}{suffix}", k * 2, 4 * s.Hidden, s.Hidden);
                Add($"lstm.bias_ih_l{l}{suffix}", k, 4 * s.Hidden);
                Add($"lstm.bias_hh_l{l}{suffix}", k, 4 * s.Hidden);
            }

            inSize = 2 * s.Hidden;
        }

        Add("fc.weight", 1.0 / Math.Sqrt(2 * s.Hidden) * 2, s.Classes, 2 * s.Hidden);
        Add("fc.bias", 0.1, s.Classes);
        return list;
    }

    public static byte[] RandomContainer(Shape s, int seed) =>
        TestContainerWriter.Write("crnn-v0", s.Height, Charset(s.Classes), ConfigJson(s), RandomTensors(s, seed));

    /// <summary>A version 1 container with the given calibration member (JSON text).</summary>
    public static byte[] RandomContainerV1(Shape s, int seed, string calibrationJson) =>
        TestContainerWriter.Write("crnn-v0", s.Height, Charset(s.Classes), ConfigJson(s), RandomTensors(s, seed), calibrationJson: calibrationJson);

    /// <summary>A random 8-bit line image with some structure (smooth gradients plus noise), so activations are not degenerate.</summary>
    public static byte[] RandomImage(int height, int width, int seed)
    {
        var rng = new Random(seed);
        var px = new byte[height * width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double v = 128 + (100 * Math.Sin((x * 0.21) + (y * 0.13))) + (rng.NextDouble() * 40 - 20);
                px[(y * width) + x] = (byte)Math.Clamp((int)v, 0, 255);
            }
        }

        return px;
    }
}
