using System.Text.Json;
using OneImage;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Tests;

/// <summary>
/// Compares the recognizer, layer by layer, with reference outputs computed by PyTorch in fp32 from the same fp16 weights.
/// Layout of the directory: <c>model.onm</c>, <c>fixtures.json</c> (lines with <c>id</c>, <c>png</c>, <c>text</c> and a list of
/// <c>{name, shape, file}</c> layer entries) and the raw little-endian float32 files those entries name.
/// Our result and PyTorch's both carry float32 rounding error against the exact (float64) answer, and the two add. Per layer the
/// allowed difference is <c>Atol + 2 * Floor + Rtol * |reference|</c>: <c>Atol</c> and <c>Rtol</c> are twice the tolerances used
/// against the float64 reference (our own error budget), and <c>Floor</c> is PyTorch's measured fp32-vs-float64 maximum absolute
/// error for that layer over the same lines, recorded in <c>fixtures.json</c> (<c>fp32_vs_fp64_floor</c>). The floor grows with the
/// activation magnitudes and with saturated LSTM gates of a trained model, which a random-weight model does not show. Without a
/// recorded floor the term is zero. Also, our greedy decoding of our own logits must equal the fixture's <c>decoded</c> text.
/// </summary>
internal static class ParityFixtures
{
    private static readonly Dictionary<string, (double Atol, double Rtol)> Tolerance = new()
    {
        ["input"] = (1e-6, 1e-6),
        ["conv"] = (8e-6, 4e-5),
        ["lstm"] = (6e-6, 4e-5),
        ["fc"] = (4e-6, 4e-5),
    };

    public static void Run(string dir, KernelPath path)
    {
        var container = ModelContainer.Load(Path.Combine(dir, "model.onm"));
        var recognizer = new LineRecognizer(CrnnModel.FromContainer(container, OcrLimits.Default), OcrLimits.Default, path);
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "fixtures.json")));
        var floors = new Dictionary<string, double>();
        if (doc.RootElement.ValueKind == JsonValueKind.Object && doc.RootElement.TryGetProperty("fp32_vs_fp64_floor", out JsonElement floorElement))
        {
            foreach (JsonProperty f in floorElement.EnumerateObject())
            {
                floors[f.Name] = f.Value.GetProperty("max_abs").GetDouble();
            }
        }

        var worstByLayer = new Dictionary<string, double>();
        JsonElement lines = doc.RootElement.ValueKind == JsonValueKind.Array ? doc.RootElement : doc.RootElement.GetProperty("lines");
        Assert.True(lines.GetArrayLength() > 0, "fixtures.json lists no lines.");
        foreach (JsonElement line in lines.EnumerateArray())
        {
            string id = line.GetProperty("id").GetString()!;
            Image image = Image.Load(Path.Combine(dir, line.GetProperty("png").GetString()!));
            int width = image.Width;
            Assert.Equal(recognizer.InputHeight, image.Height);
            var px = new Gray8[width * image.Height];
            image.CopyPixelsTo<Gray8>(px, new ImageRect(0, 0, width, image.Height));
            byte[] gray = px.Select(p => p.V).ToArray();

            var trace = new ForwardTrace();
            float[] logits = recognizer.Forward(gray, width, 1, trace, CancellationToken.None, out int steps);

            foreach (JsonElement entry in FindLayerEntries(line))
            {
                string name = entry.GetProperty("name").GetString()!;
                int[] shape = entry.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                byte[] raw = File.ReadAllBytes(Path.Combine(dir, entry.GetProperty("file").GetString()!));
                var expected = new float[raw.Length / 4];
                Buffer.BlockCopy(raw, 0, expected, 0, raw.Length);
                Assert.Equal(shape.Aggregate(1, (a, b) => a * b), expected.Length);

                var (gotShape, got) = trace.Layers[name];
                Assert.Equal(shape, gotShape);
                string kind = name.StartsWith("conv", StringComparison.Ordinal) ? "conv" : name.StartsWith("lstm", StringComparison.Ordinal) ? "lstm" : name;
                var (atol, rtol) = Tolerance[kind];
                double floor = floors.GetValueOrDefault(name);
                double worst = 0, maxAbs = 0;
                for (int i = 0; i < got.Length; i++)
                {
                    double diff = Math.Abs(got[i] - expected[i]);
                    maxAbs = Math.Max(maxAbs, diff);
                    worst = Math.Max(worst, diff / (atol + (2 * floor) + (rtol * Math.Abs(expected[i]))));
                }

                worstByLayer[name] = Math.Max(worstByLayer.GetValueOrDefault(name), maxAbs);

                Assert.True(worst <= 1, $"{id}/{name}: error is {worst:F2}x the tolerance (atol {atol}, floor {floor}, rtol {rtol}).");
            }

            if (line.TryGetProperty("decoded", out JsonElement decoded))
            {
                Assert.Equal(decoded.GetString(), recognizer.Decode(logits, steps).Text);
            }

            // The decoded text of our logits must equal the greedy decoding of the reference logits when they are given.
            JsonElement[] fc = FindLayerEntries(line).Where(e => e.GetProperty("name").GetString() == "fc").ToArray();
            if (fc.Length == 1)
            {
                byte[] raw = File.ReadAllBytes(Path.Combine(dir, fc[0].GetProperty("file").GetString()!));
                var reference = new float[raw.Length / 4];
                Buffer.BlockCopy(raw, 0, reference, 0, raw.Length);
                Assert.Equal(recognizer.Decode(reference, steps).Text, recognizer.Decode(logits, steps).Text);
            }
        }

        if (Environment.GetEnvironmentVariable("ONEOCR_PARITY_REPORT") is "1")
        {
            foreach (var (layer, abs) in worstByLayer)
            {
                Console.Error.WriteLine($"PARITY {layer}: max abs difference {abs:E2} (PyTorch floor {floors.GetValueOrDefault(layer):E2})");
            }
        }
    }

    private static IEnumerable<JsonElement> FindLayerEntries(JsonElement line)
    {
        foreach (JsonProperty p in line.EnumerateObject())
        {
            if (p.Value.ValueKind == JsonValueKind.Array && p.Value.GetArrayLength() > 0
                && p.Value[0].ValueKind == JsonValueKind.Object && p.Value[0].TryGetProperty("file", out _) && p.Value[0].TryGetProperty("name", out _))
            {
                return p.Value.EnumerateArray();
            }
        }

        throw new InvalidDataException("A fixture line has no list of layer entries.");
    }
}
