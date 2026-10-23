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
/// Tolerance per layer is twice the one used against the float64 reference: PyTorch's own fp32 error is of the same size as
/// ours, and the two errors can add.
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
                double worst = 0;
                for (int i = 0; i < got.Length; i++)
                {
                    worst = Math.Max(worst, Math.Abs(got[i] - expected[i]) / (atol + (rtol * Math.Abs(expected[i]))));
                }

                Assert.True(worst <= 1, $"{id}/{name}: error is {worst:F2}x the tolerance (atol {atol}, rtol {rtol}).");
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
