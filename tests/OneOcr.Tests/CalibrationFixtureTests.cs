using System.Globalization;
using System.Text.Json;
using OneImage;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Tests;

/// <summary>
/// Reads the held-out calibration fixture written next to a trained version 1 model: lines, the words the reference decoded from
/// them, each word's script, raw confidence and calibrated confidence (with the float32 bit patterns). Set
/// <c>ONEOCR_CALIBRATION_DIR</c> to the directory holding <c>calibration.json</c> and <c>lines/</c>; the model is
/// <c>model-v1.onm</c> in that directory or its parent.
/// </summary>
public sealed class CalibrationFixtureTests
{
    /// <summary>
    /// The reference raw confidence comes from an fp32 PyTorch forward pass; ours from a different accumulation order. A word's raw
    /// confidence is a mean of per-character maxima of a softmax, so it inherits logit error scaled by at most one; the measured
    /// difference over the fixture is below 1e-5 (measured by bisection: 1e-5 passes, 1e-6 fails), so this bound keeps a factor of two.
    /// </summary>
    private const double RawTolerance = 2e-5;

    [Fact]
    public void CalibrationReproducesTheReferenceAndRawConfidenceIsClose()
    {
        string? dir = Environment.GetEnvironmentVariable("ONEOCR_CALIBRATION_DIR");
        if (string.IsNullOrEmpty(dir))
        {
            Assert.Skip("ONEOCR_CALIBRATION_DIR is not set.");
        }

        string modelPath = Path.Combine(dir, "model-v1.onm");
        if (!File.Exists(modelPath))
        {
            modelPath = Path.Combine(dir, "..", "model-v1.onm");
        }

        ModelContainer container = ModelContainer.Load(modelPath);
        Assert.NotNull(container.Calibration);
        var recognizer = new LineRecognizer(CrnnModel.FromContainer(container, OcrLimits.Default), OcrLimits.Default, Dispatch.Best);

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "calibration.json")));
        int words = 0;
        double worstRaw = 0;
        foreach (JsonElement line in doc.RootElement.GetProperty("lines").EnumerateArray())
        {
            string id = line.GetProperty("id").GetString()!;
            Image image = Image.Load(Path.Combine(dir, line.GetProperty("png").GetString()!));
            var px = new Gray8[image.Width * image.Height];
            image.CopyPixelsTo<Gray8>(px, new ImageRect(0, 0, image.Width, image.Height));
            byte[] gray = px.Select(p => p.V).ToArray();
            LineWords got = recognizer.RecognizeWords(gray, image.Width, 1, CancellationToken.None, withChars: false);

            JsonElement expected = line.GetProperty("words");
            Assert.True(expected.GetArrayLength() == got.Words.Count, $"{id}: {got.Words.Count} words, reference has {expected.GetArrayLength()}.");
            int k = 0;
            foreach (JsonElement w in expected.EnumerateArray())
            {
                RecognizedWord g = got.Words[k++];
                string where = $"{id} word {w.GetProperty("index").GetInt32()}";
                Assert.True(w.GetProperty("text").GetString() == g.Text, $"{where}: '{g.Text}' vs '{w.GetProperty("text").GetString()}'.");
                Assert.True(w.GetProperty("script").GetString() == g.Script.ToString(), $"{where}: script {g.Script}.");

                float refRaw = BitsToFloat(w.GetProperty("raw_bits").GetString()!);
                float refCal = BitsToFloat(w.GetProperty("calibrated_bits").GetString()!);
                Assert.Equal(refRaw, (float)w.GetProperty("raw").GetDouble());

                // The map alone, on the reference's own raw value, must give the reference's calibrated bits exactly.
                float mapped = container.Calibration!.Apply(g.Script, refRaw);
                Assert.True(BitConverter.SingleToInt32Bits(mapped) == BitConverter.SingleToInt32Bits(refCal), $"{where}: map gave {mapped:R}, reference {refCal:R}.");

                double diff = Math.Abs(g.RawConfidence - (double)refRaw);
                worstRaw = Math.Max(worstRaw, diff);
                Assert.True(diff <= RawTolerance, $"{where}: raw {g.RawConfidence:R} vs reference {refRaw:R}.");
                words++;
            }
        }

        Assert.Equal(943, words);
        TestContext.Current.SendDiagnosticMessage($"calibration fixture: {words} words, calibrated bits exact, worst raw difference {worstRaw.ToString("E2", CultureInfo.InvariantCulture)} (tolerance {RawTolerance:E1})");
    }

    private static float BitsToFloat(string hex) =>
        BitConverter.Int32BitsToSingle(int.Parse(hex.AsSpan(2), NumberStyles.HexNumber, CultureInfo.InvariantCulture));
}
