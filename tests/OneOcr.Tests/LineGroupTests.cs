using OneImage;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Tests;

// Reading lines as a group runs the recurrent layers over all of them at once. Each line's result must be exactly what it gets alone.
public class LineGroupTests
{
    private static LineRecognizer Create(byte[] file, KernelPath path) =>
        new(CrnnModel.FromContainer(ModelContainer.Parse(file), OcrLimits.Default), OcrLimits.Default, path);

    private static LineImage Line(int width, int seed)
    {
        byte[] bytes = TestModels.RandomImage(32, width, seed);
        var pixels = new Gray8[bytes.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = new Gray8(bytes[i]);
        }

        return new LineImage(pixels, width);
    }

    private static void AssertSame(LineWords expected, LineWords got, string what)
    {
        Assert.True(expected.Text == got.Text, $"{what}: text '{expected.Text}' vs '{got.Text}'");
        Assert.Equal(BitConverter.SingleToInt32Bits(expected.RawConfidence), BitConverter.SingleToInt32Bits(got.RawConfidence));
        Assert.Equal(expected.Words.Count, got.Words.Count);
        for (int i = 0; i < expected.Words.Count; i++)
        {
            RecognizedWord a = expected.Words[i], b = got.Words[i];
            Assert.True(a.Text == b.Text && a.X0 == b.X0 && a.X1 == b.X1 && a.Characters == b.Characters, $"{what}: word {i}");
            Assert.Equal(BitConverter.SingleToInt32Bits(a.Confidence), BitConverter.SingleToInt32Bits(b.Confidence));
            Assert.Equal(BitConverter.SingleToInt32Bits(a.RawConfidence), BitConverter.SingleToInt32Bits(b.RawConfidence));
            Assert.Equal(a.Chars.Count, b.Chars.Count);
            for (int k = 0; k < a.Chars.Count; k++)
            {
                Assert.True(a.Chars[k] == b.Chars[k], $"{what}: char {k} of word {i}");
            }
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void GroupResultsAreBitIdenticalToReadingEachLineAlone(int shapeIndex)
    {
        var shape = shapeIndex == 0 ? TestModels.Small : TestModels.SmokeLike;
        byte[] file = TestModels.RandomContainer(shape, seed: 21);
        // Widths of every kind: too narrow for the network, odd, equal (ties in the length order), short and long, and more lines than a tile holds.
        int[] widths = [1, 20, 61, 100, 97, 100, 257, 40, 33, 100, 180, 64, 61, 129];
        LineImage[] lines = [.. widths.Select((w, i) => Line(w, 100 + i))];
        LineRecognizer scalar = Create(file, KernelPath.Scalar);
        LineWords[] alone = [.. lines.Select(l => scalar.RecognizeWords(PixelSpans.AsBytes(l.Pixels.AsSpan(0, l.Width * 32)), l.Width, 1, CancellationToken.None))];
        foreach (KernelPath path in Dispatch.Available())
        {
            LineRecognizer recognizer = Create(file, path);
            LineWords[] all = recognizer.RecognizeWordsGroup(lines, CancellationToken.None);
            for (int i = 0; i < lines.Length; i++)
            {
                AssertSame(alone[i], all[i], $"{path} width {widths[i]} in the full group");
            }

            // Another grouping, in another order, must change nothing.
            int[] pick = [13, 6, 2, 9, 0, 4];
            LineWords[] some = recognizer.RecognizeWordsGroup([.. pick.Select(i => lines[i])], CancellationToken.None, withChars: true);
            for (int k = 0; k < pick.Length; k++)
            {
                AssertSame(alone[pick[k]], some[k], $"{path} width {widths[pick[k]]} in a subset");
            }

            LineWords[] one = recognizer.RecognizeWordsGroup([lines[6]], CancellationToken.None);
            AssertSame(alone[6], one[0], $"{path} alone in a group");
        }
    }

    [Fact]
    public void GroupChecksWidthsAndHonoursCancellation()
    {
        byte[] file = TestModels.RandomContainer(TestModels.Small, seed: 5);
        LineRecognizer r = Create(file, Dispatch.Best);
        Assert.Throws<OcrLimitException>(() => r.RecognizeWordsGroup([new LineImage(new Gray8[32 * 9000], 9000)], CancellationToken.None));
        Assert.Throws<ArgumentOutOfRangeException>(() => r.RecognizeWordsGroup([new LineImage(new Gray8[32], 0)], CancellationToken.None));
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        Assert.Throws<OperationCanceledException>(() => r.RecognizeWordsGroup([Line(100, 1), Line(90, 2)], cancelled.Token));
        Assert.Empty(r.RecognizeWordsGroup([], CancellationToken.None));
    }
}
