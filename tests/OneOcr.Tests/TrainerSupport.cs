using System.Security.Cryptography;
using System.Threading.Tasks;
using OneImage;
using OneOcr.Kernels;
using OneOcr.Llm;
using OneOcr.Model;
using OneOcr.Recognition;
using OneOcr.Training;

namespace OneOcr.Tests;

/// <summary>
/// Pages of made-up bitmap letters in two styles. The test "font" needs no font file: each letter is a fixed random 5x7 pattern;
/// style A draws it plainly, style B slants and thickens it, which is the shifted font a start model trained on A has not seen.
/// </summary>
internal static class TestNotes
{
    /// <summary>Appends a line to the file named by ONEOCR_TEST_NOTES, when set, so a run can report numbers.</summary>
    public static void Add(string line)
    {
        string? path = Environment.GetEnvironmentVariable("ONEOCR_TEST_NOTES");
        if (path is not null)
        {
            File.AppendAllText(path, line + "\n");
        }
    }
}

internal static class GlyphPages
{
    public const string Alphabet = "abcdefgh";

    public static string[] Charset { get; } = ["<blank>", " ", .. Alphabet.Select(static c => c.ToString())];

    public static TestModels.Shape Shape { get; } = TestModels.Small with { Classes = Charset.Length };

    private static readonly bool[][] Glyphs = [.. Alphabet.Select(static c =>
    {
        var rng = new Random(c * 7919);
        return Enumerable.Range(0, 35).Select(_ => rng.Next(100) < 55).ToArray();
    })];

    public static string[] Lines(int seed, int count)
    {
        var rng = new Random(seed);
        var lines = new string[count];
        for (int i = 0; i < count; i++)
        {
            string Word() => new([.. Enumerable.Range(0, 3 + rng.Next(2)).Select(_ => Alphabet[rng.Next(Alphabet.Length)])]);
            lines[i] = Word() + " " + Word();
        }

        return lines;
    }

    public static Image Page(IReadOnlyList<string> lines, bool styleB)
    {
        const int Cell = 4;
        int w = 40 + (10 * 6 * Cell) + 20, h = 30 + (lines.Count * 70);
        var px = new Gray8[w * h];
        Array.Fill(px, new Gray8(255));
        for (int li = 0; li < lines.Count; li++)
        {
            int x0 = 30, y0 = 30 + (li * 70);
            foreach (char c in lines[li])
            {
                int g = Alphabet.IndexOf(c, StringComparison.Ordinal);
                if (g >= 0)
                {
                    for (int r = 0; r < 7; r++)
                    {
                        int shear = styleB ? (6 - r) : 0;
                        for (int col = 0; col < 5; col++)
                        {
                            if (!Glyphs[g][(r * 5) + col])
                            {
                                continue;
                            }

                            int extra = styleB ? 1 : 0;
                            for (int dy = 0; dy < Cell; dy++)
                            {
                                for (int dx = 0; dx < Cell + extra; dx++)
                                {
                                    px[((y0 + (r * Cell) + dy) * w) + x0 + (col * Cell) + dx + shear] = new Gray8(0);
                                }
                            }
                        }
                    }
                }

                x0 += 6 * Cell;
            }
        }

        return Image.FromPixels<Gray8>(px, w, h);
    }

    public static string Text(IReadOnlyList<string> lines) => string.Join("\n", lines);
}

/// <summary>A start model that reads style A well, built once per test run.</summary>
internal static class PretrainedA
{
    public static readonly Lazy<byte[]> Model = new(Build);

    private static byte[] Build()
    {
        byte[] random = TestContainerWriter.Write("crnn-v0", 32, GlyphPages.Charset, TestModels.ConfigJson(GlyphPages.Shape), TestModels.RandomTensors(GlyphPages.Shape, 3));
        ModelContainer container = ModelContainer.Parse(random);
        OcrEngine engine = OcrEngine.Create(container, null, Dispatch.Best);
        var model = CrnnModel.FromContainer(container, OcrLimits.Default);
        var trainer = new CrnnTrainer(model, Dispatch.Best, new AdamWOptions { LearningRate = 3e-3, WeightDecay = 0 }, null, 5);
        var data = new List<TrainLine>();
        for (int p = 0; p < 40; p++)
        {
            string[] lines = GlyphPages.Lines(1000 + p, 4);
            List<OcrEngine.LineSample> samples = engine.ReadLineSamples(GlyphPages.Page(lines, false), 1, default);
            Assert.Equal(lines.Length, samples.Count);
            for (int i = 0; i < lines.Length; i++)
            {
                data.Add(new TrainLine(samples[i].Gray, samples[i].CropWidth, trainer.Encode(lines[i])));
            }
        }

        var rng = new Random(5);
        for (int step = 0; step < 400; step++)
        {
            trainer.Step([.. Enumerable.Range(0, 16).Select(_ => data[rng.Next(data.Count)])], 4, step < 20 ? 3e-3 * (step + 1) / 20 : 3e-3);
        }

        return ContainerWriter.Write(container, model, "test pretraining");
    }

    /// <summary>Greedy character error rate of a model on pages of the given style (lines found by the engine itself).</summary>
    public static double Cer(byte[] modelBytes, bool styleB, int seed)
    {
        OcrEngine engine = OcrEngine.Create(ModelContainer.Parse(modelBytes), null, Dispatch.Best);
        int errors = 0, chars = 0;
        for (int p = 0; p < 6; p++)
        {
            string[] lines = GlyphPages.Lines(seed + p, 4);
            List<OcrEngine.LineSample> samples = engine.ReadLineSamples(GlyphPages.Page(lines, styleB), 1, default);
            for (int i = 0; i < lines.Length && i < samples.Count; i++)
            {
                errors += LevenshteinForTests(lines[i], samples[i].Text);
                chars += lines[i].Length;
            }
        }

        return errors / (double)chars;
    }

    private static int LevenshteinForTests(string a, string b)
    {
        var d = new int[a.Length + 1, b.Length + 1];
        for (int i = 0; i <= a.Length; i++)
        {
            d[i, 0] = i;
        }

        for (int j = 0; j <= b.Length; j++)
        {
            d[0, j] = j;
        }

        for (int i = 1; i <= a.Length; i++)
        {
            for (int j = 1; j <= b.Length; j++)
            {
                d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1), d[i - 1, j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
            }
        }

        return d[a.Length, b.Length];
    }
}

public class TrainerPretrainSanityTests
{
    [Fact]
    public void The_test_start_model_reads_the_plain_style_and_is_weaker_on_the_shifted_one()
    {
        double a = PretrainedA.Cer(PretrainedA.Model.Value, false, 5000);
        double b = PretrainedA.Cer(PretrainedA.Model.Value, true, 5000);
        TestNotes.Add($"pretrained: A cer {a:F3}, B cer {b:F3}");
        string? export = Environment.GetEnvironmentVariable("ONEOCR_TEST_EXPORT");
        if (export is not null)
        {
            // Leaves a start model and style-B pages with their true text, to try the command-line tool by hand.
            Directory.CreateDirectory(Path.Combine(export, "pages"));
            File.WriteAllBytes(Path.Combine(export, "start.onm"), PretrainedA.Model.Value);
            var texts = new List<string>();
            for (int p = 0; p < 12; p++)
            {
                string[] lines = GlyphPages.Lines(200 + p, 4);
                GlyphPages.Page(lines, true).Save(Path.Combine(export, "pages", $"page{p:D2}.png"));
                texts.Add(GlyphPages.Text(lines));
            }

            File.WriteAllText(Path.Combine(export, "texts.json"), System.Text.Json.JsonSerializer.Serialize(texts));
        }

        Assert.True(a < 0.15, $"style A cer {a}");
        Assert.True(b > a + 0.1, $"style B cer {b} vs A {a}");
    }
}
