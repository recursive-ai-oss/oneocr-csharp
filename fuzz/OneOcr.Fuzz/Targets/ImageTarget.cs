using OneImage;
using OneOcr.Tests;

namespace OneOcr.Fuzz;

/// <summary>
/// Image file bytes through OneImage's decoder and then through the whole page path of <see cref="OcrEngine"/>: gray conversion,
/// orientation and skew, line finding or the page detector, line preprocessing, recognition, layout and the page model. The last
/// input byte selects the engine (classical line finder, or a small random page detector) and is not part of the image.
/// Decoding belongs to OneImage: its typed <see cref="ImageException"/> is an accepted outcome here, and any other exception it
/// throws is reported as a finding for that library. Everything after decoding is ours: only <see cref="OcrLimitException"/> may
/// come out, and a page that is returned must write to JSON that reads back to the same bytes.
/// </summary>
internal static class ImageTarget
{
    private static readonly OcrLimits Limits = new()
    {
        MaxModelFileBytes = 1 << 20,
        MaxImagePixels = 1 << 20,
        MaxPagesPerCall = 2,
        MaxLinesPerPage = 64,
        MaxBlocksPerPage = 256,
        MaxWordsPerPage = 4096,
        MaxCharsPerPage = 1 << 14,
        MaxPageElements = 1024,
        MaxPageStringLength = 1024,
        MaxLineWidth = 512,
        MaxActivationElements = 1 << 20,
        MaxDetectorSide = 512,
    };

    private static readonly LoadOptions Load = new() { Limits = new ImageLimits { MaxPixelsPerFrame = 1 << 20, MaxFrames = 2 } };

    private static readonly OcrOptions Options = new() { MaxDegreeOfParallelism = 1 };

    private static OcrEngine? s_plain, s_detector;

    /// <summary>Inputs that decoded and went through the engine (progress statistic for local runs).</summary>
    public static long Pages;

    private static OcrEngine Plain() =>
        s_plain ??= OcrEngine.Load(
            new MemoryStream(TestModels.RandomContainer(new TestModels.Shape(32, [(4, 2, 2), (4, 2, 2), (8, 2, 1), (8, 2, 1), (8, 2, 1)], 8, 1, 41), 3)), Limits);

    private static OcrEngine WithDetector()
    {
        if (s_detector is null)
        {
            var shape = new TestDetectorModels.Shape(16, [16, 24, 40, 56], [1, 2, 1, 1], 20, 12, 28, 2, TestDetectorModels.AllClasses);
            s_detector = Plain().WithDetector(new MemoryStream(TestDetectorModels.Random(shape, 2)), Limits);
        }

        return s_detector;
    }

    public static void Run(ReadOnlySpan<byte> data)
    {
        if (data.Length < 2)
        {
            return;
        }

        OcrEngine engine = (data[^1] & 1) == 0 ? Plain() : WithDetector();
        byte[] bytes = data[..^1].ToArray();
        try
        {
            Image image = Image.Load(bytes, Load);
            OcrPage page = engine.Ocr(image, Options);
            Pages++;
            string first = OcrPageJson.ToJson([page]);
            string second = OcrPageJson.ToJson(OcrPageJson.Read(System.Text.Encoding.UTF8.GetBytes(first)));
            if (first != second)
            {
                throw new InvalidOperationException("page JSON does not write back to the same bytes");
            }
        }
        catch (ImageException)
        {
        }
        catch (OcrLimitException)
        {
        }
    }

    /// <summary>Writes small valid images in several formats, gray and colour, as seeds.</summary>
    public static void WriteSeeds(string dir)
    {
        Directory.CreateDirectory(dir);
        var rng = new Random(5);
        foreach (var (w, h, name) in new[] { (96, 64, "small"), (200, 120, "wide"), (33, 47, "odd") })
        {
            var px = new Gray8[w * h];
            for (int i = 0; i < px.Length; i++)
            {
                px[i] = new Gray8(250);
            }

            // Dark strokes in rows, so the line finder and the detector both have something to find.
            for (int row = 8; row + 8 < h; row += 20)
            {
                for (int x = 6; x < w - 6; x++)
                {
                    if (rng.Next(3) != 0)
                    {
                        for (int y = row; y < row + 6; y++)
                        {
                            px[(y * w) + x] = new Gray8(20);
                        }
                    }
                }
            }

            Image image = Image.FromPixels<Gray8>(px, w, h);
            foreach (var (ext, format) in new (string, IImageFormat)[] { ("png", PngFormat.Instance), ("jpg", JpegFormat.Instance), ("bmp", BmpFormat.Instance) })
            {
                using var ms = new MemoryStream();
                image.Save(ms, format, new SaveOptions(), System.Threading.CancellationToken.None);
                ms.WriteByte(name == "odd" ? (byte)1 : (byte)0);
                File.WriteAllBytes(Path.Combine(dir, $"{name}.{ext}.seed"), ms.ToArray());
            }
        }
    }
}
