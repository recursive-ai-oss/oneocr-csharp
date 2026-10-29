using System.Numerics;
using OneImage;
using OneImage.Text;

namespace OneOcr.Fuzz;

/// <summary>
/// The image target with the shipped models: image file bytes through OneImage's decoder and the whole page path of the bundled
/// engine (real detector, recognizer, word joiner, leader and table code), which random weights never reach. Pages are limited to
/// 2^18 pixels so an input stays fast. Same contract as <see cref="ImageTarget"/>; the last byte is ignored. Needs a build with the
/// model files (see fuzz/README.md).
/// </summary>
internal static class ImageRealTarget
{
    private static readonly OcrLimits Limits = new() { MaxImagePixels = 1 << 18, MaxPagesPerCall = 2 };

    private static readonly LoadOptions Load = new() { Limits = new ImageLimits { MaxPixelsPerFrame = 1 << 18, MaxFrames = 2 } };

    private static readonly OcrOptions Options = new() { MaxDegreeOfParallelism = 1 };

    private static OcrEngine? s_engine;

    public static long Pages;

    public static void Run(ReadOnlySpan<byte> data)
    {
        if (data.Length < 2)
        {
            return;
        }

        s_engine ??= OcrEngine.LoadBundled(Limits);
        try
        {
            Image image = Image.Load(data[..^1].ToArray(), Load);
            OcrPage page = s_engine.Ocr(image, Options);
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

    /// <summary>Small text pages (needs a font file; ONEOCR_TEST_FONT or DejaVu Sans), plain, ruled and photographed-looking.</summary>
    public static void WriteSeeds(string dir)
    {
        Directory.CreateDirectory(dir);
        string font = Environment.GetEnvironmentVariable("ONEOCR_TEST_FONT") ?? "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
        Font f = new FontCollection().Add(font).CreateFont(15);
        string[] lines =
        [
            "Invoice 4711 issued 12 March 2024", "Acme Holdings Ltd, 14 Long Street", "Widget ........ 2 x 12.50     25.00",
            "Gadget ........ 1 x  7.20      7.20", "Total due: EUR 32.20 (incl. 19% VAT)", "Pay within 14 days to IBAN DE89 3704 0044 0532 0130 00",
        ];
        foreach (var (name, w, h, rule, noise) in new[] { ("plain", 380, 170, false, 0), ("ruled", 380, 170, true, 0), ("noisy", 380, 170, false, 40) })
        {
            var px = new Rgba32[w * h];
            Array.Fill(px, new Rgba32(255, 255, 255, 255));
            if (rule)
            {
                for (int y = 120; y < 122; y++)
                {
                    for (int x = 8; x < w - 8; x++)
                    {
                        px[(y * w) + x] = new Rgba32(0, 0, 0, 255);
                    }
                }
            }

            Image image = Image.FromPixels<Rgba32>(px, w, h);
            for (int i = 0; i < lines.Length; i++)
            {
                image = image.DrawText(lines[i], new TextOptions(f) { Origin = new Vector2(12, 10 + (i * 26)) }, new Rgba32(0, 0, 0, 255));
            }

            if (noise > 0)
            {
                var gray = new Gray8[w * h];
                image.CopyPixelsTo<Gray8>(gray, new ImageRect(0, 0, w, h), 1);
                var rng = new Random(7);
                for (int i = 0; i < gray.Length; i++)
                {
                    gray[i] = new Gray8((byte)Math.Clamp(gray[i].V + rng.Next(-noise, noise + 1), 0, 255));
                }

                image = Image.FromPixels<Gray8>(gray, w, h);
            }

            foreach (var (ext, format) in new (string, IImageFormat)[] { ("png", PngFormat.Instance), ("jpg", JpegFormat.Instance) })
            {
                using var ms = new MemoryStream();
                image.Save(ms, format, new SaveOptions(), CancellationToken.None);
                ms.WriteByte(0);
                File.WriteAllBytes(Path.Combine(dir, $"{name}.{ext}.seed"), ms.ToArray());
            }
        }
    }
}
