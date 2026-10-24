using OneOcr.Pdf;
using OneOcr.Tests;

namespace OneOcr.Fuzz;

/// <summary>
/// PDF bytes through the whole PDF path: OnePDF parsing, text-layer conversion, image location, rendering at a low resolution and
/// recognition of the regions, under tight limits. Only the typed exceptions may come out; whatever is read must write to page JSON
/// that reads back to the same bytes.
/// </summary>
internal static class PdfTarget
{
    private static readonly OcrLimits Limits = new()
    {
        MaxModelFileBytes = 1 << 20,
        MaxImagePixels = 1 << 22,
        MaxPagesPerCall = 4,
        MaxLinesPerPage = 64,
        MaxBlocksPerPage = 256,
        MaxWordsPerPage = 4096,
        MaxCharsPerPage = 1 << 14,
        MaxPageElements = 1024,
        MaxPageStringLength = 1024,
        MaxLineWidth = 512,
        MaxActivationElements = 1 << 20,
    };

    private static readonly PdfOcrOptions Options = new() { RenderDpi = 72, MaxFileBytes = 1 << 20, CrossCheckLines = 1, PageCount = 4 };

    private static OcrEngine? s_engine;

    public static long Pages;

    private static OcrEngine Engine() =>
        s_engine ??= OcrEngine.Load(
            new MemoryStream(TestModels.RandomContainer(new TestModels.Shape(32, [(4, 2, 2), (4, 2, 2), (8, 2, 1), (8, 2, 1), (8, 2, 1)], 8, 1, 41), 3)), Limits);

    public static void Run(ReadOnlySpan<byte> data)
    {
        IReadOnlyList<OcrPage> pages;
        try
        {
            pages = PdfOcr.Read(data.ToArray(), Engine(), Options);
        }
        catch (OcrFormatException)
        {
            return;
        }
        catch (OcrLimitException)
        {
            return;
        }

        Pages += pages.Count;
        string first = OcrPageJson.ToJson(pages);
        string second = OcrPageJson.ToJson(OcrPageJson.Read(System.Text.Encoding.UTF8.GetBytes(first)));
        if (first != second)
        {
            throw new InvalidOperationException("page JSON does not write back to the same bytes");
        }
    }
}
