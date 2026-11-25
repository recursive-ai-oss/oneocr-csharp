using System.Text;
using OneOcr.Pdf;
using OneOcr.Tests;

namespace OneOcr.Fuzz;

/// <summary>
/// Page content streams through the invisible-text remover and the /ActualText marker, alone and inside a one-page PDF that goes through MakeSearchable.
/// The remover may return null, never throw; a result means at least one show was cut and the stream got no longer than the input
/// plus a fixed allowance per cut. MakeSearchable may end only in the typed exceptions of the PDF path.
/// </summary>
internal static class InvisibleTextTarget
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

    private static OcrEngine? s_engine;

    private static OcrEngine Engine() =>
        s_engine ??= OcrEngine.Load(
            new MemoryStream(TestModels.RandomContainer(new TestModels.Shape(32, [(4, 2, 2), (4, 2, 2), (8, 2, 1), (8, 2, 1), (8, 2, 1)], 8, 1, 41), 3)), Limits);

    public static IEnumerable<(string Name, byte[] Bytes)> Seeds()
    {
        string[] contents =
        [
            "BT /F1 12 Tf 3 Tr 40 700 Td (hidden) Tj ET\nBT /F1 12 Tf 0 Tr 40 660 Td (shown) Tj ET\n",
            "q BT /F1 12 Tf 3 Tr 40 700 Td [(a) -20 (b)] TJ T* (c) ' 1 2 (d) \" ET Q\n",
            "BT 3 Tr /F1 10 Tf 0 0 Td (x) Tj 0 Tr (y) Tj ET q q q Q Q Q\n",
            "BI /W 2 /H 2 /BPC 8 /CS /G ID \u0001\u0002\u0003\u0004 EI BT 3 Tr (z) Tj ET\n",
            "BT /F1 12 Tf 3 Tr <48656c6c6f> Tj (esc \\) \\( \\\\ ) Tj % comment (\n ET\n",
        ];
        for (int i = 0; i < contents.Length; i++)
        {
            yield return ("content-" + i, Encoding.Latin1.GetBytes(contents[i]));
        }
    }

    public static void Run(ReadOnlySpan<byte> data)
    {
        byte[] content = data.ToArray();
        byte[]? cut = InvisibleTextRemover.Remove(content, out int removed);
        if (cut is null)
        {
            if (removed != 0)
            {
                throw new InvalidOperationException("a null result reports removed shows");
            }
        }
        else if (removed <= 0)
        {
            throw new InvalidOperationException("a rewrite without a removed show");
        }
        else if (cut.Length > content.Length + 64L * removed)
        {
            throw new InvalidOperationException("the rewrite grew beyond its allowance");
        }

        // The /ActualText marker reads the same untrusted stream: it may return null, never throw, and each wrapped show adds a fixed amount.
        byte[]? marked = ActualTextMarker.Mark(content, [(-1e4, -1e4, 1e4, 1e4), (0, 0, 612, 792)], out int wrapped);
        if (marked is null ? wrapped != 0 : wrapped <= 0 || marked.Length != content.Length + ((long)wrapped * ActualTextMarker.AddedBytesPerShow))
        {
            throw new InvalidOperationException("the marker's result and count disagree");
        }

        var pdf = new PdfBuilder();
        int catalog = pdf.Reserve(), pages = pdf.Reserve();
        int font = pdf.Add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
        int stream = pdf.AddStream("", content);
        int page = pdf.Add($"<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font} 0 R >> >> /Contents {stream} 0 R >>");
        pdf.Set(pages, $"<< /Type /Pages /Count 1 /Kids [{page} 0 R] >>");
        pdf.Set(catalog, $"<< /Type /Catalog /Pages {pages} 0 R >>");
        byte[] file = pdf.Build(catalog);
        try
        {
            SearchablePdf.MakeSearchable(file, new MemoryStream(), Engine(), new SearchablePdfOptions { Pdf = new PdfOcrOptions { RenderDpi = 72, MaxFileBytes = 1 << 20, CrossCheckLines = 1, PageCount = 1 } });
        }
        catch (OcrFormatException)
        {
        }
        catch (OcrLimitException)
        {
        }
    }
}
