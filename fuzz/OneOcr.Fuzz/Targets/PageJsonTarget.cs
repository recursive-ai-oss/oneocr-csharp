using System.Text;
using OneOcr;

namespace OneOcr.Fuzz;

/// <summary>
/// oneocr.page JSON bytes through the reader. Only the typed exceptions may come out; whatever parses must write to JSON that
/// reads back and writes again to the same bytes.
/// </summary>
internal static class PageJsonTarget
{
    private static readonly OcrLimits Limits = new()
    {
        MaxPageJsonBytes = 1 << 20, MaxJsonDepth = 16, MaxPagesPerCall = 8, MaxBlocksPerPage = 256, MaxLinesPerPage = 256,
        MaxWordsPerPage = 2048, MaxCharsPerPage = 4096, MaxPageElements = 1024, MaxPageStringLength = 1024,
    };

    public static void Run(ReadOnlySpan<byte> data)
    {
        IReadOnlyList<OcrPage> pages;
        try
        {
            pages = OcrPageJson.Read(data, Limits);
        }
        catch (OcrFormatException)
        {
            return;
        }
        catch (OcrLimitException)
        {
            return;
        }

        string first = OcrPageJson.ToJson(pages);
        string second = OcrPageJson.ToJson(OcrPageJson.Read(Encoding.UTF8.GetBytes(first), Limits));
        if (first != second)
        {
            throw new InvalidOperationException("page JSON does not write back to the same bytes");
        }
    }
}
