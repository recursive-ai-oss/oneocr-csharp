using OneImage;
using OnePdf;

namespace OneOcr.Pdf;

/// <summary>Maps the failures of the PDF and image libraries onto OneOcr's typed exceptions and diagnostics.</summary>
internal static class PdfErrors
{
    /// <summary>The failures a damaged or hostile PDF is expected to cause in OnePDF and the OneImage PDF bridge.</summary>
    public static bool IsLibraryFailure(Exception e) => e is PdfException or ImageException;

    public static OcrException Wrap(Exception e) => e switch
    {
        PdfLimitException => new OcrLimitException("OnePdf", "The PDF exceeds a limit: " + e.Message, e),
        ImageLimitException => new OcrLimitException("OneImage", "The PDF exceeds an image limit: " + e.Message, e),
        _ => new OcrFormatException("The PDF cannot be read: " + e.Message, e),
    };

    /// <summary>Strict reads throw; lenient reads note the failure on the page.</summary>
    public static void Fail(Exception e, PdfOcrOptions options, List<string> diagnostics, string what)
    {
        if (options.Strict)
        {
            throw Wrap(e);
        }

        diagnostics.Add($"{what}: {e.Message}");
    }
}
