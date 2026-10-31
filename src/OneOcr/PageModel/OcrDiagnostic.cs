using System.Diagnostics.CodeAnalysis;

namespace OneOcr;

/// <summary>How much attention a <see cref="OcrDiagnostic"/> asks for.</summary>
public enum OcrDiagnosticSeverity
{
    /// <summary>Something the library did or decided that the caller may want to know; the result is as good as it gets.</summary>
    Info = 0,

    /// <summary>Something that may make the result worse than the caller expects, or that the caller can fix by passing an option.</summary>
    Warning = 1,
}

/// <summary>
/// A note about one result: a recoverable condition the library handled instead of throwing. The <see cref="Code"/> is stable
/// across releases and safe to switch on; the <see cref="Message"/> is for people and may change in wording.
/// </summary>
public sealed record OcrDiagnostic
{
    /// <summary>Creates an empty diagnostic; set <see cref="Code"/>, <see cref="Severity"/> and <see cref="Message"/>.</summary>
    public OcrDiagnostic()
    {
    }

    /// <summary>Creates a diagnostic.</summary>
    /// <param name="code">The stable code, for example one of <see cref="OcrDiagnosticCodes"/>.</param>
    /// <param name="severity">How much attention the note asks for.</param>
    /// <param name="message">The note in plain words.</param>
    /// <exception cref="ArgumentNullException"><paramref name="code"/> or <paramref name="message"/> is null.</exception>
    [SetsRequiredMembers]
    public OcrDiagnostic(string code, OcrDiagnosticSeverity severity, string message)
    {
        ArgumentNullException.ThrowIfNull(code);
        ArgumentNullException.ThrowIfNull(message);
        Code = code;
        Severity = severity;
        Message = message;
    }

    /// <summary>The stable code, for example <c>OCR103</c> (see <see cref="OcrDiagnosticCodes"/>).</summary>
    public required string Code { get; init; }

    /// <summary>How much attention the note asks for.</summary>
    public required OcrDiagnosticSeverity Severity { get; init; }

    /// <summary>The note in plain words. This is the text written to the <c>diagnostics</c> array of the page JSON.</summary>
    public required string Message { get; init; }

    /// <summary>The code and the message, for display.</summary>
    /// <returns><c>CODE: message</c>.</returns>
    public override string ToString() => Code + ": " + Message;
}

/// <summary>The codes of the diagnostics the library produces. New codes may be added in later releases; existing ones keep their meaning.</summary>
public static class OcrDiagnosticCodes
{
    /// <summary>The code of a diagnostic read back from page JSON, which stores the message only.</summary>
    public const string Unspecified = "OCR000";

    /// <summary>The page's resolution was estimated from the size of its text (the file declared none and <see cref="OcrOptions.Dpi"/> was not set).</summary>
    public const string DpiEstimated = "OCR101";

    /// <summary>The page has no text to estimate its resolution from, so <see cref="OcrPage.UnitsPerInch"/> is 72.</summary>
    public const string DpiDefaulted = "OCR102";

    /// <summary>The page content was turned by 90, 180 or 270 degrees; <see cref="OcrPage.Orientation"/> says by how much.</summary>
    public const string PageTurned = "OCR103";

    /// <summary>Tilted text lines were straightened before reading; <see cref="OcrPage.SkewDegrees"/> says by how much.</summary>
    public const string PageStraightened = "OCR104";

    /// <summary>Tilted text lines were read as they are because the straightened page would exceed <see cref="OcrLimits.MaxImagePixels"/>.</summary>
    public const string StraightenSkipped = "OCR105";

    /// <summary>The first reading was weak, so the page was read at several angles and the best was kept.</summary>
    public const string AngleSearched = "OCR106";

    /// <summary>Words read from specks, grain or stray marks were dropped as noise.</summary>
    public const string NoiseDropped = "OCR107";

    /// <summary>Stray accents and dots were taken off letters that were read from grain.</summary>
    public const string StrayAccentsFixed = "OCR108";

    /// <summary>A line was skipped because it would be wider than <see cref="OcrLimits.MaxLineWidth"/> at the model's input height.</summary>
    public const string LineSkipped = "OCR109";

    /// <summary>The model carries no calibration, so calibrated confidence equals raw confidence.</summary>
    public const string NoCalibration = "OCR110";

    /// <summary>The image has several frames and only some were read, or a frame was ignored.</summary>
    public const string FramesIgnored = "OCR111";

    /// <summary>The extraction model reads the page as another document type than the one asked for.</summary>
    public const string DocumentTypeDiffers = "OCR201";

    /// <summary>A property of the requested type has no class in the extraction model and keeps its default.</summary>
    public const string PropertyUnmapped = "OCR202";

    /// <summary>A value could not be converted to a property's type.</summary>
    public const string ValueNotConverted = "OCR203";

    /// <summary>The page has no words, so nothing was extracted.</summary>
    public const string PageHasNoWords = "OCR204";

    /// <summary>A note from merging a language model's reading into a page.</summary>
    public const string LlmRead = "OCR401";

    /// <summary>A note from reading a PDF page (see <c>RecursiveAI.OneOcr.Pdf</c>).</summary>
    public const string Pdf = "OCR301";
}

/// <summary>Builds the diagnostics the library itself produces.</summary>
internal static class Diag
{
    public static OcrDiagnostic Info(string code, string message) => new(code, OcrDiagnosticSeverity.Info, message);

    public static OcrDiagnostic Warn(string code, string message) => new(code, OcrDiagnosticSeverity.Warning, message);

    /// <summary>Wraps plain messages (read from JSON) as diagnostics with the unspecified code.</summary>
    public static OcrDiagnostic FromMessage(string message) => new(OcrDiagnosticCodes.Unspecified, OcrDiagnosticSeverity.Info, message);
}
