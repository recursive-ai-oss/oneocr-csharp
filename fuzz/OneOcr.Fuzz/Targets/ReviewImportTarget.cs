using System.Text;
using OneOcr.Training;

namespace OneOcr.Fuzz;

/// <summary>
/// Review-import bytes through the parser under tight limits. Only the typed exceptions may come out, and whatever parses must
/// respect the limits and carry no control characters.
/// </summary>
internal static class ReviewImportTarget
{
    private static readonly OcrLimits Limits = new()
    {
        MaxReviewImportBytes = 1 << 16, MaxReviewItems = 64, MaxReviewTextLength = 64, MaxJsonDepth = 8,
    };

    /// <summary>Inputs that parsed (progress statistic for local runs).</summary>
    public static long Parsed;

    public static void Run(ReadOnlySpan<byte> data)
    {
        ReviewFile file;
        try
        {
            file = ReviewImport.Parse(data, Limits);
        }
        catch (OcrFormatException)
        {
            return;
        }
        catch (OcrLimitException)
        {
            return;
        }

        Parsed++;
        if (file.Corrections.Count + file.Unreviewed > Limits.MaxReviewItems)
        {
            throw new InvalidOperationException("more items than the limit");
        }

        foreach (ReviewCorrection c in file.Corrections)
        {
            if (c.Id.Length is 0 or > 128 || c.Text.Length > Limits.MaxReviewTextLength)
            {
                throw new InvalidOperationException("an id or text outside its bounds was accepted");
            }

            foreach (Rune r in (c.Id + c.Text).EnumerateRunes())
            {
                if (Rune.IsControl(r) || r.Value == 0xFFFD)
                {
                    throw new InvalidOperationException("a control character was accepted");
                }
            }
        }
    }
}
