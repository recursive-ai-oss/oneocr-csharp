using OneOcr.Decoding;

namespace OneOcr.Fuzz;

/// <summary>
/// Word frequency file bytes through the parser and, when they parse, through lookups of the words they hold and of mutations of
/// them, under tight limits. Only the typed exceptions may come out; anything else is a finding.
/// </summary>
internal static class WordFreqTarget
{
    private static readonly OcrLimits Limits = new() { MaxLanguageModelFileBytes = 1 << 20, MaxWordModelWords = 1 << 14 };

    /// <summary>Inputs that parsed (progress statistic for local runs).</summary>
    public static long Parsed;

    public static void Run(ReadOnlySpan<byte> data)
    {
        try
        {
            WordFrequencyModel model = WordFrequencyModel.Parse(data, Limits);
            Parsed++;
            var probe = new byte[41];
            for (int i = 0; i < probe.Length && i < data.Length; i++)
            {
                probe[i] = data[(data.Length - 1 - i) % data.Length];
            }

            for (int length = 0; length <= probe.Length; length++)
            {
                _ = model.TryFind(probe.AsSpan(0, length), out _);
            }

            _ = model.Quarter("condition");
            _ = model.Contains("a");
        }
        catch (OcrModelException)
        {
        }
        catch (OcrLimitException)
        {
        }
    }
}
