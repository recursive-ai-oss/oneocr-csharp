namespace OneOcr.Training;

/// <summary>How a local line compares with what the language model transcribed.</summary>
public enum LineAgreement
{
    /// <summary>The model's text for the line is within the agreement tolerance of the local reading: the line trains the recognizer.</summary>
    Agreed,

    /// <summary>The model's text for the line differs from the local reading by more than the tolerance: the line goes to the review queue.</summary>
    Disagreed,

    /// <summary>No stretch of the model's text resembles the local reading closely enough to compare: the line goes to the review queue.</summary>
    Unaligned,
}

/// <summary>Matches each local line to the stretch of the language model's transcription it most likely corresponds to.</summary>
internal static class LineLabelling
{
    /// <summary>How far ahead of the last match a line may start, in words. A line the model skipped is recovered within this distance.</summary>
    private const int LookAhead = 80;

    /// <summary>How many words more or fewer than the local line a candidate stretch may have (words split or joined by recognition errors).</summary>
    private const int LengthSlack = 3;

    public readonly record struct Match(LineAgreement Agreement, string? LlmText, double Cer);

    /// <summary>
    /// Walks the local lines in order and finds for each the best stretch of <paramref name="llmText"/> (by character error rate
    /// against the normalized local text), searching forward from the end of the previous match. The line agrees when the rate is at
    /// most <paramref name="maxAgreementCer"/>, and is unaligned when even the best stretch is worse than <paramref name="alignMaxCer"/>.
    /// An exact match is taken at once; otherwise the earliest best stretch wins.
    /// </summary>
    public static Match[] Align(IReadOnlyList<string> localLines, string llmText, double maxAgreementCer, double alignMaxCer)
    {
        string[] tokens = TextMetrics.Words(TextMetrics.Normalize(llmText));
        var result = new Match[localLines.Count];
        int cursor = 0;
        for (int i = 0; i < localLines.Count; i++)
        {
            string local = TextMetrics.Normalize(localLines[i]);
            string[] lw = TextMetrics.Words(local);
            if (lw.Length == 0 || tokens.Length == 0)
            {
                result[i] = new Match(LineAgreement.Unaligned, null, 1);
                continue;
            }

            double best = double.PositiveInfinity;
            int bestStart = -1, bestLen = 0;
            int lastStart = Math.Min(tokens.Length - 1, cursor + LookAhead);
            for (int start = cursor; start <= lastStart && best > 0; start++)
            {
                for (int len = Math.Max(1, lw.Length - LengthSlack); len <= lw.Length + LengthSlack && start + len <= tokens.Length; len++)
                {
                    double cer = TextMetrics.Cer(local, string.Join(' ', tokens, start, len));
                    if (cer < best)
                    {
                        best = cer;
                        bestStart = start;
                        bestLen = len;
                    }
                }
            }

            if (bestStart < 0 || best > alignMaxCer)
            {
                result[i] = new Match(LineAgreement.Unaligned, null, best);
                continue;
            }

            cursor = bestStart + bestLen;
            string text = string.Join(' ', tokens, bestStart, bestLen);
            result[i] = new Match(best <= maxAgreementCer ? LineAgreement.Agreed : LineAgreement.Disagreed, text, best);
        }

        return result;
    }
}
