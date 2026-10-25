namespace OneOcr.Layout;

/// <summary>The thresholds of <see cref="JunkFilter"/>.</summary>
/// <param name="MaxConfidence">A word is a candidate only when its calibrated confidence is below this.</param>
/// <param name="RelativeHeight">... and its line's ink is shorter than this fraction of the page's reference line height.</param>
/// <param name="MaxHeight">... and shorter than this many pixels (text taller than this is never removed for being small).</param>
/// <param name="AnchorConfidence">A line is a reference line when its mean word confidence is at least this.</param>
/// <param name="MinAnchorLines">Pages with fewer reference lines are left alone: there is no text to be small against.</param>
internal sealed record JunkParameters(
    float MaxConfidence = 0.5f,
    double RelativeHeight = 0.6,
    int MaxHeight = 12,
    float AnchorConfidence = 0.9f,
    int MinAnchorLines = 3);

/// <summary>
/// Removes detections that are noise read as text: specks, scan grain and edges that the detector cut as tiny lines and the
/// recognizer read as a few unsure characters. A word goes only when it is both unsure and small next to the page's own text,
/// so small print that is read with confidence stays, and so does anything on a page where all the text is small.
/// </summary>
internal static class JunkFilter
{
    /// <summary>
    /// Drops the unsure words of small lines from <paramref name="strips"/> (entries become null when no word is left) and returns
    /// how many words were dropped. The reference height is the median ink height of the lines that were read with high confidence.
    /// </summary>
    public static int Apply(OrderStrip?[] strips, JunkParameters p)
    {
        var anchors = new List<int>();
        foreach (OrderStrip? s in strips)
        {
            if (s is not null && s.Words.Count > 0 && MeanConfidence(s) >= p.AnchorConfidence)
            {
                anchors.Add((int)Math.Round(s.Box.Bottom - s.Box.Top));
            }
        }

        if (anchors.Count < p.MinAnchorLines)
        {
            return 0;
        }

        anchors.Sort();
        double reference = anchors[anchors.Count / 2];
        double limit = Math.Min(p.RelativeHeight * reference, p.MaxHeight);
        int dropped = 0;
        for (int i = 0; i < strips.Length; i++)
        {
            OrderStrip? s = strips[i];
            if (s is null || s.Box.Bottom - s.Box.Top >= limit)
            {
                continue;
            }

            var kept = new List<OcrWord>(s.Words.Count);
            foreach (OcrWord w in s.Words)
            {
                if (w.Confidence >= p.MaxConfidence)
                {
                    kept.Add(w);
                }
            }

            if (kept.Count == s.Words.Count)
            {
                continue;
            }

            dropped += s.Words.Count - kept.Count;
            strips[i] = kept.Count == 0 ? null : new OrderStrip { Leaf = s.Leaf, Box = s.Box, Size = s.Size, Words = kept };
        }

        return dropped;
    }

    private static double MeanConfidence(OrderStrip s)
    {
        double sum = 0;
        foreach (OcrWord w in s.Words)
        {
            sum += w.Confidence;
        }

        return sum / s.Words.Count;
    }
}
