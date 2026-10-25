namespace OneOcr.Layout;

/// <summary>The thresholds of <see cref="JunkFilter"/>.</summary>
/// <param name="MaxConfidence">A word is a candidate only when its calibrated confidence is below this.</param>
/// <param name="RelativeHeight">... and its line's ink is shorter than this fraction of the page's reference line height.</param>
/// <param name="MaxHeight">... and shorter than this many pixels (text taller than this is never removed for being small).</param>
/// <param name="AnchorConfidence">A line is a reference line when its mean word confidence is at least this.</param>
/// <param name="TinyHeight">Lines under this many pixels (and under the relative limit) are specks, not text: their words need <paramref name="TinyConfidence"/> to stay.</param>
/// <param name="TinyConfidence">The confidence a word in a tiny line needs to be kept.</param>
/// <param name="MinAnchorLines">Pages with fewer reference lines are left alone: there is no text to be small against.</param>
/// <param name="SpeckPageMin">Dots and ticks are only dropped on a page with at least this many of them: a dusty page has many, a clean page has none or a stray one.</param>
/// <param name="SpeckConfidence">A word made only of dots, commas, quotes, colons and similar marks is a speck read as text when its confidence is below this. 0 turns the rule off.</param>
/// <param name="ControlConfidence">A line that is one short letter-and-bracket word such as <c>O</c>, <c>L]</c> or <c>D</c> is a radio button or checkbox read as text when its confidence is below this. 0 turns the rule off.</param>
internal sealed record JunkParameters(
    float MaxConfidence = 0.5f,
    double RelativeHeight = 0.6,
    int MaxHeight = 12,
    float AnchorConfidence = 0.9f,
    int MinAnchorLines = 3,
    int TinyHeight = 8,
    float TinyConfidence = 0.99f,
    float SpeckConfidence = 0.9f,
    int SpeckPageMin = 3,
    float ControlConfidence = 0.9f);

/// <summary>
/// Removes detections that are noise read as text: specks, scan grain and edges that the detector cut as tiny lines and the
/// recognizer read as a few unsure characters. A word goes only when it is both unsure and small next to the page's own text,
/// so small print that is read with confidence stays, and so does anything on a page where all the text is small.
/// </summary>
internal static class JunkFilter
{
    /// <summary>
    /// Drops noise words from <paramref name="strips"/>: the unsure words of small lines, and lone dots and ticks that are unsure.
    /// Entries become null when no word is left. Returns how many words were dropped.
    /// </summary>
    public static int Apply(OrderStrip?[] strips, JunkParameters p)
    {
        int dropped = DropSmall(strips, p);
        if (p.SpeckConfidence > 0)
        {
            dropped += DropSpecks(strips, p.SpeckConfidence, p.SpeckPageMin);
        }

        if (p.ControlConfidence > 0)
        {
            dropped += DropControls(strips, p.ControlConfidence);
        }

        return dropped;
    }

    // Dust, grain and stray ink are read as dots and ticks. Such a mark on its own is a word to the layout and to the text, and it
    // is nearly never one in the truth: a real full stop or comma is read with its word. Dashes, bullets and separators such as
    // the middle dot are not in the set, since they are words of their own on real pages.
    private static int DropSpecks(OrderStrip?[] strips, float below, int pageMin)
    {
        int found = 0;
        foreach (OrderStrip? s in strips)
        {
            if (s is not null)
            {
                foreach (OcrWord w in s.Words)
                {
                    found += w.Confidence < below && IsSpeck(w.Text) ? 1 : 0;
                }
            }
        }

        if (found < pageMin)
        {
            return 0;
        }

        int dropped = 0;
        for (int i = 0; i < strips.Length; i++)
        {
            OrderStrip? s = strips[i];
            if (s is null)
            {
                continue;
            }

            List<OcrWord>? kept = null;
            for (int k = 0; k < s.Words.Count; k++)
            {
                OcrWord w = s.Words[k];
                if (w.Confidence < below && IsSpeck(w.Text))
                {
                    kept ??= new List<OcrWord>(s.Words.Take(k));
                    dropped++;
                }
                else
                {
                    kept?.Add(w);
                }
            }

            if (kept is not null)
            {
                strips[i] = kept.Count == 0 ? null : new OrderStrip { Leaf = s.Leaf, Box = s.Box, Size = s.Size, Words = kept };
            }
        }

        return dropped;
    }

    // Radio buttons and checkboxes are drawn rings and squares. The detector finds them as lines of their own and the recognizer
    // reads a ring as O, a box as L], D or [], with little confidence. They are controls, not text: a line that is only such a
    // glyph read without confidence goes. Real single letters on a line of their own (a grade, a column label) are read with
    // confidence, and a word such as "Do" has a letter outside the set.
    private const string ControlFirst = "OQLDC[";
    private const string ControlSecond = "])[lI|";

    private static int DropControls(OrderStrip?[] strips, float below)
    {
        int dropped = 0;
        for (int i = 0; i < strips.Length; i++)
        {
            if (strips[i] is not { } s || s.Words.Count == 0)
            {
                continue;
            }

            List<OcrWord>? kept = null;
            for (int k = 0; k < s.Words.Count; k++)
            {
                if (IsControlWord(s.Words, k, below))
                {
                    kept ??= new List<OcrWord>(s.Words.Take(k));
                    dropped++;
                }
                else
                {
                    kept?.Add(s.Words[k]);
                }
            }

            if (kept is not null)
            {
                strips[i] = kept.Count == 0 ? null : new OrderStrip { Leaf = s.Leaf, Box = s.Box, Size = s.Size, Words = kept };
            }
        }

        return dropped;
    }

    // The glyph is alone on its line, or it stands at one end of the line with a gap to its neighbour of at least half its own
    // height (a form's control sits apart from its label; a real letter in running text is a word space away).
    private static bool IsControlWord(IReadOnlyList<OcrWord> words, int k, float below)
    {
        OcrWord w = words[k];
        if (!(w.Confidence < below) || !IsControl(w.Text))
        {
            return false;
        }

        if (words.Count == 1)
        {
            return true;
        }

        double height = w.Box.Bottom - w.Box.Top;
        if (k == 0)
        {
            return words[1].Box.Left - w.Box.Right >= 0.5 * height;
        }

        return k == words.Count - 1 && w.Box.Left - words[k - 1].Box.Right >= 0.5 * height;
    }

    private static bool IsControl(string text) =>
        text.Length is 1 or 2 && ControlFirst.Contains(text[0]) && (text.Length == 1 || ControlSecond.Contains(text[1]));

    private const string SpeckMarks = ".,:;'\"`\u00B4\u2018\u2019\u201C\u201D";

    private static bool IsSpeck(string text)
    {
        if (text.Length == 0)
        {
            return false;
        }

        foreach (char c in text)
        {
            if (!SpeckMarks.Contains(c))
            {
                return false;
            }
        }

        return true;
    }

    // Drops the unsure words of small lines (entries become null when no word is left). The reference height is the median ink
    // height of the lines that were read with high confidence.
    private static int DropSmall(OrderStrip?[] strips, JunkParameters p)
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
        double relative = p.RelativeHeight * reference;
        double limit = Math.Min(relative, p.MaxHeight);
        double tiny = Math.Min(relative, p.TinyHeight);
        int dropped = 0;
        for (int i = 0; i < strips.Length; i++)
        {
            OrderStrip? s = strips[i];
            if (s is null || s.Box.Bottom - s.Box.Top >= limit)
            {
                continue;
            }

            var kept = new List<OcrWord>(s.Words.Count);
            float bar = s.Box.Bottom - s.Box.Top < tiny ? Math.Max(p.MaxConfidence, p.TinyConfidence) : p.MaxConfidence;
            foreach (OcrWord w in s.Words)
            {
                if (w.Confidence >= bar)
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
