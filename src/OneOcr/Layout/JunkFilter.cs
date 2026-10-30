using OneOcr.Decoding;

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
/// <param name="ControlBracketGap">A checkbox read with a closing bracket (<c>D]</c>, <c>Dl]</c>) at the start or end of a line is a control when it is separated from its neighbour by at least this many heights of its own (a plain letter needs half a height).</param>
/// <param name="StrokeConfidence">A single narrow letter, digit or bracket read below this confidence is a stroke (a table rule's edge, a stem) read as a character. 0 turns the rule off.</param>
/// <param name="CleanPageShare">The stroke and contained-fragment rules apply only to a page where at least this share of the characters is in words read with confidence of 0.9 or more: on a clean page a lone narrow character read without confidence is nearly always a stroke, while on a photo of a receipt or a noisy scan thin real characters (<c>1</c>, <c>I</c>, <c>T</c>) are read with little confidence too.</param>
/// <param name="StrokeAspect">A stroke is at most this wide, as a fraction of its height.</param>
/// <param name="ContainedFraction">A word of at most <paramref name="ContainedMaxLength"/> characters whose box lies at least this much inside the box of a longer word is a fragment of that word read twice. 0 turns the rule off.</param>
/// <param name="ContainedMaxLength">The longest word the contained-fragment rule may drop.</param>
/// <param name="AccentConfidence">A word read with an accent or dot under a letter is a candidate for <see cref="StrayAccents"/> only when its confidence is below this. 0 turns the rule off.</param>
/// <param name="AccentMargin">... and the plain spelling must be at least this many quarter-bits (four per doubling) more frequent than the accented one.</param>
/// <param name="AccentPageShare">The rule applies only to a page on which at most this share of the words holding letters has an accented letter.</param>
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
    float ControlConfidence = 0.9f,
    double ControlBracketGap = 0.2,
    float StrokeConfidence = 0.2f,
    double StrokeAspect = 0.5,
    double ContainedFraction = 0.6,
    int ContainedMaxLength = 1,
    double CleanPageShare = 0.95,
    float AccentConfidence = 0.97f,
    int AccentMargin = 12,
    double AccentPageShare = 0.12);

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
        => Apply(strips, p, null, out _);

    /// <summary>
    /// Like <see cref="Apply(OrderStrip?[], JunkParameters)"/>, and with word frequencies also takes stray accents off letters (see
    /// <see cref="StrayAccents"/>); <paramref name="accents"/> is the number of words that changed.
    /// </summary>
    public static int Apply(OrderStrip?[] strips, JunkParameters p, WordFrequencyModel? frequencies, out int accents)
    {
        accents = frequencies is not null && p.AccentConfidence > 0 ? StrayAccents.Apply(strips, p, frequencies) : 0;
        int dropped = DropSmall(strips, p);
        if (p.SpeckConfidence > 0)
        {
            dropped += DropSpecks(strips, p.SpeckConfidence, p.SpeckPageMin);
        }

        if (p.ControlConfidence > 0)
        {
            dropped += DropControls(strips, p.ControlConfidence, p.ControlBracketGap);
        }

        if ((p.StrokeConfidence > 0 || p.ContainedFraction > 0) && ConfidentShare(strips) >= p.CleanPageShare)
        {
            if (p.StrokeConfidence > 0)
            {
                dropped += DropWords(strips, w => IsStroke(w, p.StrokeConfidence, p.StrokeAspect));
            }

            if (p.ContainedFraction > 0)
            {
                dropped += DropContained(strips, p.ContainedFraction, p.ContainedMaxLength);
            }
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
    private const string ControlFirst = "OQLDCE\u0413[]";
    private const string ControlSecond = "])[lI|";

    private static int DropControls(OrderStrip?[] strips, float below, double bracketGap)
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
                if (IsControlWord(s.Words, k, below, bracketGap))
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
    private static bool IsControlWord(IReadOnlyList<OcrWord> words, int k, float below, double bracketGap)
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

        // A box read with its closing bracket is shaped like nothing else on a page; it needs less room than a lone letter.
        double height = w.Box.Bottom - w.Box.Top;
        double needed = (w.Text[^1] is ']' or ')' ? bracketGap : 0.5) * height;
        if (k == 0)
        {
            return words[1].Box.Left - w.Box.Right >= needed;
        }

        return k == words.Count - 1 && w.Box.Left - words[k - 1].Box.Right >= needed;
    }

    // One glyph, two (a box and its bracket) or three (a box with a stray stroke inside, read as <c>Dl]</c>).
    private static bool IsControl(string text) =>
        text.Length is 1 or 2 && ControlFirst.Contains(text[0]) && (text.Length == 1 || ControlSecond.Contains(text[1]))
        || text.Length == 3 && ControlFirst.Contains(text[0]) && "lI|1".Contains(text[1]) && text[2] is ']' or ')';

    // The share of the page's characters that sit in words read with confidence of 0.9 or more (0 for a page without text).
    private static double ConfidentShare(OrderStrip?[] strips)
    {
        long sure = 0, all = 0;
        foreach (OrderStrip? s in strips)
        {
            if (s is null)
            {
                continue;
            }

            foreach (OcrWord w in s.Words)
            {
                all += w.Text.Length;
                sure += w.Confidence >= 0.9f ? w.Text.Length : 0;
            }
        }

        return all == 0 ? 0 : (double)sure / all;
    }

    // A table rule's edge, a letter's stem or a page-edge line is cut by the detector as a thin line of its own and the recognizer reads
    // it as one narrow character ("1", "I", "l", "]") with the lowest confidence it gives. Real narrow characters (a lone "I" or "1")
    // are read with confidence. Dashes, dots and bullets are not strokes: they are wide, and real separators.
    private static bool IsStroke(OcrWord w, float below, double aspect)
    {
        if (w.Text.Length != 1 || !(w.Confidence < below) || !(char.IsLetterOrDigit(w.Text[0]) || w.Text[0] is '[' or ']' or '(' or ')' or '|'))
        {
            return false;
        }

        double height = w.Box.Bottom - w.Box.Top;
        return height > 0 && w.Box.Right - w.Box.Left <= aspect * height;
    }

    // Words are removed where <paramref name="drop"/> says so; strips left without words become null. Returns the number dropped.
    private static int DropWords(OrderStrip?[] strips, Func<OcrWord, bool> drop)
    {
        int dropped = 0;
        for (int i = 0; i < strips.Length; i++)
        {
            if (strips[i] is not { } s)
            {
                continue;
            }

            List<OcrWord>? kept = null;
            for (int k = 0; k < s.Words.Count; k++)
            {
                if (drop(s.Words[k]))
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

    private const int GridCell = 128;

    // A short word that lies inside a longer word's box is the same ink read twice: the detector cut a stem or a letter of the word as
    // a line of its own ("p" over "perfect", "(" over "(with"). Two printed words never share a box like that, so the fragment goes.
    // The longer words are found through a coarse grid, so a page costs one pass however many words it has.
    private static int DropContained(OrderStrip?[] strips, double fraction, int maxLength)
    {
        var all = new List<OcrWord>();
        foreach (OrderStrip? s in strips)
        {
            if (s is not null)
            {
                all.AddRange(s.Words);
            }
        }

        var grid = new Dictionary<long, List<int>>();
        bool any = false;
        for (int i = 0; i < all.Count; i++)
        {
            OcrWord w = all[i];
            any |= w.Text.Length <= maxLength;
            if (w.Text.Length < 2)
            {
                continue;
            }

            ForCells(w.Box, (cx, cy) =>
            {
                long key = (cx << 32) ^ (uint)cy;
                if (!grid.TryGetValue(key, out List<int>? cell))
                {
                    grid[key] = cell = [];
                }

                cell.Add(i);
            });
        }

        if (!any)
        {
            return 0;
        }

        var doomed = new HashSet<OcrWord>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < all.Count; i++)
        {
            OcrWord w = all[i];
            double area = (w.Box.Right - w.Box.Left) * (w.Box.Bottom - w.Box.Top);
            if (w.Text.Length > maxLength || !(area > 0))
            {
                continue;
            }

            bool inside = false;
            ForCells(w.Box, (cx, cy) =>
            {
                if (inside || !grid.TryGetValue((cx << 32) ^ (uint)cy, out List<int>? cell))
                {
                    return;
                }

                foreach (int j in cell)
                {
                    OcrWord o = all[j];
                    if (j == i || o.Text.Length <= w.Text.Length)
                    {
                        continue;
                    }

                    double iw = Math.Min(w.Box.Right, o.Box.Right) - Math.Max(w.Box.Left, o.Box.Left);
                    double ih = Math.Min(w.Box.Bottom, o.Box.Bottom) - Math.Max(w.Box.Top, o.Box.Top);
                    if (iw > 0 && ih > 0 && iw * ih >= fraction * area)
                    {
                        inside = true;
                        return;
                    }
                }
            });
            if (inside)
            {
                doomed.Add(w);
            }
        }

        return doomed.Count == 0 ? 0 : DropWords(strips, doomed.Contains);
    }

    private static void ForCells(OcrBox box, Action<long, long> visit)
    {
        long x0 = (long)Math.Floor(box.Left / GridCell), x1 = (long)Math.Floor(box.Right / GridCell);
        long y0 = (long)Math.Floor(box.Top / GridCell), y1 = (long)Math.Floor(box.Bottom / GridCell);
        for (long cx = x0; cx <= x1; cx++)
        {
            for (long cy = y0; cy <= y1; cy++)
            {
                visit(cx, cy);
            }
        }
    }

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
