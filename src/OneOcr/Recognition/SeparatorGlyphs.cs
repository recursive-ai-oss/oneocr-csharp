namespace OneOcr.Recognition;

/// <summary>
/// Reads the small separators of interface text from their pixels (<c>docs/spec/oneocr-page-v1.md</c>, "Separators"). The
/// recognizer turns the middle dot, the bullet and the vertical bar between menu items, bylines and breadcrumbs into hyphens and
/// other marks. A one-character word between two words whose ink is a small round blob at the middle of the line's letters is a
/// middle dot or a bullet (by size); one whose ink is a thin bar taller than every other mark of the line is a vertical bar. Like
/// <see cref="DigitContext"/> it changes a character and nothing else: the word keeps its box and confidence, and words that
/// hold more than one character and words the network read as a letter or digit are never touched; the first and last word of a line are
/// judged only when read as a bullet-like mark (<see cref="EdgeDotReadAs"/>).
/// All arithmetic is on integers, so the result is the same on every path and at every degree of parallelism.
/// </summary>
internal static class SeparatorGlyphs
{
    /// <summary>The marks the network gives a middle dot or a bullet.</summary>
    internal const string DotReadAs = "-\u2013\u2014.,'*\u00B0~_:;^`+=\u00A8\u02D9\u2027\u2219\u22C5\u25CF\u25CB\u25AA\u25E6\u00A4\u00BA\u00AA";

    /// <summary>The marks the network gives a bullet at the start or end of a line (a hyphen, a full stop and the letters are left as read there: a list may be marked with a hyphen).</summary>
    internal const string EdgeDotReadAs = "\u00BA\u00AA\u00B0\u02D9\u2027\u2219\u22C5\u25CF\u25CB\u25AA\u25E6\u00A4";

    /// <summary>The marks the network gives a vertical bar.</summary>
    internal const string BarReadAs = "Il1!\u00A6/\\()[]";

    /// <summary>The middle dot.</summary>
    internal const string MiddleDot = "\u00B7";

    /// <summary>The bullet.</summary>
    internal const string Bullet = "\u2022";

    /// <summary>A line needs this many ink columns outside the glyph before its letters are a reference.</summary>
    internal const int MinReferenceColumns = 8;

    /// <summary>A dot's width and height are within this ratio of each other (percent: width / height from 45 to 160).</summary>
    internal const int MinAspectPercent = 45, MaxAspectPercent = 160;

    /// <summary>A hyphen or dash read with at least this confidence is a dot only when its blob is at most <see cref="MaxSureAspectPercent"/> wide for its height.</summary>
    internal const int MaxSureAspectPercent = 130;

    /// <summary>See <see cref="MaxSureAspectPercent"/>.</summary>
    internal const float SureConfidence = 0.8f;

    /// <summary>A blob up to this percent of the letters' median height wide may be up to twice as wide as it is tall and still be a dot.</summary>
    internal const int TinyWidthPercent = 25;

    /// <summary>A dot fills at least this percent of its bounding box (a disc fills 78).</summary>
    internal const int MinFillPercent = 55;

    /// <summary>A dot's centre is within this percent of the letters' median height from the middle of the letters.</summary>
    internal const int CentrePercent = 25;

    /// <summary>A dot's diameter is at least <see cref="MinDotPercent"/> and at most <see cref="MaxDotPercent"/> percent of the letters' median height; from <see cref="BulletPercent"/> up it is a bullet.</summary>
    internal const int MinDotPercent = 10, MaxDotPercent = 90, BulletPercent = 38;

    /// <summary>A dot the network read as a middle dot becomes a bullet from <see cref="ToBulletPercent"/>; one it read as a bullet becomes a dot up to <see cref="ToDotPercent"/>.</summary>
    internal const int ToBulletPercent = 45, ToDotPercent = 28;

    /// <summary>A bar is taller than every other mark of the line by this percent.</summary>
    internal const int BarTallerPercent = 112;

    /// <summary>A bar is at most this percent of its height wide (and at least 2 pixels are allowed), and at least this percent of the line image high.</summary>
    internal const int BarWidthPercent = 15, BarMinHeightPercent = 50;

    /// <summary>Returns <paramref name="line"/> with the separators among its one-character words read from the pixels, or <paramref name="line"/> itself when none changes.</summary>
    /// <param name="line">The line with its per-character lists.</param>
    /// <param name="gray">The scaled line image, row-major.</param>
    /// <param name="width">Its width in pixels.</param>
    /// <param name="height">Its height in pixels.</param>
    internal static LineWords Apply(LineWords line, ReadOnlySpan<byte> gray, int width, int height)
    {
        IReadOnlyList<RecognizedWord> source = line.Words;
        if (source.Count < 2 || width < 1 || gray.Length < width * height)
        {
            return line;
        }

        bool any = false;
        for (int k = 0; k < source.Count && !any; k++)
        {
            any = IsCandidate(source, k);
        }

        if (!any || WordSegmenter.InkPixels(gray, width, height) is not { } ink)
        {
            return line;
        }

        // Per column: the first and last row with ink (top is -1 for an empty column).
        var top = new int[width];
        var bottom = new int[width];
        for (int x = 0; x < width; x++)
        {
            top[x] = -1;
            for (int y = 0; y < height; y++)
            {
                if (ink[(y * width) + x])
                {
                    top[x] = top[x] < 0 ? y : top[x];
                    bottom[x] = y;
                }
            }
        }

        // Other separators are no reference for the letters: a line of bars would otherwise hold the tallest mark itself.
        var separatorColumns = new bool[width];
        for (int k = 0; k < source.Count; k++)
        {
            if (IsCandidate(source, k))
            {
                for (int x = Math.Max(0, (int)Math.Floor(source[k].X0) - 1); x <= Math.Min(width - 1, (int)Math.Ceiling(source[k].X1)); x++)
                {
                    separatorColumns[x] = true;
                }
            }
        }

        List<RecognizedWord>? words = null;
        for (int k = 0; k < source.Count; k++)
        {
            RecognizedWord w = source[k];
            string? read = IsCandidate(source, k) ? Read(w, ink, top, bottom, separatorColumns, width, height) : null;
            if (read is null)
            {
                words?.Add(w);
                continue;
            }

            words ??= [.. source.Take(k)];
            words.Add(w with { Text = read, Chars = [w.Chars[0] with { Text = read }] });
        }

        return words is null ? line : new LineWords(line.Text, line.RawConfidence, words);
    }

    // A one-character word between two words may be any separator the network misreads; the first and last word of a line only a
    // bullet-like mark (a list marker at the start, a trailing dot of a breadcrumb), since a lone letter, digit or hyphen there is
    // as likely to be what it was read as.
    private static bool IsCandidate(IReadOnlyList<RecognizedWord> words, int k)
    {
        RecognizedWord w = words[k];
        if (w.Text.Length != 1 || w.Chars.Count != 1 || w.Characters != 1 || w.Chars[0].Text != w.Text)
        {
            return false;
        }

        char c = w.Text[0];
        if (k > 0 && k < words.Count - 1)
        {
            return DotReadAs.Contains(c, StringComparison.Ordinal) || c is '·' or '•' || BarReadAs.Contains(c, StringComparison.Ordinal);
        }

        return EdgeDotReadAs.Contains(c, StringComparison.Ordinal) || c is '·' or '•';
    }

    // The glyph's blob, judged against the other ink of the line. Null when it is neither a dot nor a bar.
    private static string? Read(RecognizedWord w, bool[] ink, int[] top, int[] bottom, bool[] separatorColumns, int width, int height)
    {
        int x0 = Math.Max(0, (int)Math.Floor(w.X0)), x1 = Math.Min(width - 1, (int)Math.Ceiling(w.X1) - 1);
        if (x1 < x0 || x1 - x0 + 1 > 4 * height)
        {
            return null;
        }

        // The network's extent of a character can cut a glyph short, so the window is the glyph's whole run of ink columns.
        int seed = -1;
        for (int x = x0; x <= x1 && seed < 0; x++)
        {
            seed = top[x] >= 0 ? x : -1;
        }

        if (seed < 0)
        {
            return null;
        }

        x0 = seed;
        x1 = seed;
        while (x0 > 0 && top[x0 - 1] >= 0)
        {
            x0--;
        }

        while (x1 < width - 1 && top[x1 + 1] >= 0)
        {
            x1++;
        }

        if (x1 - x0 + 1 > height)
        {
            return null;
        }

        // The reference: ink columns outside the glyph's window.
        var extents = new List<int>();
        var middles = new List<int>();
        int tallest = 0;
        for (int x = 0; x < width; x++)
        {
            if (top[x] < 0 || separatorColumns[x] || (x >= x0 - 1 && x <= x1 + 1))
            {
                continue;
            }

            int extent = bottom[x] - top[x] + 1;
            extents.Add(extent);
            middles.Add(top[x] + bottom[x]);
            tallest = Math.Max(tallest, extent);
        }

        if (extents.Count < MinReferenceColumns)
        {
            return null;
        }

        extents.Sort();
        middles.Sort();
        int letterHeight = extents[extents.Count / 2];
        int middle2 = middles[middles.Count / 2];

        if (Blob(ink, width, height, x0, x1, out int bx0, out int by0, out int bw, out int bh, out int area) is false)
        {
            return null;
        }

        char c = w.Text[0];
        if (!BarReadAs.Contains(c, StringComparison.Ordinal))
        {
            int d2 = bw + bh; // twice the diameter
            int centre2 = (2 * by0) + bh - 1; // twice the vertical centre
            // A hyphen the network is sure of is a hyphen unless the blob is clearly as tall as wide: a short hyphen and a dot look alike at this size.
            bool sure = c is '-' or '–' or '—' && w.Confidence >= SureConfidence;
            bool small = bw * 100 <= TinyWidthPercent * letterHeight; // a dot only a pixel or two across is as wide as it is tall, give or take a pixel
            bool round = bw * 100 >= MinAspectPercent * bh && (bw * 100 <= (sure ? MaxSureAspectPercent : MaxAspectPercent) * bh || (small && !sure && bw <= 2 * bh)) && area * 100 >= MinFillPercent * bw * bh && bw >= 1 && bh >= 1;
            bool centred = Math.Abs(centre2 - middle2) * 100 <= 2 * CentrePercent * letterHeight;
            if (!round || !centred || d2 * 100 < 2 * MinDotPercent * letterHeight || d2 * 100 > 2 * MaxDotPercent * letterHeight)
            {
                return null;
            }

            // The network's own dot or bullet is kept unless its size clearly says the other: it gets a margin on both sides.
            string chosen = c == '·' ? (d2 * 100 >= 2 * ToBulletPercent * letterHeight ? Bullet : MiddleDot)
                : c == '•' ? (d2 * 100 <= 2 * ToDotPercent * letterHeight ? MiddleDot : Bullet)
                : d2 * 100 >= 2 * BulletPercent * letterHeight ? Bullet : MiddleDot;
            return chosen[0] == c ? null : chosen;
        }

        bool thin = bw <= Math.Max(2, bh * BarWidthPercent / 100) && area * 100 >= 80 * bw * bh;
        bool tall = bh * 100 >= BarTallerPercent * tallest && bh * 100 >= BarMinHeightPercent * height;
        return thin && tall ? "|" : null;
    }

    // The largest 8-connected piece of ink in columns x0..x1; false unless it holds at least 80% of the window's ink.
    private static bool Blob(bool[] ink, int width, int height, int x0, int x1, out int bx0, out int by0, out int bw, out int bh, out int area)
    {
        int w = x1 - x0 + 1;
        var seen = new bool[w * height];
        var stack = new Stack<int>();
        int total = 0, best = 0, bestX0 = 0, bestY0 = 0, bestX1 = 0, bestY1 = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (!ink[(y * width) + x0 + x])
                {
                    continue;
                }

                total++;
                if (seen[(y * w) + x])
                {
                    continue;
                }

                int count = 0, minX = x, maxX = x, minY = y, maxY = y;
                seen[(y * w) + x] = true;
                stack.Push((y * w) + x);
                while (stack.Count > 0)
                {
                    int p = stack.Pop();
                    int px = p % w, py = p / w;
                    count++;
                    minX = Math.Min(minX, px);
                    maxX = Math.Max(maxX, px);
                    minY = Math.Min(minY, py);
                    maxY = Math.Max(maxY, py);
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int nx = px + dx, ny = py + dy;
                            if (nx >= 0 && nx < w && ny >= 0 && ny < height && !seen[(ny * w) + nx] && ink[(ny * width) + x0 + nx])
                            {
                                seen[(ny * w) + nx] = true;
                                stack.Push((ny * w) + nx);
                            }
                        }
                    }
                }

                if (count > best)
                {
                    best = count;
                    bestX0 = minX;
                    bestX1 = maxX;
                    bestY0 = minY;
                    bestY1 = maxY;
                }
            }
        }

        bx0 = x0 + bestX0;
        by0 = bestY0;
        bw = bestX1 - bestX0 + 1;
        bh = bestY1 - bestY0 + 1;
        area = best;
        return best > 0 && best * 100 >= 80 * total;
    }
}
