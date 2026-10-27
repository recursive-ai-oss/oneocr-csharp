namespace OneOcr.Layout;

internal static partial class ReadingOrder
{
    // Columns whose gutter is not empty. On a scan a stray word, a speck read as a word, or a word whose box runs into the next column
    // keeps every stripe between the columns from being clear, and justified columns set a hair apart overlap by a few pixels at
    // some lines, so neither the full nor the narrow empty-stripe cut finds them and the page is read row by row across its columns.
    //
    // The stripes are found among the words, not the fragments: a stripe is an interval inside the text where at most a tenth of the
    // peak number of words lie, with text standing up to it on both sides. Words are assigned to columns by their centres and a
    // fragment that runs across a cut is cut between its words. The columns must be running text, about as wide as each other. If
    // they are not, the weakest stripe (the one with the most words in it) is dropped and the rest tried again. When lines of text
    // (not a word or two) run across the stripes, there is no column cut: a table or figure above the columns needs the horizontal
    // cut first.
    //
    // A header or footer is not part of the columns. A band of a few short fragments at the top or bottom, set off by an empty band
    // and spread over the region, is read before or after them, and so is a line that runs across a stripe when at most a few lines
    // lie beyond it.
    private static List<int[]>? RobustColumns(List<OrderFragment> f, IReadOnlyList<OrderStrip> strips, int[] all, double ms)
    {
        if (all.Length < 2 * LayoutParameters.RobustMinLines || DistinctLineCount(f, all) < LayoutParameters.RobustMinLines)
        {
            return null;
        }

        int[] reg = all;
        int[] peeledHead = [], peeledFoot = [];
        double pageLeft = all.Min(i => f[i].X0), pageRight = all.Max(i => f[i].X1);
        // A running header or a footer is a few fragments spread over the width with wide gaps between; the first lines of two columns
        // fill the width.
        bool LooksLikeMargin(int[] g) =>
            g.Max(i => f[i].X1) - g.Min(i => f[i].X0) >= LayoutParameters.RobustMarginSpan * (pageRight - pageLeft) &&
            g.Sum(i => f[i].X1 - f[i].X0) <= LayoutParameters.RobustMarginFill * DistinctLineCount(f, g) * (pageRight - pageLeft);

        if (Partition(f, all, false, LayoutParameters.RobustMarginGapEm * ms, null, false) is { } bands)
        {
            if (DistinctLineCount(f, bands[0]) <= LayoutParameters.RobustMarginLines && LooksLikeMargin(bands[0]))
            {
                peeledHead = bands[0];
                bands.RemoveAt(0);
            }

            if (bands.Count > 1 && DistinctLineCount(f, bands[^1]) <= LayoutParameters.RobustMarginLines && LooksLikeMargin(bands[^1]))
            {
                peeledFoot = bands[^1];
                bands.RemoveAt(bands.Count - 1);
            }

            reg = [.. bands.SelectMany(g => g)];
            if (reg.Length < 2 * LayoutParameters.RobustMinLines || DistinctLineCount(f, reg) < LayoutParameters.RobustMinLines)
            {
                return null;
            }
        }

        List<(double Position, int Words, double Width)> stripes = FindStripes(f, strips, reg, ms);

        bool first = true;
        while (stripes.Count > 0)
        {
            List<double> cuts = [.. stripes.Select(s => s.Position)];
            if (CutAtStripes(f, strips, reg, cuts, peeledHead, peeledFoot, ms, out bool crossed) is { } groups)
            {
                return groups;
            }

            // Lines of text, not a stray word, run across the stripes (a table above the columns): the horizontal cut comes first.
            if (crossed && first)
            {
                return null;
            }

            first = false;

            int weakest = 0;
            for (int i = 1; i < stripes.Count; i++)
            {
                if (stripes[i].Words > stripes[weakest].Words || (stripes[i].Words == stripes[weakest].Words && stripes[i].Width < stripes[weakest].Width))
                {
                    weakest = i;
                }
            }

            stripes.RemoveAt(weakest);
        }

        return null;
    }

    // The stripes inside a region's text where few words lie: position (the middle of the longest run at the lowest word count),
    // that lowest count, and the width of the run.
    private static List<(double Position, int Words, double Width)> FindStripes(List<OrderFragment> f, IReadOnlyList<OrderStrip> strips, int[] reg, double ms)
    {
        var edges = new List<(double X, int Delta)>();
        foreach (int id in reg)
        {
            OrderFragment fr = f[id];
            IReadOnlyList<OcrWord> words = strips[fr.Strip].Words;
            for (int k = 0; k < fr.WordCount; k++)
            {
                OcrBox b = words[fr.WordStart + k].Box;
                edges.Add((b.Left, 1));
                edges.Add((b.Right, -1));
            }
        }

        // Closings before openings at one position, so touching words leave no stripe.
        edges.Sort((a, b) =>
        {
            int c = a.X.CompareTo(b.X);
            return c != 0 ? c : a.Delta.CompareTo(b.Delta);
        });

        // Coverage steps: `Count` words cover the span from one edge to the next.
        var steps = new List<(double From, double To, int Count)>();
        int count = 0, peak = 0;
        for (int i = 0; i < edges.Count - 1; i++)
        {
            count += edges[i].Delta;
            peak = Math.Max(peak, count);
            if (edges[i + 1].X > edges[i].X)
            {
                steps.Add((edges[i].X, edges[i + 1].X, count));
            }
        }

        int tolerance = Math.Max(1, (int)Math.Ceiling(LayoutParameters.RobustCrossing * peak));
        double width = LayoutParameters.RobustGutterEm * ms;
        var stripes = new List<(double, int, double)>();
        int s = 0;
        while (s < steps.Count)
        {
            if (steps[s].Count > tolerance)
            {
                s++;
                continue;
            }

            int e = s;
            int low = int.MaxValue;
            while (e < steps.Count && steps[e].Count <= tolerance)
            {
                low = Math.Min(low, steps[e].Count);
                e++;
            }

            // Inside the text, not its margin, and wide enough where it holds that few words.
            if (s > 0 && e < steps.Count && steps[e - 1].To - steps[s].From >= width)
            {
                double bestFrom = 0, bestTo = 0;
                for (int a = s; a < e;)
                {
                    if (steps[a].Count != low)
                    {
                        a++;
                        continue;
                    }

                    int z = a;
                    while (z + 1 < e && steps[z + 1].Count == low)
                    {
                        z++;
                    }

                    if (steps[z].To - steps[a].From > bestTo - bestFrom)
                    {
                        (bestFrom, bestTo) = (steps[a].From, steps[z].To);
                    }

                    a = z + 1;
                }

                // Text must stand on both sides of a stripe, up to it: the ragged right edge of long lines, with a few longer lines
                // reaching beyond, is not a gutter.
                double middle = (bestFrom + bestTo) / 2;
                double reach = LayoutParameters.RobustSideEm * ms;
                if (MeanCoverage(steps, middle - reach, bestFrom) >= LayoutParameters.RobustSideShare * peak &&
                    MeanCoverage(steps, bestTo, middle + reach) >= LayoutParameters.RobustSideShare * peak)
                {
                    stripes.Add((middle, low, bestTo - bestFrom));
                }
            }

            s = e;
        }

        return stripes;
    }

    // The mean number of words over [from, to] (zero outside the text).
    private static double MeanCoverage(List<(double From, double To, int Count)> steps, double from, double to)
    {
        if (to <= from)
        {
            return 0;
        }

        double area = 0;
        foreach ((double a, double b, int count) in steps)
        {
            double overlap = Math.Min(b, to) - Math.Max(a, from);
            if (overlap > 0)
            {
                area += overlap * count;
            }
        }

        return area / (to - from);
    }

    // Cuts the region at the given positions: header, then the columns left to right, then footer. Null when the columns are not
    // all running text.
    private static List<int[]>? CutAtStripes(List<OrderFragment> f, IReadOnlyList<OrderStrip> strips, int[] reg, List<double> cuts, int[] peeledHead, int[] peeledFoot, double ms, out bool crossed)
    {
        crossed = false;
        // A fragment crosses the cuts when its words are on both sides of one (by their centres); one whose words all lie on one side
        // merely reaches into the stripe.
        bool Crosses(OrderFragment fr)
        {
            IReadOnlyList<OcrWord> words = strips[fr.Strip].Words;
            return ZoneOf(cuts, words[fr.WordStart].Box) != ZoneOf(cuts, words[fr.WordStart + fr.WordCount - 1].Box);
        }

        var crossingLines = new HashSet<int>();
        foreach (int id in reg)
        {
            if (Crosses(f[id]))
            {
                crossingLines.Add(f[id].Line);
            }
        }

        // A line that runs across a stripe is a header (footer) of the region when at most a few other lines lie above (below) it:
        // everything from it outwards is read before (after) the columns.
        var bodyBases = new List<double>();
        var seen = new HashSet<int>();
        foreach (int id in reg)
        {
            if (!crossingLines.Contains(f[id].Line) && seen.Add(f[id].Line))
            {
                bodyBases.Add(f[id].Base);
            }
        }

        int margin = Math.Max(LayoutParameters.RobustMarginLines, (int)(LayoutParameters.RobustMarginShare * bodyBases.Count));
        double headEdge = double.MinValue, footEdge = double.MaxValue;
        foreach (int id in reg)
        {
            if (!crossingLines.Contains(f[id].Line))
            {
                continue;
            }

            double y = f[id].Base;
            int above = bodyBases.Count(v => v < y);
            int below = bodyBases.Count - above;

            // More than a few lines beyond the line still make it a header or footer when an empty band sets it off from the text
            // between it and the columns (a table under the columns, then the page footer): inside running text a line that
            // crosses a stripe is noise, not a margin.
            if (above <= LayoutParameters.RobustMarginLines || (above <= margin && BandBeside(f, reg, crossingLines, id, true, ms)))
            {
                headEdge = Math.Max(headEdge, y);
            }
            else if (below <= LayoutParameters.RobustMarginLines || (below <= margin && BandBeside(f, reg, crossingLines, id, false, ms)))
            {
                footEdge = Math.Min(footEdge, y);
            }
        }

        var head = new List<int>(peeledHead);
        var foot = new List<int>(peeledFoot);
        var body = new List<int>();
        foreach (int id in reg)
        {
            if (f[id].Base <= headEdge + (0.5 * ms))
            {
                head.Add(id);
            }
            else if (f[id].Base >= footEdge - (0.5 * ms))
            {
                foot.Add(id);
            }
            else
            {
                body.Add(id);
            }
        }

        var crossedLines = new HashSet<int>();
        foreach (int id in body)
        {
            if (Crosses(f[id]))
            {
                crossedLines.Add(f[id].Line);
            }
        }

        if (crossedLines.Count > margin)
        {
            crossed = true;
            return null;
        }

        int before = f.Count;
        var zones = new List<int>[cuts.Count + 1];
        for (int z = 0; z < zones.Length; z++)
        {
            zones[z] = [];
        }

        foreach (int id in body)
        {
            OrderFragment fr = f[id];
            IReadOnlyList<OcrWord> words = strips[fr.Strip].Words;
            int start = fr.WordStart;
            int last = fr.WordStart + fr.WordCount;
            int zone = ZoneOf(cuts, words[start].Box);
            for (int k = start + 1; k <= last; k++)
            {
                int next = k < last ? ZoneOf(cuts, words[k].Box) : -1;
                if (next == zone)
                {
                    continue;
                }

                if (start == fr.WordStart && k == last)
                {
                    zones[zone].Add(id);
                }
                else
                {
                    OrderFragment piece = MakeFragment(strips[fr.Strip], fr.Strip, start, k);
                    piece.Line = fr.Line;
                    piece.Column = fr.Column;
                    piece.Cell = fr.Cell;
                    f.Add(piece);
                    zones[zone].Add(f.Count - 1);
                }

                start = k;
                zone = next;
            }
        }

        List<int[]> columns = [.. zones.Where(g => g.Count > 0).Select(g => g.ToArray())];
        // Columns of one page are about as wide as each other. A group much wider than another has a gutter missing in it (the three
        // columns under a table, with only the last gutter found): it is not cut here, and the horizontal cut gets its turn.
        double narrowest = double.MaxValue, widest = 0;
        foreach (int[] g in columns)
        {
            double w = g.Max(i => f[i].X1) - g.Min(i => f[i].X0);
            narrowest = Math.Min(narrowest, w);
            widest = Math.Max(widest, w);
        }

        if (columns.Count < 2 || widest > LayoutParameters.RobustColumnRatio * narrowest || !columns.All(g => IsRunningText(f, g, ms)))
        {
            f.RemoveRange(before, f.Count - before);
            return null;
        }

        var groups = new List<int[]>();
        if (head.Count > 0)
        {
            groups.Add([.. head]);
        }

        groups.AddRange(columns);
        if (foot.Count > 0)
        {
            groups.Add([.. foot]);
        }

        return groups;
    }

    // Whether an empty band of at least BlockGapEm lies between fragment `id` and the nearest body fragment on the inner side
    // (below it for a header, above it for a footer).
    private static bool BandBeside(List<OrderFragment> f, int[] reg, HashSet<int> crossingLines, int id, bool header, double ms)
    {
        double y = f[id].Base;
        double nearest = header ? double.MaxValue : double.MinValue;
        foreach (int other in reg)
        {
            if (crossingLines.Contains(f[other].Line))
            {
                continue;
            }

            if (header && f[other].Base > y + (0.5 * ms))
            {
                nearest = Math.Min(nearest, f[other].Top);
            }
            else if (!header && f[other].Base < y - (0.5 * ms))
            {
                nearest = Math.Max(nearest, f[other].Bot);
            }
        }

        return header ? nearest - f[id].Bot >= LayoutParameters.BlockGapEm * ms : f[id].Top - nearest >= LayoutParameters.BlockGapEm * ms;
    }

    // Which of the stripes' cuts a word lies after: the number of cuts left of its centre.
    private static int ZoneOf(List<double> cuts, OcrBox box)
    {
        double centre = (box.Left + box.Right) / 2;
        int zone = 0;
        foreach (double c in cuts)
        {
            if (c < centre)
            {
                zone++;
            }
        }

        return zone;
    }

    // A column of running text: enough lines, and the widest quarter of its fragments are full lines (not labels or cells).
    private static bool IsRunningText(List<OrderFragment> f, int[] g, double ms)
    {
        var widths = new double[g.Length];
        for (int i = 0; i < g.Length; i++)
        {
            widths[i] = (f[g[i]].X1 - f[g[i]].X0) / Math.Max(ms, 1e-6);
        }

        Array.Sort(widths);
        return DistinctLineCount(f, g) >= LayoutParameters.SoftColumnLines && widths[(int)(0.75 * (widths.Length - 1))] >= LayoutParameters.SoftColumnWidthEm;
    }
}
