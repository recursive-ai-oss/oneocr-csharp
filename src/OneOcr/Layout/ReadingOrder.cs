// Adapted from OnePDF (src/OnePdf/Text/Extraction/TextLayout.cs, LayoutParameters.cs): the fragment, X–Y cut, table test and
// line-kind split of its reading-order analysis. OnePDF works on glyphs with exact boxes; here the input is recognized words
// whose boxes are ink extents, so the stages that build words and lines out of glyphs (baseline clustering, script runs, bidi,
// vertical writing) are replaced by splitting a recognized line at wide gaps and joining the cells of staggered strips by their
// vertical centre. The thresholds keep OnePDF's values and meaning, in units of the size of the text concerned (em). Only
// left-to-right horizontal text is handled.

namespace OneOcr.Layout;

/// <summary>The thresholds of the layout analysis; distances are in units of the size of the text concerned (em).</summary>
internal static class LayoutParameters
{
    /// <summary>A gap wider than this between two words of a recognized line splits the line into fragments (cells or columns).</summary>
    internal const double FragmentGapEm = 1.25;

    /// <summary>An empty vertical stripe at least this wide, crossing every fragment of a region, separates columns.</summary>
    internal const double ColumnGutterEm = 1.25;

    /// <summary>An empty horizontal band at least this tall separates blocks.</summary>
    internal const double BlockGapEm = 0.5;

    /// <summary>Two strips (pieces of lines found in different regions) are on one line when their centres differ by at most this fraction of the smaller size.</summary>
    internal const double LineToleranceEm = 0.3;

    /// <summary>A region is a table when its fragments' median width is at most this many em ...</summary>
    internal const double TableCellWidthEm = 10;

    /// <summary>... it has at least this many rows ...</summary>
    internal const int TableMinRows = 3;

    /// <summary>... and at least this fraction of the rows have fragments in more than one column.</summary>
    internal const double TableRowFraction = 0.6;

    /// <summary>The topmost band of a region is a header row (a bar of menu items) when it is at most this many em tall ...</summary>
    internal const double HeaderHeightEm = 2;

    /// <summary>... spans at least this fraction of the region's width ...</summary>
    internal const double HeaderSpan = 0.6;

    /// <summary>... and stands clear of the rest by at least this many em.</summary>
    internal const double HeaderGapEm = 3;

    /// <summary>A column is inside another when the other has text above and below it, each at least this many em away.</summary>
    internal const double EmbedMarginEm = 2;

    /// <summary>Rows of fields or cards are separated by empty bands of at least this many em.</summary>
    internal const double RowGapEm = 1.5;

    /// <summary>A gap inside a row band that is at most this fraction of the gaps on both sides does not separate rows (a label above its value).</summary>
    internal const double WeakGapRatio = 0.8;

    /// <summary>A table's columns are filled in at least this fraction of its rows, on average.</summary>
    internal const double TableMinColumnFill = 0.6;

    /// <summary>A table has no row of fragments taller than this multiple of the median row (a card's large number).</summary>
    internal const double TableMaxRowSizeRatio = 1.6;

    /// <summary>Lines whose spacing alternates short and long by at least this factor are fields (label above value), not table rows.</summary>
    internal const double FieldPitchRatio = 1.25;
}

/// <summary>A recognized strip handed to the reading-order analysis: one line found in one leaf region, with its words left to right.</summary>
internal sealed class OrderStrip
{
    public required int Leaf { get; init; }
    public required OcrBox Box { get; init; }

    /// <summary>The text size (em) of the strip.</summary>
    public required double Size { get; init; }

    public required IReadOnlyList<OcrWord> Words { get; init; }
}

/// <summary>A run of words of one strip with no wide gap inside, with the line (across strips) it belongs to.</summary>
internal sealed class OrderFragment
{
    public int Strip;
    public int WordStart;
    public int WordCount;
    public int Line;

    /// <summary>The region of the page the fragment's strip was cut from.</summary>
    public int Leaf;
    public double X0, X1, Top, Bot, Base, Size;

    /// <summary>Column index inside a table block; 0 elsewhere.</summary>
    public int Column;

    /// <summary>Index into <see cref="OrderBlock.Cells"/> of the table cell the fragment lies in; -1 outside a ruled table.</summary>
    public int Cell = -1;
}

/// <summary>A cell of a table whose structure was read: from its ruling lines, or from the gaps between its columns.</summary>
internal sealed record OrderCell(int Row, int Column, int RowSpan, int ColumnSpan, OcrBox Box);

/// <summary>One line of a block: its fragments left to right.</summary>
internal sealed class OrderLine
{
    public required List<OrderFragment> Fragments { get; init; }
}

/// <summary>A block in reading order.</summary>
internal sealed class OrderBlock
{
    public required bool IsTable { get; init; }
    public required int Columns { get; init; }
    public required List<OrderLine> Lines { get; init; }

    /// <summary>The cells of a table whose structure was read, empty ones included; null for every other block.</summary>
    public IReadOnlyList<OrderCell>? Cells { get; init; }

    /// <summary>Number of rows of a table whose cells are given.</summary>
    public int Rows { get; init; }

    /// <summary>Probability that the block is a table, when its structure was read; null for ruled tables, which have a fixed value.</summary>
    public float? TableConfidence { get; init; }
}

internal static partial class ReadingOrder
{
    /// <summary>Puts the strips of a page into reading order: blocks of lines, with tables found.</summary>
    /// <param name="strips">The recognized strips; strips without words are ignored.</param>
    /// <param name="cancellationToken">Cancels the analysis.</param>
    /// <returns>The blocks in reading order.</returns>
    public static List<OrderBlock> Arrange(IReadOnlyList<OrderStrip> strips, CancellationToken cancellationToken)
    {
        List<OrderFragment> frags = Fragment(strips);
        if (frags.Count == 0)
        {
            return [];
        }

        AssignLines(strips, frags);
        var blocks = new List<OrderBlock>();
        foreach ((int[] ids, bool table, int columns) in XyCut(frags, cancellationToken))
        {
            OrderBlock? block = MakeBlock(frags, ids, table, columns);
            if (block is not null)
            {
                blocks.Add(block);
            }
        }

        return blocks;
    }

    // ---- fragments ----------------------------------------------------------------------------------------------------------

    // OnePDF TextLayout.Fragment: a line is cut where the gap to the previous word is wider than FragmentGapEm of the text size.
    private static List<OrderFragment> Fragment(IReadOnlyList<OrderStrip> strips)
    {
        var frags = new List<OrderFragment>();
        for (int s = 0; s < strips.Count; s++)
        {
            OrderStrip strip = strips[s];
            if (strip.Words.Count == 0)
            {
                continue;
            }

            int start = 0;
            double right = strip.Words[0].Box.Right;
            for (int i = 1; i < strip.Words.Count; i++)
            {
                OcrBox b = strip.Words[i].Box;
                if (b.Left - right > LayoutParameters.FragmentGapEm * strip.Size)
                {
                    frags.Add(MakeFragment(strip, s, start, i));
                    start = i;
                    right = b.Right;
                }
                else
                {
                    right = Math.Max(right, b.Right);
                }
            }

            frags.Add(MakeFragment(strip, s, start, strip.Words.Count));
        }

        return frags;
    }

    private static OrderFragment MakeFragment(OrderStrip strip, int index, int start, int end)
    {
        double x0 = double.MaxValue, x1 = double.MinValue, top = double.MaxValue, bot = double.MinValue;
        for (int i = start; i < end; i++)
        {
            OcrBox b = strip.Words[i].Box;
            x0 = Math.Min(x0, b.Left);
            x1 = Math.Max(x1, b.Right);
            top = Math.Min(top, b.Top);
            bot = Math.Max(bot, b.Bottom);
        }

        return new OrderFragment
        {
            Strip = index,
            Leaf = strip.Leaf,
            WordStart = start,
            WordCount = end - start,
            X0 = x0,
            X1 = x1,
            Top = top,
            Bot = bot,
            Base = (strip.Box.Top + strip.Box.Bottom) / 2,
            Size = strip.Size,
        };
    }

    // Strips cut from different regions that sit on one row (the cells of a table row, found as separate regions) are one line.
    // The strips are clustered by their vertical centre, as OnePDF's ClusterLines clusters glyphs by baseline; two strips of the
    // same region never join, because the region's own line cut already separated them.
    private static void AssignLines(IReadOnlyList<OrderStrip> strips, List<OrderFragment> frags)
    {
        var order = new List<int>();
        for (int s = 0; s < strips.Count; s++)
        {
            if (strips[s].Words.Count > 0)
            {
                order.Add(s);
            }
        }

        order.Sort((a, b) =>
        {
            int c = Center(strips[a]).CompareTo(Center(strips[b]));
            return c != 0 ? c : a.CompareTo(b);
        });

        var lineOf = new int[strips.Count];
        int lines = 0;
        double sum = 0, minSize = 0;
        int count = 0;
        var leaves = new HashSet<int>();
        foreach (int s in order)
        {
            double cy = Center(strips[s]);
            if (count > 0 && !leaves.Contains(strips[s].Leaf) &&
                Math.Abs(cy - (sum / count)) <= LayoutParameters.LineToleranceEm * Math.Min(minSize, strips[s].Size))
            {
                sum += cy;
                count++;
                minSize = Math.Min(minSize, strips[s].Size);
                leaves.Add(strips[s].Leaf);
            }
            else
            {
                lines++;
                sum = cy;
                count = 1;
                minSize = strips[s].Size;
                leaves.Clear();
                leaves.Add(strips[s].Leaf);
            }

            lineOf[s] = lines - 1;
        }

        foreach (OrderFragment f in frags)
        {
            f.Line = lineOf[f.Strip];
        }
    }

    private static double Center(OrderStrip s) => (s.Box.Top + s.Box.Bottom) / 2;

    // ---- X–Y cut ------------------------------------------------------------------------------------------------------------

    // OnePDF TextLayout.XyCut: cut a region at empty vertical stripes (columns) when there are any, else at empty horizontal
    // bands (blocks), else where lines with several fragments alternate with single-fragment lines. A region whose columns are
    // the cells of a table is kept whole. The explicit stack keeps the work independent of how deeply the page nests.
    private static List<(int[] Ids, bool Table, int Columns)> XyCut(List<OrderFragment> f, CancellationToken ct)
    {
        var leaves = new List<(int[], bool, int)>();
        var stack = new Stack<int[]>();
        var all = new int[f.Count];
        for (int i = 0; i < all.Length; i++)
        {
            all[i] = i;
        }

        stack.Push(all);
        int guard = 0;
        while (stack.Count > 0)
        {
            if ((++guard & 255) == 0)
            {
                ct.ThrowIfCancellationRequested();
            }

            int[] reg = stack.Pop();
            if (reg.Length == 0)
            {
                continue;
            }

            if (reg.Length == 1 || DistinctLines(f, reg) <= 1)
            {
                leaves.Add((reg, false, 1));
                continue;
            }

            double ms = Median(reg.Select(i => f[i].Size).ToArray());
            List<int[]>? groups = Partition(f, reg, true, LayoutParameters.ColumnGutterEm * ms);
            if (groups is not null)
            {
                if (IsTable(f, reg, groups, ms))
                {
                    for (int c = 0; c < groups.Count; c++)
                    {
                        foreach (int id in groups[c])
                        {
                            f[id].Column = c;
                        }
                    }

                    leaves.Add((reg, true, groups.Count));
                    continue;
                }
            }

            if (groups is null)
            {
                groups = Partition(f, reg, false, LayoutParameters.BlockGapEm * ms);
                if (groups is not null)
                {
                    groups = MergeColumnRuns(f, groups, ms);
                }
            }
            else
            {
                groups = ChooseCut(f, reg, groups, ms);
            }

            groups ??= SplitByLineKind(f, reg);
            if (groups is null)
            {
                leaves.Add((reg, false, 1));
                continue;
            }

            for (int j = groups.Count - 1; j >= 0; j--)
            {
                stack.Push(groups[j]);
            }
        }

        return leaves;
    }

    private static int DistinctLines(List<OrderFragment> f, int[] reg)
    {
        int first = f[reg[0]].Line;
        foreach (int id in reg)
        {
            if (f[id].Line != first)
            {
                return 2;
            }
        }

        return 1;
    }

    // Sorts the region along one axis and cuts it at every empty stripe of at least `gap`. Null when there is none.
    private static List<int[]>? Partition(List<OrderFragment> f, int[] reg, bool xAxis, double gap, List<double>? gaps = null)
    {
        int[] sorted = [.. reg];
        Array.Sort(sorted, (a, b) =>
        {
            int c = (xAxis ? f[a].X0 : f[a].Top).CompareTo(xAxis ? f[b].X0 : f[b].Top);
            return c != 0 ? c : a.CompareTo(b);
        });

        // A horizontal cut must not run through a column. When the region holds several leaf regions of the page (the columns found
        // by the ink-level cut) and some leaf has fragments on both sides of a gap, the gap is only a paragraph gap that two
        // columns happen to share, so it is not a cut: the columns are cut apart first.
        Dictionary<int, int>? total = null, before = null;
        int straddling = 0;
        if (!xAxis)
        {
            total = [];
            foreach (int id in sorted)
            {
                total[f[id].Leaf] = total.GetValueOrDefault(f[id].Leaf) + 1;
            }

            if (total.Count < 2)
            {
                total = null;
            }
            else
            {
                before = [];
            }
        }

        var groups = new List<int[]>();
        int start = 0;
        double curEnd = xAxis ? f[sorted[0]].X1 : f[sorted[0]].Bot;
        for (int i = 0; i < sorted.Length; i++)
        {
            if (i > 0)
            {
                double key = xAxis ? f[sorted[i]].X0 : f[sorted[i]].Top;
                if (key - curEnd >= gap && straddling == 0)
                {
                    groups.Add(sorted[start..i]);
                    gaps?.Add(key - curEnd);
                    start = i;
                }

                curEnd = Math.Max(curEnd, xAxis ? f[sorted[i]].X1 : f[sorted[i]].Bot);
            }

            if (total is not null && before is not null)
            {
                int leaf = f[sorted[i]].Leaf;
                int b = before.GetValueOrDefault(leaf);
                if (b == 0 && total[leaf] > 1)
                {
                    straddling++;
                }

                if (b + 1 == total[leaf] && total[leaf] > 1)
                {
                    straddling--;
                }

                before[leaf] = b + 1;
            }
        }

        groups.Add(sorted[start..]);
        return groups.Count >= 2 ? groups : null;
    }

    // OnePDF TextLayout.IsTable: at least three rows, most of them with fragments in more than one column, and short fragments.
    private static bool IsTable(List<OrderFragment> f, int[] reg, List<int[]> columns, double ms)
    {
        var firstColumn = new Dictionary<int, int>();
        var multi = new HashSet<int>();
        for (int c = 0; c < columns.Count; c++)
        {
            foreach (int id in columns[c])
            {
                int line = f[id].Line;
                if (!firstColumn.TryGetValue(line, out int fc))
                {
                    firstColumn[line] = c;
                }
                else if (fc != c)
                {
                    multi.Add(line);
                }
            }
        }

        int rows = firstColumn.Count;
        if (rows < LayoutParameters.TableMinRows || multi.Count < LayoutParameters.TableRowFraction * rows)
        {
            return false;
        }

        var widths = new double[reg.Length];
        for (int i = 0; i < reg.Length; i++)
        {
            widths[i] = (f[reg[i]].X1 - f[reg[i]].X0) / Math.Max(ms, 1e-6);
        }

        if (Median(widths) > LayoutParameters.TableCellWidthEm)
        {
            return false;
        }

        return !(SparseColumns(f, columns) || UnevenRows(f, reg) || AlternatingPitch(f, reg));
    }

    // OnePDF TextLayout.SplitByLineKind: lines with a gutter (several fragments) alternating with single-fragment lines are cut
    // into runs of one kind, in the order of the lines' positions.
    private static List<int[]>? SplitByLineKind(List<OrderFragment> f, int[] reg)
    {
        var count = new Dictionary<int, int>();
        var top = new Dictionary<int, double>();
        foreach (int id in reg)
        {
            int line = f[id].Line;
            count[line] = count.GetValueOrDefault(line) + 1;
            top[line] = top.TryGetValue(line, out double t) ? Math.Min(t, f[id].Base) : f[id].Base;
        }

        List<int> lines = [.. count.Keys];
        lines.Sort((a, b) =>
        {
            int c = top[a].CompareTo(top[b]);
            return c != 0 ? c : a.CompareTo(b);
        });

        var runOf = new Dictionary<int, int>();
        int run = 0;
        bool previousMulti = count[lines[0]] > 1;
        foreach (int line in lines)
        {
            bool multi = count[line] > 1;
            if (multi != previousMulti)
            {
                run++;
                previousMulti = multi;
            }

            runOf[line] = run;
        }

        if (run == 0)
        {
            return null;
        }

        var groups = new List<int>[run + 1];
        for (int i = 0; i <= run; i++)
        {
            groups[i] = [];
        }

        foreach (int id in reg)
        {
            groups[runOf[f[id].Line]].Add(id);
        }

        return [.. groups.Select(g => g.ToArray())];
    }

    internal static double Median(double[] values)
    {
        Array.Sort(values);
        int n = values.Length;
        return n % 2 == 1 ? values[n / 2] : (values[(n / 2) - 1] + values[n / 2]) / 2;
    }

    // ---- blocks -------------------------------------------------------------------------------------------------------------

    // OnePDF TextLayout.MakeBlock: lines in the order of their first fragment's position, fragments left to right.
    private static OrderBlock? MakeBlock(List<OrderFragment> f, int[] ids, bool table, int columns)
    {
        var byLine = new Dictionary<int, List<OrderFragment>>();
        var firstSeen = new List<int>();
        foreach (int id in ids)
        {
            int line = f[id].Line;
            if (!byLine.TryGetValue(line, out List<OrderFragment>? list))
            {
                byLine[line] = list = [];
                firstSeen.Add(line);
            }

            list.Add(f[id]);
        }

        var lines = new List<(double Key, int Seen, OrderLine Line)>();
        for (int i = 0; i < firstSeen.Count; i++)
        {
            List<OrderFragment> list = byLine[firstSeen[i]];
            list.Sort((a, b) =>
            {
                int c = a.X0.CompareTo(b.X0);
                return c != 0 ? c : a.Strip.CompareTo(b.Strip);
            });
            lines.Add((list[0].Base, i, new OrderLine { Fragments = list }));
        }

        lines.Sort((a, b) =>
        {
            int c = a.Key.CompareTo(b.Key);
            return c != 0 ? c : a.Seen.CompareTo(b.Seen);
        });

        return lines.Count == 0 ? null : new OrderBlock { IsTable = table, Columns = columns, Lines = [.. lines.Select(l => l.Line)] };
    }
}
