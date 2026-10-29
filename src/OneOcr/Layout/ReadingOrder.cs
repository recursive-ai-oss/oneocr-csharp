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

    /// <summary>
    /// A narrower empty stripe than <see cref="ColumnGutterEm"/> still separates columns when it runs clear through a region of text
    /// with many lines and every column it leaves is a column of running text (see <see cref="SoftColumnLines"/> and
    /// <see cref="SoftColumnWidthEm"/>). Print gutters are about one em of the font, which is a little over the measured text height,
    /// so a stripe of 1.25 text heights misses the gutters of tightly set pages by a pixel or two, more so on blurred scans, and the
    /// page was then read row by row across its columns. A stripe that no fragment of a dozen lines touches is not a gap between words.
    /// The bar is 0.4: a page whose gutter measured 0.46 text heights (word boxes are ink extents, a little short of the print) was a
    /// pixel below 0.5 with one detector and above it with another, and its columns flipped.
    /// </summary>
    internal const double SoftGutterEm = 0.4;

    /// <summary>
    /// Columns whose gutter has a few words in it (a stray word or speck on a scan, justified lines that run a few pixels into the next
    /// column): a stripe counts as a gutter when at most this fraction of the peak number of words (at least one) lie in it ...
    /// </summary>
    internal const double RobustCrossing = 0.10;

    /// <summary>... it is at least this wide ...</summary>
    internal const double RobustGutterEm = 0.2;

    /// <summary>... and over this many text heights on each side of it the mean number of words is at least <see cref="RobustSideShare"/> of the peak ...</summary>
    internal const double RobustSideEm = 3;

    /// <summary>... (text stands up to it on both sides; the ragged right edge of long lines, with a few longer lines reaching beyond, is not a gutter).</summary>
    internal const double RobustSideShare = 0.25;

    /// <summary>Such columns are cut only in a region of at least this many distinct lines (a page of text, not a card or a form on a screen) ...</summary>
    internal const int RobustMinLines = 30;

    /// <summary>... and are at most this many times as wide as each other (a wider group has a gutter missing in it).</summary>
    internal const double RobustColumnRatio = 1.6;

    /// <summary>
    /// When no upright gutter is found, gutters that lean are tried: the column edges move sideways by this much per unit of height
    /// (a skew of about a seventh of a degree) ...
    /// </summary>
    internal const double RobustShearStep = 0.0025;

    /// <summary>... up to this many steps either way (about 0.7 degrees; a page is deskewed before it is read, and a lean is what the estimate leaves).</summary>
    internal const int RobustShearSteps = 5;

    /// <summary>A line that crosses such a stripe is a header or footer of its region when at most this many other lines lie above or below it ...</summary>
    internal const int RobustMarginLines = 5;

    /// <summary>... or this fraction of the region's lines when an empty band of <see cref="BlockGapEm"/> sets the line off (a table under the columns, then the footer).</summary>
    internal const double RobustMarginShare = 0.1;

    /// <summary>A band of at most <see cref="RobustMarginLines"/> lines at the top or bottom of the region is a header or footer when an empty band this tall sets it off (a paragraph gap is not enough) ...</summary>
    internal const double RobustMarginGapEm = 1;

    /// <summary>... it spans at least this fraction of the region's width ...</summary>
    internal const double RobustMarginSpan = 0.6;

    /// <summary>... and on each of its lines its fragments fill at most this fraction of that width (the first lines of two columns fill it).</summary>
    internal const double RobustMarginFill = 0.4;

    /// <summary>
    /// A group of at most this many fragments, and at most <see cref="SpeckShare"/> of the fragments of a region of at least
    /// <see cref="SpeckMinFragments"/>, that the full-gutter cut leaves between two others is a speck, not a column. It joins the
    /// nearer neighbour when it lies inside that neighbour's vertical extent, at least <see cref="SpeckInsideEm"/> from its top and bottom
    /// (a menu item at the top of a screen is not a speck).
    /// </summary>
    internal const int SpeckMaxFragments = 2;

    /// <summary>The share of a region's fragments a speck stays under.</summary>
    internal const double SpeckShare = 0.03;

    /// <summary>The distance, in text heights, a speck lies inside the vertical extent of the group it joins.</summary>
    internal const double SpeckInsideEm = 2;

    /// <summary>A region needs at least this many fragments before it has specks.</summary>
    internal const int SpeckMinFragments = 30;

    /// <summary>A region needs at least this many distinct lines before a narrow stripe can separate its columns.</summary>
    internal const int SoftMinLines = 10;

    /// <summary>Every column cut at a narrow stripe has at least this many lines ...</summary>
    internal const int SoftColumnLines = 5;

    /// <summary>... and the widest quarter of its fragments are at least this many em wide (full lines of running text, not a list of labels or cells).</summary>
    internal const double SoftColumnWidthEm = 12;

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
    internal const double HeaderHeightEm = 2.5;

    /// <summary>... spans at least this fraction of the region's width ...</summary>
    internal const double HeaderSpan = 0.6;

    /// <summary>... and stands clear of the rest by at least this many em.</summary>
    internal const double HeaderGapEm = 2.5;

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

    /// <summary>A column is a sidebar only when at most this fraction of its lines sit on a line shared with the rest of the region.</summary>
    internal const double SidebarSharedLines = 0.5;

    /// <summary>A sidebar is at most this fraction of the region's width.</summary>
    internal const double SidebarMaxWidth = 0.35;

    /// <summary>The body starts within this fraction of the region's width after the sidebar's right edge.</summary>
    internal const double SidebarMaxGutter = 0.25;

    /// <summary>The gaps between a sidebar's items differ from their median by at most this fraction.</summary>
    internal const double SidebarPitchTolerance = 0.3;

    /// <summary>A sidebar has at least this many lines.</summary>
    internal const int SidebarMinLines = 3;

    /// <summary>A column of short fragments with a vertical gap this wide is not a text column: its text comes in cells.</summary>
    internal const double CellColumnGapEm = 3;

    /// <summary>A column whose runs of lines (gaps under half an em) are at most this long is a column of cells, not of paragraphs.</summary>
    internal const int CellColumnMaxRun = 2;

    /// <summary>... and its fragments are at most this many em wide at the median (lines of text are longer).</summary>
    internal const double CellColumnWidthEm = 15;

    /// <summary>A row of fields may sit under at most this many one-fragment heading lines in one region.</summary>
    internal const int FieldHeadingLines = 1;

    /// <summary>A column is a column of labels when this fraction of its fragments ends in a colon.</summary>
    internal const double LabelColumnShare = 0.6;

    /// <summary>A field's value starts within this many em of its label's left edge.</summary>
    internal const double FieldAlignEm = 2;
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

    /// <summary>The fragment ends in a colon: the label of a "Total: 45.00" row.</summary>
    public bool Label;
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

    /// <summary>The block is part of a screen's menu bar or sidebar, or one field of a form: it is read on its own and no table is built across it.</summary>
    public bool IsStandalone { get; init; }
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
        foreach ((int[] ids, bool table, int columns, bool chrome) in XyCut(frags, strips, cancellationToken))
        {
            OrderBlock? block = MakeBlock(frags, ids, table, columns, chrome);
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
            Label = strip.Words[end - 1].Text.EndsWith(':'),
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
    private static List<(int[] Ids, bool Table, int Columns, bool Chrome)> XyCut(List<OrderFragment> f, IReadOnlyList<OrderStrip> strips, CancellationToken ct)
    {
        var leaves = new List<(int[], bool, int, bool)>();
        var stack = new Stack<int[]>();
        var standalone = new HashSet<int[]>(ReferenceEqualityComparer.Instance);
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
            bool chrome = standalone.Contains(reg);
            if (reg.Length == 0)
            {
                continue;
            }

            if (reg.Length == 1 || DistinctLines(f, reg) <= 1)
            {
                leaves.Add((reg, false, 1, chrome));
                continue;
            }

            if (TrailingFields(f, reg) is { } fieldCells)
            {
                foreach ((int[] ids, bool alone) in fieldCells)
                {
                    leaves.Add((ids, false, 1, alone));
                }

                continue;
            }

            bool firstIsStandalone = false;
            double ms = Median(reg.Select(i => f[i].Size).ToArray());
            List<int[]>? groups = Partition(f, reg, true, LayoutParameters.ColumnGutterEm * ms);
            if (groups is not null)
            {
                groups = AbsorbSpecks(f, reg, groups, ms);
            }

            // A speck at the edge of the page that stands clear of the text is cut off on its own: it is neither a column nor a reason to
            // read the rest in bands, and the text it leaves is cut into columns in its turn.
            bool narrowCut = groups is not null && IsEdgeSpeck(f, reg, groups, ms);
            if (groups is null && PeelMargins(f, reg, ms) is { } peeled)
            {
                groups = peeled;
                narrowCut = true;
            }

            if (groups is null && SoftColumns(f, reg, ms) is { } narrow)
            {
                groups = narrow;
                narrowCut = true;
            }

            if (groups is null && RobustColumns(f, strips, reg, ms) is { } robust)
            {
                groups = robust;
                narrowCut = true;
            }

            if (narrowCut)
            {
                // Columns of running text: no label, table or sidebar reading applies to them.
            }
            else if (groups is not null && LabelBlocks(f, groups) is { } blocks)
            {
                groups = blocks;
            }
            else if (groups is not null)
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

                    leaves.Add((reg, true, groups.Count, chrome));
                    continue;
                }
            }

            if (groups is null)
            {
                groups = Partition(f, reg, false, LayoutParameters.BlockGapEm * ms);
                if (groups is not null)
                {
                    groups = MergeFieldCells(f, MergeColumnRuns(f, groups, ms), ms, standalone);
                }
            }
            else if (!narrowCut)
            {
                (groups, firstIsStandalone) = ChooseCut(f, reg, groups, ms);
            }

            groups ??= SplitByLineKind(f, reg);
            if (groups is null)
            {
                leaves.Add((reg, false, 1, chrome));
                continue;
            }

            for (int j = groups.Count - 1; j >= 0; j--)
            {
                if (chrome || (j == 0 && firstIsStandalone))
                {
                    standalone.Add(groups[j]);
                }

                stack.Push(groups[j]);
            }
        }

        return leaves;
    }

    // Columns separated by a stripe narrower than a full gutter, for a region of many lines whose columns are all running text.
    // Null when there is no such stripe or the groups it leaves are not all columns of text.
    private static List<int[]>? SoftColumns(List<OrderFragment> f, int[] reg, double ms)
    {
        if (reg.Length < 2 * LayoutParameters.SoftMinLines || DistinctLineCount(f, reg) < LayoutParameters.SoftMinLines)
        {
            return null;
        }

        List<int[]>? groups = Partition(f, reg, true, LayoutParameters.SoftGutterEm * ms);
        if (groups is null)
        {
            return null;
        }

        foreach (int[] g in groups)
        {
            var widths = new double[g.Length];
            for (int i = 0; i < g.Length; i++)
            {
                widths[i] = (f[g[i]].X1 - f[g[i]].X0) / Math.Max(ms, 1e-6);
            }

            Array.Sort(widths);
            if (DistinctLineCount(f, g) < LayoutParameters.SoftColumnLines || widths[(int)(0.75 * (widths.Length - 1))] < LayoutParameters.SoftColumnWidthEm)
            {
                return null;
            }
        }

        return groups;
    }

    // A group of a fragment or two in a region of many is a speck of noise (or a stray mark in the margin) that happens to stand
    // clear of the columns on both sides, not a column: left as a group of its own it breaks the reading of the columns on either
    // side of it (it looks like a column embedded between parts of another). It joins the nearer neighbour. Null when no cut is left.
    // A region of many lines with no full-gutter cut: a running header or footer is cut off first (see MarginBands), so that the narrow
    // cuts below do not take a footer at the left for the end of the left column. Null when there is none.
    private static List<int[]>? PeelMargins(List<OrderFragment> f, int[] reg, double ms)
    {
        if (reg.Length < 2 * LayoutParameters.RobustMinLines || DistinctLineCount(f, reg) < LayoutParameters.RobustMinLines || MarginBands(f, reg, ms) is not { } margins)
        {
            return null;
        }

        (int[] head, int[] body, int[] foot) = margins;
        if (DistinctLineCount(f, body) < LayoutParameters.RobustMinLines)
        {
            return null;
        }

        var groups = new List<int[]>(3);
        if (head.Length > 0)
        {
            groups.Add(head);
        }

        groups.Add(body);
        if (foot.Length > 0)
        {
            groups.Add(foot);
        }

        return groups;
    }

    private static List<int[]>? AbsorbSpecks(List<OrderFragment> f, int[] reg, List<int[]> groups, double ms)
    {
        if (groups.Count < 3 || reg.Length < LayoutParameters.SpeckMinFragments)
        {
            return groups;
        }

        int max = Math.Min(LayoutParameters.SpeckMaxFragments, (int)(LayoutParameters.SpeckShare * reg.Length));
        var kept = new List<int[]>(groups);
        for (int i = kept.Count - 1; i >= 0 && kept.Count > 1; i--)
        {
            if (kept[i].Length > max)
            {
                continue;
            }

            int into;
            if (i == 0)
            {
                into = 1;
            }
            else if (i == kept.Count - 1)
            {
                into = i - 1;
            }
            else
            {
                double left = kept[i].Min(id => f[id].X0) - kept[i - 1].Max(id => f[id].X1);
                double right = kept[i + 1].Min(id => f[id].X0) - kept[i].Max(id => f[id].X1);
                into = left <= right ? i - 1 : i + 1;
            }

            double top = kept[i].Min(id => f[id].Top), bottom = kept[i].Max(id => f[id].Bot);
            if (top - kept[into].Min(id => f[id].Top) < LayoutParameters.SpeckInsideEm * ms || kept[into].Max(id => f[id].Bot) - bottom < LayoutParameters.SpeckInsideEm * ms)
            {
                continue;
            }

            kept[into] = [.. kept[into], .. kept[i]];
            kept.RemoveAt(i);
        }

        return kept.Count >= 2 ? kept : null;
    }

    // Whether the full-gutter cut only splits a speck (see AbsorbSpecks) off the edge of a region of many fragments.
    private static bool IsEdgeSpeck(List<OrderFragment> f, int[] reg, List<int[]> groups, double ms)
    {
        if (groups.Count != 2 || reg.Length < LayoutParameters.SpeckMinFragments)
        {
            return false;
        }

        int max = Math.Min(LayoutParameters.SpeckMaxFragments, (int)(LayoutParameters.SpeckShare * reg.Length));
        int[] small = groups[0].Length <= groups[1].Length ? groups[0] : groups[1];
        int[] rest = ReferenceEquals(small, groups[0]) ? groups[1] : groups[0];
        if (small.Length > max)
        {
            return false;
        }

        double top = small.Min(id => f[id].Top), bottom = small.Max(id => f[id].Bot);
        return top - rest.Min(id => f[id].Top) >= LayoutParameters.SpeckInsideEm * ms && rest.Max(id => f[id].Bot) - bottom >= LayoutParameters.SpeckInsideEm * ms;
    }

    private static int DistinctLineCount(List<OrderFragment> f, int[] reg)
    {
        var lines = new HashSet<int>();
        foreach (int id in reg)
        {
            lines.Add(f[id].Line);
        }

        return lines.Count;
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
    private static List<int[]>? Partition(List<OrderFragment> f, int[] reg, bool xAxis, double gap, List<double>? gaps = null, bool byLeaf = true)
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
        if (!xAxis && byLeaf)
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

        // Rows of "label: value" are read across even when there are only two, and their uneven spacing (an address block, then a
        // payment block) does not make them fields: a label that ends in a colon is a label whatever its size.
        bool keyValue = LabelColumns(f, columns).Count(l => l) == 1;
        int rows = firstColumn.Count;
        if (rows < (keyValue ? 2 : LayoutParameters.TableMinRows) || multi.Count < LayoutParameters.TableRowFraction * rows)
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

        return !(SparseColumns(f, columns) || UnevenRows(f, reg) || (!keyValue && AlternatingPitch(f, reg)));
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
    private static OrderBlock? MakeBlock(List<OrderFragment> f, int[] ids, bool table, int columns, bool chrome)
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

        return lines.Count == 0 ? null : new OrderBlock { IsTable = table, IsStandalone = chrome, Columns = columns, Lines = [.. lines.Select(l => l.Line)] };
    }
}
