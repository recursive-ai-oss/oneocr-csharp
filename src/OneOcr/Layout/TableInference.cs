using OneOcr.Detection;

namespace OneOcr.Layout;

/// <summary>
/// A source of column separators for a table region, other than the whitespace between the columns: the place for a model that
/// learns where columns divide. The default (none) leaves the whitespace analysis in charge.
/// </summary>
internal interface ITableSeparators
{
    /// <summary>The x positions, ascending and strictly inside the region, where columns divide; null or empty when there are none.</summary>
    IReadOnlyList<double>? ColumnBoundaries(OcrBox region);
}

/// <summary>
/// The structure of tables without a complete rule grid. Columns are the stretches of the page that are free of text in nearly every
/// row (aligned whitespace gaps, as in the projection-profile methods of the table-recognition surveys: Zanibbi, Blostein and Cordy,
/// "A survey of table recognition", IJDAR 2004, sec. 5); rows are the text lines. A fragment that reaches across a gap is a cell with
/// a column span; a column with nothing in a row is an empty cell. The words and their order are not changed.
/// </summary>
internal static partial class TableStructure
{
    // A column separator is free of text in all but this share of the rows (spanning headers and stray fragments cross it).
    private const double GapTolerance = 0.25;

    // A separator is at least this many text heights wide.
    private const double MinGapHeights = 0.5;

    // A fragment belongs to a column when it covers at least this share of the smaller of itself and the column's text.
    private const double MinOverlapShare = 0.3;

    // Most fragments analysed in one table; a larger one is left as it is.
    private const int MaxFragments = 4000;

    /// <summary>
    /// Reads the structure of the table blocks the reading order found and of the blocks inside the detector's table regions
    /// (merged into one table, read row by row). A block that has no two columns is left alone.
    /// </summary>
    /// <param name="plan">The blocks in reading order.</param>
    /// <param name="regions">The page's detected regions; only those named "table" are used.</param>
    /// <param name="separators">Learned separators, or null.</param>
    /// <param name="rules">Finds the page's horizontal rules (called at most once, and only when a table is looked at); null when the
    /// page's pixels are not available. Rules that cross a table separate its rows into bands read cell by cell.</param>
    public static List<OrderBlock> Infer(List<OrderBlock> plan, IReadOnlyList<DetectedRegion> regions, ITableSeparators? separators, Func<IReadOnlyList<RuleLine>>? rules = null)
    {
        IReadOnlyList<RuleLine>? pageRules = null;
        IReadOnlyList<RuleLine> Rules() => pageRules ??= rules?.Invoke() ?? [];
        var slots = new OrderBlock?[plan.Count];
        for (int i = 0; i < plan.Count; i++)
        {
            slots[i] = plan[i];
        }

        foreach (DetectedRegion region in regions)
        {
            if (region.Name != "table")
            {
                continue;
            }

            List<int> narrow = InsideBlocks(slots, region, includeStandalone: false);
            if (narrow.Count == 0)
            {
                continue;
            }

            var box = new OcrBox(region.Left, region.Top, region.Right, region.Bottom);
            var rows = RowsOf(slots, narrow);
            List<double> cuts = rules is null ? [] : Cuts(rows, Rules());
            List<int> inside = narrow;
            OrderBlock? made = null;
            if (cuts.Count > 0)
            {
                // Rules cut the table into bands. A block that the reading order set apart as a field of a form (a wrapped paragraph in a
                // cell reads like one) belongs to the table when it lies within one band, and only if it ends up in a band read cell by
                // cell: a form's label and its input box are also rows between rules, and read line by line they lose the pairing the
                // field blocks give them.
                List<int> wide = InsideBlocks(slots, region, includeStandalone: true, cuts);
                if (wide.Count > narrow.Count)
                {
                    var added = new HashSet<OrderFragment>();
                    foreach (int i in wide.Except(narrow))
                    {
                        foreach (OrderLine l in slots[i]!.Lines)
                        {
                            added.UnionWith(l.Fragments);
                        }
                    }

                    var wideRows = RowsOf(slots, wide);
                    (wideRows, bool[]? wideGrouped) = Bands(wideRows, cuts);
                    var info = new HashSet<OrderFragment>();
                    OrderBlock? candidate = Build(wideRows, box, separators, wideGrouped, info);
                    if (candidate is not null && info.Overlaps(added))
                    {
                        made = candidate;
                        inside = wide;
                    }
                }
            }

            if (made is null)
            {
                rows = RowsOf(slots, narrow);
                (rows, bool[]? grouped) = Bands(rows, cuts);
                made = Build(rows, box, separators, grouped);
            }

            if (made is null)
            {
                continue;
            }

            slots[inside[0]] = made;
            for (int k = 1; k < inside.Count; k++)
            {
                slots[inside[k]] = null;
            }
        }

        var result = new List<OrderBlock>(plan.Count);
        foreach (OrderBlock? b in slots)
        {
            if (b is null)
            {
                continue;
            }

            if (b.IsTable && b.Cells is null)
            {
                var rows = b.Lines.Select(l => l.Fragments).ToList();
                (double top, double left, double right) = Extent(b);
                double bottom = b.Lines.SelectMany(l => l.Fragments).Max(f => f.Bot);
                (rows, bool[]? grouped) = Bands(rows, rules is null ? [] : Cuts(rows, Rules()));
                result.Add(Build(rows, new OcrBox(left, top, right, bottom), separators, grouped) ?? b);
            }
            else
            {
                result.Add(b);
            }
        }

        return result;
    }

    private static List<int> InsideBlocks(OrderBlock?[] slots, DetectedRegion region, bool includeStandalone, List<double>? cuts = null)
    {
        var inside = new List<int>();
        for (int i = 0; i < slots.Length; i++)
        {
            if (slots[i] is { Cells: null } b && (!b.IsStandalone || (includeStandalone && !Crossed(b, cuts))) && Inside(region, b))
            {
                inside.Add(i);
            }
        }

        return inside;
    }

    // The rows are the lines of the page: fragments of different blocks that sit on one line are one row.
    private static List<List<OrderFragment>> RowsOf(OrderBlock?[] slots, List<int> inside)
    {
        var byLine = new SortedDictionary<int, List<OrderFragment>>();
        foreach (int i in inside)
        {
            foreach (OrderLine l in slots[i]!.Lines)
            {
                foreach (OrderFragment f in l.Fragments)
                {
                    if (!byLine.TryGetValue(f.Line, out List<OrderFragment>? list))
                    {
                        byLine[f.Line] = list = [];
                    }

                    list.Add(f);
                }
            }
        }

        return byLine.Values.OrderBy(r => r.Average(f => f.Base)).ToList();
    }

    private static bool Inside(DetectedRegion r, OrderBlock b)
    {
        (double top, double left, double right) = Extent(b);
        double bottom = double.MinValue;
        foreach (OrderLine l in b.Lines)
        {
            foreach (OrderFragment f in l.Fragments)
            {
                bottom = Math.Max(bottom, f.Bot);
            }
        }

        double cx = (left + right) / 2, cy = (top + bottom) / 2;
        return cx >= r.Left && cx <= r.Right && cy >= r.Top && cy <= r.Bottom;
    }

    // Builds the table block for rows of fragments (each row left to right), or null when there are not two columns.
    // When given, groupedFragments receives the fragments of the rows that are read cell by cell.
    private static OrderBlock? Build(List<List<OrderFragment>> rows, OcrBox box, ITableSeparators? separators, bool[]? grouped = null, HashSet<OrderFragment>? groupedFragments = null)
    {
        int total = rows.Sum(r => r.Count);
        if (rows.Count < 2 || total > MaxFragments)
        {
            return null;
        }

        for (int i = 0; i < rows.Count; i++)
        {
            if (grouped is null || !grouped[i])
            {
                rows[i].Sort((a, b) => a.X0.CompareTo(b.X0));
            }
        }

        (double Left, double Right)[]? columns = Columns(rows, box, separators);
        if (columns is null || columns.Length < 2)
        {
            return null;
        }

        if (grouped is not null)
        {
            (rows, grouped) = SplitAligned(rows, grouped, columns);
            if (groupedFragments is not null)
            {
                for (int r = 0; r < rows.Count; r++)
                {
                    if (grouped[r])
                    {
                        groupedFragments.UnionWith(rows[r]);
                    }
                }
            }
        }

        var cells = new List<OrderCell>();
        var lines = new List<OrderLine>(rows.Count);
        int multi = 0;
        for (int r = 0; r < rows.Count; r++)
        {
            List<OrderFragment> row = rows[r];
            double top = row.Min(f => f.Top), bottom = row.Max(f => f.Bot);
            // The cell of each fragment: the columns it reaches, joined with a fragment before it that reaches the same column.
            var covered = new bool[columns.Length];
            OrderCell? last = null;
            int lastIndex = -1, lastEnd = -1;
            var rowCells = new List<int>();
            foreach (OrderFragment f in grouped is not null && grouped[r] ? Array.Empty<OrderFragment>() : (IEnumerable<OrderFragment>)row)
            {
                (int c0, int c1) = Reach(columns, f);
                if (last is not null && c0 <= lastEnd)
                {
                    f.Cell = lastIndex;
                    OcrBox merged = OcrBox.Union(last.Box, new OcrBox(f.X0, f.Top, f.X1, f.Bot));
                    last = last with { ColumnSpan = Math.Max(last.Column + last.ColumnSpan, c1 + 1) - last.Column, Box = merged };
                    cells[lastIndex] = last;
                    lastEnd = Math.Max(lastEnd, c1);
                }
                else
                {
                    last = new OrderCell(r, c0, 1, c1 - c0 + 1, new OcrBox(f.X0, f.Top, f.X1, f.Bot));
                    lastIndex = cells.Count;
                    lastEnd = c1;
                    f.Cell = lastIndex;
                    cells.Add(last);
                    rowCells.Add(lastIndex);
                }
            }

            if (grouped is not null && grouped[r])
            {
                GroupCells(columns, row, r, cells, rowCells);
            }

            foreach (int ci in rowCells)
            {
                OrderCell c = cells[ci];
                for (int k = c.Column; k < c.Column + c.ColumnSpan && k < covered.Length; k++)
                {
                    covered[k] = true;
                }
            }

            // Columns with nothing in this row: empty cells.
            for (int k = 0; k < columns.Length; k++)
            {
                if (!covered[k])
                {
                    cells.Add(new OrderCell(r, k, 1, 1, new OcrBox(columns[k].Left, top, columns[k].Right, bottom)));
                }
            }

            if (row.Count > 1)
            {
                multi++;
            }


            lines.Add(new OrderLine { Fragments = row });
        }

        // Row-major, as the page model wants; the fragments' cell numbers follow the reordering.
        var order = Enumerable.Range(0, cells.Count).OrderBy(i => cells[i].Row).ThenBy(i => cells[i].Column).ToArray();
        var newIndex = new int[cells.Count];
        var sorted = new List<OrderCell>(cells.Count);
        for (int i = 0; i < order.Length; i++)
        {
            newIndex[order[i]] = i;
            sorted.Add(cells[order[i]]);
        }

        foreach (List<OrderFragment> row in rows)
        {
            foreach (OrderFragment f in row)
            {
                f.Cell = newIndex[f.Cell];
            }
        }

        return new OrderBlock
        {
            IsTable = true,
            Columns = columns.Length,
            Rows = rows.Count,
            Lines = lines,
            Cells = sorted,
            TableConfidence = (float)multi / rows.Count,
        };
    }

    // A band of fewer columns than this is read line by line: with two columns a row is mostly a name and its value, and cells that wrap in
    // both columns are rare.
    private const int MinCellColumns = 3;

    // A band whose lines hold at least this share of the (line, column) pairs they could are rows read line by line.
    private const double AlignedShare = 0.75;

    // A rule separates two bands when it runs across at least this share of the table's width: a rule under one column is the edge of
    // one cell, and a short rule is an underline.
    private const double BandRuleShare = 0.6;

    // Rules closer together than this many text heights are one separator (a double rule, a thick rule drawn as two lines).
    private const double BandRuleMerge = 0.5;

    /// <summary>
    /// The rules that cut a table into bands: those that lie between the table's first and last fragment and run across at least
    /// <see cref="BandRuleShare"/> of its text extent, merged when closer than half a text height. Empty when no rule qualifies.
    /// </summary>
    internal static List<double> Cuts(List<List<OrderFragment>> rows, IReadOnlyList<RuleLine> rules)
    {
        var cuts = new List<double>();
        if (rules.Count == 0 || rows.Count < 2)
        {
            return cuts;
        }

        double top = double.MaxValue, bottom = double.MinValue, left = double.MaxValue, right = double.MinValue;
        var heights = new List<double>();
        foreach (List<OrderFragment> row in rows)
        {
            foreach (OrderFragment f in row)
            {
                top = Math.Min(top, f.Top);
                bottom = Math.Max(bottom, f.Bot);
                left = Math.Min(left, f.X0);
                right = Math.Max(right, f.X1);
                heights.Add(f.Bot - f.Top);
            }
        }

        double height = Median(heights);
        foreach (RuleLine r in rules.OrderBy(r => r.Pos))
        {
            double overlap = Math.Min(r.Hi, right) - Math.Max(r.Lo, left);
            if (overlap >= BandRuleShare * (right - left) && r.Pos > top && r.Pos < bottom && (cuts.Count == 0 || r.Pos - cuts[^1] >= BandRuleMerge * height))
            {
                cuts.Add(r.Pos);
            }
        }

        return cuts;
    }

    // Whether a cut runs through the vertical extent of a block's text.
    private static bool Crossed(OrderBlock b, List<double>? cuts)
    {
        if (cuts is null || cuts.Count == 0)
        {
            return false;
        }

        double top = double.MaxValue, bottom = double.MinValue;
        foreach (OrderLine l in b.Lines)
        {
            foreach (OrderFragment f in l.Fragments)
            {
                top = Math.Min(top, f.Top);
                bottom = Math.Max(bottom, f.Bot);
            }
        }

        foreach (double c in cuts)
        {
            if (c > top && c < bottom)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Groups the rows (text lines) of a table into bands: the stretches of the table between the cutting rules (<see cref="Cuts"/>). A band
    /// whose lines are cells of different columns is read cell by cell (its cells' lines together, top to bottom), not line by line across
    /// the columns. A band of one line, and a band in which a fragment spans columns that other fragments also reach (so cells cannot be
    /// told apart), stays as the lines it holds, as without rules. Returns the rows unchanged, with no band flags, when there are no cuts.
    /// </summary>
    internal static (List<List<OrderFragment>> Rows, bool[]? Grouped) Bands(List<List<OrderFragment>> rows, List<double> cuts)
    {
        if (cuts.Count == 0)
        {
            return (rows, null);
        }

        // Fragments go to the band their centre lies in; the lines of a band, by the page's line numbers.
        var bands = new SortedDictionary<int, List<OrderFragment>>();
        foreach (List<OrderFragment> row in rows)
        {
            foreach (OrderFragment f in row)
            {
                double centre = (f.Top + f.Bot) / 2;
                int band = 0;
                while (band < cuts.Count && cuts[band] < centre)
                {
                    band++;
                }

                if (!bands.TryGetValue(band, out List<OrderFragment>? list))
                {
                    bands[band] = list = [];
                }

                list.Add(f);
            }
        }

        var result = new List<List<OrderFragment>>();
        var flags = new List<bool>();
        foreach (List<OrderFragment> band in bands.Values)
        {
            int lines = band.Select(f => f.Line).Distinct().Count();
            if (lines <= 1)
            {
                result.Add(band);
                flags.Add(false);
                continue;
            }

            result.Add(band);
            flags.Add(true);
        }

        return (result, [.. flags]);
    }

    // A band whose lines run across all of its columns at once (a list of rows: a name and a price on every line) is a set of rows that the
    // rules happen to fence together, and is read line by line; a band whose columns hold their own numbers of lines, so that the lines of
    // different columns do not share baselines, is a row of cells of several lines. The share of (line, column) pairs present among all
    // pairs of the band's lines and the columns it uses tells them apart.
    private static (List<List<OrderFragment>> Rows, bool[] Grouped) SplitAligned(List<List<OrderFragment>> rows, bool[] grouped, (double Left, double Right)[] columns)
    {
        var result = new List<List<OrderFragment>>(rows.Count);
        var flags = new List<bool>(rows.Count);
        for (int r = 0; r < rows.Count; r++)
        {
            if (!grouped[r])
            {
                result.Add(rows[r]);
                flags.Add(false);
                continue;
            }

            var byLine = new SortedDictionary<int, List<OrderFragment>>();
            var used = new HashSet<int>();
            var present = new HashSet<(int Line, int Column)>();
            foreach (OrderFragment f in rows[r])
            {
                (int c0, _) = Reach(columns, f);
                used.Add(c0);
                present.Add((f.Line, c0));
                if (!byLine.TryGetValue(f.Line, out List<OrderFragment>? list))
                {
                    byLine[f.Line] = list = [];
                }

                list.Add(f);
            }

            if (used.Count < MinCellColumns || present.Count >= AlignedShare * byLine.Count * used.Count)
            {
                foreach (List<OrderFragment> line in byLine.Values.OrderBy(l => l.Average(f => f.Base)))
                {
                    line.Sort((a, b) => a.X0.CompareTo(b.X0));
                    result.Add(line);
                    flags.Add(false);
                }
            }
            else
            {
                result.Add(rows[r]);
                flags.Add(true);
            }
        }

        return (result, [.. flags]);
    }

    // Cells of a band read as units: one cell per first column; its fragments (the cell's lines) top to bottom. When a fragment spans columns
    // that another fragment of the band also reaches, the cells cannot be told apart and the band is read by its lines, as without rules.
    private static void GroupCells(
        (double Left, double Right)[] columns, List<OrderFragment> band, int row, List<OrderCell> cells, List<int> rowCells)
    {
        var reach = new (int First, int Last)[band.Count];
        bool conflict = false;
        for (int i = 0; i < band.Count; i++)
        {
            reach[i] = Reach(columns, band[i]);
        }

        for (int i = 0; i < band.Count && !conflict; i++)
        {
            if (reach[i].Last == reach[i].First)
            {
                continue;
            }

            for (int j = 0; j < band.Count; j++)
            {
                if (j != i && reach[j].First <= reach[i].Last && reach[j].Last >= reach[i].First && band[j].Line != band[i].Line)
                {
                    conflict = true;
                    break;
                }
            }
        }

        if (conflict)
        {
            // Back to lines: the band's fragments row by row, cells as in a plain row.
            band.Sort((a, b) => a.Line != b.Line ? a.Base.CompareTo(b.Base) : a.X0.CompareTo(b.X0));
            OrderCell? last = null;
            int lastIndex = -1, lastEnd = -1, currentLine = int.MinValue;
            foreach (OrderFragment f in band)
            {
                (int c0, int c1) = Reach(columns, f);
                if (f.Line == currentLine && last is not null && c0 <= lastEnd)
                {
                    f.Cell = lastIndex;
                    last = last with { ColumnSpan = Math.Max(last.Column + last.ColumnSpan, c1 + 1) - last.Column, Box = OcrBox.Union(last.Box, new OcrBox(f.X0, f.Top, f.X1, f.Bot)) };
                    cells[lastIndex] = last;
                    lastEnd = Math.Max(lastEnd, c1);
                }
                else
                {
                    last = new OrderCell(row, c0, 1, c1 - c0 + 1, new OcrBox(f.X0, f.Top, f.X1, f.Bot));
                    lastIndex = cells.Count;
                    lastEnd = c1;
                    currentLine = f.Line;
                    f.Cell = lastIndex;
                    cells.Add(last);
                    rowCells.Add(lastIndex);
                }
            }

            return;
        }

        var byColumn = new SortedDictionary<int, int>();
        for (int i = 0; i < band.Count; i++)
        {
            OrderFragment f = band[i];
            var fragmentBox = new OcrBox(f.X0, f.Top, f.X1, f.Bot);
            if (byColumn.TryGetValue(reach[i].First, out int index))
            {
                OrderCell c = cells[index];
                cells[index] = c with { ColumnSpan = Math.Max(c.Column + c.ColumnSpan, reach[i].Last + 1) - c.Column, Box = OcrBox.Union(c.Box, fragmentBox) };
                f.Cell = index;
            }
            else
            {
                f.Cell = cells.Count;
                byColumn[reach[i].First] = cells.Count;
                rowCells.Add(cells.Count);
                cells.Add(new OrderCell(row, reach[i].First, 1, reach[i].Last - reach[i].First + 1, fragmentBox));
            }
        }

        // Cells left to right, the lines of a cell top to bottom.
        band.Sort((a, b) =>
        {
            int c = cells[a.Cell].Column.CompareTo(cells[b.Cell].Column);
            if (c != 0)
            {
                return c;
            }

            c = a.Top.CompareTo(b.Top);
            return c != 0 ? c : a.X0.CompareTo(b.X0);
        });
    }

    // The first and last column a fragment belongs to.
    private static (int First, int Last) Reach((double Left, double Right)[] columns, OrderFragment f)
    {
        int first = -1, last = -1;
        double width = f.X1 - f.X0;
        for (int c = 0; c < columns.Length; c++)
        {
            double overlap = Math.Min(f.X1, columns[c].Right) - Math.Max(f.X0, columns[c].Left);
            if (overlap > 0 && overlap >= MinOverlapShare * Math.Min(width, columns[c].Right - columns[c].Left))
            {
                first = first < 0 ? c : first;
                last = c;
            }
        }

        if (first >= 0)
        {
            return (first, last);
        }

        // Entirely in a gap: the nearest column.
        double centre = (f.X0 + f.X1) / 2;
        int best = 0;
        double bestDistance = double.MaxValue;
        for (int c = 0; c < columns.Length; c++)
        {
            double d = centre < columns[c].Left ? columns[c].Left - centre : centre > columns[c].Right ? centre - columns[c].Right : 0;
            if (d < bestDistance)
            {
                bestDistance = d;
                best = c;
            }
        }

        return (best, best);
    }

    // The columns' extents (the stretches of text between separators), left to right; null when there is no separator.
    private static (double Left, double Right)[]? Columns(List<List<OrderFragment>> rows, OcrBox box, ITableSeparators? separators)
    {
        double minX = double.MaxValue, maxX = double.MinValue;
        foreach (List<OrderFragment> r in rows)
        {
            foreach (OrderFragment f in r)
            {
                minX = Math.Min(minX, f.X0);
                maxX = Math.Max(maxX, f.X1);
            }
        }

        if (separators?.ColumnBoundaries(box) is { Count: > 0 } given)
        {
            var edges = new List<double> { minX };
            foreach (double b in given.Order())
            {
                if (b > edges[^1] && b < maxX)
                {
                    edges.Add(b);
                }
            }

            edges.Add(maxX);
            var learned = new (double, double)[edges.Count - 1];
            for (int i = 0; i < learned.Length; i++)
            {
                learned[i] = (edges[i], edges[i + 1]);
            }

            return learned;
        }

        // Coverage: for each stretch between neighbouring fragment edges, the number of rows that have text across it.
        var points = new SortedSet<double>();
        foreach (List<OrderFragment> r in rows)
        {
            foreach (OrderFragment f in r)
            {
                points.Add(f.X0);
                points.Add(f.X1);
            }
        }

        double[] xs = [.. points];
        var coverage = new int[xs.Length - 1];
        var stamp = new int[xs.Length - 1];
        Array.Fill(stamp, -1);
        for (int r = 0; r < rows.Count; r++)
        {
            foreach (OrderFragment f in rows[r])
            {
                int from = Array.BinarySearch(xs, f.X0), to = Array.BinarySearch(xs, f.X1);
                for (int i = from; i < to; i++)
                {
                    if (stamp[i] != r)
                    {
                        stamp[i] = r;
                        coverage[i]++;
                    }
                }
            }
        }

        double height = Median(rows.SelectMany(r => r).Select(f => f.Bot - f.Top));
        int tolerance = (int)(GapTolerance * rows.Count);
        var result = new List<(double, double)>();
        double start = double.NaN, end = double.NaN;
        double gapStart = double.NaN;
        for (int i = 0; i < coverage.Length; i++)
        {
            bool text = coverage[i] > tolerance;
            if (text)
            {
                if (double.IsNaN(start))
                {
                    start = xs[i];
                }

                end = xs[i + 1];
                gapStart = double.NaN;
            }
            else
            {
                if (double.IsNaN(gapStart))
                {
                    gapStart = xs[i];
                }

                // A gap wide enough closes the column in progress.
                if (!double.IsNaN(start) && xs[i + 1] - gapStart >= MinGapHeights * height)
                {
                    result.Add((start, end));
                    start = double.NaN;
                }
            }
        }

        if (!double.IsNaN(start))
        {
            result.Add((start, end));
        }

        return result.Count < 2 ? null : [.. result];
    }

    private static double Median(IEnumerable<double> values)
    {
        double[] v = [.. values.Order()];
        return v.Length == 0 ? 0 : v[v.Length / 2];
    }
}
