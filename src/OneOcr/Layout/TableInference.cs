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
    public static List<OrderBlock> Infer(List<OrderBlock> plan, IReadOnlyList<DetectedRegion> regions, ITableSeparators? separators)
    {
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

            var inside = new List<int>();
            for (int i = 0; i < slots.Length; i++)
            {
                if (slots[i] is { Cells: null, IsStandalone: false } b && Inside(region, b))
                {
                    inside.Add(i);
                }
            }

            if (inside.Count == 0)
            {
                continue;
            }

            // The rows are the lines of the page: fragments of different blocks that sit on one line are one row.
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

            var rows = byLine.Values.OrderBy(r => r.Average(f => f.Base)).ToList();
            OrderBlock? made = Build(rows, new OcrBox(region.Left, region.Top, region.Right, region.Bottom), separators);
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
                result.Add(Build(rows, new OcrBox(left, top, right, bottom), separators) ?? b);
            }
            else
            {
                result.Add(b);
            }
        }

        return result;
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
    private static OrderBlock? Build(List<List<OrderFragment>> rows, OcrBox box, ITableSeparators? separators)
    {
        int total = rows.Sum(r => r.Count);
        if (rows.Count < 2 || total > MaxFragments)
        {
            return null;
        }

        foreach (List<OrderFragment> r in rows)
        {
            r.Sort((a, b) => a.X0.CompareTo(b.X0));
        }

        (double Left, double Right)[]? columns = Columns(rows, box, separators);
        if (columns is null || columns.Length < 2)
        {
            return null;
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
            foreach (OrderFragment f in row)
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
