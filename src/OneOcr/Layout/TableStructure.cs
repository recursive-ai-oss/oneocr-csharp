namespace OneOcr.Layout;

/// <summary>
/// Reads the structure of tables drawn with ruling lines: strips are cut at the vertical rules, so a word is never read across a
/// column, and the words are put into the cells the rules enclose, row by row.
/// </summary>
internal static partial class TableStructure
{
    // Leaves of strips inside a grid are numbered from here, well clear of the finder's own leaf numbers.
    private const int LeafBase = 1 << 24;

    /// <summary>
    /// Cuts the strips that cross a drawn vertical rule at the rule, and gives the strips of each grid their own leaf per cell column,
    /// so the reading order sees the cells of a row as separate regions on one line. Other strips are returned as they are.
    /// </summary>
    /// <param name="strips">The strips found on the page.</param>
    /// <param name="grids">The page's ruled grids; with none the strips are returned as they are.</param>
    /// <param name="gray">The page after <see cref="TableRules.Erase"/>.</param>
    /// <param name="width">Page width.</param>
    /// <param name="height">Page height.</param>
    /// <param name="threshold">The page's binarization threshold.</param>
    /// <param name="inkIsBright">Whether ink is the bright side of the threshold.</param>
    public static List<StripLine> SplitStrips(
        List<StripLine> strips, IReadOnlyList<RuleGrid> grids, ReadOnlySpan<byte> gray, int width, int height, int threshold, bool inkIsBright)
    {
        if (grids.Count == 0)
        {
            return strips;
        }

        var result = new List<StripLine>(strips.Count);
        foreach (StripLine s in strips)
        {
            double cx = s.Box.X + (s.Box.Width / 2.0), cy = s.Box.Y + (s.Box.Height / 2.0);
            int gi = GridAt(grids, cx, cy);
            if (gi < 0)
            {
                result.Add(s);
                continue;
            }

            RuleGrid g = grids[gi];
            int row = g.RowAt(cy);
            // The cuts: drawn vertical rules strictly inside the strip.
            var cuts = new List<int>();
            for (int b = 1; b < g.Columns; b++)
            {
                if (g.Xs[b] > s.Box.X && g.Xs[b] < s.Box.Right && g.CellOfBase(row, b - 1) != g.CellOfBase(row, b))
                {
                    cuts.Add(g.Xs[b]);
                }
            }

            int from = s.Box.X;
            for (int i = 0; i <= cuts.Count; i++)
            {
                int to = i < cuts.Count ? cuts[i] : s.Box.Right;
                LineBox piece = cuts.Count == 0 ? s.Box : InkBounds(gray, width, height, from, s.Box.Y, to, s.Box.Bottom, threshold, inkIsBright);
                from = to;
                if (piece.Width <= 0)
                {
                    continue;
                }

                int cell = g.CellAt(piece.X + (piece.Width / 2.0), cy);
                int column = cell < 0 ? 0 : g.Cells[cell].Column;
                result.Add(new StripLine(piece, LeafBase + (gi * 4096) + column));
            }
        }

        return result;
    }

    private static int GridAt(IReadOnlyList<RuleGrid> grids, double x, double y)
    {
        for (int i = 0; i < grids.Count; i++)
        {
            if (grids[i].CellAt(x, y) >= 0)
            {
                return i;
            }
        }

        return -1;
    }

    // The box of the ink inside a rectangle; a zero width when there is none.
    private static LineBox InkBounds(ReadOnlySpan<byte> gray, int width, int height, int x0, int y0, int x1, int y1, int threshold, bool inkIsBright)
    {
        x0 = Math.Clamp(x0, 0, width);
        x1 = Math.Clamp(x1, x0, width);
        y0 = Math.Clamp(y0, 0, height);
        y1 = Math.Clamp(y1, y0, height);
        int minX = int.MaxValue, maxX = -1, minY = int.MaxValue, maxY = -1;
        for (int y = y0; y < y1; y++)
        {
            ReadOnlySpan<byte> row = gray.Slice((y * width) + x0, x1 - x0);
            for (int x = 0; x < row.Length; x++)
            {
                if (inkIsBright ? row[x] > threshold : row[x] <= threshold)
                {
                    minX = Math.Min(minX, x0 + x);
                    maxX = Math.Max(maxX, x0 + x);
                    minY = Math.Min(minY, y);
                    maxY = Math.Max(maxY, y);
                }
            }
        }

        return maxX < 0 ? new LineBox(x0, y0, 0, 0) : new LineBox(minX, minY, maxX - minX + 1, maxY - minY + 1);
    }

    /// <summary>
    /// Replaces, for every grid, the fragments that lie inside it by one table block whose lines are the grid's rows and whose
    /// fragments are tagged with their cell. Fragments are put into cells by their centre, so a multi-line cell holds all of its
    /// lines, top to bottom, and a cell with a span holds its words once, in its first row. The block goes where the table stands in
    /// the reading order: before the first block below it that overlaps its columns.
    /// </summary>
    /// <param name="plan">The blocks in reading order, as <see cref="ReadingOrder.Arrange"/> made them.</param>
    /// <param name="grids">The page's ruled grids.</param>
    /// <returns>The blocks, with a table block for every grid that holds text.</returns>
    public static List<OrderBlock> Apply(List<OrderBlock> plan, IReadOnlyList<RuleGrid> grids)
    {
        foreach (RuleGrid g in grids)
        {
            var taken = new List<OrderFragment>();
            var next = new List<OrderBlock>(plan.Count + 1);
            foreach (OrderBlock b in plan)
            {
                var lines = new List<OrderLine>(b.Lines.Count);
                bool changed = false;
                foreach (OrderLine l in b.Lines)
                {
                    var keep = new List<OrderFragment>(l.Fragments.Count);
                    foreach (OrderFragment f in l.Fragments)
                    {
                        if (g.CellAt((f.X0 + f.X1) / 2, (f.Top + f.Bot) / 2) >= 0)
                        {
                            taken.Add(f);
                            changed = true;
                        }
                        else
                        {
                            keep.Add(f);
                        }
                    }

                    if (keep.Count > 0)
                    {
                        lines.Add(keep.Count == l.Fragments.Count ? l : new OrderLine { Fragments = keep });
                    }
                }

                if (!changed)
                {
                    next.Add(b);
                }
                else if (lines.Count > 0)
                {
                    // What is left of a block that held a table's text is ordinary text.
                    next.Add(new OrderBlock { IsTable = false, Columns = 0, Lines = lines });
                }
            }

            if (taken.Count == 0)
            {
                plan = next;
                continue;
            }

            OrderBlock table = MakeBlock(g, taken);
            int at = next.Count;
            for (int i = 0; i < next.Count; i++)
            {
                (double top, double left, double right) = Extent(next[i]);
                double overlap = Math.Min(right, g.Right) - Math.Max(left, g.Left);
                if (top >= g.Top && overlap >= 0.3 * Math.Min(right - left, g.Right - g.Left))
                {
                    at = i;
                    break;
                }
            }

            next.Insert(at, table);
            plan = next;
        }

        return plan;
    }

    private static (double Top, double Left, double Right) Extent(OrderBlock b)
    {
        double top = double.MaxValue, left = double.MaxValue, right = double.MinValue;
        foreach (OrderLine l in b.Lines)
        {
            foreach (OrderFragment f in l.Fragments)
            {
                top = Math.Min(top, f.Top);
                left = Math.Min(left, f.X0);
                right = Math.Max(right, f.X1);
            }
        }

        return (top, left, right);
    }

    private static OrderBlock MakeBlock(RuleGrid g, List<OrderFragment> taken)
    {
        var cells = new List<OrderCell>(g.Cells.Count);
        foreach (GridCell c in g.Cells)
        {
            cells.Add(new OrderCell(c.Row, c.Column, c.RowSpan, c.ColumnSpan, new OcrBox(g.Xs[c.Column], g.Ys[c.Row], g.Xs[c.Column + c.ColumnSpan], g.Ys[c.Row + c.RowSpan])));
        }

        var byRow = new List<OrderFragment>?[g.Rows];
        foreach (OrderFragment f in taken)
        {
            int cell = g.CellAt((f.X0 + f.X1) / 2, (f.Top + f.Bot) / 2);
            f.Cell = cell;
            (byRow[cells[cell].Row] ??= []).Add(f);
        }

        var lines = new List<OrderLine>();
        foreach (List<OrderFragment>? row in byRow)
        {
            if (row is null)
            {
                continue;
            }

            // Cells left to right, the lines of one cell top to bottom.
            row.Sort((a, b) =>
            {
                int c = cells[a.Cell].Column.CompareTo(cells[b.Cell].Column);
                if (c != 0)
                {
                    return c;
                }

                c = a.Cell.CompareTo(b.Cell);
                if (c != 0)
                {
                    return c;
                }

                c = a.Top.CompareTo(b.Top);
                return c != 0 ? c : a.X0.CompareTo(b.X0);
            });
            lines.Add(new OrderLine { Fragments = row });
        }

        return new OrderBlock { IsTable = true, Columns = g.Columns, Rows = g.Rows, Lines = lines, Cells = cells };
    }
}
