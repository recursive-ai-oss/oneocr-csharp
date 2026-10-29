namespace OneOcr.Layout;

/// <summary>A straight ruling line: its centre position across the line, its extent along it (half-open) and its thickness.</summary>
internal readonly record struct RuleLine(int Pos, int Lo, int Hi, int Thickness)
{
    public int First => Pos - (Thickness / 2);

    public int Last => First + Thickness - 1;
}

/// <summary>A cell of a <see cref="RuleGrid"/>: the grid rows and columns it covers.</summary>
internal readonly record struct GridCell(int Row, int Column, int RowSpan, int ColumnSpan);

/// <summary>
/// A table drawn with ruling lines: the column and row boundaries and the cells they enclose. A cell is a rectangle of the
/// elementary grid cells that are not separated by a rule, so a rule that is missing between two cells makes one cell with a span.
/// </summary>
internal sealed class RuleGrid
{
    private readonly int[] _cellOfBase;

    public RuleGrid(int[] xs, int[] ys, List<GridCell> cells, int[] cellOfBase, List<RuleLine> horizontal, List<RuleLine> vertical)
    {
        Xs = xs;
        Ys = ys;
        Cells = cells;
        _cellOfBase = cellOfBase;
        Horizontal = horizontal;
        Vertical = vertical;
    }

    /// <summary>Centres of the vertical rules, left to right; one more than <see cref="Columns"/>.</summary>
    public int[] Xs { get; }

    /// <summary>Centres of the horizontal rules, top to bottom; one more than <see cref="Rows"/>.</summary>
    public int[] Ys { get; }

    public int Columns => Xs.Length - 1;

    public int Rows => Ys.Length - 1;

    /// <summary>The cells in row-major order of their top-left elementary cell.</summary>
    public List<GridCell> Cells { get; }

    /// <summary>The ruling lines the grid was built from.</summary>
    public List<RuleLine> Horizontal { get; }

    public List<RuleLine> Vertical { get; }

    public int Left => Xs[0];

    public int Right => Xs[^1];

    public int Top => Ys[0];

    public int Bottom => Ys[^1];

    /// <summary>The cell that holds a point, or -1 when the point is outside the grid.</summary>
    public int CellAt(double x, double y)
    {
        if (x < Left || x >= Right || y < Top || y >= Bottom)
        {
            return -1;
        }

        return _cellOfBase[(Bound(Ys, y) * Columns) + Bound(Xs, x)];
    }

    /// <summary>The elementary row that holds y (clamped to the grid).</summary>
    public int RowAt(double y) => Bound(Ys, y);

    /// <summary>The cell that holds the elementary cell at a row and column.</summary>
    public int CellOfBase(int row, int column) => _cellOfBase[(row * Columns) + column];

    // The index of the interval that holds v: Xs[i] <= v < Xs[i + 1].
    private static int Bound(int[] bounds, double v)
    {
        int lo = 0, hi = bounds.Length - 2;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) / 2;
            if (bounds[mid] <= v)
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1;
            }
        }

        return lo;
    }
}

/// <summary>
/// Finds tables drawn with ruling lines. Horizontal and vertical ink runs of at least <see cref="MinRun"/> pixels are collected in
/// one pass over the binarized page (the run-length form of the morphological line filters; Zanibbi, Blostein and Cordy, "A survey of
/// table recognition", IJDAR 2004, sec. 4), runs in neighbouring rows or columns are joined into lines of limited thickness, lines
/// that touch form a component, and a component with at least two rules each way is a grid. Where a rule does not cover the edge
/// between two cells, the cells are one cell with a span.
/// </summary>
internal static class TableRules
{
    /// <summary>Shortest ink run, in pixels, that can belong to a ruling line.</summary>
    internal const int MinRun = 24;

    // A line thicker than this is a filled bar, not a rule.
    private const int MaxThickness = 12;

    // A line is a rule only when it is at least this many times longer than thick.
    private const int MinSlenderness = 10;

    // Most runs of one direction kept; a page with more is noise (a dithered photograph), not a document with tables.
    private const int MaxRuns = 1 << 17;

    // Most lines of one direction that are tested for grids.
    private const int MaxLines = 4096;

    // Most elementary cells of one grid.
    private const int MaxCells = 1 << 16;

    // Most ink a table's box may hold.
    private const double MaxInkShare = 0.4;

    // Smallest row height or column width of a table, in pixels.
    private const int MinCell = 16;

    // Share of a table's width or height that its frame rules must run.
    private const double FrameShare = 0.6;

    // Fraction of an edge a rule must cover for the edge to count as drawn.
    private const double EdgeCoverage = 0.6;

    // Horizontal rules that run on beyond the outermost vertical rule by this share of the grid's width mark a fragment of a table.
    private const double FragmentOverhang = 0.25;

    internal readonly record struct Run(int Pos, int Lo, int Hi);

    /// <summary>
    /// The horizontal rules of a page, whether or not they belong to a ruled grid: the lines that separate the rows of a table whose
    /// columns are not drawn. Nothing is changed. Empty when the page has no contrast or is noise.
    /// </summary>
    /// <param name="gray">The page, row-major.</param>
    /// <param name="width">Page width.</param>
    /// <param name="height">Page height.</param>
    /// <param name="threshold">The page's binarization threshold; negative when the page has no contrast.</param>
    /// <param name="inkIsBright">Whether ink is the bright side of the threshold.</param>
    /// <returns>The rules, top to bottom.</returns>
    public static List<RuleLine> HorizontalRules(ReadOnlySpan<byte> gray, int width, int height, int threshold, bool inkIsBright)
    {
        if (threshold < 0 || width < MinRun || height < MinRun || !CollectRuns(gray, width, height, threshold, inkIsBright, out List<Run> hRuns, out _))
        {
            return [];
        }

        List<RuleLine> h = Merge(hRuns);
        return h.Count > MaxLines ? [] : h;
    }

    /// <summary>Finds the ruled grids of a page. Nothing is changed.</summary>
    /// <param name="gray">The page, row-major.</param>
    /// <param name="width">Page width.</param>
    /// <param name="height">Page height.</param>
    /// <param name="threshold">The page's binarization threshold; negative when the page has no contrast.</param>
    /// <param name="inkIsBright">Whether ink is the bright side of the threshold.</param>
    /// <returns>The grids, top to bottom.</returns>
    public static List<RuleGrid> Find(ReadOnlySpan<byte> gray, int width, int height, int threshold, bool inkIsBright)
    {
        var grids = new List<RuleGrid>();
        if (threshold < 0 || width < MinRun || height < MinRun)
        {
            return grids;
        }

        if (!CollectRuns(gray, width, height, threshold, inkIsBright, out List<Run> hRuns, out List<Run> vRuns))
        {
            return grids;
        }

        List<RuleLine> h = Merge(hRuns), v = Merge(vRuns);
        if (h.Count < 2 || v.Count < 2 || h.Count > MaxLines || v.Count > MaxLines)
        {
            return grids;
        }

        // Lines that touch belong to one table.
        var parent = new int[h.Count + v.Count];
        for (int i = 0; i < parent.Length; i++)
        {
            parent[i] = i;
        }

        for (int i = 0; i < h.Count; i++)
        {
            RuleLine a = h[i];
            for (int j = 0; j < v.Count; j++)
            {
                RuleLine b = v[j];
                int tol = Math.Max(3, Math.Max(a.Thickness, b.Thickness) * 2);
                if (b.Pos >= a.Lo - tol && b.Pos <= a.Hi + tol && a.Pos >= b.Lo - tol && a.Pos <= b.Hi + tol)
                {
                    Union(parent, i, h.Count + j);
                }
            }
        }

        var members = new SortedDictionary<int, (List<RuleLine> H, List<RuleLine> V)>();
        for (int i = 0; i < parent.Length; i++)
        {
            int root = Find(parent, i);
            if (!members.TryGetValue(root, out (List<RuleLine> H, List<RuleLine> V) m))
            {
                members[root] = m = ([], []);
            }

            (i < h.Count ? m.H : m.V).Add(i < h.Count ? h[i] : v[i - h.Count]);
        }

        foreach ((List<RuleLine> mh, List<RuleLine> mv) in members.Values)
        {
            if (mh.Count >= 2 && mv.Count >= 2 && Build(mh, mv) is { } grid && !IsFragment(grid, mh) && InkShare(gray, width, height, grid, threshold, inkIsBright) <= MaxInkShare)
            {
                grids.Add(grid);
            }
        }

        grids.Sort((a, b) => a.Top != b.Top ? a.Top.CompareTo(b.Top) : a.Left.CompareTo(b.Left));
        return grids;
    }

    /// <summary>
    /// Paints the ruling lines of the grids with the paper next to them, so that crops and profiles no longer contain them.
    /// Each rule pixel and its anti-aliased fringe take the value of the paper-side neighbour two pixels beyond the line.
    /// </summary>
    public static void Erase(Span<byte> gray, int width, int height, bool inkIsBright, IReadOnlyList<RuleGrid> grids)
    {
        foreach (RuleGrid g in grids)
        {
            foreach (RuleLine line in g.Horizontal)
            {
                int y0 = Math.Max(0, line.First - 1), y1 = Math.Min(height - 1, line.Last + 1);
                int above = Math.Max(0, y0 - 2), below = Math.Min(height - 1, y1 + 2);
                for (int x = Math.Max(0, line.Lo); x < Math.Min(width, line.Hi); x++)
                {
                    byte fill = Paper(gray[(above * width) + x], gray[(below * width) + x], inkIsBright);
                    for (int y = y0; y <= y1; y++)
                    {
                        gray[(y * width) + x] = fill;
                    }
                }
            }
        }

        foreach (RuleGrid g in grids)
        {
            foreach (RuleLine line in g.Vertical)
            {
                int x0 = Math.Max(0, line.First - 1), x1 = Math.Min(width - 1, line.Last + 1);
                int left = Math.Max(0, x0 - 2), right = Math.Min(width - 1, x1 + 2);
                for (int y = Math.Max(0, line.Lo); y < Math.Min(height, line.Hi); y++)
                {
                    int row = y * width;
                    byte fill = Paper(gray[row + left], gray[row + right], inkIsBright);
                    for (int x = x0; x <= x1; x++)
                    {
                        gray[row + x] = fill;
                    }
                }
            }
        }
    }

    // The share of ink inside the grid's box. A table is mostly paper; a light panel on a dark page, read as ink, is not a table.
    // A grid built from the verticals of part of a table: at least two of the horizontal rules run on well beyond the outer vertical rules
    // on one side, so the table has columns whose rules are not drawn (or too faint to be found). Rows and cells of such a grid would leave
    // those columns out; the table is read without a grid, row bands from the horizontal rules and columns from the gaps.
    private static bool IsFragment(RuleGrid g, List<RuleLine> horizontal)
    {
        int overhang = (int)(FragmentOverhang * (g.Right - g.Left));
        int left = 0, right = 0;
        foreach (RuleLine h in horizontal)
        {
            if (g.Left - h.Lo > overhang)
            {
                left++;
            }

            if (h.Hi - g.Right > overhang)
            {
                right++;
            }
        }

        return left >= 2 || right >= 2;
    }

    private static double InkShare(ReadOnlySpan<byte> gray, int width, int height, RuleGrid g, int threshold, bool inkIsBright)
    {
        int x0 = Math.Clamp(g.Left, 0, width), x1 = Math.Clamp(g.Right, x0, width), y0 = Math.Clamp(g.Top, 0, height), y1 = Math.Clamp(g.Bottom, y0, height);
        long ink = 0, total = (long)(x1 - x0) * (y1 - y0);
        for (int y = y0; y < y1; y++)
        {
            foreach (byte v in gray.Slice((y * width) + x0, x1 - x0))
            {
                if (inkIsBright ? v > threshold : v <= threshold)
                {
                    ink++;
                }
            }
        }

        return total == 0 ? 1 : (double)ink / total;
    }

    // Of two neighbours the one farther from ink: the paper, even when text touches the line on one side.
    private static byte Paper(byte a, byte b, bool inkIsBright) => inkIsBright ? Math.Min(a, b) : Math.Max(a, b);

    // Collects the horizontal and vertical ink runs of at least MinRun pixels. Paper is skipped with the vectorized range search of the
    // base library and only ink pixels are visited: the columns that were ink in the row above are the only ones whose vertical run can
    // end in this row, so they are tracked in a list instead of every column being looked at in every row. The runs found are the
    // same as a pixel-by-pixel scan finds (Merge sorts them by position, which is unique per run).
    internal static bool CollectRuns(ReadOnlySpan<byte> gray, int width, int height, int threshold, bool inkIsBright, out List<Run> hRuns, out List<Run> vRuns)
    {
        hRuns = [];
        vRuns = [];
        var vLen = new int[width];
        // Whether a gray value is ink, for the columns that were ink in the row above.
        bool[] isInk = new bool[256];
        for (int v = 0; v < 256; v++)
        {
            isInk[v] = inkIsBright ? v > threshold : v <= threshold;
        }

        // The ink values form one range: [0, threshold] for dark ink, [threshold + 1, 255] for bright ink.
        byte lo = inkIsBright ? (byte)(threshold + 1) : (byte)0, hi = inkIsBright ? (byte)255 : (byte)threshold;
        int[] active = new int[width], next = new int[width];
        int activeCount = 0;
        for (int y = 0; y < height; y++)
        {
            ReadOnlySpan<byte> row = gray.Slice(y * width, width);
            int nextCount = 0;
            for (int x = 0; x < row.Length;)
            {
                int skip = row.Slice(x).IndexOfAnyInRange(lo, hi);
                if (skip < 0)
                {
                    break;
                }

                int start = x + skip;
                int length = row.Slice(start).IndexOfAnyExceptInRange(lo, hi);
                int end = length < 0 ? row.Length : start + length;
                if (end - start >= MinRun)
                {
                    hRuns.Add(new Run(y, start, end));
                }

                for (int k = start; k < end; k++)
                {
                    vLen[k]++;
                    next[nextCount++] = k;
                }

                x = end;
            }

            // A column that was ink above and is not now ends its vertical run here.
            for (int a = 0; a < activeCount; a++)
            {
                int x = active[a];
                if (!isInk[row[x]])
                {
                    if (vLen[x] >= MinRun)
                    {
                        vRuns.Add(new Run(x, y - vLen[x], y));
                    }

                    vLen[x] = 0;
                }
            }

            (active, next) = (next, active);
            activeCount = nextCount;
            if (hRuns.Count > MaxRuns || vRuns.Count > MaxRuns)
            {
                return false;
            }
        }

        for (int a = 0; a < activeCount; a++)
        {
            int x = active[a];
            if (vLen[x] >= MinRun)
            {
                vRuns.Add(new Run(x, height - vLen[x], height));
            }
        }

        return vRuns.Count <= MaxRuns;
    }

    // Joins runs in adjacent rows (or columns) that overlap by most of the shorter one into lines; thick ones are dropped.
    internal static List<RuleLine> Merge(List<Run> runs)
    {
        runs.Sort((a, b) => a.Pos != b.Pos ? a.Pos.CompareTo(b.Pos) : a.Lo.CompareTo(b.Lo));
        var done = new List<RuleLine>();
        var open = new List<(int First, int Last, int Lo, int Hi)>();
        int index = 0;
        while (index < runs.Count)
        {
            int pos = runs[index].Pos;
            // Lines that were not extended in the previous row are finished.
            for (int i = open.Count - 1; i >= 0; i--)
            {
                if (open[i].Last < pos - 1)
                {
                    Close(done, open[i]);
                    open.RemoveAt(i);
                }
            }

            var extended = new List<(int First, int Last, int Lo, int Hi)>();
            for (; index < runs.Count && runs[index].Pos == pos; index++)
            {
                Run r = runs[index];
                int match = -1;
                for (int i = 0; i < open.Count; i++)
                {
                    int overlap = Math.Min(r.Hi, open[i].Hi) - Math.Max(r.Lo, open[i].Lo);
                    int shorter = Math.Min(r.Hi - r.Lo, open[i].Hi - open[i].Lo);
                    if (open[i].Last == pos - 1 && overlap * 10 >= shorter * 6)
                    {
                        match = i;
                        break;
                    }
                }

                if (match >= 0)
                {
                    (int first, _, int lo, int hi) = open[match];
                    open[match] = (first, pos, Math.Min(lo, r.Lo), Math.Max(hi, r.Hi));
                }
                else
                {
                    extended.Add((pos, pos, r.Lo, r.Hi));
                }
            }

            open.AddRange(extended);
        }

        foreach (var l in open)
        {
            Close(done, l);
        }

        return done;
    }

    private static void Close(List<RuleLine> done, (int First, int Last, int Lo, int Hi) l)
    {
        int thickness = l.Last - l.First + 1;
        // A rule is slender; the stem or bar of a large letter is as long as a short rule but several times thicker.
        if (thickness <= MaxThickness && (l.Hi - l.Lo) >= MinSlenderness * thickness)
        {
            done.Add(new RuleLine(l.First + (thickness / 2), l.Lo, l.Hi, thickness));
        }
    }

    private static int Find(int[] parent, int i)
    {
        while (parent[i] != i)
        {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }

        return i;
    }

    private static void Union(int[] parent, int a, int b)
    {
        a = Find(parent, a);
        b = Find(parent, b);
        if (a != b)
        {
            parent[Math.Max(a, b)] = Math.Min(a, b);
        }
    }

    // Lines of one direction closer together than a rule is thick are one boundary.
    private static int[] Boundaries(List<RuleLine> lines)
    {
        var sorted = lines.OrderBy(l => l.Pos).ToList();
        var result = new List<int>();
        int sum = 0, count = 0, last = int.MinValue;
        foreach (RuleLine l in sorted)
        {
            if (count > 0 && l.Pos - last > Math.Max(3, l.Thickness * 2))
            {
                result.Add((int)Math.Round((double)sum / count));
                sum = 0;
                count = 0;
            }

            sum += l.Pos;
            count++;
            last = l.Pos;
        }

        if (count > 0)
        {
            result.Add((int)Math.Round((double)sum / count));
        }

        return [.. result];
    }

    // Whether the lines near boundary `pos` together cover most of [lo, hi).
    private static bool Covers(List<RuleLine> lines, int pos, int lo, int hi, int slack)
    {
        int a = lo + slack, b = hi - slack;
        if (b <= a)
        {
            a = lo;
            b = hi;
        }

        var spans = new List<(int, int)>();
        foreach (RuleLine l in lines)
        {
            if (Math.Abs(l.Pos - pos) <= Math.Max(3, l.Thickness * 2) && l.Hi > a && l.Lo < b)
            {
                spans.Add((Math.Max(a, l.Lo), Math.Min(b, l.Hi)));
            }
        }

        spans.Sort();
        int covered = 0, reach = a;
        foreach ((int s, int e) in spans)
        {
            int from = Math.Max(s, reach);
            if (e > from)
            {
                covered += e - from;
                reach = e;
            }
        }

        return covered >= EdgeCoverage * Math.Max(1, b - a);
    }

    // How many of the lines run at least most of `span`.
    private static int Longest(List<RuleLine> lines, int span)
    {
        int n = 0;
        foreach (RuleLine l in lines)
        {
            if ((l.Hi - l.Lo) >= FrameShare * span)
            {
                n++;
            }
        }

        return n;
    }

    private static int Narrowest(int[] bounds)
    {
        int min = int.MaxValue;
        for (int i = 1; i < bounds.Length; i++)
        {
            min = Math.Min(min, bounds[i] - bounds[i - 1]);
        }

        return min;
    }

    private static RuleGrid? Build(List<RuleLine> mh, List<RuleLine> mv)
    {
        int[] xs = Boundaries(mv), ys = Boundaries(mh);
        int cols = xs.Length - 1, rows = ys.Length - 1;
        if (rows < 2 || cols < 2 || (long)cols * rows > MaxCells)
        {
            return null;
        }

        // Strokes of large letters line up into rules too; a table is a frame: at least two rules each way run most of its width or
        // height, and its rows and columns are wide enough to hold text.
        int width = xs[^1] - xs[0], height = ys[^1] - ys[0];
        if (Longest(mh, width) < 2 || Longest(mv, height) < 2 || Narrowest(xs) < MinCell || Narrowest(ys) < MinCell)
        {
            return null;
        }

        int thick = 0;
        foreach (RuleLine l in mh)
        {
            thick = Math.Max(thick, l.Thickness);
        }

        foreach (RuleLine l in mv)
        {
            thick = Math.Max(thick, l.Thickness);
        }

        // vEdge[b][r]: the vertical boundary b is drawn along row r; hEdge[b][c] likewise.
        var parent = new int[rows * cols];
        for (int i = 0; i < parent.Length; i++)
        {
            parent[i] = i;
        }

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (c + 1 < cols && !Covers(mv, xs[c + 1], ys[r], ys[r + 1], thick + 1))
                {
                    Union(parent, (r * cols) + c, (r * cols) + c + 1);
                }

                if (r + 1 < rows && !Covers(mh, ys[r + 1], xs[c], xs[c + 1], thick + 1))
                {
                    Union(parent, (r * cols) + c, ((r + 1) * cols) + c);
                }
            }
        }

        // A merged cell must be a rectangle; a group that is not falls apart into elementary cells.
        var bounds = new Dictionary<int, (int R0, int C0, int R1, int C1, int N)>();
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                int root = Find(parent, (r * cols) + c);
                bounds[root] = bounds.TryGetValue(root, out var b)
                    ? (Math.Min(b.R0, r), Math.Min(b.C0, c), Math.Max(b.R1, r), Math.Max(b.C1, c), b.N + 1)
                    : (r, c, r, c, 1);
            }
        }

        var cells = new List<GridCell>();
        var cellOf = new int[rows * cols];
        var assigned = new Dictionary<int, int>();
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                int root = Find(parent, (r * cols) + c);
                (int r0, int c0, int r1, int c1, int n) = bounds[root];
                bool rect = n == (r1 - r0 + 1) * (c1 - c0 + 1);
                if (!rect)
                {
                    cellOf[(r * cols) + c] = cells.Count;
                    cells.Add(new GridCell(r, c, 1, 1));
                }
                else if (assigned.TryGetValue(root, out int existing))
                {
                    cellOf[(r * cols) + c] = existing;
                }
                else
                {
                    assigned[root] = cells.Count;
                    cellOf[(r * cols) + c] = cells.Count;
                    cells.Add(new GridCell(r0, c0, r1 - r0 + 1, c1 - c0 + 1));
                }
            }
        }

        return new RuleGrid(xs, ys, cells, cellOf, mh, mv);
    }
}
