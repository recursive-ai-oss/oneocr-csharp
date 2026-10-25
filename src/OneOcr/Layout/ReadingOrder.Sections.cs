// Reading order beyond the plain X–Y cut, for screens and multi-column pages. The plain cut (OnePDF's) cuts at empty vertical
// stripes first. That reads a screen's menu bar, sidebar and cards in the wrong order, because every stripe between menu items
// or cards runs through the rows below, and it reads two text columns in slices whenever their paragraphs end at the same
// height. The rules here pick the cut that keeps each visual unit whole (Nagy and Seth 1984, recursive X–Y cut; the choice of
// axis follows O'Gorman 1993, Docstrum: blocks are the units that stay together). All distances are in em of the text size.

namespace OneOcr.Layout;

internal static partial class ReadingOrder
{
    // A region that the vertical cut would split into columns, but that is not a table: decide whether the columns are the right
    // reading of it. Order of the rules: a header row is read first; fields or list rows that the columns tie together are read
    // row by row; a column that sits inside another (cards between a title and a list) means the columns are not the structure.
    private static (List<int[]> Groups, bool FirstIsStandalone) ChooseCut(List<OrderFragment> f, int[] reg, List<int[]> columns, double ms)
    {
        var bandGaps = new List<double>();
        List<int[]>? bands = Partition(f, reg, false, LayoutParameters.BlockGapEm * ms, bandGaps);
        if (bands is null)
        {
            return (columns, false);
        }

        if (IsHeaderRow(f, reg, bands, ms))
        {
            return ([bands[0], [.. bands.Skip(1).SelectMany(b => b)]], true);
        }

        if (PeelSidebar(f, reg, columns) is { } sidebar)
        {
            return (sidebar, true);
        }

        var rowGaps = new List<double>();
        List<int[]>? coarse = Partition(f, reg, false, LayoutParameters.RowGapEm * ms, rowGaps);
        List<int[]>? rows = coarse is null ? null : MergeWeak(coarse, rowGaps);
        if (rows is not null && AreTiedRows(f, columns, rows))
        {
            return (rows, false);
        }

        return (HasEmbedded(f, columns, LayoutParameters.EmbedMarginEm * ms) ? rows ?? bands : columns, false);
    }

    // The topmost band of a region is a header row when it is a single row of several fragments (a bar of menu items, a running
    // head) that spans most of the region's width and stands clear of the rest by a wide gap. It is read first, whole, before
    // any column cut. The label row of a field grid has the same shape but a gap of two em or less, so it is not taken.
    private static bool IsHeaderRow(List<OrderFragment> f, int[] reg, List<int[]> bands, double ms)
    {
        int[] top = bands[0];
        if (top.Length < 2)
        {
            return false;
        }

        double headTop = double.MaxValue, headBottom = double.MinValue, headLeft = double.MaxValue, headRight = double.MinValue;
        foreach (int id in top)
        {
            headTop = Math.Min(headTop, f[id].Top);
            headBottom = Math.Max(headBottom, f[id].Bot);
            headLeft = Math.Min(headLeft, f[id].X0);
            headRight = Math.Max(headRight, f[id].X1);
        }

        if (headBottom - headTop > LayoutParameters.HeaderHeightEm * ms)
        {
            return false;
        }

        double left = double.MaxValue, right = double.MinValue;
        foreach (int id in reg)
        {
            left = Math.Min(left, f[id].X0);
            right = Math.Max(right, f[id].X1);
        }

        if (headRight - headLeft < LayoutParameters.HeaderSpan * (right - left))
        {
            return false;
        }

        double next = double.MaxValue;
        foreach (int id in bands[1])
        {
            next = Math.Min(next, f[id].Top);
        }

        return next - headBottom >= LayoutParameters.HeaderGapEm * ms;
    }

    // The columns are tied by rows when the smaller one holds a few items (at least three, in at least half as many row bands),
    // and each row band that has any of them also has other text: the small column is a value per row (a setting and its value,
    // a field and its neighbour), not a column of its own. The first column is never the small one: a value follows its label, while
    // a narrow column at the start is a sidebar or an index, which is read before the body.
    private static bool AreTiedRows(List<OrderFragment> f, List<int[]> columns, List<int[]> rows)
    {
        int[] small = columns[0];
        foreach (int[] c in columns)
        {
            if (c.Length <= small.Length)
            {
                small = c;
            }
        }

        if (ReferenceEquals(small, columns[0]))
        {
            return false;
        }

        var bandOf = new Dictionary<int, int>();
        for (int r = 0; r < rows.Count; r++)
        {
            foreach (int id in rows[r])
            {
                bandOf[id] = r;
            }
        }

        var withSmall = new HashSet<int>();
        foreach (int id in small)
        {
            withSmall.Add(bandOf[id]);
        }

        if (withSmall.Count < 3 || withSmall.Count * 2 < small.Length)
        {
            return false;
        }

        var smallSet = new HashSet<int>(small);
        foreach (int r in withSmall)
        {
            bool other = false;
            foreach (int id in rows[r])
            {
                other |= !smallSet.Contains(id);
            }

            if (!other)
            {
                return false;
            }
        }

        return true;
    }

    // Bands are joined across a gap that is clearly smaller than the gaps on both sides: the label above its value inside a cell,
    // while cells are separated by the larger gaps. With evenly spaced rows no gap is weak and the bands stay as they are.
    private static List<int[]> MergeWeak(List<int[]> bands, List<double> gaps)
    {
        var rows = new List<int[]>();
        var current = new List<int>(bands[0]);
        for (int i = 0; i < gaps.Count; i++)
        {
            double before = i > 0 ? gaps[i - 1] : double.MaxValue;
            double after = i + 1 < gaps.Count ? gaps[i + 1] : double.MaxValue;
            bool weak = (before != double.MaxValue || after != double.MaxValue) && gaps[i] <= LayoutParameters.WeakGapRatio * Math.Min(before, after);
            if (!weak)
            {
                rows.Add([.. current]);
                current.Clear();
            }

            current.AddRange(bands[i + 1]);
        }

        rows.Add([.. current]);
        return rows.Count >= 2 ? rows : bands;
    }

    // A column cut is doubtful when one group sits wholly between two parts of another group: reading the other group first
    // would read across it. The margin keeps a sidebar, which starts a little below its neighbour's title, out of this.
    private static bool HasEmbedded(List<OrderFragment> f, List<int[]> groups, double margin)
    {
        var top = new double[groups.Count];
        var bottom = new double[groups.Count];
        for (int g = 0; g < groups.Count; g++)
        {
            top[g] = double.MaxValue;
            bottom[g] = double.MinValue;
            foreach (int id in groups[g])
            {
                top[g] = Math.Min(top[g], f[id].Top);
                bottom[g] = Math.Max(bottom[g], f[id].Bot);
            }
        }

        for (int inner = 0; inner < groups.Count; inner++)
        {
            for (int outer = 0; outer < groups.Count; outer++)
            {
                if (outer == inner)
                {
                    continue;
                }

                bool above = false, below = false;
                foreach (int id in groups[outer])
                {
                    above |= f[id].Bot <= top[inner] - margin;
                    below |= f[id].Top >= bottom[inner] + margin;
                }

                if (above && below)
                {
                    return true;
                }
            }
        }

        return false;
    }

    // Columns whose paragraphs happen to end at the same height leave empty bands that cross both of them. Cutting at every such
    // band reads the columns in slices (left slice, right slice, next left slice ...). Consecutive bands that share an empty
    // vertical stripe are one section again, so the section is cut into its columns as a whole. A section that would be a table
    // stays in bands, as it was.
    private static List<int[]> MergeColumnRuns(List<OrderFragment> f, List<int[]> bands, double ms)
    {
        double gutter = LayoutParameters.ColumnGutterEm * ms;
        var result = new List<int[]>();
        int i = 0;
        while (i < bands.Count)
        {
            var occupied = new List<(double From, double To)>();
            int last = i;
            for (int k = i; k < bands.Count; k++)
            {
                AddSpans(occupied, f, bands[k]);
                if (!HasStripe(occupied, gutter))
                {
                    break;
                }

                last = k;
            }

            if (last == i)
            {
                result.Add(bands[i]);
                i++;
                continue;
            }

            var section = new List<int>();
            for (int k = i; k <= last; k++)
            {
                section.AddRange(bands[k]);
            }

            int[] union = [.. section];
            List<int[]>? columns = Partition(f, union, true, gutter);
            if (columns is not null && !IsTable(f, union, columns, ms) && !HasCellColumn(f, columns, ms))
            {
                result.Add(union);
            }
            else
            {
                // Rows of a table: each stays a band. The whole run is passed at once so that a long table is looked at once.
                for (int k = i; k <= last; k++)
                {
                    result.Add(bands[k]);
                }
            }

            i = last + 1;
        }

        return result.Count >= 2 ? result : bands;
    }

    // Adds the horizontal extents of the fragments to the sorted list of occupied intervals, merging those that touch.
    private static void AddSpans(List<(double From, double To)> occupied, List<OrderFragment> f, int[] ids)
    {
        foreach (int id in ids)
        {
            occupied.Add((f[id].X0, f[id].X1));
        }

        occupied.Sort();
        int w = 0;
        for (int r = 0; r < occupied.Count; r++)
        {
            if (w > 0 && occupied[r].From <= occupied[w - 1].To)
            {
                occupied[w - 1] = (occupied[w - 1].From, Math.Max(occupied[w - 1].To, occupied[r].To));
            }
            else
            {
                occupied[w++] = occupied[r];
            }
        }

        occupied.RemoveRange(w, occupied.Count - w);
    }

    private static bool HasStripe(List<(double From, double To)> occupied, double gutter)
    {
        for (int r = 1; r < occupied.Count; r++)
        {
            if (occupied[r].From - occupied[r - 1].To >= gutter)
            {
                return true;
            }
        }

        return false;
    }

    // A sidebar (a list of navigation items down the left edge) is read before the body. The body's rows do not line up with the
    // sidebar's items, which tells it from the label column of a list of settings or fields, whose values sit on the same lines.
    // Cutting it off first keeps the tied-rows rule from reading across it.
    private static List<int[]>? PeelSidebar(List<OrderFragment> f, int[] reg, List<int[]> columns)
    {
        int[] first = columns[0];
        var lines = new HashSet<int>();
        double left = double.MaxValue, right = double.MinValue;
        foreach (int id in reg)
        {
            left = Math.Min(left, f[id].X0);
            right = Math.Max(right, f[id].X1);
        }

        double fRight = double.MinValue;
        foreach (int id in first)
        {
            if (!lines.Add(f[id].Line))
            {
                return null;
            }

            fRight = Math.Max(fRight, f[id].X1);
        }

        if (lines.Count < LayoutParameters.SidebarMinLines || fRight - left > LayoutParameters.SidebarMaxWidth * (right - left))
        {
            return null;
        }

        // A sidebar's items are set at an even pitch.
        var centres = first.Select(id => (f[id].Top + f[id].Bot) / 2).Order().ToArray();
        var pitches = new double[centres.Length - 1];
        for (int i = 0; i < pitches.Length; i++)
        {
            pitches[i] = centres[i + 1] - centres[i];
        }

        double typical = Median([.. pitches]);
        foreach (double p in pitches)
        {
            if (Math.Abs(p - typical) > LayoutParameters.SidebarPitchTolerance * typical)
            {
                return null;
            }
        }

        // A sidebar is followed closely by the body; a label column is followed by its values, which may stand far off.
        double restLeft = double.MaxValue;
        foreach (int[] c in columns.Skip(1))
        {
            foreach (int id in c)
            {
                restLeft = Math.Min(restLeft, f[id].X0);
            }
        }

        if (restLeft - fRight > LayoutParameters.SidebarMaxGutter * (right - left))
        {
            return null;
        }

        var firstSet = new HashSet<int>(first);
        var shared = new HashSet<int>();
        var restLines = new HashSet<int>();
        foreach (int id in reg)
        {
            if (firstSet.Contains(id))
            {
                continue;
            }

            restLines.Add(f[id].Line);
            if (lines.Contains(f[id].Line))
            {
                shared.Add(f[id].Line);
            }
        }

        // Few of the sidebar's lines have a partner, and few of the body's lines have one either: rows that line up on both
        // sides are a label column and its values.
        if (shared.Count > LayoutParameters.SidebarSharedLines * lines.Count || shared.Count > LayoutParameters.SidebarSharedLines * restLines.Count)
        {
            return null;
        }

        return [first, [.. columns.Skip(1).SelectMany(c => c)]];
    }

    // A text column is made of long lines. A column of the cells of a form is made of one or two lines (a label,
    // a value) with wide gaps between them, in some rows only. A run of bands with such a column is not a section of columns.
    private static bool HasCellColumn(List<OrderFragment> f, List<int[]> columns, double ms)
    {
        foreach (int[] c in columns)
        {
            int[] sorted = [.. c];
            Array.Sort(sorted, (a, b) =>
            {
                int k = f[a].Top.CompareTo(f[b].Top);
                return k != 0 ? k : a.CompareTo(b);
            });
            double end = f[sorted[0]].Bot, widest = 0;
            int run = 1, longest = 1;
            var widths = new double[sorted.Length];
            widths[0] = (f[sorted[0]].X1 - f[sorted[0]].X0) / Math.Max(ms, 1e-6);
            for (int i = 1; i < sorted.Length; i++)
            {
                widths[i] = (f[sorted[i]].X1 - f[sorted[i]].X0) / Math.Max(ms, 1e-6);
                double gap = f[sorted[i]].Top - end;
                widest = Math.Max(widest, gap);
                run = gap < LayoutParameters.BlockGapEm * ms ? run + 1 : 1;
                longest = Math.Max(longest, run);
                end = Math.Max(end, f[sorted[i]].Bot);
            }

            if (widest >= LayoutParameters.CellColumnGapEm * ms && longest <= LayoutParameters.CellColumnMaxRun && Median(widths) <= LayoutParameters.CellColumnWidthEm)
            {
                return true;
            }
        }

        return false;
    }

    // Fields in a grid are a label above a value, several side by side: a band of labels, then a band of values, each value under
    // its label and closer to it than to the fields around. Read band by band that is all labels, then all values; the cells are
    // read one at a time instead. Evenly spaced rows (a table) have no such close pairs and are left alone.
    private static List<int[]> MergeFieldCells(List<OrderFragment> f, List<int[]> bands, double ms, HashSet<int[]> standalone)
    {
        int n = bands.Count;
        var top = new double[n];
        var bot = new double[n];
        for (int i = 0; i < n; i++)
        {
            top[i] = double.MaxValue;
            bot[i] = double.MinValue;
            foreach (int id in bands[i])
            {
                top[i] = Math.Min(top[i], f[id].Top);
                bot[i] = Math.Max(bot[i], f[id].Bot);
            }
        }

        var result = new List<int[]>();
        bool any = false;
        for (int i = 0; i < n; i++)
        {
            if (i + 1 < n && FieldCells(f, bands[i], bands[i + 1], ms) is { } cells)
            {
                // The gap under the pair is the reference: a heading above a label may sit closer than the fields do.
                double gap = top[i + 1] - bot[i];
                double reference = i + 2 < n ? top[i + 2] - bot[i + 1] : i > 0 ? top[i] - bot[i - 1] : double.MaxValue;
                if (reference != double.MaxValue && gap <= LayoutParameters.WeakGapRatio * reference)
                {
                    foreach (int[] cell in cells)
                    {
                        standalone.Add(cell);
                    }

                    result.AddRange(cells);
                    i++;
                    any = true;
                    continue;
                }
            }

            result.Add(bands[i]);
        }

        return any ? result : bands;
    }

    // The columns whose fragments mostly end in a colon ("Invoice no:", "Date:").
    private static bool[] LabelColumns(List<OrderFragment> f, List<int[]> columns)
    {
        var result = new bool[columns.Count];
        for (int c = 0; c < columns.Count; c++)
        {
            int labels = 0;
            foreach (int id in columns[c])
            {
                labels += f[id].Label ? 1 : 0;
            }

            result[c] = labels >= LayoutParameters.LabelColumnShare * columns[c].Length;
        }

        return result;
    }

    // Several label columns (a label and its value, a label and its value, side by side) are blocks, each read through before the
    // next: a block is a label column with the columns up to the next label column. Null with fewer than two label columns.
    private static List<int[]>? LabelBlocks(List<OrderFragment> f, List<int[]> columns)
    {
        bool[] label = LabelColumns(f, columns);
        if (label.Count(l => l) < 2 || !label[0])
        {
            return null;
        }

        var blocks = new List<List<int>>();
        for (int c = 0; c < columns.Count; c++)
        {
            if (label[c])
            {
                blocks.Add([]);
            }

            blocks[^1].AddRange(columns[c]);
        }

        return [.. blocks.Select(b => b.ToArray())];
    }

    // The last two lines of a region are a row of fields (labels, then the value under each), possibly under headings that are
    // single fragments (a section title). The headings are read first, then the row cell by cell; the cells are standalone.
    private static List<(int[] Ids, bool Standalone)>? TrailingFields(List<OrderFragment> f, int[] reg)
    {
        var byLine = new Dictionary<int, List<int>>();
        foreach (int id in reg)
        {
            if (!byLine.TryGetValue(f[id].Line, out List<int>? list))
            {
                byLine[f[id].Line] = list = [];
            }

            list.Add(id);
        }

        if (byLine.Count < 2 || byLine.Count > LayoutParameters.FieldHeadingLines + 2)
        {
            return null;
        }

        List<List<int>> lines = [.. byLine.Values.OrderBy(l => f[l[0]].Base).ThenBy(l => l[0])];
        for (int i = 0; i < lines.Count - 2; i++)
        {
            if (lines[i].Count != 1)
            {
                return null;
            }
        }

        double ms = Median([.. reg.Select(i => f[i].Size)]);
        List<int[]>? cells = FieldCells(f, [.. lines[^2]], [.. lines[^1]], ms);
        if (cells is null)
        {
            return null;
        }

        var result = new List<(int[], bool)>();
        for (int i = 0; i < lines.Count - 2; i++)
        {
            result.Add(([lines[i][0]], false));
        }

        foreach (int[] c in cells)
        {
            result.Add((c, true));
        }

        return result;
    }

    // The cells of a label band and the value band under it: the same number of fragments each, in equal numbers, each value starting
    // under its label. Null when they are not a row of fields.
    private static List<int[]>? FieldCells(List<OrderFragment> f, int[] labels, int[] values, double ms)
    {
        if (labels.Length < 1 || labels.Length != values.Length || DistinctLines(f, labels) > 1 || DistinctLines(f, values) > 1)
        {
            return null;
        }

        // "Total: 45.00" lines are rows of a list, not a label above a value.
        if (labels.Count(i => f[i].Label) * 2 >= labels.Length)
        {
            return null;
        }

        int[] a = [.. labels.OrderBy(i => f[i].X0).ThenBy(i => i)];
        int[] b = [.. values.OrderBy(i => f[i].X0).ThenBy(i => i)];
        var cells = new List<int[]>(a.Length);
        for (int i = 0; i < a.Length; i++)
        {
            if (Math.Abs(f[a[i]].X0 - f[b[i]].X0) > LayoutParameters.FieldAlignEm * ms)
            {
                return null;
            }

            cells.Add([a[i], b[i]]);
        }

        return cells;
    }

    // ---- what is not a table ------------------------------------------------------------------------------------------------

    // The columns of a table are filled in most of its rows. Short blocks side by side that share a few rows (cards, a list with a
    // time at the end of each row) leave most columns empty in most rows.
    private static bool SparseColumns(List<OrderFragment> f, List<int[]> columns)
    {
        var rows = new HashSet<int>();
        foreach (int[] c in columns)
        {
            foreach (int id in c)
            {
                rows.Add(f[id].Line);
            }
        }

        double sum = 0;
        foreach (int[] c in columns)
        {
            var own = new HashSet<int>();
            foreach (int id in c)
            {
                own.Add(f[id].Line);
            }

            sum += own.Count / (double)rows.Count;
        }

        return sum / columns.Count < LayoutParameters.TableMinColumnFill;
    }

    // The rows of a table are set in one size. A row of figures much taller than the other rows (a card's large number between a
    // small label and a small note) means the blocks are cards.
    private static bool UnevenRows(List<OrderFragment> f, int[] reg)
    {
        var byLine = new Dictionary<int, List<double>>();
        foreach (int id in reg)
        {
            if (!byLine.TryGetValue(f[id].Line, out List<double>? list))
            {
                byLine[f[id].Line] = list = [];
            }

            list.Add(f[id].Bot - f[id].Top);
        }

        var heights = new List<double>();
        foreach (List<double> v in byLine.Values)
        {
            if (v.Count > 1)
            {
                heights.Add(Median([.. v]));
            }
        }

        if (heights.Count < 3)
        {
            return false;
        }

        double max = heights.Max();
        if (max >= LayoutParameters.TableMaxRowSizeRatio * Median([.. heights]))
        {
            return true;
        }

        // The text size of a line of figures against a line of small print: card text, not table rows.
        var sizes = new Dictionary<int, double>();
        foreach (int id in reg)
        {
            sizes[f[id].Line] = Math.Max(sizes.GetValueOrDefault(f[id].Line), f[id].Size);
        }

        return sizes.Values.Max() >= LayoutParameters.TableMaxRowSizeRatio * sizes.Values.Min();
    }

    // The rows of a table are evenly spaced. When the distance between consecutive lines alternates short, long, short (a small
    // label above its value, then the next pair), the lines are fields, not table rows.
    private static bool AlternatingPitch(List<OrderFragment> f, int[] reg)
    {
        var sums = new SortedDictionary<int, (double Sum, int N)>();
        foreach (int id in reg)
        {
            (double sum, int n) = sums.GetValueOrDefault(f[id].Line);
            sums[f[id].Line] = (sum + ((f[id].Top + f[id].Bot) / 2), n + 1);
        }

        var centres = new List<double>(sums.Count);
        foreach ((double sum, int n) in sums.Values)
        {
            centres.Add(sum / n);
        }

        centres.Sort();
        if (centres.Count < 4)
        {
            return false;
        }

        var pitch = new double[centres.Count - 1];
        var even = new List<double>();
        var odd = new List<double>();
        for (int i = 0; i < pitch.Length; i++)
        {
            pitch[i] = centres[i + 1] - centres[i];
            (i % 2 == 0 ? even : odd).Add(pitch[i]);
        }

        double medianEven = Median([.. even]), medianOdd = Median([.. odd]);
        double low = Math.Min(medianEven, medianOdd), high = Math.Max(medianEven, medianOdd);
        if (low <= 0 || high < LayoutParameters.FieldPitchRatio * low)
        {
            return false;
        }

        double middle = (low + high) / 2;
        int fits = 0;
        for (int i = 0; i < pitch.Length; i++)
        {
            bool isShort = pitch[i] < middle;
            fits += isShort == ((i % 2 == 0) == (medianEven < medianOdd)) ? 1 : 0;
        }

        return fits >= 0.85 * pitch.Length;
    }
}
