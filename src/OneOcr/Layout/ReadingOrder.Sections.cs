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
    private static List<int[]> ChooseCut(List<OrderFragment> f, int[] reg, List<int[]> columns, double ms)
    {
        var bandGaps = new List<double>();
        List<int[]>? bands = Partition(f, reg, false, LayoutParameters.BlockGapEm * ms, bandGaps);
        if (bands is null)
        {
            return columns;
        }

        if (IsHeaderRow(f, reg, bands, ms))
        {
            return [bands[0], [.. bands.Skip(1).SelectMany(b => b)]];
        }

        var rowGaps = new List<double>();
        List<int[]>? coarse = Partition(f, reg, false, LayoutParameters.RowGapEm * ms, rowGaps);
        List<int[]>? rows = coarse is null ? null : MergeWeak(coarse, rowGaps);
        if (rows is not null && AreTiedRows(f, columns, rows))
        {
            return rows;
        }

        return HasEmbedded(f, columns, LayoutParameters.EmbedMarginEm * ms) ? rows ?? bands : columns;
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
            if (columns is not null && !IsTable(f, union, columns, ms))
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
        return max >= LayoutParameters.TableMaxRowSizeRatio * Median([.. heights]);
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
