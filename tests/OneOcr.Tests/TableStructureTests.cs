using OneOcr.Detection;
using OneOcr.Layout;

namespace OneOcr.Tests;

/// <summary>Table structure: ruled grids found in pixels, cells with spans, words put into cells, and tables without rules.</summary>
public class TableStructureTests
{
    // A white page with dark rules.
    private sealed class Canvas(int width, int height)
    {
        public readonly byte[] Pixels = Enumerable.Repeat((byte)255, width * height).ToArray();

        public void Fill(int x0, int y0, int x1, int y1, byte value = 40)
        {
            for (int y = y0; y < y1; y++)
            {
                for (int x = x0; x < x1; x++)
                {
                    Pixels[(y * width) + x] = value;
                }
            }
        }

        public void H(int y, int x0, int x1, int t = 2) => Fill(x0, y, x1, y + t);

        public void V(int x, int y0, int y1, int t = 2) => Fill(x, y0, x + t, y1);

        public List<RuleGrid> Find()
        {
            (int threshold, bool bright) = LineFinder.Binarize(Pixels);
            return TableRules.Find(Pixels, width, height, threshold, bright);
        }
    }

    // A frame of 3 rows by 4 columns: columns 40, 140, 240, 340, 440 and rows 40, 100, 160, 220. `skipV` and `skipH` list drawn edges to leave out.
    private static Canvas Table(HashSet<(int Column, int Row)>? skipV = null, HashSet<(int Row, int Column)>? skipH = null)
    {
        var c = new Canvas(520, 300);
        int[] xs = [40, 140, 240, 340, 440], ys = [40, 100, 160, 220];
        for (int b = 0; b < xs.Length; b++)
        {
            for (int r = 0; r < 3; r++)
            {
                if (skipV is null || !skipV.Contains((b, r)))
                {
                    c.V(xs[b], ys[r], ys[r + 1] + 2);
                }
            }
        }

        for (int b = 0; b < ys.Length; b++)
        {
            for (int col = 0; col < 4; col++)
            {
                if (skipH is null || !skipH.Contains((b, col)))
                {
                    c.H(ys[b], xs[col], xs[col + 1] + 2);
                }
            }
        }

        return c;
    }

    [Fact]
    public void FindsEveryCellOfAPlainGrid()
    {
        List<RuleGrid> grids = Table().Find();
        RuleGrid g = Assert.Single(grids);
        Assert.Equal(3, g.Rows);
        Assert.Equal(4, g.Columns);
        Assert.Equal(12, g.Cells.Count);
        Assert.All(g.Cells, c => Assert.True(c.RowSpan == 1 && c.ColumnSpan == 1));
        // Centres of the rules, to the pixel.
        Assert.Equal([41, 141, 241, 341, 441], g.Xs);
        Assert.Equal([41, 101, 161, 221], g.Ys);
    }

    [Fact]
    public void AMissingRuleMakesASpan()
    {
        // No vertical rule between columns 1 and 2 in row 0, and no horizontal rule between rows 1 and 2 under column 0.
        List<RuleGrid> grids = Table(skipV: [(2, 0)], skipH: [(2, 0)]).Find();
        RuleGrid g = Assert.Single(grids);
        Assert.Equal(10, g.Cells.Count);
        Assert.Contains(new GridCell(0, 1, 1, 2), g.Cells);
        Assert.Contains(new GridCell(1, 0, 2, 1), g.Cells);
        Assert.Equal(g.CellAt(150, 60), g.CellAt(250, 60));
        Assert.NotEqual(g.CellAt(150, 60), g.CellAt(350, 60));
        Assert.Equal(g.CellAt(60, 120), g.CellAt(60, 180));
    }

    [Fact]
    public void ARuleBrokenByATextGapStillCountsAsDrawn()
    {
        // A vertical rule that stops short of the edge by two pixels at both ends (a gap at a crossing) is still the cell's border.
        var c = new Canvas(520, 300);
        int[] xs = [40, 140, 240], ys = [40, 100, 160];
        foreach (int y in ys)
        {
            c.H(y, 40, 242);
        }

        foreach (int x in xs)
        {
            c.V(x, 42, 158);
        }

        RuleGrid g = Assert.Single(c.Find());
        Assert.Equal(4, g.Cells.Count);
    }

    [Fact]
    public void LargeLettersAndBoxesAreNotTables()
    {
        // A lone box, a filled bar, and the strokes of a big "H" and "E" are not grids.
        var c = new Canvas(600, 300);
        c.H(20, 20, 220);
        c.H(120, 20, 220);
        c.V(20, 20, 122);
        c.V(218, 20, 122);
        c.Fill(260, 20, 560, 80);
        c.V(300, 150, 250, 14);
        c.V(380, 150, 250, 14);
        c.H(195, 300, 394, 14);
        Assert.Empty(c.Find());
    }

    [Fact]
    public void ErasingPaintsTheRulesOutAndKeepsText()
    {
        Canvas c = Table();
        c.Fill(60, 60, 90, 70);
        List<RuleGrid> grids = c.Find();
        (int threshold, bool bright) = LineFinder.Binarize(c.Pixels);
        TableRules.Erase(c.Pixels, 520, 300, bright, grids);
        // Row 100 (a horizontal rule) and column 140 (a vertical rule) are paper now; the text block is still there.
        Assert.All(Enumerable.Range(42, 396), x => Assert.True(c.Pixels[(101 * 520) + x] > threshold));
        Assert.All(Enumerable.Range(42, 176), y => Assert.True(c.Pixels[(y * 520) + 141] > threshold));
        Assert.Equal(40, c.Pixels[(65 * 520) + 70]);
        Assert.Empty(c.Find());
    }

    [Fact]
    public void StripsAreCutAtDrawnRulesOnly()
    {
        Canvas c = Table(skipV: [(2, 0)]);
        List<RuleGrid> grids = c.Find();
        (int threshold, bool bright) = LineFinder.Binarize(c.Pixels);
        TableRules.Erase(c.Pixels, 520, 300, bright, grids);
        // Text across the whole of row 0 and of row 1 (one strip each, as a finder that sees no gutter would cut).
        c.Fill(60, 65, 90, 75);
        c.Fill(160, 65, 190, 75);
        c.Fill(260, 65, 290, 75);
        c.Fill(60, 125, 90, 135);
        c.Fill(160, 125, 190, 135);
        c.Fill(260, 125, 290, 135);
        c.Fill(360, 125, 390, 135);
        var strips = new List<StripLine>
        {
            new(new LineBox(60, 65, 230, 10), 0),
            new(new LineBox(60, 125, 330, 10), 0),
        };
        List<StripLine> cut = TableStructure.SplitStrips(strips, grids, c.Pixels, 520, 300, threshold, bright);
        // Row 0: columns 0 | 1+2 (no rule between them); row 1: four cells.
        Assert.Equal(2 + 4, cut.Count);
        Assert.Equal([60, 160], cut.Take(2).Select(s => s.Box.X));
        Assert.Equal(60, cut[0].Box.X);
        Assert.Equal(30, cut[0].Box.Width);
        Assert.Equal(130, cut[1].Box.Width);
        // Every piece has the ink's own box and its own region number per column.
        Assert.Equal(4, cut.Skip(2).Select(s => s.Leaf).Distinct().Count());
    }

    [Fact]
    public void PathologicalPagesNeitherCrashNorSlowDown()
    {
        var rng = new Random(7);
        var clock = System.Diagnostics.Stopwatch.StartNew();
        // Noise, stripes in both directions, a checkerboard, a page full of short rules, hatching and an empty page.
        for (int kind = 0; kind < 6; kind++)
        {
            var c = new Canvas(1200, 900);
            for (int y = 0; y < 900; y++)
            {
                for (int x = 0; x < 1200; x++)
                {
                    bool ink = kind switch
                    {
                        0 => rng.Next(2) == 0,
                        1 => y % 4 < 2,
                        2 => x % 4 < 2,
                        3 => ((x / 8) + (y / 8)) % 2 == 0,
                        4 => (x + y) % 6 < 2,
                        _ => false,
                    };
                    if (ink)
                    {
                        c.Pixels[(y * 1200) + x] = 0;
                    }
                }
            }

            if (kind == 5)
            {
                for (int i = 0; i < 3000; i++)
                {
                    c.H(rng.Next(0, 890), rng.Next(0, 1100), 1200 - rng.Next(0, 50));
                    c.V(rng.Next(0, 1190), rng.Next(0, 800), 900 - rng.Next(0, 50));
                }
            }

            _ = c.Find();
        }

        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(20), clock.Elapsed.ToString());
    }

    // ---- fragments in cells -------------------------------------------------------------------------------------------------

    private static OrderFragment Frag(int strip, double x0, double x1, double top, double bot, int line = 0) =>
        new() { Strip = strip, WordStart = 0, WordCount = 1, Line = line, X0 = x0, X1 = x1, Top = top, Bot = bot, Base = (top + bot) / 2, Size = bot - top };

    private static OrderBlock Plan(params OrderFragment[] fragments) =>
        new() { IsTable = false, Columns = 0, Lines = [new OrderLine { Fragments = [.. fragments] }] };

    [Fact]
    public void WordsGoIntoTheCellsOfTheGridRowByRow()
    {
        Canvas c = Table(skipH: [(2, 1)]);
        RuleGrid g = Assert.Single(c.Find());
        // The reading order saw the columns of a table as separate blocks, column by column.
        OrderFragment a0 = Frag(0, 50, 90, 55, 70), a1 = Frag(1, 50, 90, 115, 130), a2 = Frag(2, 50, 90, 175, 190);
        OrderFragment b0 = Frag(3, 150, 190, 55, 70), b1 = Frag(4, 150, 190, 115, 130), b1b = Frag(5, 150, 190, 135, 150);
        OrderFragment d2 = Frag(6, 350, 390, 175, 190);
        var plan = new List<OrderBlock> { Plan(a0), Plan(a1), Plan(a2), Plan(b0), Plan(b1), Plan(b1b), Plan(d2) };
        List<OrderBlock> arranged = TableStructure.Apply(plan, [g]);
        OrderBlock table = Assert.Single(arranged);
        Assert.True(table.IsTable);
        Assert.Equal(3, table.Rows);
        Assert.Equal(4, table.Columns);
        // Rows hold their cells left to right; the two lines of the cell under column 1 are one cell, top to bottom.
        Assert.Equal(3, table.Lines.Count);
        Assert.Equal([0, 3], table.Lines[0].Fragments.Select(f => f.Strip));
        OrderLine row1 = table.Lines[1];
        Assert.Equal([1, 4, 5], row1.Fragments.Select(f => f.Strip));
        Assert.Equal(row1.Fragments[1].Cell, row1.Fragments[2].Cell);
        Assert.NotEqual(row1.Fragments[0].Cell, row1.Fragments[1].Cell);
        // The cell under row 1 and 2 in column 1 spans two rows when its rule is missing; its words are listed once, in row 1.
        OrderCell tall = table.Cells![row1.Fragments[1].Cell];
        Assert.Equal((1, 1, 2, 1), (tall.Row, tall.Column, tall.RowSpan, tall.ColumnSpan));
        // Empty cells are kept: 12 cells of the grid less the one the span swallowed.
        Assert.Equal(11, table.Cells.Count);
    }

    [Fact]
    public void ABlockKeepsTheTextOutsideTheGrid()
    {
        Canvas c = Table();
        RuleGrid g = Assert.Single(c.Find());
        OrderFragment title = Frag(0, 40, 300, 10, 25), cell = Frag(1, 50, 90, 55, 70, line: 1);
        var plan = new List<OrderBlock>
        {
            new() { IsTable = false, Columns = 0, Lines = [new OrderLine { Fragments = [title] }] },
            Plan(cell),
        };
        List<OrderBlock> arranged = TableStructure.Apply(plan, [g]);
        Assert.Equal(2, arranged.Count);
        Assert.False(arranged[0].IsTable);
        Assert.True(arranged[1].IsTable);
    }

    // ---- tables without rules -----------------------------------------------------------------------------------------------

    private static OrderBlock Rows(params double[][][] rows)
    {
        var lines = new List<OrderLine>();
        int strip = 0, y = 0;
        foreach (double[][] row in rows)
        {
            var frags = row.Select(e => Frag(strip++, e[0], e[1], y, y + 14, lines.Count)).ToList();
            lines.Add(new OrderLine { Fragments = frags });
            y += 30;
        }

        return new OrderBlock { IsTable = true, Columns = 3, Lines = lines };
    }

    [Fact]
    public void ColumnsComeFromAlignedGapsAndSpansFromFragmentsThatCrossThem()
    {
        // Three columns near x 0, 200 and 400. Row 0: a header over the last two columns. Row 2 has nothing in the middle column.
        OrderBlock block = Rows(
            [[0, 60], [200, 520]],
            [[0, 50], [200, 260], [400, 470]],
            [[0, 70], [400, 440]],
            [[0, 45], [205, 250], [405, 480]],
            [[0, 55], [200, 240], [402, 460]]);
        List<OrderBlock> arranged = TableStructure.Infer([block], [], null);
        OrderBlock t = Assert.Single(arranged);
        Assert.Equal(3, t.Columns);
        Assert.Equal(5, t.Rows);
        OrderCell[] cells = [.. t.Cells!];
        Assert.Contains(cells, c => c is { Row: 0, Column: 1, ColumnSpan: 2 });
        Assert.DoesNotContain(cells, c => c is { Row: 0, Column: 2 });
        // The empty cell of row 2 is there, with no fragment in it.
        int empty = Array.FindIndex(cells, c => c is { Row: 2, Column: 1 });
        Assert.True(empty >= 0);
        Assert.DoesNotContain(t.Lines[2].Fragments, f => f.Cell == empty);
        // 1 + 2 + ... cells: row 0 has two (column 0 and the span), rows 1, 3, 4 three, row 2 three (one empty).
        Assert.Equal(2 + 3 + 3 + 3 + 3, cells.Length);
        // Row-major order, and the fragments point at their own cell.
        Assert.Equal(cells.OrderBy(c => c.Row).ThenBy(c => c.Column), cells);
        Assert.Equal((1, 0), (cells[t.Lines[1].Fragments[0].Cell].Row, cells[t.Lines[1].Fragments[0].Cell].Column));
    }

    [Fact]
    public void WithoutTwoColumnsABlockIsLeftAlone()
    {
        OrderBlock block = Rows([[0, 100]], [[0, 90]], [[0, 80]]);
        Assert.Same(block, Assert.Single(TableStructure.Infer([block], [], null)));
    }

    [Fact]
    public void ADetectedTableRegionGathersBlocksAndReadsThemRowByRow()
    {
        // Two columns found as two blocks (column by column) inside a region the detector calls a table.
        var left = new OrderBlock
        {
            IsTable = false,
            Columns = 0,
            Lines = [new OrderLine { Fragments = [Frag(0, 0, 60, 0, 14, 0)] }, new OrderLine { Fragments = [Frag(1, 0, 60, 30, 44, 1)] }, new OrderLine { Fragments = [Frag(2, 0, 60, 60, 74, 2)] }],
        };
        var right = new OrderBlock
        {
            IsTable = false,
            Columns = 0,
            Lines = [new OrderLine { Fragments = [Frag(3, 300, 380, 0, 14, 0)] }, new OrderLine { Fragments = [Frag(4, 300, 380, 30, 44, 1)] }, new OrderLine { Fragments = [Frag(5, 300, 380, 60, 74, 2)] }],
        };
        var region = new DetectedRegion(3, "table", -10, -10, 400, 100, 0.9f);
        List<OrderBlock> arranged = TableStructure.Infer([left, right], [region], null);
        OrderBlock t = Assert.Single(arranged);
        Assert.Equal(3, t.Rows);
        Assert.Equal(2, t.Columns);
        Assert.Equal([0, 3, 1, 4, 2, 5], t.Lines.SelectMany(l => l.Fragments).Select(f => f.Strip));
    }

    private sealed class FixedSeparators(params double[] at) : ITableSeparators
    {
        public IReadOnlyList<double>? ColumnBoundaries(OcrBox region) => at;
    }

    [Fact]
    public void LearnedSeparatorsOverrideTheGaps()
    {
        // One column of text by the gaps, so no table; a learned separator at 50 divides it, and text that crosses it spans both columns.
        OrderBlock block = Rows([[0, 100]], [[0, 90]], [[0, 80]]);
        Assert.Same(block, Assert.Single(TableStructure.Infer([block], [], null)));
        OrderBlock t = Assert.Single(TableStructure.Infer([block], [], new FixedSeparators(50)));
        Assert.Equal(2, t.Columns);
        Assert.All(t.Cells!, c => Assert.Equal(2, c.ColumnSpan));
    }
}
