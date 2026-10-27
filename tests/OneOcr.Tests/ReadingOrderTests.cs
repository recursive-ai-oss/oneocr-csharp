using OneOcr.Layout;

namespace OneOcr.Tests;

/// <summary>Reading order of screens and multi-column pages, on hand-built strips (one strip per line, ink height 10).</summary>
public class ReadingOrderTests
{
    private static int leaf;

    // A line of words starting at x; every strip is a leaf of its own, as for lines found one by one on a screen.
    private static OrderStrip Line(double x, double y, string text, double height = 10, int? sharedLeaf = null)
    {
        var words = new List<OcrWord>();
        double at = x;
        foreach (string w in text.Split(' '))
        {
            double width = 6 * w.Length;
            words.Add(new OcrWord(w, new OcrBox(at, y, at + width, y + height), 0.99f, 0.99f, OcrSource.Ocr, []));
            at += width + 8;
        }

        return new OrderStrip
        {
            Leaf = sharedLeaf ?? Interlocked.Increment(ref leaf),
            Box = new OcrBox(x, y, at - 8, y + height),
            Size = height,
            Words = words,
        };
    }

    private static string Read(params OrderStrip[] strips) => Read(out _, strips);

    private static string Read(out List<OrderBlock> blocks, params OrderStrip[] strips)
    {
        blocks = ReadingOrder.Arrange(strips, CancellationToken.None);
        var parts = new List<string>();
        foreach (OrderBlock b in blocks)
        {
            foreach (OrderLine l in b.Lines)
            {
                foreach (OrderFragment f in l.Fragments)
                {
                    for (int k = 0; k < f.WordCount; k++)
                    {
                        parts.Add(strips[f.Strip].Words[f.WordStart + k].Text);
                    }
                }
            }
        }

        return string.Join(' ', parts);
    }

    private static OrderStrip[] MenuBar() =>
    [
        Line(20, 10, "Logo"), Line(120, 10, "Home"), Line(200, 10, "Orders"), Line(290, 10, "Help"), Line(700, 10, "Jo Park"),
    ];

    [Fact]
    public void MenuBarIsReadWholeBeforeTheBody()
    {
        // The user name at the far right of the bar lines up with the right-hand column of the body; it still belongs to the bar.
        OrderStrip[] bar = MenuBar();
        OrderStrip[] body =
        [
            Line(30, 80, "Settings", 14),
            Line(30, 130, "Alpha one with a longer line"), Line(30, 150, "beta two with a longer line"), Line(30, 170, "epsilon five with a longer line"),
            Line(500, 130, "Gamma three with a longer line"), Line(500, 150, "delta four with a longer line"), Line(500, 170, "zeta six with a longer line"),
        ];

        Assert.Equal(
            "Logo Home Orders Help Jo Park Settings Alpha one with a longer line beta two with a longer line epsilon five with a longer line " +
            "Gamma three with a longer line delta four with a longer line zeta six with a longer line",
            Read([.. bar, .. body]));
    }

    [Fact]
    public void CardsAreReadCardByCard()
    {
        // Three cards side by side under a title: label, a large number, a note; a list below ends each row with a time.
        var strips = new List<OrderStrip>(MenuBar()) { Line(30, 80, "Overview", 14) };
        for (int c = 0; c < 3; c++)
        {
            double x = 30 + (c * 300);
            strips.Add(Line(x, 130, $"Label{c}", 8));
            strips.Add(Line(x, 150, $"Number{c}", 24));
            strips.Add(Line(x, 190, $"note{c}", 8));
        }

        strips.Add(Line(30, 260, "Recent activity", 14));
        for (int r = 0; r < 3; r++)
        {
            strips.Add(Line(30, 310 + (r * 40), $"entry{r} was approved"));
            strips.Add(Line(800, 310 + (r * 40), $"1{r}:30"));
        }

        string expected = "Logo Home Orders Help Jo Park Overview " +
                          "Label0 Number0 note0 Label1 Number1 note1 Label2 Number2 note2 Recent activity " +
                          "entry0 was approved 10:30 entry1 was approved 11:30 entry2 was approved 12:30";
        Assert.Equal(expected, Read([.. strips]));
    }

    [Fact]
    public void ListRowsWithAValueAtTheEndAreReadRowByRow()
    {
        // A setting has a name and a description on two lines and its value, centred, at the right.
        var strips = new List<OrderStrip> { Line(30, 20, "Preferences", 14) };
        string expected = "Preferences";
        for (int r = 0; r < 4; r++)
        {
            double y = 80 + (r * 70);
            strips.Add(Line(40, y, $"Name{r}"));
            strips.Add(Line(40, y + 18, $"description of setting {r}"));
            strips.Add(Line(800, y + 9, $"value{r}"));
            expected += $" Name{r} description of setting {r} value{r}";
        }

        Assert.Equal(expected, Read([.. strips]));
    }

    [Fact]
    public void FieldsAreReadCellByCell()
    {
        // Two columns of fields, each a small label above its value; the cells of a row are read left to right.
        var strips = new List<OrderStrip> { Line(30, 20, "Billing", 14) };
        string expected = "Billing";
        for (int r = 0; r < 3; r++)
        {
            double y = 80 + (r * 70);
            for (int c = 0; c < 2; c++)
            {
                double x = 30 + (c * 500);
                strips.Add(Line(x, y, $"Label{r}{c}", 8));
                strips.Add(Line(x, y + 22, $"value {r}{c}"));
                expected += $" Label{r}{c} value {r}{c}";
            }
        }

        Assert.Equal(expected, Read([.. strips]));
    }

    [Fact]
    public void ColumnsSharingParagraphGapsAreNotReadInSlices()
    {
        // Two text columns whose paragraphs happen to end at the same heights, under a full-width title.
        var strips = new List<OrderStrip> { Line(30, 20, "A very long full width title that spans both of the columns", 14) };
        strips[0] = Line(30, 20, "A very long full width title that spans both of the columns below it and more", 14);
        string left = "", right = "";
        for (int p = 0; p < 3; p++)
        {
            for (int l = 0; l < 3; l++)
            {
                double y = 70 + (p * 60) + (l * 14);
                strips.Add(Line(30, y, $"left{p}{l} text of the left column"));
                strips.Add(Line(330, y, $"right{p}{l} text of the right column"));
                left += $" left{p}{l} text of the left column";
                right += $" right{p}{l} text of the right column";
            }
        }

        string title = "A very long full width title that spans both of the columns below it and more";
        Assert.Equal(title + left + right, Read([.. strips]));
    }

    [Fact]
    public void SidebarIsReadBeforeTheBodyAndTheMenuBarBeforeBoth()
    {
        var strips = new List<OrderStrip>(MenuBar());
        string expected = "Logo Home Orders Help Jo Park";
        for (int i = 0; i < 4; i++)
        {
            strips.Add(Line(20, 80 + (i * 34), $"Side{i}"));
            expected += $" Side{i}";
        }

        strips.Add(Line(220, 80, "Heading", 14));
        expected += " Heading";
        for (int p = 0; p < 3; p++)
        {
            for (int l = 0; l < 2; l++)
            {
                strips.Add(Line(220, 120 + (p * 55) + (l * 20), $"body{p}{l} some words of the body"));
                expected += $" body{p}{l} some words of the body";
            }
        }

        Assert.Equal(expected, Read([.. strips]));
    }

    [Fact]
    public void TableStaysATableAndIsReadRowByRow()
    {
        var strips = new List<OrderStrip> { Line(30, 20, "Invoices", 14) };
        string expected = "Invoices";
        for (int r = 0; r < 6; r++)
        {
            double y = 70 + (r * 26);
            for (int c = 0; c < 4; c++)
            {
                strips.Add(Line(30 + (c * 150), y, r == 0 ? $"Head{c}" : $"cell{r}{c}", 10, sharedLeaf: 1000 + c));
                expected += r == 0 ? $" Head{c}" : $" cell{r}{c}";
            }
        }

        Assert.Equal(expected, Read(out List<OrderBlock> blocks, [.. strips]));
        Assert.Contains(blocks, b => b.IsTable && b.Lines.Count >= 6);
    }

    private static (List<OrderStrip> Strips, string Expected) LabelValueRows(double labelX, double valueX, double y0, int rows, string[] labels, string[] values, double pitch = 22)
    {
        var strips = new List<OrderStrip>();
        string expected = "";
        for (int r = 0; r < rows; r++)
        {
            strips.Add(Line(labelX, y0 + (r * pitch), labels[r]));
            strips.Add(Line(valueX, y0 + (r * pitch), values[r]));
            expected += $" {labels[r]} {values[r]}";
        }

        return (strips, expected);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    public void LabelAndValueOnOneLineAreReadAcross(int rows)
    {
        // "Invoice no: 123" lines: a label column and a value column that share every baseline, under a title and a paragraph.
        string[] labels = ["Invoice no:", "Date:", "Total:", "Currency:", "Terms:"];
        string[] values = ["123", "2026-10-01", "45.00", "EUR", "net 30"];
        var strips = new List<OrderStrip> { Line(30, 20, "Invoice", 14), Line(30, 50, "Thank you for your order with us") };
        (List<OrderStrip> body, string expected) = LabelValueRows(30, 150, 100, rows, labels, values);
        strips.AddRange(body);
        Assert.Equal("Invoice Thank you for your order with us" + expected, Read([.. strips]));
    }

    [Fact]
    public void LabelAndValueRowsAloneOnThePageAreReadAcross()
    {
        (List<OrderStrip> body, string expected) = LabelValueRows(30, 150, 40, 3, ["Invoice no:", "Date:", "Total:"], ["123", "2026-10-01", "45.00"]);
        Assert.Equal(expected.TrimStart(), Read([.. body]));
    }

    [Fact]
    public void LabelAndValueRowsWithUnevenSpacingAreReadAcross()
    {
        // Not a table: the rows are grouped in twos by their spacing, as the lines of an address block and a payment block are.
        var strips = new List<OrderStrip>();
        string expected = "";
        string[] labels = ["Name:", "Street:", "Bank:", "IBAN:"];
        string[] values = ["Acme Ltd", "1 High Road", "First Bank", "DE00 1234"];
        double[] ys = [40, 62, 120, 142];
        for (int r = 0; r < 4; r++)
        {
            strips.Add(Line(30, ys[r], labels[r]));
            strips.Add(Line(130, ys[r], values[r]));
            expected += $" {labels[r]} {values[r]}";
        }

        Assert.Equal(expected.TrimStart(), Read([.. strips]));
    }

    [Fact]
    public void TwoLabelValueBlocksSideBySideAreReadBlockByBlock()
    {
        // Seller on the left, buyer on the right, rows sharing baselines: each block is read down (every row across) before the
        // next block, because the gap between the blocks is wider than the gap between a label and its value.
        var strips = new List<OrderStrip>();
        string left = "", right = "";
        string[] labels = ["Name:", "Street:", "City:", "VAT:"];
        for (int r = 0; r < 4; r++)
        {
            double y = 40 + (r * 22);
            strips.Add(Line(30, y, labels[r]));
            strips.Add(Line(110, y, $"seller{r}"));
            strips.Add(Line(420, y, labels[r]));
            strips.Add(Line(500, y, $"buyer{r}"));
            left += $" {labels[r]} seller{r}";
            right += $" {labels[r]} buyer{r}";
        }

        Assert.Equal((left + right).TrimStart(), Read([.. strips]));
    }

    [Fact]
    public void SettingsListIsNotASidebar()
    {
        // A column of names with a value far to the right of each is one list read row by row, not a sidebar.
        var strips = new List<OrderStrip>();
        string expected = "";
        for (int r = 0; r < 5; r++)
        {
            double y = 40 + (r * 30);
            strips.Add(Line(30, y, $"Setting{r}"));
            strips.Add(Line(700, y, $"On{r}"));
            expected += $" Setting{r} On{r}";
        }

        Assert.Equal(expected.TrimStart(), Read([.. strips]));
    }

    [Fact]
    public void FieldRowsUnderAHeadingAreReadCellByCell()
    {
        // Three cells in a row, each a label over a value, under a section title; none is read down the page.
        var strips = new List<OrderStrip> { Line(30, 20, "Payment", 14) };
        string expected = "Payment";
        for (int c = 0; c < 3; c++)
        {
            strips.Add(Line(30 + (c * 220), 60, $"Label{c}", 8));
            strips.Add(Line(30 + (c * 220), 82, $"value number {c}"));
            expected += $" Label{c} value number {c}";
        }

        Assert.Equal(expected, Read([.. strips]));
    }

    // A word of the test columns: names its column, row and place; lengths 3 to 6 so that word gaps do not line up from row to row.
    private static string ColumnWord(int column, int row, int word) =>
        $"{(char)('a' + column)}{row:x}{word}".PadRight(3 + ((row + (2 * word) + column) % 4), 'x');

    // Column pitch: the widest row of any column, plus a gutter of 0.9 of the text height (9 of 10).
    private static readonly double Pitch = Enumerable.Range(0, 3).SelectMany(c => Enumerable.Range(0, 12).Select(r => ColumnRow(c, r))).Max(t => t.Split(' ').Sum(w => 6.0 * w.Length) + (8.0 * 4)) + 9;

    private static string ColumnRow(int column, int row) => string.Join(' ', Enumerable.Range(0, 5).Select(w => ColumnWord(column, row, w)));

    [Fact]
    public void ColumnsWithATightGutterAreReadColumnByColumn()
    {
        // Three columns of twelve lines whose gutters are 0.9 of the text height: narrower than the usual 1.25, but empty through every
        // line of the region and each column is running text. They are read one after the other, not row by row across the page.
        var strips = new List<OrderStrip>();
        var expected = new List<string>();
        for (int c = 0; c < 3; c++)
        {
            for (int r = 0; r < 12; r++)
            {
                strips.Add(Line(20 + (c * Pitch), 20 + (r * 16), ColumnRow(c, r)));
                expected.Add(ColumnRow(c, r));
            }
        }

        Assert.Equal(string.Join(' ', expected), Read([.. strips]));
    }

    [Fact]
    public void ABulletColumnIsNotACutAtANarrowStripe()
    {
        // A bullet and its text 0.9 of the height apart on twelve lines: the narrow stripe is empty all the way down, but the bullets
        // are not a column of text, so each bullet is still read with its line.
        var strips = new List<OrderStrip>();
        var expected = new List<string>();
        for (int r = 0; r < 12; r++)
        {
            string text = ColumnRow(0, r);
            strips.Add(Line(20, 20 + (r * 16), "o"));
            strips.Add(Line(20 + 6 + 9, 20 + (r * 16), text));
            expected.Add("o " + text);
        }

        Assert.Equal(string.Join(' ', expected), Read([.. strips]));
    }

    private static string Columns(IEnumerable<int> cols, int rows) =>
        string.Join(' ', cols.SelectMany(c => Enumerable.Range(0, rows).Select(r => ColumnRow(c, r))));

    // Column pitch for a page of `rows` lines: the widest row of any column, plus a gutter of 0.9 of the text height.
    private static double PitchFor(int columns, int rows) =>
        Enumerable.Range(0, columns).SelectMany(c => Enumerable.Range(0, rows).Select(r => ColumnRow(c, r))).Max(t => t.Split(' ').Sum(w => 6.0 * w.Length) + (8.0 * 4)) + 9;

    [Fact]
    public void AStrayWordInATightGutterDoesNotMergeTheColumns()
    {
        // Three columns of a page of text with a gutter of 0.9 of the text height, but the end of one line of the first column runs
        // into the gutter (a word the recognizer put a little too far right, or a speck read as a word): no stripe is empty any more.
        const int rows = 34;
        double pitch = PitchFor(3, rows);
        var strips = new List<OrderStrip>();
        for (int c = 0; c < 3; c++)
        {
            for (int r = 0; r < rows; r++)
            {
                strips.Add(Line(20 + (c * pitch), 20 + (r * 16), c == 0 && r == 5 ? ColumnRow(c, r) + " zz" : ColumnRow(c, r)));
            }
        }

        Assert.Equal(Columns([0, 1, 2], rows), Read([.. strips]).Replace(" zz", string.Empty, StringComparison.Ordinal));
    }

    [Fact]
    public void AHeaderAndAFooterAcrossTheColumnsAreReadFirstAndLast()
    {
        // Two columns with a tight gutter, a running header above and a footer below that run across the gutter: the columns are
        // cut apart and the header and the footer stay whole, before and after them.
        const int rows = 34;
        double pitch = PitchFor(2, rows);
        var strips = new List<OrderStrip> { Line(20, 0, "Publication 100 Chapter 4 Heading of the page") };
        for (int c = 0; c < 2; c++)
        {
            for (int r = 0; r < rows; r++)
            {
                strips.Add(Line(20 + (c * pitch), 30 + (r * 16), ColumnRow(c, r)));
            }
        }

        strips.Add(Line(20, 30 + (rows * 16) + 20, "Footer words that cross the page 12"));
        Assert.Equal("Publication 100 Chapter 4 Heading of the page " + Columns([0, 1], rows) + " Footer words that cross the page 12", Read([.. strips]));
    }

    [Fact]
    public void ASpeckBetweenTwoColumnsIsNotAThirdColumn()
    {
        // Two columns with a wide gutter and one stray mark in the middle of it, half way down. Left alone it is a group of its own
        // that sits between two parts of the column beside it, and the page would be read in bands instead of column by column.
        double pitch = PitchFor(2, 20);
        var strips = new List<OrderStrip>();
        for (int c = 0; c < 2; c++)
        {
            for (int r = 0; r < 20; r++)
            {
                strips.Add(Line(20 + (c * (pitch + 40)), 20 + (r * 16), ColumnRow(c, r)));
            }
        }

        strips.Add(Line(20 + pitch + 20, 20 + (9 * 16) + 2, "x"));
        Assert.Equal(Columns([0, 1], 20), Read([.. strips]).Replace(" x ", " ", StringComparison.Ordinal));
    }
}
