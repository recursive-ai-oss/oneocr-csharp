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
                double y = 70 + (p * 85) + (l * 20);
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
}
