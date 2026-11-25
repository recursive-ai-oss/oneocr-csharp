using System.Text.RegularExpressions;
using OneOcr.Layout;

namespace OneOcr.Tests;

/// <summary>Page structure: paragraph breaks, running headers and footers, offsets of words into text, indented page JSON.</summary>
public class PageStructureTests
{
    // A line of words starting at `left`, the words 6 units per character wide and 6 apart; `right` (when given) stretches the words to end there.
    private static OcrLine Line(string text, double left, double top, double height = 20, double? right = null)
    {
        string[] parts = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        double natural = parts.Sum(p => 6.0 * p.Length) + (6.0 * (parts.Length - 1));
        double scale = right is null ? 1 : (right.Value - left) / natural;
        var words = new List<OcrWord>();
        double at = left;
        foreach (string p in parts)
        {
            double w = 6.0 * p.Length * scale;
            words.Add(new OcrWord(p, new OcrBox(at, top, at + w, top + height), 0.95f, 0.95f, OcrSource.Ocr, []));
            at += w + (6.0 * scale);
        }

        return new OcrLine(text, words, [0], new OcrBox(left, top, at - (6.0 * scale), top + height), 0.95f, OcrSource.Ocr);
    }

    private static List<int> Sizes(List<List<OcrLine>> paragraphs) => [.. paragraphs.Select(p => p.Count)];

    // Lines of one block, `count` of them at a steady pitch from y; `full` lines end at 600.
    private static List<OcrLine> Run(string text, double left, ref double y, int count, double pitch = 24, double right = 600)
    {
        var lines = new List<OcrLine>();
        for (int i = 0; i < count; i++)
        {
            lines.Add(Line(text, left, y, 20, right));
            y += pitch;
        }

        return lines;
    }

    [Fact]
    public void ExtraSpaceBetweenParagraphsStartsANewParagraph()
    {
        // Line spacing 24; the paragraph break adds 5, which leaves a gap of 9 (under half a text height of 20), so the reader's cut keeps one block.
        double y = 100;
        var lines = Run("the sale of goods under this agreement continues.", 100, ref y, 3);
        y += 5;
        lines.AddRange(Run("the buyer pays the price in full.", 100, ref y, 3));
        Assert.Equal([3, 3], Sizes(ParagraphSplitter.Split(lines)));
    }

    [Fact]
    public void AnIndentedFirstLineStartsANewParagraph()
    {
        double y = 100;
        var lines = Run("the sale of goods under this agreement continues.", 100, ref y, 3);
        lines.Add(Line("the buyer pays the price in full and on time.", 130, y, 20, 600));
        y += 24;
        lines.AddRange(Run("the seller delivers within thirty days.", 100, ref y, 2));
        Assert.Equal([3, 3], Sizes(ParagraphSplitter.Split(lines)));
    }

    [Fact]
    public void ALineThatStopsShortBeforeAFullLineEndsAParagraph()
    {
        // Line 2 ends well before the margin although the next line's first word (30 wide) would have fitted; the lines around it are full.
        double y = 100;
        var lines = Run("the sale of goods under this agreement continues", 100, ref y, 2);
        lines.Add(Line("and ends here.", 100, y, 20, 330));
        y += 24;
        lines.Add(Line("goods are delivered to the buyer at the seller's dock.", 100, y, 20, 600));
        y += 24;
        lines.AddRange(Run("the buyer pays the price in full and on time.", 100, ref y, 2));
        Assert.Equal([3, 3], Sizes(ParagraphSplitter.Split(lines)));
    }

    [Fact]
    public void RaggedRightTextAndAddressesStayWhole()
    {
        // A line is short because the next word is long: it would not have fitted, so nothing ends there.
        double y = 100;
        var lines = Run("the sale of goods under this agreement continues", 100, ref y, 2);
        lines.Add(Line("and the seller's obligations.", 100, y, 20, 560));
        y += 24;
        lines.AddRange(Run("internationalization considerations apply throughout the term", 100, ref y, 2));
        Assert.Equal([5], Sizes(ParagraphSplitter.Split(lines)));

        // An address: short lines one under the other, none followed by a full line.
        y = 100;
        var address = new List<OcrLine> { Line("Harbor Supply Ltd.", 100, y, 20), Line("12 Quay Street.", 100, y + 24, 20), Line("Rotterdam.", 100, y + 48, 20) };
        Assert.Equal([3], Sizes(ParagraphSplitter.Split(address)));
    }

    [Fact]
    public void ListItemsAndTableRowsAreNotCut()
    {
        double y = 100;
        var items = Run("the sale of goods under this agreement continues.", 100, ref y, 2);
        y += 5;
        items.Add(Line("1. the buyer pays the price in full.", 100, y, 20, 600));
        Assert.Equal([3], Sizes(ParagraphSplitter.Split(items)));

        var row = new OcrLine("a\tb", [.. Line("a b", 100, 100).Words], [0, 1], new OcrBox(100, 100, 130, 120), 0.9f, OcrSource.Ocr);
        Assert.Equal([2], Sizes(ParagraphSplitter.Split([row, row])));
    }

    [Fact]
    public void ParagraphsAreSeparateBlocksOfTheAssembledPage()
    {
        // Lines of height 20 at pitch 24 (an empty band of 4); the paragraph break leaves 4 more of space and the band (8) is under the 10 that cuts blocks.
        var strips = new List<OrderStrip>();
        double y = 100;
        const string Text = "alpha beta gamma delta epsilon zeta eta theta iota.";
        strips.Add(StripOf(Text, 100, y));
        strips.Add(StripOf(Text, 100, y += 24));
        strips.Add(StripOf(Text, 100, y += 24));
        strips.Add(StripOf(Text, 130, y += 28));
        strips.Add(StripOf(Text, 100, y += 24));
        strips.Add(StripOf(Text, 100, y += 24));
        List<OrderBlock> plan = ReadingOrder.Arrange(strips, CancellationToken.None);
        Assert.Single(plan);
        OcrPage page = PageReader.Assemble(plan, strips, [], 0, 800, 1000, 72, []);
        Assert.Equal(2, page.Blocks.Count);
        Assert.Equal([3, 3], page.Blocks.Select(b => b.Lines.Count));
        Assert.Equal(string.Join('\n', page.Blocks.SelectMany(b => b.Lines).Select(l => l.Text)), page.ToText());
        Assert.Contains("\n\n", page.Text, StringComparison.Ordinal);
    }

    private static int leaf;

    private static OrderStrip StripOf(string text, double x, double y, double height = 20)
    {
        var words = new List<OcrWord>();
        double at = x;
        foreach (string w in text.Split(' '))
        {
            double width = 6 * w.Length;
            words.Add(new OcrWord(w, new OcrBox(at, y, at + width, y + height), 0.99f, 0.99f, OcrSource.Ocr, []));
            at += width + 6;
        }

        // Lines of running text reach a common right edge (a justified column).
        return new OrderStrip { Leaf = Interlocked.Increment(ref leaf), Box = new OcrBox(x, y, at - 6, y + height), Size = height, Words = words };
    }

    // ---- running headers and footers ----

    private static OcrBlock Block(OcrBlockRole role, params OcrLine[] lines)
    {
        OcrBox box = lines[0].Box;
        foreach (OcrLine l in lines)
        {
            box = OcrBox.Union(box, l.Box);
        }

        return new OcrBlock(role, false, box, lines, 0.9f);
    }

    // A letter-size page at 72 units per inch: a header and a footer set off from a body of ten lines.
    private static OcrPage Letter(string header, string footer, string bodyOpening, int index = 0, double bodyTop = 100)
    {
        var blocks = new List<OcrBlock>();
        if (header.Length > 0)
        {
            blocks.Add(Block(OcrBlockRole.Body, Line(header, 72, 24, 12)));
        }

        for (int i = 0; i < 5; i++)
        {
            blocks.Add(Block(OcrBlockRole.Body, Line($"{bodyOpening} paragraph {i} line one of text.", 72, bodyTop + (i * 70), 12), Line("line two of the same paragraph.", 72, bodyTop + (i * 70) + 14, 12)));
        }

        if (footer.Length > 0)
        {
            blocks.Add(Block(OcrBlockRole.Body, Line(footer, 280, 755, 12)));
        }

        return new OcrPage(index, 612, 792, 72, blocks);
    }

    [Fact]
    public void ABlockAtTheEdgeOfThePageSetOffFromTheTextIsFurniture()
    {
        OcrPage page = Furniture.MarkWithinPage(Letter("Harbor Supply annual report", "Page 3 of 9", "first"));
        Assert.Equal(OcrBlockRole.Header, page.Blocks[0].Role);
        Assert.All(page.Blocks.Skip(1).Take(5), b => Assert.Equal(OcrBlockRole.Body, b.Role));
        Assert.Equal(OcrBlockRole.Footer, page.Blocks[^1].Role);

        // Markdown leaves furniture out and keeps the text.
        string md = page.ToMarkdown();
        Assert.DoesNotContain("Harbor Supply annual report", md, StringComparison.Ordinal);
        Assert.DoesNotContain("Page 3 of 9", md, StringComparison.Ordinal);
        Assert.Contains("first paragraph 0 line one", md, StringComparison.Ordinal);
        Assert.Contains("Page 3 of 9", page.ToText(), StringComparison.Ordinal);
        Assert.Contains("Harbor Supply annual report", page.ToMarkdown(new MarkdownOptions { IncludeHeadersAndFooters = true }), StringComparison.Ordinal);
    }

    [Fact]
    public void ATitleOrABlockWithoutABodyIsNotFurniture()
    {
        // A heading near the top is larger than the body.
        var big = new OcrPage(0, 612, 792, 72, [Block(OcrBlockRole.Body, Line("Annual Report", 72, 20, 30)), .. Letter("", "", "x").Blocks]);
        Assert.Same(big, Furniture.MarkWithinPage(big));

        // A page of three lines has no body to be set off from (a receipt, a short note).
        var note = new OcrPage(0, 612, 792, 72, [Block(OcrBlockRole.Body, Line("Harbor Supply", 72, 20, 12)), Block(OcrBlockRole.Body, Line("one", 72, 100, 12), Line("two", 72, 114, 12))]);
        Assert.Same(note, Furniture.MarkWithinPage(note));

        // A block close to the text is part of it.
        var close = new OcrPage(0, 612, 792, 72, [Block(OcrBlockRole.Body, Line("Opening line", 72, 40, 12)), .. Letter("", "", "x", bodyTop: 56).Blocks]);
        Assert.Same(close, Furniture.MarkWithinPage(close));
    }

    [Fact]
    public void BlocksThatRepeatAcrossPagesAreFurniture()
    {
        // The running title is in the band but too close to the text to pass alone; it repeats on every page, as does the page number.
        OcrPage[] pages =
        [
            Letter("Harbor Supply annual report", "- 1 -", "alpha", 0),
            Letter("Harbor Supply annual report", "- 2 -", "beta", 1),
            Letter("Harbor Supply annual report", "- 10 -", "gamma", 2),
        ];
        IReadOnlyList<OcrPage> marked = Furniture.MarkAcrossPages(pages);
        Assert.All(marked, p =>
        {
            Assert.Equal(OcrBlockRole.Header, p.Blocks[0].Role);
            Assert.Equal(OcrBlockRole.Footer, p.Blocks[^1].Role);
            Assert.All(p.Blocks.Skip(1).Take(5), b => Assert.Equal(OcrBlockRole.Body, b.Role));
        });

        // A line that is only on one page (a letterhead) stays body text.
        OcrPage[] letterhead = [pages[0], Letter("", "- 2 -", "beta", 1), Letter("", "- 3 -", "gamma", 2)];
        IReadOnlyList<OcrPage> second = Furniture.MarkAcrossPages(letterhead);
        Assert.Equal(OcrBlockRole.Body, second[0].Blocks[0].Role);
        Assert.Equal(OcrBlockRole.Footer, second[1].Blocks[^1].Role);

        // One page alone has nothing to repeat with.
        IReadOnlyList<OcrPage> alone = [pages[0]];
        Assert.Same(alone, Furniture.MarkAcrossPages(alone));
    }

    [Fact]
    public void RepeatingTextAtADifferentPlaceIsNotFurniture()
    {
        OcrPage a = Letter("Summary", "", "alpha", 0);
        var movedBlocks = Letter("", "", "beta", 1).Blocks.ToList();
        movedBlocks.Insert(0, Block(OcrBlockRole.Body, Line("Summary", 72, 70, 12)));
        var b = new OcrPage(1, 612, 792, 72, movedBlocks);
        IReadOnlyList<OcrPage> marked = Furniture.MarkAcrossPages([a, b]);
        Assert.Equal(OcrBlockRole.Body, marked[0].Blocks[0].Role);
        Assert.Equal(OcrBlockRole.Body, marked[1].Blocks[0].Role);
    }

    // ---- offsets of words into text ----

    [Fact]
    public void WordSpansPointIntoTheTextOfALineAndOfAPage()
    {
        OcrLine first = Line("alpha beta", 10, 10);
        OcrLine cells = new("name\tgamma delta", [.. Line("name gamma delta", 10, 40).Words], [0, 1], new OcrBox(10, 40, 200, 60), 0.9f, OcrSource.Ocr);
        OcrLine other = Line("epsilon", 10, 200);
        var page = new OcrPage(0, 300, 300, 72, [Block(OcrBlockRole.Body, first, cells), Block(OcrBlockRole.Body, other)]);

        IReadOnlyList<OcrTextSpan> line = cells.GetWordSpans();
        Assert.Equal([new OcrTextSpan(0, 4), new OcrTextSpan(5, 5), new OcrTextSpan(11, 5)], line);

        string text = page.Text;
        Assert.Equal("alpha beta\nname\tgamma delta\n\nepsilon", text);
        IReadOnlyList<OcrTextSpan> spans = page.GetWordSpans();
        Assert.Equal(page.Words.Count, spans.Count);
        for (int i = 0; i < spans.Count; i++)
        {
            Assert.Equal(page.Words[i].Text, text.Substring(spans[i].Start, spans[i].Length));
        }

        // The text that ToText() returns has one line feed between blocks.
        string lines = page.ToText();
        IReadOnlyList<OcrTextSpan> plain = page.GetWordSpans(OcrTextLayout.Lines);
        for (int i = 0; i < plain.Count; i++)
        {
            Assert.Equal(page.Words[i].Text, lines.Substring(plain[i].Start, plain[i].Length));
        }

        IReadOnlyList<OcrTextSpan> lineSpans = page.GetLineSpans();
        Assert.Equal(["alpha beta", "name\tgamma delta", "epsilon"], lineSpans.Select(s => text.Substring(s.Start, s.Length)));
        Assert.Equal(15, new OcrTextSpan(10, 5).End);
    }

    [Fact]
    public void WordSpansOfALineWhoseTextBreaksTheRuleAreSearchedFor()
    {
        OcrLine odd = new("x  alpha ~ missing", [.. Line("alpha beta", 10, 10).Words], [0], new OcrBox(10, 10, 100, 30), 0.9f, OcrSource.Ocr);
        Assert.Equal([new OcrTextSpan(3, 5), new OcrTextSpan(-1, 0)], odd.GetWordSpans());
    }

    // ---- indented JSON ----

    [Fact]
    public void IndentedJsonKeepsEveryBoundsAndQuadOnOneLine()
    {
        OcrLine tilted = Line("alpha", 10, 10) with { Quad = new OcrQuad(new OcrPoint(10, 10), new OcrPoint(40, 11), new OcrPoint(40, 31), new OcrPoint(10, 30)) };
        OcrPage page = new(
            0, 300, 300, 72, [Block(OcrBlockRole.Body, tilted, Line("beta gamma", 10, 60))], toPdf: new OcrMatrix(1, 0, 0, -1, 0, 300),
            tables: [new OcrTable(new OcrBox(0, 0, 10, 10), 1, 1, [new OcrTableCell(0, 0, 1, 1, new OcrBox(0, 0, 10, 10), [0, 1])], 0.9f, OcrSource.Ocr)]);
        string compact = OcrPageJson.ToJson([page]);
        string indented = OcrPageJson.ToJson([page], new OcrPageJsonOptions { Indented = true });

        Assert.Contains("\"bounds\":[10,10,40,30]", compact, StringComparison.Ordinal);
        Assert.Contains("\"toPdf\":[1,0,0,-1,0,300]", compact, StringComparison.Ordinal);
        Assert.Matches(new Regex("\"bounds\": \\[10, 10, 40, 30\\]"), indented);
        Assert.Matches(new Regex("\"quad\": \\[10, 10, 40, 11, 40, 31, 10, 30\\]"), indented);
        Assert.Matches(new Regex("\"toPdf\": \\[1, 0, 0, -1, 0, 300\\]"), indented);
        Assert.Matches(new Regex("\"words\": \\[0, 1\\]"), indented);

        // No line holds only a number or an opening bracket of such an array.
        foreach (string line in indented.Split('\n'))
        {
            string t = line.Trim().TrimEnd(',');
            Assert.False(Regex.IsMatch(t, "^-?[0-9.]+$"), line);
        }

        // Both forms read back as the same page.
        Assert.Equal(OcrPageJson.ToJson(OcrPageJson.Parse(compact)), compact);
        Assert.Equal(OcrPageJson.ToJson(OcrPageJson.Parse(indented)), compact);
    }
}
