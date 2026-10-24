using OneOcr.Pdf;

namespace OneOcr.Tests;

public class MarkdownTests
{
    // A line of words at a known place; words are spread evenly over the line's width.
    private static OcrLine Line(string text, double left, double top, double height = 12, double width = 300, params int[] cellStarts)
    {
        string[] parts = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var words = new List<OcrWord>();
        double step = width / Math.Max(1, parts.Length);
        for (int i = 0; i < parts.Length; i++)
        {
            words.Add(new OcrWord(parts[i], new OcrBox(left + (i * step), top, left + ((i + 1) * step) - 2, top + height), 0.9f, 0.9f, OcrSource.Ocr, []));
        }

        return new OcrLine(text, words, cellStarts.Length == 0 ? [0] : cellStarts, new OcrBox(left, top, left + width, top + height), 0.9f, OcrSource.Ocr);
    }

    private static OcrBlock Block(OcrBlockRole role, bool table, params OcrLine[] lines)
    {
        OcrBox box = lines[0].Box;
        foreach (OcrLine l in lines)
        {
            box = OcrBox.Union(box, l.Box);
        }

        return new OcrBlock(role, table, box, lines, 0.9f);
    }

    private static OcrBlock Body(params OcrLine[] lines) => Block(OcrBlockRole.Body, false, lines);

    private static OcrPage Page(params OcrBlock[] blocks) => new(0, 612, 792, 72, blocks);

    // Body text: enough 12-point lines that the median line height is 12.
    private static OcrBlock Filler(double top) =>
        Body(Line("The quick brown fox jumps over the lazy dog and keeps running", 72, top), Line("through the field until the sun goes down behind the hills.", 72, top + 14));

    [Fact]
    public void HeadingsFollowTypeSizeAndHyphenatedWordsAreJoined()
    {
        OcrPage page = Page(
            Body(Line("Annual Report", 72, 60, 30)),
            Body(Line("Overview", 72, 110, 20)),
            Body(Line("Lorem ipsum dolor sit amet, consec-", 72, 150), Line("tetur adipiscing elit sed do eiusmod.", 72, 164)),
            Filler(220),
            Filler(260));
        Assert.Equal(
            "# Annual Report\n\n## Overview\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit sed do eiusmod.\n\n"
            + "The quick brown fox jumps over the lazy dog and keeps running through the field until the sun goes down behind the hills.\n\n"
            + "The quick brown fox jumps over the lazy dog and keeps running through the field until the sun goes down behind the hills.\n",
            page.ToMarkdown());
        // Without the size rule the same blocks are paragraphs.
        Assert.StartsWith("Annual Report\n\nOverview\n\nLorem", page.ToMarkdown(new MarkdownOptions { HeadingsFromSize = false }), StringComparison.Ordinal);
    }

    [Fact]
    public void LayoutRolesBeatTheSizeHeuristic()
    {
        OcrPage plain = Page(Body(Line("Chapter two", 72, 60, 12)), Filler(100), Filler(140));
        Assert.StartsWith("Chapter two\n\n", plain.ToMarkdown(), StringComparison.Ordinal);
        var withRoles = new OcrPage(
            0, 612, 792, 72, plain.Blocks,
            [new OcrRegion(OcrRegionRole.Title, plain.Blocks[0].Box, 0.9f, OcrSource.Ocr, [0]), new OcrRegion(OcrRegionRole.Text, plain.Blocks[1].Box, 0.9f, OcrSource.Ocr, [1, 2])]);
        Assert.StartsWith("# Chapter two\n\n", withRoles.ToMarkdown(), StringComparison.Ordinal);
        // A big line in a text region stays a paragraph.
        var bigButText = new OcrPage(
            0, 612, 792, 72, [Body(Line("Large but text", 72, 60, 30)), Filler(100), Filler(140)],
            [new OcrRegion(OcrRegionRole.Text, new OcrBox(0, 0, 612, 100), 0.9f, OcrSource.Ocr, [0])]);
        Assert.StartsWith("Large but text\n\n", bigButText.ToMarkdown(), StringComparison.Ordinal);
    }

    [Fact]
    public void ListsKeepTheirMarkersNestingAndWrappedLines()
    {
        OcrPage page = Page(
            Body(
                Line("\u2022 First item that wraps", 72, 100),
                Line("onto a second line", 90, 114),
                Line("\u2022 Second item", 72, 128),
                Line("\u2013 nested one", 100, 142),
                Line("\u2013 nested two", 100, 156),
                Line("\u2022 Third item", 72, 170)),
            Body(Line("1. alpha", 72, 220), Line("2. beta", 72, 234), Line("10) tenth", 72, 248)),
            Filler(300),
            Filler(340));
        string md = page.ToMarkdown();
        Assert.StartsWith(
            "- First item that wraps onto a second line\n- Second item\n    - nested one\n    - nested two\n- Third item\n\n1. alpha\n2. beta\n10. tenth\n\nThe quick",
            md, StringComparison.Ordinal);
        Assert.StartsWith("\u2022 First item", page.ToMarkdown(new MarkdownOptions { Lists = false }), StringComparison.Ordinal);
    }

    [Fact]
    public void NumbersThatAreNotMarkersStayText()
    {
        OcrPage page = Page(Body(Line("1.5 million visitors came in 2021. Then", 72, 100), Line("-5 degrees and a dash - here", 72, 114)), Filler(160), Filler(200));
        Assert.StartsWith("1.5 million visitors came in 2021. Then -5 degrees and a dash - here\n\n", page.ToMarkdown(), StringComparison.Ordinal);
    }

    [Fact]
    public void TextThatLooksLikeMarkupIsEscaped()
    {
        OcrPage page = Page(Body(Line("# not a heading, a_b *x* [link] <tag> `code` 3. nope", 72, 100)), Filler(140), Filler(180));
        Assert.StartsWith("\\# not a heading, a\\_b \\*x\\* \\[link\\] \\<tag\\> \\`code\\` 3. nope\n\n", page.ToMarkdown(), StringComparison.Ordinal);
        OcrPage ordered = Page(Body(Line("2021. A year", 72, 100)), Filler(140), Filler(180));
        // "2021. " is a list marker with four digits; a four-digit run is not one, but it still must not render as a list.
        Assert.StartsWith("2021\\. A year\n\n", ordered.ToMarkdown(), StringComparison.Ordinal);
    }

    [Fact]
    public void TablesBecomePipeTablesAndSpansLeaveTheCoveredCellsEmpty()
    {
        // Rows: [Quarter (spans 2 columns) | Total], [Q1 | 10 | 30], [Region (spans 2 rows) | a|b | 5], [ | 7 | 6]
        var lines = new List<OcrLine>
        {
            Line("Quarter Total", 72, 100, 12, 300, 0, 1),
            Line("Q1 10 30", 72, 120, 12, 300, 0, 1, 2),
            Line("Region a|b 5", 72, 140, 12, 300, 0, 1, 2),
            Line("7 6", 72, 160, 12, 300, 0, 1),
        };
        OcrBlock block = Block(OcrBlockRole.Body, true, [.. lines]);
        // Word numbers in reading order: Quarter 0, Total 1, Q1 2, 10 3, 30 4, Region 5, a|b 6, 5 7, 7 8, 6 9.
        var cells = new List<OcrTableCell>
        {
            new(0, 0, 1, 2, lines[0].Box, [0]), new(0, 2, 1, 1, lines[0].Box, [1]),
            new(1, 0, 1, 1, lines[1].Box, [2]), new(1, 1, 1, 1, lines[1].Box, [3]), new(1, 2, 1, 1, lines[1].Box, [4]),
            new(2, 0, 2, 1, lines[2].Box, [5]), new(2, 1, 1, 1, lines[2].Box, [6]), new(2, 2, 1, 1, lines[2].Box, [7]),
            new(3, 1, 1, 1, lines[3].Box, [8]), new(3, 2, 1, 1, lines[3].Box, [9]),
        };
        var table = new OcrTable(block.Box, 4, 3, cells, 0.9f, OcrSource.Ocr, 0);
        var page = new OcrPage(0, 612, 792, 72, [block, Filler(300), Filler(340)], null, [table]);
        Assert.Equal(
            "| Quarter |  | Total |\n| --- | --- | --- |\n| Q1 | 10 | 30 |\n| Region | a\\|b | 5 |\n|  | 7 | 6 |\n\nThe quick brown fox jumps over the lazy dog and keeps running through the field until the sun goes down behind the hills.\n\n"
            + "The quick brown fox jumps over the lazy dog and keeps running through the field until the sun goes down behind the hills.\n",
            page.ToMarkdown());
    }

    [Fact]
    public void ATableBlockWithoutATableEntryIsBuiltFromItsCells()
    {
        OcrBlock block = Block(OcrBlockRole.Body, true, Line("Item Price", 72, 100, 12, 300, 0, 1), Line("Tea 3.50", 72, 120, 12, 300, 0, 1));
        string md = Page(block, Filler(200), Filler(240)).ToMarkdown();
        Assert.StartsWith("| Item | Price |\n| --- | --- |\n| Tea | 3.50 |\n\n", md, StringComparison.Ordinal);
    }

    [Fact]
    public void TwoColumnPagesReadColumnByColumnAndDropFurniture()
    {
        OcrBlock header = Block(OcrBlockRole.Header, false, Line("Company confidential", 72, 20));
        OcrBlock left = Body(Line("Left column starts here and", 72, 100, 12, 240), Line("goes on for a little while.", 72, 114, 12, 240));
        OcrBlock right = Body(Line("Right column follows the left", 340, 100, 12, 240), Line("one in reading order.", 340, 114, 12, 240));
        OcrBlock footer = Block(OcrBlockRole.Footer, false, Line("Harbor authority", 72, 760));
        OcrBlock number = Body(Line("Page 7 of 12", 280, 765, 12, 60));
        OcrPage page = Page(header, left, right, footer, number, Filler(300));
        Assert.Equal(
            "Left column starts here and goes on for a little while.\n\nRight column follows the left one in reading order.\n\n"
            + "The quick brown fox jumps over the lazy dog and keeps running through the field until the sun goes down behind the hills.\n",
            page.ToMarkdown());
        string kept = page.ToMarkdown(new MarkdownOptions { IncludeHeadersAndFooters = true, DropPageNumbers = false });
        Assert.StartsWith("Company confidential\n\nLeft column", kept, StringComparison.Ordinal);
        Assert.Contains("Harbor authority\n\nPage 7 of 12", kept, StringComparison.Ordinal);
        // A number in the middle of the page is content.
        OcrPage middle = Page(Body(Line("7", 300, 400, 12, 10)), Filler(100), Filler(140));
        Assert.StartsWith("7\n\n", middle.ToMarkdown(), StringComparison.Ordinal);
    }

    [Fact]
    public void ParagraphsSplitAtLargeGapsAndPagesAreSeparated()
    {
        OcrPage page = Page(Body(Line("First paragraph line one", 72, 100), Line("and line two.", 72, 114), Line("Second paragraph starts after a gap", 72, 150)));
        Assert.Equal("First paragraph line one and line two.\n\nSecond paragraph starts after a gap\n", page.ToMarkdown());
        string two = new[] { page, new OcrPage(1, 612, 792, 72, []), page }.ToMarkdown();
        Assert.Equal(page.ToMarkdown() + "\n---\n\n" + page.ToMarkdown(), two);
        Assert.Equal("", new OcrPage(0, 10, 10, 72, []).ToMarkdown());
    }

    [Fact]
    public void RotatedPagesUseTheirQuadsForSizesAndSkipPageNumberGuesses()
    {
        // A page turned 90 degrees: the lines' boxes in the image are tall and narrow, but their quads say how tall the text is.
        OcrLine Rot(string text, double x, double h) => Line(text, x, 72, 300, 12) with { Box = new OcrBox(x, 72, x + h, 372), Quad = new OcrQuad(new OcrPoint(x + h, 72), new OcrPoint(x + h, 372), new OcrPoint(x, 372), new OcrPoint(x, 72)) };
        var blocks = new List<OcrBlock> { Body(Rot("Big title", 500, 30)), Body(Rot("small text one", 440, 12)), Body(Rot("small text two", 400, 12)), Body(Rot("small text three", 360, 12)) };
        var page = new OcrPage(0, 612, 792, 72, blocks, null, null, null, null, null, 90, 0);
        Assert.StartsWith("# Big title\n\nsmall text one\n\n", page.ToMarkdown(), StringComparison.Ordinal);
    }

    [Fact]
    public void ArgumentsAreChecked()
    {
        Assert.Throws<ArgumentNullException>(() => OcrMarkdown.ToMarkdown((OcrPage)null!));
        Assert.Throws<ArgumentNullException>(() => OcrMarkdown.ToMarkdown((IEnumerable<OcrPage>)null!));
    }

    [Fact]
    public void ATextLayerPdfGivesTheHandCheckedMarkdown()
    {
        var engine = OcrEngine.Load(new MemoryStream(TestModels.RandomContainer(
            new TestModels.Shape(32, [(8, 2, 2), (16, 2, 2), (16, 2, 1), (32, 2, 1), (32, 2, 1)], 24, 2, 41), 3)));
        IReadOnlyList<OcrPage> pages = PdfOcr.Read(PdfFixtures.MarkdownPage(), engine, new PdfOcrOptions { ReadImages = false, CrossCheckLines = 0 });
        const string Expected =
            "# Harbor Report\n\n"
            + "Cargo volumes rose by twelve percent over the spring months and the cranes finished their training ahead of schedule.\n\n"
            + "## Cranes\n\n"
            + "- Crane operators\n- Warehouse seven\n- Customs clearance\n\n"
            + "1. Open the east gate\n2. Sign the log\n";
        Assert.Equal(Expected, pages[0].ToMarkdown());
        Assert.Equal(Expected, pages.ToMarkdown());
    }
}
