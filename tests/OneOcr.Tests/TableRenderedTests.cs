using System.Numerics;
using OneImage;
using OneImage.Text;

namespace OneOcr.Tests;

/// <summary>
/// Tables rendered with a real font and read by a real recognizer: the cell grid, the spans and the words in each cell must come back
/// exactly. Runs when ONEOCR_TABLE_MODEL names a recognizer container; ONEOCR_TEST_FONT names the font (default: DejaVu Sans).
/// </summary>
public class TableRenderedTests
{
    private sealed record Spec(int Row, int Column, int RowSpan, int ColumnSpan, params string[] Lines);

    private const int ColumnWidth = 300, RowHeight = 90, Margin = 60, FontSize = 30;

    private static OcrEngine? Engine()
    {
        string? model = Environment.GetEnvironmentVariable("ONEOCR_TABLE_MODEL");
        if (model is null || !File.Exists(model))
        {
            return null;
        }

        using FileStream stream = File.OpenRead(model);
        return OcrEngine.Load(stream);
    }

    private static string? FontFile()
    {
        string path = Environment.GetEnvironmentVariable("ONEOCR_TEST_FONT") ?? "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
        return File.Exists(path) ? path : null;
    }

    // Draws the cells' text and, when ruled, each cell's border (a spanned cell has its outer border only).
    private static Image Render(string font, int rows, int columns, IReadOnlyList<Spec> cells, bool ruled, int columnWidth = ColumnWidth)
    {
        int w = (2 * Margin) + (columns * columnWidth), h = (2 * Margin) + (rows * RowHeight);
        var px = new Rgba32[w * h];
        Array.Fill(px, new Rgba32(255, 255, 255, 255));
        void Fill(int x0, int y0, int x1, int y1)
        {
            for (int y = y0; y < y1; y++)
            {
                for (int x = x0; x < x1; x++)
                {
                    px[(y * w) + x] = new Rgba32(70, 70, 70, 255);
                }
            }
        }

        if (ruled)
        {
            foreach (Spec c in cells)
            {
                int x0 = Margin + (c.Column * columnWidth), x1 = Margin + ((c.Column + c.ColumnSpan) * columnWidth);
                int y0 = Margin + (c.Row * RowHeight), y1 = Margin + ((c.Row + c.RowSpan) * RowHeight);
                Fill(x0, y0, x1 + 2, y0 + 2);
                Fill(x0, y1, x1 + 2, y1 + 2);
                Fill(x0, y0, x0 + 2, y1 + 2);
                Fill(x1, y0, x1 + 2, y1 + 2);
            }
        }

        Image image = Image.FromPixels<Rgba32>(px, w, h);
        var family = new FontCollection().Add(font);
        Font f = family.CreateFont(FontSize);
        foreach (Spec c in cells)
        {
            float y = Margin + (c.Row * RowHeight) + 14;
            foreach (string line in c.Lines)
            {
                image = image.DrawText(line, new TextOptions(f) { Origin = new Vector2(Margin + (c.Column * columnWidth) + 16, y) }, new Rgba32(0, 0, 0, 255));
                y += 34;
            }
        }

        return image;
    }

    private static List<(int, int, int, int, string)> Read(OcrEngine engine, Image image, out OcrTable table, out OcrPage page)
    {
        page = engine.Ocr(image, new OcrOptions { DetectOrientation = false, Deskew = false, MaxDegreeOfParallelism = 1 });
        OcrPage p = page;
        table = Assert.Single(page.Tables);
        return [.. table.Cells.Select(c => (c.Row, c.Column, c.RowSpan, c.ColumnSpan, string.Join(' ', c.WordIndices.Select(i => p.Words[i].Text))))];
    }

    [Fact]
    public void RuledTableWithSpansEmptyAndMultiLineCells()
    {
        OcrEngine? engine = Engine();
        string? font = FontFile();
        Assert.SkipWhen(engine is null || font is null, "ONEOCR_TABLE_MODEL (a recognizer container) or the test font is not available.");
        Spec[] cells =
        [
            new(0, 0, 1, 1, "Region"), new(0, 1, 1, 2, "Totals"),
            new(1, 0, 2, 1, "North"), new(1, 1, 1, 1, "Alpha", "Beta"), new(1, 2, 1, 1, "420"),
            new(2, 1, 1, 1), new(2, 2, 1, 1, "310"),
            new(3, 0, 1, 1, "South"), new(3, 1, 1, 1, "Gamma"), new(3, 2, 1, 1, "150"),
        ];
        Image image = Render(font!, 4, 3, cells, ruled: true);
        List<(int, int, int, int, string)> got = Read(engine!, image, out OcrTable table, out OcrPage page);
        Assert.Equal((4, 3), (table.Rows, table.Columns));
        var expected = cells.OrderBy(c => c.Row).ThenBy(c => c.Column).Select(c => (c.Row, c.Column, c.RowSpan, c.ColumnSpan, string.Join(' ', c.Lines))).ToList();
        Assert.Equal(expected, got);
        // Reading order: row by row, cell by cell, the lines of a cell top to bottom; a spanning cell once, in its first row.
        Assert.Equal("Region Totals North Alpha Beta 420 310 South Gamma 150", string.Join(' ', page.Words.Select(w => w.Text)));
        Assert.Single(page.Blocks);
        // The same table as Markdown (spans leave the covered cells empty) and through the page JSON.
        Assert.Contains("| Region | Totals |  |\n| --- | --- | --- |\n| North | Alpha Beta | 420 |\n|  |  | 310 |\n| South | Gamma | 150 |", page.ToMarkdown());
        OcrPage again = Assert.Single(OcrPageJson.Parse(OcrPageJson.ToJson([page])));
        Assert.Equal(table.Cells.Select(c => (c.Row, c.Column, c.RowSpan, c.ColumnSpan, c.WordIndices.Count)), again.Tables[0].Cells.Select(c => (c.Row, c.Column, c.RowSpan, c.ColumnSpan, c.WordIndices.Count)));
    }

    [Fact]
    public void TableWithoutRulesKeepsColumnsSpansAndEmptyCells()
    {
        OcrEngine? engine = Engine();
        string? font = FontFile();
        Assert.SkipWhen(engine is null || font is null, "ONEOCR_TABLE_MODEL (a recognizer container) or the test font is not available.");
        Spec[] cells =
        [
            new(0, 0, 1, 1, "Region"), new(0, 1, 1, 2, "Totals for the whole year"),
            new(1, 0, 1, 1, "North"), new(1, 1, 1, 1, "Alpha"), new(1, 2, 1, 1, "420"),
            new(2, 0, 1, 1, "East"), new(2, 1, 1, 1), new(2, 2, 1, 1, "310"),
            new(3, 0, 1, 1, "South"), new(3, 1, 1, 1, "Gamma"), new(3, 2, 1, 1, "150"),
        ];
        // Columns 170 pixels apart: the recognizer reads text separated by a much wider gap as one word ("Alpha420"), and a word is never
        // split between cells.
        Image image = Render(font!, 4, 3, cells, ruled: false, columnWidth: 170);
        List<(int, int, int, int, string)> got = Read(engine!, image, out OcrTable table, out _);
        Assert.Equal((4, 3), (table.Rows, table.Columns));
        var expected = cells.OrderBy(c => c.Row).ThenBy(c => c.Column).Select(c => (c.Row, c.Column, c.RowSpan, c.ColumnSpan, string.Join(' ', c.Lines))).ToList();
        Assert.Equal(expected, got);
    }
}
