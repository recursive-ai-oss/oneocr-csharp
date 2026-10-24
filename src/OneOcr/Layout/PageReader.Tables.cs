using OneImage;
using OneOcr.Kernels;

namespace OneOcr.Layout;

internal static partial class PageReader
{
    // Probability reported for a table whose structure was read from drawn rules: a grid of at least two by two cells enclosed by
    // rules is a table in every case the page generators produce. Not calibrated.
    private const float RuledTableConfidence = 0.95f;

    // Finds the tables drawn with ruling lines and paints the rules out of the page, so that the crops the recognizer reads hold no
    // rule. This runs before any finder that erases rules itself. The page's own binarization is used, not the finder's: a rule is
    // often lighter than text, and whether it counts as ink must not depend on which finder ran.
    private static List<RuleGrid> FindGrids(Gray8[] gray, int width, int height, int degree)
    {
        int count = checked(width * height);
        (int level, bool inkIsBright) = LineFinder.Binarize(gray, count, degree);
        List<RuleGrid> grids = TableRules.Find(PixelSpans.AsBytes(gray.AsSpan(0, count)), width, height, level, inkIsBright);
        if (grids.Count > 0)
        {
            TableRules.Erase(PixelSpans.AsWritableBytes(gray.AsSpan(0, count)), width, height, inkIsBright, grids);
        }

        return grids;
    }

    // Reads one ruled table block: a line per row that holds text, a cell per grid cell (empty ones too), spans from the grid.
    private static void AddGridBlock(OrderBlock block, List<OrderStrip> strips, ref int wordIndex, List<OcrBlock> blocks, List<OcrTable> tables)
    {
        IReadOnlyList<OrderCell> cells = block.Cells!;
        var cellWords = new List<int>?[cells.Count];
        var lines = new List<OcrLine>(block.Lines.Count);
        foreach (OrderLine ol in block.Lines)
        {
            var words = new List<OcrWord>();
            var cellStarts = new List<int>();
            var text = new System.Text.StringBuilder();
            int current = -1;
            foreach (OrderFragment frag in ol.Fragments)
            {
                if (frag.Cell != current)
                {
                    cellStarts.Add(words.Count);
                    if (text.Length > 0)
                    {
                        text.Append('\t');
                    }

                    current = frag.Cell;
                }
                else
                {
                    // Another line of the same cell.
                    text.Append(' ');
                }

                for (int k = 0; k < frag.WordCount; k++)
                {
                    OcrWord word = strips[frag.Strip].Words[frag.WordStart + k];
                    if (k > 0)
                    {
                        text.Append(' ');
                    }

                    text.Append(word.Text);
                    (cellWords[frag.Cell] ??= []).Add(wordIndex + words.Count);
                    words.Add(word);
                }
            }

            wordIndex += words.Count;
            lines.Add(MakeLine(words, cellStarts, text.ToString()));
        }

        var tableCells = new List<OcrTableCell>(cells.Count);
        OcrBox? box = null;
        for (int i = 0; i < cells.Count; i++)
        {
            OrderCell c = cells[i];
            tableCells.Add(new OcrTableCell(c.Row, c.Column, c.RowSpan, c.ColumnSpan, c.Box, cellWords[i] ?? []));
            box = box is { } b ? OcrBox.Union(b, c.Box) : c.Box;
        }

        OcrBlock made = MakeBlock(block, lines);
        tables.Add(new OcrTable(box!.Value, block.Rows, block.Columns, tableCells, block.TableConfidence ?? RuledTableConfidence, OcrSource.Ocr, blocks.Count));
        blocks.Add(made);
    }
}
