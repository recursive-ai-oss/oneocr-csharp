using System.Text;

namespace OneOcr.Pdf;

/// <summary>Building and rewriting page-model elements: moving recognized regions onto the PDF page and dropping words.</summary>
internal static class PageParts
{
    /// <summary>x' = x * ScaleX + OffsetX, y' = y * ScaleY + OffsetY.</summary>
    internal readonly record struct Affine(double ScaleX, double ScaleY, double OffsetX, double OffsetY)
    {
        public OcrBox Map(OcrBox b) => new(
            (b.Left * ScaleX) + OffsetX, (b.Top * ScaleY) + OffsetY, (b.Right * ScaleX) + OffsetX, (b.Bottom * ScaleY) + OffsetY);

        public OcrPoint Map(OcrPoint p) => new((p.X * ScaleX) + OffsetX, (p.Y * ScaleY) + OffsetY);

        public OcrQuad? Map(OcrQuad? q) => q is { } v ? new OcrQuad(Map(v.P0), Map(v.P1), Map(v.P2), Map(v.P3)) : null;
    }

    public static OcrWord Map(OcrWord w, Affine a)
    {
        var chars = new OcrChar[w.Chars.Count];
        for (int i = 0; i < chars.Length; i++)
        {
            OcrChar c = w.Chars[i];
            chars[i] = c with { Box = a.Map(c.Box), Quad = a.Map(c.Quad) };
        }

        return w with { Box = a.Map(w.Box), Chars = chars, Quad = a.Map(w.Quad) };
    }

    public static OcrBlock Map(OcrBlock b, Affine a)
    {
        var lines = new OcrLine[b.Lines.Count];
        for (int i = 0; i < lines.Length; i++)
        {
            OcrLine l = b.Lines[i];
            var words = new OcrWord[l.Words.Count];
            for (int j = 0; j < words.Length; j++)
            {
                words[j] = Map(l.Words[j], a);
            }

            lines[i] = l with { Box = a.Map(l.Box), Words = words, Quad = a.Map(l.Quad) };
        }

        return b with { Box = a.Map(b.Box), Lines = lines, Quad = a.Map(b.Quad) };
    }

    public static OcrRegion Map(OcrRegion r, Affine a) => r with { Box = a.Map(r.Box), Quad = a.Map(r.Quad) };

    public static OcrTable Map(OcrTable t, Affine a)
    {
        var cells = new OcrTableCell[t.Cells.Count];
        for (int i = 0; i < cells.Length; i++)
        {
            cells[i] = t.Cells[i] with { Box = a.Map(t.Cells[i].Box) };
        }

        return t with { Box = a.Map(t.Box), Cells = cells, Quad = a.Map(t.Quad) };
    }

    public static OcrKeyValue Map(OcrKeyValue k, Affine a) => k with { Box = a.Map(k.Box) };

    /// <summary>A line from its words: text joined by a space within a cell and a tab between cells, box and confidence derived.</summary>
    public static OcrLine MakeLine(
        IReadOnlyList<OcrWord> words, IReadOnlyList<int> cells, OcrSource source, bool rtl, bool vertical, OcrQuad? quad = null)
    {
        var text = new StringBuilder();
        OcrBox box = words[0].Box;
        double weighted = 0;
        long weight = 0;
        int cell = 0;
        for (int i = 0; i < words.Count; i++)
        {
            if (i > 0)
            {
                text.Append(cell + 1 < cells.Count && cells[cell + 1] == i ? '\t' : ' ');
            }

            if (cell + 1 < cells.Count && cells[cell + 1] == i)
            {
                cell++;
            }

            text.Append(words[i].Text);
            box = OcrBox.Union(box, words[i].Box);
            weighted += words[i].Confidence * (double)words[i].Text.Length;
            weight += words[i].Text.Length;
        }

        float confidence = weight == 0 ? 0f : (float)(weighted / weight);
        return new OcrLine(text.ToString(), words, cells, box, confidence, source, rtl, vertical, quad);
    }

    /// <summary>A block from its lines; the box is their union and the confidence their mean weighted by word count.</summary>
    public static OcrBlock MakeBlock(IReadOnlyList<OcrLine> lines, OcrBlockRole role, bool isTable, OcrSource source, OcrQuad? quad = null)
    {
        OcrBox box = lines[0].Box;
        double weighted = 0;
        long weight = 0;
        foreach (OcrLine line in lines)
        {
            box = OcrBox.Union(box, line.Box);
            weighted += line.Confidence * (double)line.Words.Count;
            weight += line.Words.Count;
        }

        return new OcrBlock(role, isTable, box, lines, weight == 0 ? 0f : (float)(weighted / weight), source, quad);
    }

    /// <summary>
    /// The blocks without the words <paramref name="keep"/> rejects: lines and blocks that lose all their words disappear, cells are
    /// renumbered, and line and block boxes and confidences are recomputed for the ones that lost some.
    /// </summary>
    public static List<OcrBlock> Filter(IReadOnlyList<OcrBlock> blocks, Func<OcrWord, bool> keep, out int removedWords)
    {
        removedWords = 0;
        var result = new List<OcrBlock>(blocks.Count);
        foreach (OcrBlock block in blocks)
        {
            var lines = new List<OcrLine>(block.Lines.Count);
            bool changed = false;
            foreach (OcrLine line in block.Lines)
            {
                var kept = new List<OcrWord>(line.Words.Count);
                var cells = new List<int>();
                int nextCell = 0;
                for (int i = 0; i < line.Words.Count; i++)
                {
                    bool startsCell = nextCell < line.Cells.Count && line.Cells[nextCell] == i;
                    if (startsCell)
                    {
                        nextCell++;
                    }

                    if (!keep(line.Words[i]))
                    {
                        removedWords++;
                        // A removed word that started a cell hands the start to the next kept word of that cell.
                        if (startsCell)
                        {
                            cells.Add(kept.Count);
                        }

                        continue;
                    }

                    if (startsCell || cells.Count == 0)
                    {
                        cells.Add(kept.Count);
                    }

                    kept.Add(line.Words[i]);
                }

                if (kept.Count == line.Words.Count)
                {
                    lines.Add(line);
                    continue;
                }

                changed = true;
                if (kept.Count == 0)
                {
                    continue;
                }

                // Cell starts must start at 0, rise strictly and stay inside the kept words.
                var distinct = new List<int>();
                foreach (int c in cells)
                {
                    if (c < kept.Count && (distinct.Count == 0 || c > distinct[^1]))
                    {
                        distinct.Add(c);
                    }
                }

                if (distinct.Count == 0 || distinct[0] != 0)
                {
                    distinct.Insert(0, 0);
                }

                lines.Add(MakeLine(kept, distinct, line.Source, line.IsRightToLeft, line.IsVertical, line.Quad));
            }

            if (!changed)
            {
                result.Add(block);
            }
            else if (lines.Count > 0)
            {
                result.Add(MakeBlock(lines, block.Role, block.IsTable, block.Source, block.Quad));
            }
        }

        return result;
    }

    public static int CountWords(IReadOnlyList<OcrBlock> blocks)
    {
        int n = 0;
        foreach (OcrBlock b in blocks)
        {
            foreach (OcrLine l in b.Lines)
            {
                n += l.Words.Count;
            }
        }

        return n;
    }

    public static double Area(OcrBox b) => Math.Max(0, b.Width) * Math.Max(0, b.Height);

    public static double IntersectionArea(OcrBox a, OcrBox b)
    {
        double w = Math.Min(a.Right, b.Right) - Math.Max(a.Left, b.Left);
        double h = Math.Min(a.Bottom, b.Bottom) - Math.Max(a.Top, b.Top);
        return w > 0 && h > 0 ? w * h : 0;
    }
}
