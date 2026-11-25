// Paragraph breaks inside a block. The page reader cuts blocks at empty bands of half a text height, which is what separates a heading or
// a table from the text around it, but paragraphs set with a smaller gap (a few points of space, or none at all and a first-line indent)
// end up in one block, and then in one paragraph of the page's text and Markdown. Typesetting marks a paragraph in three ways: a gap
// larger than the line spacing, an indented first line, and a last line that stops short before the next paragraph starts
// (Bringhurst, The Elements of Typographic Style, 2.3 and 2.4). Each is read here from the line boxes of a block that is a left-aligned
// run of text; blocks that are tables, rotated, or made of cells are left whole.

namespace OneOcr.Layout;

internal static class ParagraphSplitter
{
    /// <summary>A line spacing larger than this multiple of the block's median spacing starts a paragraph.</summary>
    internal const double GapRatio = 1.2;

    /// <summary>... when it is also larger by at least this share of the median line height.</summary>
    internal const double GapMinExtra = 0.22;

    /// <summary>A first line indented by at least this many line heights past the margin starts a paragraph.</summary>
    internal const double IndentEm = 0.9;

    /// <summary>... and by at most this many (a deeper indent is a quotation or a list).</summary>
    internal const double IndentMaxEm = 4;

    /// <summary>A line that stops short enough for the next line's first word to have fitted, by this many line heights more, ends a paragraph ...</summary>
    internal const double ShortSlackEm = 0.6;

    /// <summary>... when the next line reaches within this share of the measure of the full right edge.</summary>
    internal const double FullShare = 0.1;

    /// <summary>A block is left-aligned when this share of its lines start within <see cref="AlignEm"/> line heights of the margin.</summary>
    internal const double AlignedShare = 0.6;

    /// <summary>Left edges within this many line heights are one margin.</summary>
    internal const double AlignEm = 0.4;

    /// <summary>A block with this share of its lines in several cells is a table-like block and stays whole.</summary>
    internal const double CellShare = 0.25;

    /// <summary>The lines of a block cut into paragraphs: one list of lines per paragraph, in order. A single list when there is no break.</summary>
    public static List<List<OcrLine>> Split(IReadOnlyList<OcrLine> lines)
    {
        int n = lines.Count;
        var whole = new List<List<OcrLine>> { new List<OcrLine>(lines) };
        if (n < 2)
        {
            return whole;
        }

        int multi = 0;
        var heights = new double[n];
        var lefts = new double[n];
        var rights = new double[n];
        for (int i = 0; i < n; i++)
        {
            OcrLine l = lines[i];
            if (l.Quad is not null || l.Words.Count == 0)
            {
                return whole;
            }

            multi += l.Cells.Count > 1 ? 1 : 0;
            heights[i] = l.Box.Height;
            lefts[i] = l.Box.Left;
            rights[i] = l.Box.Right;
        }

        if (multi >= CellShare * n)
        {
            return whole;
        }

        double h = Quantile(heights, 0.5);
        if (h <= 0)
        {
            return whole;
        }

        double margin = Quantile(lefts, 0.1);
        int aligned = 0;
        for (int i = 0; i < n; i++)
        {
            aligned += lefts[i] - margin <= AlignEm * h ? 1 : 0;
        }

        if (aligned < AlignedShare * n)
        {
            return whole;
        }

        double fullRight = Quantile(rights, 0.8);
        double measure = fullRight - margin;
        double pitch = 0;
        bool havePitch = n >= 3;
        if (havePitch)
        {
            var pitches = new double[n - 1];
            for (int i = 0; i + 1 < n; i++)
            {
                pitches[i] = Centre(lines[i + 1]) - Centre(lines[i]);
            }

            pitch = Quantile(pitches, 0.5);
            havePitch = pitch > 0;
        }

        var result = new List<List<OcrLine>>();
        var current = new List<OcrLine> { lines[0] };
        for (int i = 0; i + 1 < n; i++)
        {
            if (Breaks(lines, i, margin, fullRight, measure, h, pitch, havePitch))
            {
                result.Add(current);
                current = [];
            }

            current.Add(lines[i + 1]);
        }

        result.Add(current);
        return result.Count == 1 ? whole : result;
    }

    // Whether a paragraph ends after line i.
    private static bool Breaks(IReadOnlyList<OcrLine> lines, int i, double margin, double fullRight, double measure, double h, double pitch, bool havePitch)
    {
        OcrLine line = lines[i], next = lines[i + 1];

        // A list item or a numbered clause keeps the block it was cut in; the Markdown writer and the reader of the page text own those.
        if (MarkdownRenderer.TryListMarker(next.Text, out _, out _))
        {
            return false;
        }

        // Extra space before the line.
        double step = Centre(next) - Centre(line);
        if (havePitch && step > GapRatio * pitch && step - pitch >= GapMinExtra * h)
        {
            return true;
        }

        // A paragraph ends in a sentence mark; a line that does not is cut by the line width, a heading or a label.
        bool ends = EndsSentence(line.Text);

        // An indented first line.
        int after = i + 2;
        double indent = next.Box.Left - margin;
        if (ends && indent >= IndentEm * h && indent <= IndentMaxEm * h && (after >= lines.Count || lines[after].Box.Left - margin < IndentEm * h))
        {
            return true;
        }

        // A short last line, then a line that fills the measure: the next word would have fitted on the short one.
        if (ends && measure > 0 && next.Words.Count > 0 && next.Box.Right >= fullRight - (FullShare * measure))
        {
            double slack = fullRight - line.Box.Right;
            double first = next.Words[0].Box.Width;
            if (slack > first + (ShortSlackEm * h) && line.Box.Left - margin < IndentEm * h)
            {
                return true;
            }
        }

        return false;
    }

    private static bool EndsSentence(string text)
    {
        string t = text.TrimEnd();
        return t.Length > 0 && t[^1] is '.' or '!' or '?' or ':' or ';' or '"' or '\u201D' or '\u2019' or ')' or ']' or '\u2026';
    }

    private static double Centre(OcrLine l) => (l.Box.Top + l.Box.Bottom) / 2;

    // The value at the given rank (lower of the two middle values for the median): sorted[(count - 1) * q] rounded down.
    private static double Quantile(double[] values, double q)
    {
        double[] sorted = [.. values];
        Array.Sort(sorted);
        return sorted[(int)((sorted.Length - 1) * q)];
    }
}
