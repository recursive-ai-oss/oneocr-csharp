using System.Globalization;
using System.Text;

namespace OneOcr;

/// <summary>
/// Markdown from the page model: headings, paragraphs, lists and pipe tables in reading order, with running headers, footers and
/// page numbers left out. It needs no model and no network and gives the same text for the same page. The rules are written down
/// in <c>docs/spec/markdown-v1.md</c>.
/// </summary>
public static class OcrMarkdown
{
    /// <summary>Renders one page.</summary>
    /// <param name="page">The page.</param>
    /// <param name="options">Rendering options; null for the defaults.</param>
    /// <returns>The Markdown, ending with one line feed; empty when the page has nothing to write.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="page"/> is null.</exception>
    public static string ToMarkdown(this OcrPage page, MarkdownOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(page);
        return new MarkdownRenderer(page, options ?? new MarkdownOptions()).Render();
    }

    /// <summary>Renders pages one after another, a thematic break (<c>---</c>) between pages that have content.</summary>
    /// <param name="pages">The pages in order.</param>
    /// <param name="options">Rendering options; null for the defaults.</param>
    /// <param name="cancellationToken">Checked before each page.</param>
    /// <returns>The Markdown; empty when no page has anything to write.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="pages"/> or one of its pages is null.</exception>
    public static string ToMarkdown(this IEnumerable<OcrPage> pages, MarkdownOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(pages);
        var sb = new StringBuilder();
        foreach (OcrPage page in pages)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string md = ToMarkdown(page, options);
            if (md.Length == 0)
            {
                continue;
            }

            if (sb.Length > 0)
            {
                sb.Append("\n---\n\n");
            }

            sb.Append(md);
        }

        return sb.ToString();
    }
}

internal sealed class MarkdownRenderer
{
    // A line taller than this many times the body's is heading-sized; two sizes within this ratio are one level.
    private const double HeadingRatio = 1.25;
    private const double SameLevelRatio = 0.92;
    private const int MaxHeadingChars = 160;
    private const int MaxHeadingLines = 2;

    // A vertical gap of more than this many body line heights between two lines of a block starts a new paragraph.
    private const double ParagraphGap = 0.75;

    // Indentation steps, in line heights, that make a list item a child or a sibling of the one before.
    private const double NestStep = 0.8;

    // A line indented at least this many line heights past a list item's marker continues the item.
    private const double ContinuationIndent = 0.3;

    private const double FurnitureBand = 0.08;

    private readonly OcrPage _page;
    private readonly MarkdownOptions _options;
    private readonly IReadOnlyList<OcrWord> _words;

    public MarkdownRenderer(OcrPage page, MarkdownOptions options)
    {
        _page = page;
        _options = options;
        _words = page.Words;
    }

    private sealed class Element
    {
        public double Top;
        public string Text = "";
        public bool IsHeading;
        public double HeadingHeight;
    }

    public string Render()
    {
        int count = _page.Blocks.Count;
        var regionOf = new OcrRegionRole?[count];
        foreach (OcrRegion region in _page.Regions)
        {
            foreach (int index in region.BlockIndices)
            {
                if ((uint)index < (uint)count)
                {
                    regionOf[index] ??= region.Role;
                }
            }
        }

        var tableOf = new OcrTable?[count];
        var loose = new List<OcrTable>();
        foreach (OcrTable table in _page.Tables)
        {
            if ((uint)table.BlockIndex < (uint)count && tableOf[table.BlockIndex] is null)
            {
                tableOf[table.BlockIndex] = table;
            }
            else
            {
                loose.Add(table);
            }
        }

        double body = BodyHeight(regionOf);
        var elements = new List<Element>();
        for (int i = 0; i < count; i++)
        {
            OcrBlock block = _page.Blocks[i];
            if (block.Lines.Count == 0)
            {
                continue;
            }

            bool furniture = block.Role is OcrBlockRole.Header or OcrBlockRole.Footer || regionOf[i] is OcrRegionRole.Header or OcrRegionRole.Footer;
            if (furniture && !_options.IncludeHeadersAndFooters)
            {
                continue;
            }

            if (!furniture && _options.DropPageNumbers && IsPageNumber(block))
            {
                continue;
            }

            if (tableOf[i] is { } table)
            {
                AddTable(elements, block.Box.Top, RenderTable(table));
                continue;
            }

            if (block.IsTable)
            {
                string? derived = RenderLineTable(block);
                if (derived is not null)
                {
                    AddTable(elements, block.Box.Top, derived);
                    continue;
                }
            }

            if (IsHeading(block, regionOf[i], body, out double height))
            {
                elements.Add(new Element { Top = block.Box.Top, Text = Escape(JoinLines(block.Lines, 0, block.Lines.Count), false), IsHeading = true, HeadingHeight = height });
                continue;
            }

            AddBody(elements, block, body);
        }

        foreach (OcrTable table in loose)
        {
            string text = RenderTable(table);
            if (text.Length == 0)
            {
                continue;
            }

            int at = elements.FindIndex(e => e.Top > table.Box.Top);
            elements.Insert(at < 0 ? elements.Count : at, new Element { Top = table.Box.Top, Text = text });
        }

        return Assemble(elements);
    }

    private static void AddTable(List<Element> elements, double top, string text)
    {
        if (text.Length > 0)
        {
            elements.Add(new Element { Top = top, Text = text });
        }
    }

    // Heading levels follow type size: the largest heading size is level 1, the next level 2, and so on (sizes within 8% are one size).
    private static string Assemble(List<Element> elements)
    {
        var sizes = new List<double>();
        foreach (Element e in elements)
        {
            if (e.IsHeading)
            {
                sizes.Add(e.HeadingHeight);
            }
        }

        sizes.Sort();
        sizes.Reverse();
        var leads = new List<double>();
        foreach (double s in sizes)
        {
            if (leads.Count == 0 || s < leads[^1] * SameLevelRatio)
            {
                leads.Add(s);
            }
        }

        var sb = new StringBuilder();
        foreach (Element e in elements)
        {
            if (e.Text.Length == 0)
            {
                continue;
            }

            if (sb.Length > 0)
            {
                sb.Append("\n\n");
            }

            if (e.IsHeading)
            {
                int level = 1;
                while (level < leads.Count && e.HeadingHeight < leads[level - 1] * SameLevelRatio)
                {
                    level++;
                }

                sb.Append('#', Math.Min(level, 6)).Append(' ');
            }

            sb.Append(e.Text);
        }

        if (sb.Length > 0)
        {
            sb.Append('\n');
        }

        return sb.ToString();
    }

    // ----- geometry -----

    private static bool IsRotated(OcrLine line) => line.Quad is not null;

    private static double LineHeight(OcrLine line)
    {
        if (line.Quad is { } q)
        {
            double dx = q.P3.X - q.P0.X, dy = q.P3.Y - q.P0.Y;
            return Math.Sqrt((dx * dx) + (dy * dy));
        }

        return line.Box.Height;
    }

    private static double Median(List<double> values)
    {
        if (values.Count == 0)
        {
            return 0;
        }

        values.Sort();
        return values[values.Count / 2];
    }

    private double BodyHeight(OcrRegionRole?[] regionOf)
    {
        var heights = new List<double>();
        var all = new List<double>();
        for (int i = 0; i < _page.Blocks.Count; i++)
        {
            OcrBlock b = _page.Blocks[i];
            foreach (OcrLine l in b.Lines)
            {
                all.Add(LineHeight(l));
                if (b.Role == OcrBlockRole.Body && !b.IsTable && regionOf[i] is null or OcrRegionRole.Text or OcrRegionRole.List)
                {
                    heights.Add(LineHeight(l));
                }
            }
        }

        return Median(heights.Count > 0 ? heights : all);
    }

    // ----- headings -----

    private bool IsHeading(OcrBlock block, OcrRegionRole? region, double body, out double height)
    {
        var heights = new List<double>(block.Lines.Count);
        foreach (OcrLine l in block.Lines)
        {
            heights.Add(LineHeight(l));
        }

        height = Median(heights);
        if (region == OcrRegionRole.Title)
        {
            return true;
        }

        if (!_options.HeadingsFromSize || region is not null || block.Lines.Count > MaxHeadingLines || body <= 0 || height < HeadingRatio * body)
        {
            return false;
        }

        int chars = 0;
        foreach (OcrLine l in block.Lines)
        {
            chars += l.Text.Length;
        }

        string last = block.Lines[^1].Text.TrimEnd();
        return chars <= MaxHeadingChars && last.Length > 0 && last[^1] is not (',' or ';') && !TryListMarker(block.Lines[0].Text, out _, out _);
    }

    // ----- paragraphs and lists -----

    private sealed class Item
    {
        public int Level;
        public string? Number;
        public readonly List<OcrLine> Lines = [];
        public double Left;
        public double Height;
        public string Text = "";
    }

    private void AddBody(List<Element> elements, OcrBlock block, double body)
    {
        var para = new List<OcrLine>();
        List<Item>? items = null;
        var stack = new List<double>();
        double top = block.Box.Top;

        void FlushParagraph()
        {
            if (para.Count > 0)
            {
                elements.Add(new Element { Top = para[0].Box.Top, Text = Escape(JoinLines(para, 0, para.Count), true) });
                para.Clear();
            }
        }

        void FlushList()
        {
            if (items is { Count: > 0 })
            {
                var sb = new StringBuilder();
                foreach (Item it in items)
                {
                    if (sb.Length > 0)
                    {
                        sb.Append('\n');
                    }

                    sb.Append(' ', 4 * it.Level).Append(it.Number is null ? "- " : it.Number + ". ").Append(Escape(it.Text, false));
                }

                elements.Add(new Element { Top = top, Text = sb.ToString() });
            }

            items = null;
            stack.Clear();
        }

        for (int i = 0; i < block.Lines.Count; i++)
        {
            OcrLine line = block.Lines[i];
            double h = LineHeight(line);
            if (_options.Lists && TryListMarker(line.Text, out string? number, out string rest))
            {
                FlushParagraph();
                items ??= [];
                double left = IsRotated(line) ? 0 : line.Box.Left;
                while (stack.Count > 0 && left < stack[^1] - (NestStep * h))
                {
                    stack.RemoveAt(stack.Count - 1);
                }

                if (stack.Count == 0 || left > stack[^1] + (NestStep * h))
                {
                    stack.Add(left);
                }

                var item = new Item { Level = stack.Count - 1, Number = number, Left = left, Height = h, Text = rest };
                item.Lines.Add(line);
                items.Add(item);
                continue;
            }

            if (items is { Count: > 0 })
            {
                Item last = items[^1];
                OcrLine previous = last.Lines[^1];
                bool indented = !IsRotated(line) && line.Box.Left >= last.Left + (ContinuationIndent * last.Height);
                bool near = IsRotated(line) || line.Box.Top - previous.Box.Bottom <= 1.5 * Math.Max(body, last.Height);
                if (indented && near)
                {
                    last.Lines.Add(line);
                    last.Text = Join(last.Text, line.Text);
                    continue;
                }

                FlushList();
            }

            if (para.Count > 0 && !IsRotated(line) && !IsRotated(para[^1]) && line.Box.Top - para[^1].Box.Bottom > ParagraphGap * Math.Max(body, h))
            {
                FlushParagraph();
            }

            para.Add(line);
        }

        FlushParagraph();
        FlushList();
    }

    private static readonly char[] BulletGlyphs = ['\u2022', '\u25CF', '\u25CB', '\u25E6', '\u25AA', '\u25AB', '\u25A0', '\u25A1', '\u2023', '\u2043', '\u2219', '\u00B7'];

    // A list marker at the start of a line: a bullet glyph (the text may follow it directly), "-", "*", "+" or an en dash followed by
    // a space, or one to three digits followed by "." or ")" and a space, or the same in parentheses.
    internal static bool TryListMarker(string text, out string? number, out string rest)
    {
        number = null;
        rest = "";
        ReadOnlySpan<char> t = text.AsSpan().TrimStart();
        if (t.Length < 2)
        {
            return false;
        }

        if (BulletGlyphs.AsSpan().Contains(t[0]))
        {
            rest = t[1..].TrimStart().ToString().TrimEnd();
            return rest.Length > 0;
        }

        if (t[0] is '-' or '*' or '+' or '\u2013')
        {
            if (!char.IsWhiteSpace(t[1]))
            {
                return false;
            }

            rest = t[2..].TrimStart().ToString().TrimEnd();
            return rest.Length > 0;
        }

        int p = 0;
        bool parenthesized = t[0] == '(';
        if (parenthesized)
        {
            p = 1;
        }

        int start = p;
        while (p < t.Length && p - start < 4 && char.IsAsciiDigit(t[p]))
        {
            p++;
        }

        int digits = p - start;
        if (digits is < 1 or > 3 || p >= t.Length - 1)
        {
            return false;
        }

        bool closed = parenthesized ? t[p] == ')' : t[p] is '.' or ')';
        if (!closed || !char.IsWhiteSpace(t[p + 1]))
        {
            return false;
        }

        rest = t[(p + 1)..].TrimStart().ToString().TrimEnd();
        if (rest.Length == 0)
        {
            return false;
        }

        number = t.Slice(start, digits).ToString();
        return true;
    }

    // Lines of a paragraph joined with one space; a hyphen at the end of a line that follows a letter, with a lower-case letter
    // starting the next line, is a hyphenation break and is removed.
    private static string JoinLines(IReadOnlyList<OcrLine> lines, int from, int to)
    {
        string text = "";
        for (int i = from; i < to; i++)
        {
            text = Join(text, lines[i].Text);
        }

        return text;
    }

    private static string Join(string text, string next)
    {
        string n = next.Replace('\t', ' ').Trim();
        if (n.Length == 0)
        {
            return text;
        }

        if (text.Length == 0)
        {
            return n;
        }

        if (text.Length >= 2 && text[^1] == '-' && char.IsLetter(text[^2]) && char.IsLower(n[0]))
        {
            return text[..^1] + n;
        }

        return text + " " + n;
    }

    // Escapes what Markdown would read as markup: \ ` * _ [ ] < > everywhere, and at the start of a line # > + - = ~ and the dot or
    // bracket of "1." / "1)".
    private static string Escape(string text, bool atLineStart)
    {
        int orderedAt = atLineStart ? OrderedMarkerAt(text) : -1;
        var sb = new StringBuilder(text.Length + 8);
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c is '\\' or '`' or '*' or '_' or '[' or ']' or '<' or '>')
            {
                sb.Append('\\');
            }
            else if (atLineStart && ((i == 0 && c is '#' or '+' or '-' or '=' or '~') || i == orderedAt))
            {
                sb.Append('\\');
            }

            sb.Append(c);
        }

        return sb.ToString();
    }

    // The index of the "." or ")" of a leading "123. " / "123) ", or -1.
    private static int OrderedMarkerAt(string text)
    {
        int p = 0;
        while (p < text.Length && char.IsAsciiDigit(text[p]))
        {
            p++;
        }

        return p is > 0 and <= 9 && p < text.Length - 1 && text[p] is '.' or ')' && text[p + 1] == ' ' ? p : -1;
    }

    // ----- page numbers -----

    private static readonly string[] PageTotalSeparators = ["of", "/"];

    private bool IsPageNumber(OcrBlock block)
    {
        if (block.Lines.Count != 1 || _page.Orientation != 0 || _page.SkewDegrees != 0 || _page.Height <= 0)
        {
            return false;
        }

        if (block.Box.Bottom > FurnitureBand * _page.Height && block.Box.Top < (1 - FurnitureBand) * _page.Height)
        {
            return false;
        }

        string t = block.Lines[0].Text.Trim().ToLowerInvariant();
        if (t.StartsWith("page ", StringComparison.Ordinal))
        {
            t = t[5..].TrimStart();
        }

        if (t.Length >= 3 && t[0] is '-' or '\u2013' or '\u2014' && t[^1] is '-' or '\u2013' or '\u2014')
        {
            t = t[1..^1].Trim();
        }

        int p = 0;
        while (p < t.Length && char.IsAsciiDigit(t[p]) && p < 5)
        {
            p++;
        }

        if (p == 0)
        {
            return false;
        }

        if (p == t.Length)
        {
            return true;
        }

        string tail = t[p..].Trim();
        foreach (string sep in PageTotalSeparators)
        {
            if (tail.StartsWith(sep, StringComparison.Ordinal))
            {
                string total = tail[sep.Length..].Trim();
                return total.Length is > 0 and <= 5 && total.All(char.IsAsciiDigit);
            }
        }

        return false;
    }

    // ----- tables -----

    private string RenderTable(OcrTable table)
    {
        if (table.Rows < 1 || table.Columns < 1 || (long)table.Rows * table.Columns > 1_000_000)
        {
            return "";
        }

        var grid = new string[table.Rows, table.Columns];
        foreach (OcrTableCell cell in table.Cells)
        {
            if ((uint)cell.Row >= (uint)table.Rows || (uint)cell.Column >= (uint)table.Columns)
            {
                continue;
            }

            var sb = new StringBuilder();
            foreach (int w in cell.WordIndices)
            {
                if ((uint)w < (uint)_words.Count)
                {
                    if (sb.Length > 0)
                    {
                        sb.Append(' ');
                    }

                    sb.Append(_words[w].Text);
                }
            }

            // Only the cell's top-left position holds its text: pipe tables have no spans, so the cells a span covers stay empty.
            grid[cell.Row, cell.Column] = sb.ToString();
        }

        var rows = new List<string[]>(table.Rows);
        for (int r = 0; r < table.Rows; r++)
        {
            var row = new string[table.Columns];
            for (int c = 0; c < table.Columns; c++)
            {
                row[c] = grid[r, c] ?? "";
            }

            rows.Add(row);
        }

        return PipeTable(rows);
    }

    // A table block that has no table entry: each line is a row, its cells split where Cells says.
    private static string? RenderLineTable(OcrBlock block)
    {
        var rows = new List<string[]>(block.Lines.Count);
        foreach (OcrLine line in block.Lines)
        {
            if (line.Cells.Count == 0)
            {
                continue;
            }

            var cells = new string[line.Cells.Count];
            for (int c = 0; c < cells.Length; c++)
            {
                int from = line.Cells[c], to = c + 1 < cells.Length ? line.Cells[c + 1] : line.Words.Count;
                var sb = new StringBuilder();
                for (int w = Math.Max(0, from); w < Math.Min(to, line.Words.Count); w++)
                {
                    if (sb.Length > 0)
                    {
                        sb.Append(' ');
                    }

                    sb.Append(line.Words[w].Text);
                }

                cells[c] = sb.ToString();
            }

            rows.Add(cells);
        }

        string text = PipeTable(rows);
        return text.Length == 0 ? null : text;
    }

    private static string PipeTable(List<string[]> rows)
    {
        int columns = 0;
        foreach (string[] r in rows)
        {
            columns = Math.Max(columns, r.Length);
        }

        // Columns with nothing in them are left out.
        var keep = new List<int>();
        for (int c = 0; c < columns; c++)
        {
            if (rows.Any(r => c < r.Length && r[c].Length > 0))
            {
                keep.Add(c);
            }
        }

        if (keep.Count == 0)
        {
            return "";
        }

        var sb = new StringBuilder();
        for (int r = 0; r < rows.Count; r++)
        {
            if (r > 0)
            {
                sb.Append('\n');
            }

            sb.Append('|');
            foreach (int c in keep)
            {
                string cell = c < rows[r].Length ? rows[r][c] : "";
                sb.Append(' ').Append(Escape(cell, false).Replace("|", "\\|", StringComparison.Ordinal)).Append(" |");
            }

            if (r == 0)
            {
                sb.Append("\n|");
                for (int k = 0; k < keep.Count; k++)
                {
                    sb.Append(" --- |");
                }
            }
        }

        return sb.ToString();
    }
}
