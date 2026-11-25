// Running headers and footers of pages read from pixels. A page from a PDF text layer says which blocks are furniture; a page read from an
// image has to be told by its geometry. Two sources of evidence, both conservative because Markdown leaves furniture out:
//
// * Within one page: a block of one or two lines in the outermost band of the page (the top or bottom 7.5%), set off from the text by
//   at least a line height and no larger than the body text. A heading sits near the top of a page too but is larger than the body,
//   and receipts and screens have no body to be set off from, so they are left alone.
// * Across the pages of one document: a block of one or two lines in the outer 12% whose text is the same on several pages once digit
//   runs are folded (a page number, "Page 3 of 9", a running title) and that stands at the same place. Furniture repeats; content does not.

namespace OneOcr.Layout;

internal static class Furniture
{
    /// <summary>The outer share of the page height in which a block can be furniture by its place alone.</summary>
    internal const double Band = 0.075;

    /// <summary>The outer share of the page height searched for blocks that repeat across pages.</summary>
    internal const double RepeatBand = 0.12;

    /// <summary>A furniture block is set off from the text by at least this many body line heights.</summary>
    internal const double GapEm = 1.0;

    /// <summary>... and its lines are at most this many times as tall as the body's.</summary>
    internal const double MaxSizeRatio = 1.15;

    /// <summary>The page needs at least this many lines outside the furniture to have a body to set it against.</summary>
    internal const int MinBodyLines = 6;

    /// <summary>A furniture block has at most this many lines.</summary>
    internal const int MaxLines = 2;

    /// <summary>A furniture block has at most this many characters.</summary>
    internal const int MaxChars = 120;

    /// <summary>Blocks that repeat across pages stand within this share of the page height of each other, vertically ...</summary>
    internal const double RepeatTolerance = 0.02;

    /// <summary>... and within this share of the page width at one of their edges or at their centre.</summary>
    internal const double RepeatTolerance2 = 0.03;

    /// <summary>A block repeats on at least this share of the pages (and on at least two).</summary>
    internal const double RepeatShare = 0.4;

    /// <summary>Marks the furniture of one page by its place on the page. The page is returned as it is when nothing changes.</summary>
    public static OcrPage MarkWithinPage(OcrPage page)
    {
        if (page.Orientation != 0 || Math.Abs(page.SkewDegrees) > 2 || page.Height <= 0 || page.Blocks.Count < 2)
        {
            return page;
        }

        int count = page.Blocks.Count;
        var roles = new OcrBlockRole[count];
        var candidate = new bool[count];
        var side = new int[count];
        int lines = 0;
        var heights = new List<double>();
        for (int i = 0; i < count; i++)
        {
            OcrBlock b = page.Blocks[i];
            roles[i] = b.Role;
            if (b.Role == OcrBlockRole.Body && IsCandidate(b))
            {
                if (b.Box.Bottom <= Band * page.Height)
                {
                    candidate[i] = true;
                    side[i] = -1;
                    continue;
                }

                if (b.Box.Top >= (1 - Band) * page.Height)
                {
                    candidate[i] = true;
                    side[i] = 1;
                    continue;
                }
            }

            if (b.Role == OcrBlockRole.Body)
            {
                foreach (OcrLine l in b.Lines)
                {
                    lines++;
                    heights.Add(l.Box.Height);
                }
            }
        }

        if (lines < MinBodyLines)
        {
            return page;
        }

        heights.Sort();
        double body = heights[heights.Count / 2];
        bool changed = false;
        for (int i = 0; i < count; i++)
        {
            if (!candidate[i])
            {
                continue;
            }

            OcrBlock b = page.Blocks[i];
            if (MedianHeight(b) > MaxSizeRatio * body)
            {
                continue;
            }

            // The distance to the nearest other block on the page's side of it.
            double gap = double.MaxValue;
            for (int j = 0; j < count; j++)
            {
                if (j == i || page.Blocks[j].Lines.Count == 0)
                {
                    continue;
                }

                OcrBox o = page.Blocks[j].Box;
                if (side[i] < 0 && o.Top >= b.Box.Bottom)
                {
                    gap = Math.Min(gap, o.Top - b.Box.Bottom);
                }
                else if (side[i] > 0 && o.Bottom <= b.Box.Top)
                {
                    gap = Math.Min(gap, b.Box.Top - o.Bottom);
                }
            }

            if (gap != double.MaxValue && gap >= GapEm * body)
            {
                roles[i] = side[i] < 0 ? OcrBlockRole.Header : OcrBlockRole.Footer;
                changed = true;
            }
        }

        return changed ? WithRoles(page, roles) : page;
    }

    /// <summary>Marks the blocks that repeat across the pages of a document. Pages that change come back as new pages; the others are the same instances.</summary>
    public static IReadOnlyList<OcrPage> MarkAcrossPages(IReadOnlyList<OcrPage> pages)
    {
        if (pages.Count < 2)
        {
            return pages;
        }

        var members = new List<Member>();
        for (int p = 0; p < pages.Count; p++)
        {
            OcrPage page = pages[p];
            if (page.Orientation != 0 || page.Height <= 0 || page.Width <= 0)
            {
                continue;
            }

            for (int b = 0; b < page.Blocks.Count; b++)
            {
                OcrBlock block = page.Blocks[b];
                if (block.Role != OcrBlockRole.Body || !IsCandidate(block))
                {
                    continue;
                }

                int side = block.Box.Bottom <= RepeatBand * page.Height ? -1 : block.Box.Top >= (1 - RepeatBand) * page.Height ? 1 : 0;
                string key = FoldDigits(block);
                if (side != 0 && key.Any(static c => char.IsLetterOrDigit(c) || c == '#'))
                {
                    members.Add(new Member(p, b, side, key, block.Box, page.Width, page.Height));
                }
            }
        }

        int needed = Math.Max(2, (int)Math.Ceiling(RepeatShare * pages.Count));
        var marked = new HashSet<(int Page, int Block)>();
        foreach (IGrouping<(int Side, string Key), Member> group in members.GroupBy(static m => (m.Side, m.Key)))
        {
            var clusters = new List<List<Member>>();
            foreach (Member m in group)
            {
                List<Member>? home = clusters.Find(c => SamePlace(c[0], m));
                if (home is null)
                {
                    clusters.Add([m]);
                }
                else
                {
                    home.Add(m);
                }
            }

            foreach (List<Member> cluster in clusters)
            {
                if (cluster.Select(static m => m.Page).Distinct().Count() >= needed)
                {
                    foreach (Member m in cluster)
                    {
                        marked.Add((m.Page, m.Block));
                    }
                }
            }
        }

        if (marked.Count == 0)
        {
            return pages;
        }

        var result = new OcrPage[pages.Count];
        for (int p = 0; p < pages.Count; p++)
        {
            OcrPage page = pages[p];
            OcrBlockRole[]? roles = null;
            foreach (Member m in members)
            {
                if (m.Page == p && marked.Contains((p, m.Block)))
                {
                    roles ??= [.. page.Blocks.Select(static b => b.Role)];
                    roles[m.Block] = m.Side < 0 ? OcrBlockRole.Header : OcrBlockRole.Footer;
                }
            }

            result[p] = roles is null ? page : WithRoles(page, roles);
        }

        return result;
    }

    private sealed record Member(int Page, int Block, int Side, string Key, OcrBox Box, double Width, double Height);

    private static bool SamePlace(Member a, Member b)
    {
        double ay = (a.Box.Top + a.Box.Bottom) / 2 / a.Height, by = (b.Box.Top + b.Box.Bottom) / 2 / b.Height;
        if (Math.Abs(ay - by) > RepeatTolerance)
        {
            return false;
        }

        double t = RepeatTolerance2;
        return Math.Abs((a.Box.Left / a.Width) - (b.Box.Left / b.Width)) <= t
            || Math.Abs((a.Box.Right / a.Width) - (b.Box.Right / b.Width)) <= t
            || Math.Abs(((a.Box.Left + a.Box.Right) / 2 / a.Width) - ((b.Box.Left + b.Box.Right) / 2 / b.Width)) <= t;
    }

    // The block's text in lower case with every run of digits as one '#', words separated by single spaces.
    private static string FoldDigits(OcrBlock block)
    {
        var sb = new System.Text.StringBuilder();
        foreach (OcrLine l in block.Lines)
        {
            if (sb.Length > 0)
            {
                sb.Append(' ');
            }

            bool digits = false, space = true;
            foreach (char c in l.Text)
            {
                if (char.IsAsciiDigit(c))
                {
                    if (!digits)
                    {
                        sb.Append('#');
                    }

                    digits = true;
                    space = false;
                }
                else if (char.IsWhiteSpace(c))
                {
                    if (!space)
                    {
                        sb.Append(' ');
                    }

                    space = true;
                    digits = false;
                }
                else
                {
                    sb.Append(char.ToLowerInvariant(c));
                    digits = false;
                    space = false;
                }
            }
        }

        return sb.ToString().Trim();
    }

    private static bool IsCandidate(OcrBlock b)
    {
        if (b.IsTable || b.Source != OcrSource.Ocr || b.Lines.Count is 0 or > MaxLines)
        {
            return false;
        }

        int chars = 0;
        foreach (OcrLine l in b.Lines)
        {
            if (l.Quad is not null)
            {
                return false;
            }

            chars += l.Text.Length;
        }

        return chars <= MaxChars;
    }

    private static double MedianHeight(OcrBlock b)
    {
        var h = new List<double>();
        foreach (OcrLine l in b.Lines)
        {
            h.Add(l.Box.Height);
        }

        h.Sort();
        return h[h.Count / 2];
    }

    private static OcrPage WithRoles(OcrPage page, OcrBlockRole[] roles)
    {
        var blocks = new OcrBlock[page.Blocks.Count];
        for (int i = 0; i < blocks.Length; i++)
        {
            blocks[i] = roles[i] == page.Blocks[i].Role ? page.Blocks[i] : page.Blocks[i] with { Role = roles[i] };
        }

        return new OcrPage(
            page.Index, page.Width, page.Height, page.UnitsPerInch, blocks, page.Regions, page.Tables, page.KeyValues, page.Diagnostics, page.ToPdf, page.Orientation, page.SkewDegrees);
    }
}
