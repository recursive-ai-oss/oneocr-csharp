using OneImage;
using OnePdf;
using OnePdf.Text;

namespace OneOcr.Pdf;

/// <summary>What a page read needs besides the page itself.</summary>
internal sealed class ReadContext
{
    public required OcrEngine Engine { get; init; }
    public required PdfOcrOptions Options { get; init; }
    public required PdfRenderer Renderer { get; init; }

    /// <summary>Threads the recognizer may use for one region.</summary>
    public int RegionDegree { get; init; }
}

/// <summary>
/// Reads one PDF page into the page model: the text layer where it can be trusted, the local recognizer on pixels where it cannot
/// or where there is no text, merged into one page with a <see cref="OcrSource"/> on every element.
/// </summary>
internal static class PdfPageReader
{
    private static readonly PdfTextOptions TextOptions = new() { Mode = PdfTextMode.ReadingOrder, RetainGlyphs = true };
    private static readonly PdfInterpretOptions InterpretOptions = new() { Annotations = PdfAnnotationMode.None, GlyphOutlines = false };

    // Two boxes "are the same word" when the overlap is at least this share of the smaller one.
    private const double SameWordOverlap = 0.4;

    // An image is already read by the text layer when words sit on at least this share of its area.
    private const double CoveredShare = 0.10;

    // A sampled line disagrees with the render when more than this share of its letters and digits differ.
    private const double LineDisagreement = 0.4;

    private sealed class TextWord
    {
        public required PdfTextWord Source;
        public required OcrBox Box;
        public bool Visible;
        public bool Garbled;
        public bool Keep = true;
    }

    private sealed class TextLine
    {
        public required PdfTextLine Source;
        public required List<TextWord> Words;
    }

    private sealed class TextBlock
    {
        public required PdfTextBlock Source;
        public required List<TextLine> Lines;
    }

    private sealed record Region(OcrBox Box, string Why);

    public static OcrPage Read(PdfDocument document, int pageIndex, ReadContext ctx, CancellationToken cancellationToken)
    {
        PdfOcrOptions options = ctx.Options;
        var diagnostics = new List<string>();
        PdfPage page = document.Pages[pageIndex];
        double userUnit = page.UserUnit;
        PdfRectangle box = page.CropBox;
        int rotate = page.Rotate;
        (double width, double height) = PdfGeometry.Size(box, rotate, userUnit);
        OcrMatrix toPdf = PdfGeometry.ToPdf(box, rotate, userUnit);

        // 1. The text layer.
        PdfPageText? text = null;
        try
        {
            text = page.ExtractText(TextOptions, cancellationToken);
        }
        catch (Exception e) when (PdfErrors.IsLibraryFailure(e))
        {
            PdfErrors.Fail(e, options, diagnostics, "The text layer could not be read");
        }

        List<TextBlock> blocks = [];
        if (text is not null)
        {
            width = text.Width;
            height = text.Height;
            foreach (PdfDiagnostic d in text.Diagnostics)
            {
                if (d.Severity != PdfSeverity.Info)
                {
                    diagnostics.Add(d.Message);
                }
            }

            blocks = Collect(text, height, ctx.Engine.Limits);
        }

        List<TextWord> all = [.. blocks.SelectMany(b => b.Lines).SelectMany(l => l.Words)];

        // 2. Invisible text is an OCR layer from somebody else; garbled text is not text.
        int invisible = 0, garbledLines = 0;
        foreach (TextWord w in all)
        {
            if (!w.Visible && !options.TrustExistingOcrLayer)
            {
                w.Keep = false;
                invisible++;
            }
        }

        if (invisible > 0)
        {
            diagnostics.Add($"{invisible} invisible text words (an existing OCR layer) were dropped and the page was read again; set TrustExistingOcrLayer to keep them.");
        }
        else if (options.TrustExistingOcrLayer && all.Any(w => !w.Visible))
        {
            diagnostics.Add("An existing invisible OCR layer was kept as it is (TrustExistingOcrLayer).");
        }

        var regions = new List<Region>();
        bool canRender = options.ReadImages;
        ImageFrame? frame = null;
        double sx = 1, sy = 1;
        if (canRender)
        {
            try
            {
                frame = ctx.Renderer.Frame(pageIndex);
                sx = frame.Width / width;
                sy = frame.Height / height;
                if (!(sx > 0 && sy > 0 && double.IsFinite(sx) && double.IsFinite(sy)))
                {
                    throw new ImageFormatException("The page has no size.");
                }
            }
            catch (ImageException e)
            {
                canRender = false;
                if (frame is not null)
                {
                    ctx.Renderer.Done();
                }

                frame = null;
                PdfErrors.Fail(e, options, diagnostics, "The page could not be rendered, so image regions and garbled text are not read");
            }
        }

        // Garbled lines are read from pixels instead.
        foreach (TextBlock b in blocks)
        {
            foreach (TextLine l in b.Lines)
            {
                bool garbled = false;
                foreach (TextWord w in l.Words)
                {
                    if (w.Keep && w.Visible && IsGarbled(w.Source.Text))
                    {
                        w.Garbled = true;
                        garbled = true;
                    }
                }

                if (garbled)
                {
                    garbledLines++;
                    foreach (TextWord w in l.Words)
                    {
                        w.Keep = false;
                    }

                    if (canRender)
                    {
                        regions.Add(new Region(Inflate(Union(l.Words), 0.15, width, height), "garbled text"));
                    }
                }
            }
        }

        if (garbledLines > 0)
        {
            diagnostics.Add(canRender
                ? $"{garbledLines} lines with unreadable characters (U+FFFD or private-use code points) were read from pixels instead."
                : $"{garbledLines} lines with unreadable characters were dropped because the page cannot be rendered.");
        }

        // 3. Does the text layer say what the page shows? A font whose character map is wrong gives clean, wrong text.
        bool wholePage = false;
        if (canRender && frame is not null && options.CrossCheckLines > 0)
        {
            wholePage = CrossCheck(blocks, frame, sx, sy, width, height, ctx, diagnostics, cancellationToken);
            if (wholePage)
            {
                foreach (TextWord w in all)
                {
                    w.Keep = false;
                }

                regions.Clear();
            }
        }

        // 4. Image regions the text does not already cover, or the whole page.
        List<OcrBox> images = [];
        if (canRender && !wholePage)
        {
            images = FindImages(page, width, height, options, diagnostics, cancellationToken);
            var kept = all.Where(w => w.Keep).ToList();
            foreach (OcrBox image in images)
            {
                if (!Covered(image, kept))
                {
                    regions.Add(new Region(image, "image"));
                }
            }
        }

        if (wholePage)
        {
            regions.Add(new Region(new OcrBox(0, 0, width, height), "page"));
        }

        // 5. Read the regions.
        var groups = new List<Group>();
        if (canRender && frame is not null)
        {
            try
            {
                foreach (Region region in regions)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    try
                    {
                        Group? g = ReadRegion(region, frame, sx, sy, ctx, pageIndex, cancellationToken);
                        if (g is not null)
                        {
                            groups.Add(g);
                        }
                    }
                    catch (OcrLimitException e) when (!options.Strict)
                    {
                        diagnostics.Add($"A {region.Why} region was not read: {e.Message}");
                    }
                    catch (Exception e) when (PdfErrors.IsLibraryFailure(e))
                    {
                        PdfErrors.Fail(e, options, diagnostics, $"A {region.Why} region could not be rendered");
                    }
                }
            }
            finally
            {
                // The page's rendered pixels are not needed again.
                ctx.Renderer.Done();
            }
        }

        // 6. Merge.
        return Merge(pageIndex, width, height, toPdf, blocks, groups, options, text, diagnostics, ctx);
    }

    private sealed class Group
    {
        public required List<OcrBlock> Blocks;
        public required List<OcrRegion> Regions;
        public required List<OcrTable> Tables;
        public required List<OcrKeyValue> KeyValues;
        public double Top;
        public double Left;
    }

    private static Group? ReadRegion(
        Region region, ImageFrame frame, double sx, double sy, ReadContext ctx, int pageIndex, CancellationToken cancellationToken)
    {
        int x0 = Math.Max(0, (int)Math.Floor(region.Box.Left * sx));
        int y0 = Math.Max(0, (int)Math.Floor(region.Box.Top * sy));
        int x1 = Math.Min(frame.Width, (int)Math.Ceiling(region.Box.Right * sx));
        int y1 = Math.Min(frame.Height, (int)Math.Ceiling(region.Box.Bottom * sy));
        if (x1 - x0 < 4 || y1 - y0 < 4)
        {
            return null;
        }

        Image crop = frame.ToImage().Crop(new ImageRect(x0, y0, x1 - x0, y1 - y0));
        OcrPage read = ctx.Engine.Ocr(
            crop,
            new OcrOptions { Dpi = ctx.Options.RenderDpi, IncludeChars = ctx.Options.IncludeChars, MaxDegreeOfParallelism = ctx.RegionDegree, PageIndex = pageIndex },
            cancellationToken);
        if (read.Blocks.Count == 0)
        {
            return null;
        }

        var a = new PageParts.Affine(1 / sx, 1 / sy, x0 / sx, y0 / sy);
        var blocks = new List<OcrBlock>(read.Blocks.Count);
        foreach (OcrBlock b in read.Blocks)
        {
            blocks.Add(PageParts.Map(b, a));
        }

        var regions = new List<OcrRegion>(read.Regions.Count);
        foreach (OcrRegion r in read.Regions)
        {
            regions.Add(PageParts.Map(r, a));
        }

        var tables = new List<OcrTable>(read.Tables.Count);
        foreach (OcrTable t in read.Tables)
        {
            tables.Add(PageParts.Map(t, a));
        }

        var keyValues = new List<OcrKeyValue>(read.KeyValues.Count);
        foreach (OcrKeyValue k in read.KeyValues)
        {
            keyValues.Add(PageParts.Map(k, a));
        }

        OcrBox first = blocks[0].Box;
        return new Group { Blocks = blocks, Regions = regions, Tables = tables, KeyValues = keyValues, Top = first.Top, Left = first.Left };
    }

    private static OcrPage Merge(
        int pageIndex, double width, double height, OcrMatrix toPdf, List<TextBlock> textBlocks, List<Group> groups,
        PdfOcrOptions options, PdfPageText? text, List<string> diagnostics, ReadContext ctx)
    {
        var keptWords = new List<OcrBox>();
        var finalBlocks = new List<OcrBlock>();
        var textOut = new List<(OcrBlock Block, double Top)>();
        foreach (TextBlock tb in textBlocks)
        {
            var lines = new List<OcrLine>();
            foreach (TextLine tl in tb.Lines)
            {
                var words = new List<OcrWord>();
                var cells = new List<int>();
                int nextCell = 0;
                for (int i = 0; i < tl.Words.Count; i++)
                {
                    bool startsCell = nextCell < tl.Source.CellStarts.Count && tl.Source.CellStarts[nextCell] == i;
                    if (startsCell)
                    {
                        nextCell++;
                    }

                    TextWord w = tl.Words[i];
                    if (!w.Keep)
                    {
                        if (startsCell)
                        {
                            cells.Add(words.Count);
                        }

                        continue;
                    }

                    if (startsCell || cells.Count == 0)
                    {
                        cells.Add(words.Count);
                    }

                    words.Add(MakeWord(w, text, height, options.IncludeChars));
                    keptWords.Add(w.Box);
                }

                if (words.Count == 0)
                {
                    continue;
                }

                var distinct = new List<int>();
                foreach (int c in cells)
                {
                    if (c < words.Count && (distinct.Count == 0 || c > distinct[^1]))
                    {
                        distinct.Add(c);
                    }
                }

                if (distinct.Count == 0 || distinct[0] != 0)
                {
                    distinct.Insert(0, 0);
                }

                lines.Add(PageParts.MakeLine(
                    words, distinct, OcrSource.TextLayer, tl.Source.Direction == PdfTextDirection.RightToLeft, tl.Source.IsVertical));
            }

            if (lines.Count > 0)
            {
                OcrBlockRole role = tb.Source.Role switch
                {
                    PdfTextRole.Header => OcrBlockRole.Header,
                    PdfTextRole.Footer => OcrBlockRole.Footer,
                    _ => OcrBlockRole.Body,
                };
                OcrBlock block = PageParts.MakeBlock(lines, role, tb.Source.IsTable, OcrSource.TextLayer);
                textOut.Add((block, block.Box.Top));
            }
        }

        // Recognized words that sit on kept text-layer words are the same words (text drawn over a picture, or a region that
        // reaches into text); the text layer wins. Recognized words from two regions that overlap are the same words too.
        var accepted = new List<OcrBox>(keptWords);
        var groupBlocks = new List<Group>();
        int duplicates = 0;
        foreach (Group g in groups)
        {
            bool Keep(OcrWord w)
            {
                foreach (OcrBox k in accepted)
                {
                    double inter = PageParts.IntersectionArea(w.Box, k);
                    if (inter > 0 && inter >= SameWordOverlap * Math.Min(PageParts.Area(w.Box), PageParts.Area(k)))
                    {
                        return false;
                    }
                }

                return true;
            }

            List<OcrBlock> filtered = PageParts.Filter(g.Blocks, Keep, out int removed);
            duplicates += removed;
            if (filtered.Count == 0)
            {
                continue;
            }

            Group ng = removed == 0
                ? g
                : new Group { Blocks = filtered, Regions = [], Tables = [], KeyValues = [], Top = filtered[0].Box.Top, Left = filtered[0].Box.Left };
            foreach (OcrBlock b in ng.Blocks)
            {
                foreach (OcrLine l in b.Lines)
                {
                    foreach (OcrWord w in l.Words)
                    {
                        accepted.Add(w.Box);
                    }
                }
            }

            groupBlocks.Add(ng);
        }

        if (duplicates > 0)
        {
            diagnostics.Add($"{duplicates} recognized words that repeat text-layer words were left out; tables of the affected regions are not kept.");
        }

        // Order: the text layer's reading order, with each recognized region before the first text block that starts below its top.
        var tables = new List<OcrTable>();
        var regionsOut = new List<OcrRegion>();
        var keyValues = new List<OcrKeyValue>();
        groupBlocks.Sort((p, q) => p.Top != q.Top ? p.Top.CompareTo(q.Top) : p.Left.CompareTo(q.Left));
        int gi = 0, wordCount = 0;
        void AddGroup(Group g)
        {
            int blockBase = finalBlocks.Count;
            int wordBase = wordCount;
            foreach (OcrBlock b in g.Blocks)
            {
                finalBlocks.Add(b);
                wordCount += PageParts.CountWords([b]);
            }

            foreach (OcrRegion r in g.Regions)
            {
                regionsOut.Add(r with { BlockIndices = [.. r.BlockIndices.Select(i => i + blockBase)] });
            }

            foreach (OcrTable t in g.Tables)
            {
                tables.Add(t with
                {
                    BlockIndex = t.BlockIndex < 0 ? -1 : t.BlockIndex + blockBase,
                    Cells = [.. t.Cells.Select(c => c with { WordIndices = [.. c.WordIndices.Select(i => i + wordBase)] })],
                });
            }

            foreach (OcrKeyValue k in g.KeyValues)
            {
                keyValues.Add(k with
                {
                    KeyWordIndices = [.. k.KeyWordIndices.Select(i => i + wordBase)],
                    ValueWordIndices = [.. k.ValueWordIndices.Select(i => i + wordBase)],
                });
            }
        }

        foreach ((OcrBlock block, double top) in textOut)
        {
            while (gi < groupBlocks.Count && groupBlocks[gi].Top <= top)
            {
                AddGroup(groupBlocks[gi++]);
            }

            finalBlocks.Add(block);
            wordCount += PageParts.CountWords([block]);
        }

        while (gi < groupBlocks.Count)
        {
            AddGroup(groupBlocks[gi++]);
        }

        int ocrWords = groupBlocks.Sum(g => PageParts.CountWords(g.Blocks));
        if (groupBlocks.Count > 0)
        {
            diagnostics.Add($"{groupBlocks.Count} regions were read from pixels at {options.RenderDpi:0.##} DPI ({ocrWords} words).");
        }

        OcrLimits limits = ctx.Engine.Limits;
        if (finalBlocks.Count > limits.MaxBlocksPerPage)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxBlocksPerPage), limits.MaxBlocksPerPage, finalBlocks.Count);
        }

        if (wordCount > limits.MaxWordsPerPage)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxWordsPerPage), limits.MaxWordsPerPage, wordCount);
        }

        return new OcrPage(pageIndex, width, height, 72, finalBlocks, regionsOut, tables, keyValues, diagnostics, toPdf);
    }

    private static OcrWord MakeWord(TextWord w, PdfPageText? text, double pageHeight, bool includeChars)
    {
        var chars = new List<OcrChar>();
        if (includeChars && text is not null)
        {
            IReadOnlyList<PdfTextGlyph> glyphs = text.Glyphs;
            foreach (int gi in w.Source.GlyphIndices)
            {
                if (glyphs is null || (uint)gi >= (uint)glyphs.Count)
                {
                    continue;
                }

                PdfTextGlyph g = glyphs[gi];
                if (string.IsNullOrEmpty(g.Text))
                {
                    continue;
                }

                PdfRectangle b = g.Bounds;
                var cb = new OcrBox(b.Left, pageHeight - b.Top, b.Right, pageHeight - b.Bottom);
                if (double.IsFinite(cb.Left + cb.Top + cb.Right + cb.Bottom))
                {
                    chars.Add(new OcrChar(g.Text, cb, 1f, OcrSource.TextLayer, false));
                }
            }
        }

        // Text layer words are exactly what the file says: confidence 1 means "not guessed", not a calibrated probability.
        return new OcrWord(w.Source.Text, w.Box, 1f, 1f, OcrSource.TextLayer, chars);
    }

    private static List<TextBlock> Collect(PdfPageText text, double height, OcrLimits limits)
    {
        var blocks = new List<TextBlock>();
        long words = 0;
        foreach (PdfTextBlock b in text.Blocks)
        {
            var lines = new List<TextLine>();
            foreach (PdfTextLine l in b.Lines)
            {
                var ws = new List<TextWord>(l.Words.Count);
                foreach (PdfTextWord w in l.Words)
                {
                    words++;
                    if (words > limits.MaxWordsPerPage)
                    {
                        throw new OcrLimitException(nameof(OcrLimits.MaxWordsPerPage), limits.MaxWordsPerPage, words);
                    }

                    PdfRectangle r = w.Bounds;
                    var box = new OcrBox(
                        Math.Min(r.Left, r.Right), height - Math.Max(r.Top, r.Bottom), Math.Max(r.Left, r.Right), height - Math.Min(r.Top, r.Bottom));
                    if (string.IsNullOrEmpty(w.Text) || !double.IsFinite(box.Left + box.Top + box.Right + box.Bottom))
                    {
                        continue;
                    }

                    ws.Add(new TextWord { Source = w, Box = box, Visible = w.IsVisible });
                }

                if (ws.Count > 0)
                {
                    lines.Add(new TextLine { Source = l, Words = ws });
                }
            }

            if (lines.Count > 0)
            {
                blocks.Add(new TextBlock { Source = b, Lines = lines });
            }
        }

        return blocks;
    }

    /// <summary>True when a word holds the replacement character, a private-use code point or a control character: not text.</summary>
    internal static bool IsGarbled(string s)
    {
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '\uFFFD' || (c >= '\uE000' && c <= '\uF8FF') || c < ' ' || (c >= '\u007F' && c <= '\u009F'))
            {
                return true;
            }

            if (char.IsHighSurrogate(c) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
            {
                int cp = char.ConvertToUtf32(c, s[i + 1]);
                if (cp >= 0xF0000)
                {
                    return true;
                }

                i++;
            }
        }

        return false;
    }

    private static OcrBox Union(List<TextWord> words)
    {
        OcrBox box = words[0].Box;
        for (int i = 1; i < words.Count; i++)
        {
            box = OcrBox.Union(box, words[i].Box);
        }

        return box;
    }

    private static OcrBox Inflate(OcrBox b, double share, double width, double height)
    {
        double padY = Math.Max(1, b.Height * share), padX = Math.Max(1, b.Height * share * 2);
        return new OcrBox(Math.Max(0, b.Left - padX), Math.Max(0, b.Top - padY), Math.Min(width, b.Right + padX), Math.Min(height, b.Bottom + padY));
    }

    private static bool Covered(OcrBox image, List<TextWord> kept)
    {
        double area = PageParts.Area(image);
        if (area <= 0)
        {
            return true;
        }

        double covered = 0;
        int invisibleInside = 0;
        foreach (TextWord w in kept)
        {
            double inter = PageParts.IntersectionArea(image, w.Box);
            covered += inter;
            // A layer the caller chose to trust sits on the scan it describes: a few of its words inside the picture are enough.
            if (!w.Visible && inter >= 0.5 * PageParts.Area(w.Box))
            {
                invisibleInside++;
            }
        }

        return covered >= CoveredShare * area || invisibleInside >= 3;
    }

    private static List<OcrBox> FindImages(
        PdfPage page, double width, double height, PdfOcrOptions options, List<string> diagnostics, CancellationToken cancellationToken)
    {
        var collector = new ImagePlacementCollector(4096);
        try
        {
            page.Interpret(collector, InterpretOptions, cancellationToken);
        }
        catch (Exception e) when (PdfErrors.IsLibraryFailure(e))
        {
            PdfErrors.Fail(e, options, diagnostics, "The page's images could not be located");
        }

        if (collector.Truncated)
        {
            diagnostics.Add("The page draws more than 4096 images; only the first 4096 were considered.");
        }

        var result = new List<OcrBox>();
        foreach (OcrBox raw in collector.Images)
        {
            var clipped = new OcrBox(Math.Max(0, raw.Left), Math.Max(0, raw.Top), Math.Min(width, raw.Right), Math.Min(height, raw.Bottom));
            if (clipped.Width >= options.MinImagePoints && clipped.Height >= options.MinImagePoints)
            {
                result.Add(clipped);
            }
        }

        return result;
    }

    // Renders a few lines of visible text-layer text, reads them, and compares. True when the text layer cannot be trusted.
    private static bool CrossCheck(
        List<TextBlock> blocks, ImageFrame frame, double sx, double sy, double width, double height, ReadContext ctx,
        List<string> diagnostics, CancellationToken cancellationToken)
    {
        var eligible = new List<TextLine>();
        foreach (TextBlock b in blocks)
        {
            foreach (TextLine l in b.Lines)
            {
                if (l.Source.IsVertical || l.Source.Direction != PdfTextDirection.LeftToRight || l.Words.Any(w => !w.Keep || !w.Visible))
                {
                    continue;
                }

                OcrBox lb = Union(l.Words);
                int letters = l.Words.Sum(w => w.Source.Text.Count(char.IsLetterOrDigit));
                if (letters >= 6 && letters <= 200 && lb.Height >= 4 && lb.Height <= 80 && lb.Width >= 8)
                {
                    eligible.Add(l);
                }
            }
        }

        int n = Math.Min(ctx.Options.CrossCheckLines, eligible.Count);
        if (n == 0)
        {
            return false;
        }

        int disagree = 0, checkedLines = 0;
        for (int i = 0; i < n; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            TextLine line = eligible[(int)(((2L * i) + 1) * eligible.Count / (2L * n))];
            OcrBox box = Inflate(Union(line.Words), 0.25, width, height);
            int x0 = Math.Max(0, (int)Math.Floor(box.Left * sx)), y0 = Math.Max(0, (int)Math.Floor(box.Top * sy));
            int x1 = Math.Min(frame.Width, (int)Math.Ceiling(box.Right * sx)), y1 = Math.Min(frame.Height, (int)Math.Ceiling(box.Bottom * sy));
            if (x1 - x0 < 4 || y1 - y0 < 4)
            {
                continue;
            }

            OcrPage read;
            try
            {
                Image crop = frame.ToImage().Crop(new ImageRect(x0, y0, x1 - x0, y1 - y0));
                read = ctx.Engine.Ocr(
                    crop, new OcrOptions { Dpi = ctx.Options.RenderDpi, IncludeChars = false, MaxDegreeOfParallelism = ctx.RegionDegree }, cancellationToken);
            }
            catch (OcrLimitException)
            {
                continue;
            }
            catch (Exception e) when (PdfErrors.IsLibraryFailure(e))
            {
                diagnostics.Add("The text layer could not be checked against the rendering: " + e.Message);
                return false;
            }

            checkedLines++;
            string expected = Letters(string.Concat(line.Words.Select(w => w.Source.Text)));
            string seen = Letters(read.ToText());
            if (expected.Length == 0 || Distance(expected, seen) > LineDisagreement * Math.Max(expected.Length, seen.Length))
            {
                disagree++;
            }
        }

        if (checkedLines > 0 && disagree * 2 >= checkedLines)
        {
            diagnostics.Add(
                $"The text layer disagrees with the rendered page on {disagree} of {checkedLines} checked lines (a font's character map is probably wrong); the page was read from pixels instead.");
            return true;
        }

        return false;
    }

    private static string Letters(string s)
    {
        var sb = new System.Text.StringBuilder(s.Length);
        foreach (char c in s)
        {
            if (char.IsLetterOrDigit(c))
            {
                sb.Append(char.ToLowerInvariant(c));
            }
        }

        return sb.ToString();
    }

    // Levenshtein distance over at most a few hundred characters; two rows.
    private static int Distance(string a, string b)
    {
        if (a.Length == 0 || b.Length == 0)
        {
            return Math.Max(a.Length, b.Length);
        }

        var prev = new int[b.Length + 1];
        var cur = new int[b.Length + 1];
        for (int j = 0; j <= b.Length; j++)
        {
            prev[j] = j;
        }

        for (int i = 1; i <= a.Length; i++)
        {
            cur[0] = i;
            for (int j = 1; j <= b.Length; j++)
            {
                int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                cur[j] = Math.Min(Math.Min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + cost);
            }

            (prev, cur) = (cur, prev);
        }

        return prev[b.Length];
    }
}
