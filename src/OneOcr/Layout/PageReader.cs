using OneImage;
using OneOcr.Detection;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Layout;

/// <summary>Turns a grayscale page into an <see cref="OcrPage"/>: finds strips, reads them, orders them, assembles the model.</summary>
internal static partial class PageReader
{
    // Page units per inch that body text of about 11 points would have, for a typical line of this many pixels (ink from the top
    // of the ascenders to the bottom of the descenders is about one em tall).
    private const double AssumedBodyPoints = 11;

    // Most lines read together as one group.
    private const int MaxGroupLines = 12;

    // Most line pixels (the sum of the widths, about four per step of the recurrence) in one group: the group holds the sequences of
    // all its lines at once, so this bounds the memory a group needs.
    private const int MaxGroupWidth = 2048;

    private static readonly double[] CommonResolutions = [72, 96, 150, 200, 300, 400, 600];

    // The ruled tables are found before the finder erases rules; a page whose strips were found earlier brings its tables with them.
    private static FoundStrips FindStrips(Gray8[] gray, int width, int height, OcrLimits limits, BufferPool<byte> masks, int degree, out List<RuleGrid> grids)
    {
        grids = FindGrids(gray, width, height, degree);
        List<StripLine> strips = StripFinder.Find(gray, width, height, limits.MaxLinesPerPage, masks, degree, out _, out int level, out bool bright);
        return new FoundStrips(strips, level, bright);
    }

    public static OcrPage Read(
        LineRecognizer recognizer, OcrLimits limits, BufferPool<Gray8> pixelPool, BufferPool<byte> masks, ResizerPool resizers, IPageDetector? detector,
        Gray8[] gray, int width, int height, int pageIndex, double? dpi, int degree, bool includeChars, PageDetection? first, bool turn,
        FoundStrips? found, CancellationToken cancellationToken, (int Level, bool InkIsBright)? known = null, JunkParameters? junk = null)
    {
        if (detector is null)
        {
            // Strips found earlier on these very pixels are used as they are.
            List<RuleGrid> ruled = found?.Grids ?? [];
            FoundStrips strips = found ?? FindStrips(gray, width, height, limits, masks, degree, out ruled);
            List<StripLine> cut = TableStructure.SplitStrips(strips.Strips, ruled, PixelSpans.AsBytes(gray.AsSpan(0, width * height)), width, height, strips.Threshold, strips.InkIsBright);
            return ReadStrips(
                recognizer, limits, pixelPool, resizers, gray, width, height, cut, strips.Threshold, strips.InkIsBright, [], [], ruled, pageIndex, dpi, degree, includeChars,
                junk, cancellationToken);
        }

        LocatedPage located = DetectorLines.Locate(detector, gray, width, height, pixelPool, resizers, degree, cancellationToken, first, turn, known);
        try
        {
            long clock = StageClock.Start();
            List<RuleGrid> ruled = FindGrids(located.Gray, located.Width, located.Height, degree, (located.Threshold, located.InkIsBright));
            StageClock.Stop("8a table rules", clock);
            clock = StageClock.Start();
            List<StripLine> cut = TableStructure.SplitStrips(
                located.Strips, ruled, PixelSpans.AsBytes(located.Gray.AsSpan(0, located.Width * located.Height)), located.Width, located.Height, located.Threshold, located.InkIsBright);
            StageClock.Stop("8b split strips", clock);
            return ReadStrips(
                recognizer, limits, pixelPool, resizers, located.Gray, located.Width, located.Height, cut, located.Threshold, located.InkIsBright,
                located.Regions, located.Notes, ruled, pageIndex, dpi, degree, includeChars, junk, cancellationToken);
        }
        finally
        {
            located.Release(pixelPool);
        }
    }

    private static OcrPage ReadStrips(
        LineRecognizer recognizer, OcrLimits limits, BufferPool<Gray8> pixelPool, ResizerPool resizers, Gray8[] gray, int width, int height,
        List<StripLine> strips, int threshold, bool inkIsBright, List<DetectedRegion> regions, List<string> notes, List<RuleGrid> grids, int pageIndex, double? dpi, int degree,
        bool includeChars, JunkParameters? junk, CancellationToken cancellationToken)
    {
        int count = checked(width * height);
        var diagnostics = new List<string>(notes);
        if (!recognizer.HasCalibration)
        {
            diagnostics.Add(ModelContainer.NoCalibrationNote);
        }

        var boxes = new List<LineBox>(strips.Count);
        foreach (StripLine s in strips)
        {
            boxes.Add(s.Box);
        }

        int typical = LinePreprocessor.TypicalHeight(boxes);
        double units;
        if (dpi is double given)
        {
            units = given;
        }
        else if (typical > 0)
        {
            double estimate = typical * 72.0 / AssumedBodyPoints;
            units = Nearest(estimate);
            diagnostics.Add($"UnitsPerInch is estimated at {units:0} from the size of the text; pass OcrOptions.Dpi to set it.");
        }
        else
        {
            units = 72;
            diagnostics.Add("UnitsPerInch is 72 because the page has no text to estimate it from; pass OcrOptions.Dpi to set it.");
        }

        if (strips.Count == 0)
        {
            return new OcrPage(pageIndex, width, height, units, [], regions.Count > 0 ? MakeRegions(regions, [], height) : null, diagnostics: diagnostics);
        }

        int modelHeight = recognizer.InputHeight;
        var read = new OrderStrip?[strips.Count];
        var skipped = new string?[strips.Count];
        var crops = new LineBox[strips.Count];
        var queue = new List<(int Index, int Width)>(strips.Count);
        for (int index = 0; index < strips.Count; index++)
        {
            LineBox ink = strips[index].Box;
            crops[index] = strips[index].Tilt is { } tilted
                ? LinePreprocessor.CropBoxInFrame(tilted.Ink, typical)
                : LinePreprocessor.CropBox(ink, typical, width, height);
            LineBox crop = crops[index];
            // A strip far wider than tall (a rule, a photo's edge) would be wider than the model's limit once scaled to its input height;
            // it is left out with a note instead of failing the page.
            int scaledWidth = PixelResizer.GetSize(crop.Width, crop.Height, null, modelHeight).Width;
            if (Math.Ceiling((double)crop.Width * modelHeight / crop.Height) > limits.MaxLineWidth || scaledWidth > limits.MaxLineWidth)
            {
                skipped[index] = SkipNote(ink, limits);
                continue;
            }

            queue.Add((index, scaledWidth));
        }

        // The recurrent layers read their weights once per step for a whole group of lines, so lines are read in groups of similar
        // width (a group runs as long as its longest line). A group is the unit of parallelism; with few lines the groups shrink so
        // that every thread has one. Which lines share a group never changes any line's result.
        queue.Sort((l, r) => l.Width != r.Width ? r.Width.CompareTo(l.Width) : l.Index.CompareTo(r.Index));
        int maxLines = degree <= 1 ? MaxGroupLines : Math.Clamp(queue.Count / (2 * degree), 1, MaxGroupLines);
        var groups = new List<(int First, int Count)>();
        for (int first = 0; first < queue.Count;)
        {
            int taken = 0, columns = 0;
            while (first + taken < queue.Count && taken < maxLines && (taken == 0 || columns + queue[first + taken].Width <= MaxGroupWidth))
            {
                columns += queue[first + taken].Width;
                taken++;
            }

            groups.Add((first, taken));
            first += taken;
        }

        void ReadGroup(int g)
        {
            cancellationToken.ThrowIfCancellationRequested();
            (int first, int lineCount) = groups[g];
            ReadOnlySpan<byte> page = PixelSpans.AsBytes(gray.AsSpan(0, count));
            var pixels = new Gray8[lineCount][];
            var images = new LineImage[lineCount];
            try
            {
                for (int k = 0; k < lineCount; k++)
                {
                    int index = queue[first + k].Index;
                    long c0 = StageClock.Start();
                    pixels[k] = strips[index].Tilt is { } tilt
                        ? TiltedLines.Prepare(page, width, height, tilt, crops[index], modelHeight, pixelPool, resizers, out int cropWidth)
                        : LinePreprocessor.Prepare(page, width, crops[index], modelHeight, pixelPool, resizers, out cropWidth);
                    images[k] = new LineImage(pixels[k], cropWidth);
                    StageClock.Stop("8c line crops", c0);
                }

                long r0 = StageClock.Start();
                LineWords[] lines = recognizer.RecognizeWordsGroup(images, cancellationToken, includeChars);
                StageClock.Stop("8d recognize", r0);
                r0 = StageClock.Start();
                for (int k = 0; k < lineCount; k++)
                {
                    int index = queue[first + k].Index;
                    LineBox ink = strips[index].Box;
                    var words = new List<OcrWord>(lines[k].Words.Count);
                    foreach (RecognizedWord w in lines[k].Words)
                    {
                        words.Add(strips[index].Tilt is { } along
                            ? MakeTiltedWord(w, page, width, height, threshold, inkIsBright, along, crops[index], images[k].Width, includeChars)
                            : MakeWord(w, page, width, height, threshold, inkIsBright, ink, crops[index], images[k].Width, includeChars));
                    }

                    read[index] = new OrderStrip
                    {
                        Leaf = strips[index].Leaf,
                        Box = new OcrBox(ink.X, ink.Y, ink.Right, ink.Bottom),
                        Size = Math.Max(ink.Height, 0.85 * typical),
                        Words = words,
                    };
                }

                StageClock.Stop("8e word boxes", r0);
            }
            finally
            {
                foreach (Gray8[]? p in pixels)
                {
                    if (p is not null)
                    {
                        pixelPool.Return(p);
                    }
                }
            }
        }

        ParallelRun.For(groups.Count, degree, ReadGroup, cancellationToken);

        // Noise is removed before ordering, so lines, blocks, tables and text are built from what is left.
        int junkWords = junk is null ? 0 : JunkFilter.Apply(read, junk);
        if (junkWords > 0)
        {
            diagnostics.Add($"{junkWords} unsure word(s) in lines much smaller than the page's text were dropped as noise; pass OcrOptions.DropNoise = false to keep them.");
        }

        var ordered = new List<OrderStrip>(strips.Count);
        for (int i = 0; i < read.Length; i++)
        {
            if (read[i] is { } strip)
            {
                ordered.Add(strip);
            }
            else if (skipped[i] is { } note)
            {
                diagnostics.Add(note);
            }
        }

        long t = StageClock.Start();
        List<OrderBlock> plan = TableStructure.Infer(TableStructure.Apply(ReadingOrder.Arrange(ordered, cancellationToken), grids), regions, null);
        OcrPage assembled = Assemble(plan, ordered, regions, pageIndex, width, height, units, diagnostics);
        StageClock.Stop("8f order+tables+assemble", t);
        return assembled;
    }

    private static double Nearest(double estimate)
    {
        double best = CommonResolutions[0];
        double bestDistance = double.MaxValue;
        foreach (double r in CommonResolutions)
        {
            double d = Math.Abs(Math.Log(estimate / r));
            if (d < bestDistance)
            {
                bestDistance = d;
                best = r;
            }
        }

        return best;
    }

    // The word's horizontal extent comes from the alignment (halfway across the gaps to its neighbours); its box is the ink inside
    // that extent and the line's rows, which gives tight edges all round. When there is no ink there (an alignment error), the
    // box is the aligned extent over the line's rows.
    private static OcrWord MakeWord(
        RecognizedWord w, ReadOnlySpan<byte> page, int pageWidth, int pageHeight, int threshold, bool inkIsBright, LineBox ink, LineBox crop,
        int cropWidth, bool includeChars)
    {
        double scale = (double)crop.Width / cropWidth;
        double xa = Math.Clamp(crop.X + (w.X0 * scale), crop.X, crop.Right);
        double xb = Math.Clamp(crop.X + (w.X1 * scale), crop.X, crop.Right);
        int from = Math.Clamp((int)Math.Floor(xa), 0, pageWidth), to = Math.Clamp((int)Math.Ceiling(xb), from, pageWidth);
        int minX = int.MaxValue, maxX = -1, minY = int.MaxValue, maxY = -1;
        for (int y = Math.Max(0, ink.Y); y <= Math.Min(pageHeight - 1, ink.Bottom - 1); y++)
        {
            ReadOnlySpan<byte> row = page.Slice((y * pageWidth) + from, to - from);
            for (int x = 0; x < row.Length; x++)
            {
                if (inkIsBright ? row[x] > threshold : row[x] <= threshold)
                {
                    minX = Math.Min(minX, from + x);
                    maxX = Math.Max(maxX, from + x);
                    minY = Math.Min(minY, y);
                    maxY = Math.Max(maxY, y);
                }
            }
        }

        OcrBox box = maxX < 0 ? new OcrBox(xa, ink.Y, xb, ink.Bottom) : new OcrBox(minX, minY, maxX + 1, maxY + 1);
        IReadOnlyList<OcrChar> chars = [];
        if (includeChars)
        {
            var list = new List<OcrChar>(w.Chars.Count);
            chars = list;
            // A character's edge toward a neighbour is halfway across the gap between their runs; the ends are the word's.
            double left = box.Left;
            for (int i = 0; i < w.Chars.Count; i++)
            {
                RecognizedChar c = w.Chars[i];
                double right = box.Right;
                if (i + 1 < w.Chars.Count)
                {
                    double edge = crop.X + ((c.X1 + w.Chars[i + 1].X0) / 2 * scale);
                    right = Math.Clamp(edge, left, box.Right);
                }

                list.Add(new OcrChar(c.Text, new OcrBox(left, box.Top, right, box.Bottom), c.Probability));
                left = right;
            }
        }

        return new OcrWord(w.Text, box, w.Confidence, w.RawConfidence, OcrSource.Ocr, chars);
    }

    private static string SkipNote(LineBox ink, OcrLimits limits) =>
        $"A line at x {ink.X}, y {ink.Y} ({ink.Width} by {ink.Height} pixels) was skipped: scaled to the model's input height it would be wider than OcrLimits.MaxLineWidth ({limits.MaxLineWidth}).";

    private static OcrPage Assemble(
        List<OrderBlock> plan, List<OrderStrip> strips, List<DetectedRegion> detected, int pageIndex, int width, int height, double units,
        List<string> diagnostics)
    {
        var blocks = new List<OcrBlock>(plan.Count);
        var tables = new List<OcrTable>();
        int wordIndex = 0;
        foreach (OrderBlock block in plan)
        {
            if (block.Cells is not null)
            {
                AddGridBlock(block, strips, ref wordIndex, blocks, tables);
                continue;
            }

            var lines = new List<OcrLine>(block.Lines.Count);
            var cells = new Dictionary<(int Row, int Column), (OcrBox Box, List<int> Words)>();
            int multi = 0;
            for (int row = 0; row < block.Lines.Count; row++)
            {
                OrderLine ol = block.Lines[row];
                var words = new List<OcrWord>();
                var cellStarts = new List<int>();
                var text = new System.Text.StringBuilder();
                if (ol.Fragments.Count > 1)
                {
                    multi++;
                }

                foreach (OrderFragment frag in ol.Fragments)
                {
                    cellStarts.Add(words.Count);
                    if (text.Length > 0)
                    {
                        text.Append('\t');
                    }

                    for (int k = 0; k < frag.WordCount; k++)
                    {
                        OcrWord word = strips[frag.Strip].Words[frag.WordStart + k];
                        if (k > 0)
                        {
                            text.Append(' ');
                        }

                        text.Append(word.Text);
                        if (block.IsTable)
                        {
                            (int, int) key = (row, frag.Column);
                            if (cells.TryGetValue(key, out (OcrBox Box, List<int> Words) existing))
                            {
                                existing.Words.Add(wordIndex + words.Count);
                                cells[key] = (OcrBox.Union(existing.Box, word.Box), existing.Words);
                            }
                            else
                            {
                                cells[key] = (word.Box, [wordIndex + words.Count]);
                            }
                        }

                        words.Add(word);
                    }
                }

                wordIndex += words.Count;
                lines.Add(MakeLine(words, cellStarts, text.ToString()));
            }

            OcrBlock made = MakeBlock(block, lines);
            if (block.IsTable)
            {
                var tableCells = new List<OcrTableCell>(cells.Count);
                foreach (var ((row, column), (box, ws)) in cells.OrderBy(kv => kv.Key.Row).ThenBy(kv => kv.Key.Column))
                {
                    tableCells.Add(new OcrTableCell(row, column, 1, 1, box, ws));
                }

                // Not calibrated: the share of rows whose fragments lie in more than one column, the evidence the table test uses.
                tables.Add(new OcrTable(made.Box, block.Lines.Count, block.Columns, tableCells, (float)multi / block.Lines.Count, OcrSource.Ocr, blocks.Count));
            }

            blocks.Add(made);
        }

        IReadOnlyList<OcrRegion>? regions = null;
        if (detected.Count > 0)
        {
            regions = MakeRegions(detected, blocks, height);
            for (int i = 0; i < blocks.Count; i++)
            {
                OcrBlockRole role = FurnitureRole(detected, blocks[i], height);
                if (role != OcrBlockRole.Body)
                {
                    blocks[i] = blocks[i] with { Role = role };
                }
            }
        }

        return new OcrPage(pageIndex, width, height, units, blocks, regions, tables, null, diagnostics);
    }

    private static OcrLine MakeLine(List<OcrWord> words, List<int> cellStarts, string text)
    {
        OcrBox box = words[0].Box;
        double weighted = 0, chars = 0;
        foreach (OcrWord w in words)
        {
            box = OcrBox.Union(box, w.Box);
            weighted += w.Confidence * (double)w.Text.Length;
            chars += w.Text.Length;
        }

        return new OcrLine(text, words, cellStarts, box, (float)(weighted / chars), OcrSource.Ocr, Quad: EnclosingQuad(words));
    }

    private static OcrBlock MakeBlock(OrderBlock block, List<OcrLine> lines)
    {
        OcrBox box = lines[0].Box;
        double weighted = 0;
        int words = 0;
        foreach (OcrLine l in lines)
        {
            box = OcrBox.Union(box, l.Box);
            weighted += l.Confidence * (double)l.Words.Count;
            words += l.Words.Count;
        }

        return new OcrBlock(OcrBlockRole.Body, block.IsTable, box, lines, (float)(weighted / words), OcrSource.Ocr);
    }

    // The page model's regions from the detector's, with the blocks whose centres lie in each. A header_footer region is a header in the
    // upper half of the page and a footer in the lower half; classes the page model has no role for are left out.
    private static List<OcrRegion> MakeRegions(List<DetectedRegion> detected, List<OcrBlock> blocks, int pageHeight)
    {
        var result = new List<OcrRegion>(detected.Count);
        foreach (DetectedRegion d in detected)
        {
            OcrRegionRole? role = RoleOf(d, pageHeight);
            if (role is not { } r)
            {
                continue;
            }

            var inside = new List<int>();
            for (int i = 0; i < blocks.Count; i++)
            {
                if (Contains(d, blocks[i].Box))
                {
                    inside.Add(i);
                }
            }

            result.Add(new OcrRegion(r, new OcrBox(d.Left, d.Top, d.Right, d.Bottom), d.Score, OcrSource.Ocr, inside));
        }

        return result;
    }

    private static OcrRegionRole? RoleOf(DetectedRegion d, int pageHeight) => d.Name switch
    {
        "text" => OcrRegionRole.Text,
        "title" => OcrRegionRole.Title,
        "list" => OcrRegionRole.List,
        "table" => OcrRegionRole.Table,
        "figure" => OcrRegionRole.Figure,
        "header_footer" => (d.Top + d.Bottom) / 2 < pageHeight / 2.0 ? OcrRegionRole.Header : OcrRegionRole.Footer,
        "form_field" => OcrRegionRole.FormField,
        "checkbox" => OcrRegionRole.Checkbox,
        _ => null,
    };

    private static OcrBlockRole FurnitureRole(List<DetectedRegion> detected, OcrBlock block, int pageHeight)
    {
        foreach (DetectedRegion d in detected)
        {
            if (d.Name == "header_footer" && Contains(d, block.Box))
            {
                return RoleOf(d, pageHeight) == OcrRegionRole.Header ? OcrBlockRole.Header : OcrBlockRole.Footer;
            }
        }

        return OcrBlockRole.Body;
    }

    private static bool Contains(DetectedRegion d, OcrBox box)
    {
        double cx = (box.Left + box.Right) / 2, cy = (box.Top + box.Bottom) / 2;
        return cx >= d.Left && cx <= d.Right && cy >= d.Top && cy <= d.Bottom;
    }
}
