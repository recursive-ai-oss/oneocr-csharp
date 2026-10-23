using OneImage;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Layout;

/// <summary>Turns a grayscale page into an <see cref="OcrPage"/>: finds strips, reads them, orders them, assembles the model.</summary>
internal static class PageReader
{
    // Page units per inch that body text of about 11 points would have, for a typical line of this many pixels (ink from the top
    // of the ascenders to the bottom of the descenders is about one em tall).
    private const double AssumedBodyPoints = 11;

    private static readonly double[] CommonResolutions = [72, 96, 150, 200, 300, 400, 600];

    public static OcrPage Read(
        LineRecognizer recognizer, OcrLimits limits, BufferPool<Gray8> pixelPool, BufferPool<byte> masks, ResizerPool resizers, Gray8[] gray, int width, int height,
        int pageIndex, double? dpi, int degree, bool includeChars, CancellationToken cancellationToken)
    {
        int count = checked(width * height);
        List<StripLine> strips = StripFinder.Find(gray, width, height, limits.MaxLinesPerPage, masks, out _, out int threshold, out bool inkIsBright);
        var diagnostics = new List<string>();
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
            return new OcrPage(pageIndex, width, height, units, [], diagnostics: diagnostics);
        }

        int modelHeight = recognizer.InputHeight;
        var read = new OrderStrip?[strips.Count];
        var skipped = new string?[strips.Count];
        void ReadStrip(int index)
        {
            cancellationToken.ThrowIfCancellationRequested();
            LineBox ink = strips[index].Box;
            LineBox crop = LinePreprocessor.CropBox(ink, typical, width, height);
            // A strip far wider than tall (a rule, a photo's edge) would be wider than the model's limit once scaled to its input height;
            // it is left out with a note instead of failing the page.
            if (Math.Ceiling((double)crop.Width * modelHeight / crop.Height) > limits.MaxLineWidth)
            {
                skipped[index] = SkipNote(ink, limits);
                return;
            }

            ReadOnlySpan<byte> page = PixelSpans.AsBytes(gray.AsSpan(0, count));
            Gray8[] cropPixels = LinePreprocessor.Prepare(page, width, crop, modelHeight, pixelPool, resizers, out int cropWidth);
            try
            {
                // Lines are the unit of parallelism, so each is recognized on one thread.
                LineWords line;
                try
                {
                    line = recognizer.RecognizeWords(PixelSpans.AsBytes(cropPixels.AsSpan(0, cropWidth * modelHeight)), cropWidth, 1, cancellationToken);
                }
                catch (OcrLimitException ex) when (ex.LimitName == nameof(OcrLimits.MaxLineWidth))
                {
                    skipped[index] = SkipNote(ink, limits);
                    return;
                }

                var words = new List<OcrWord>(line.Words.Count);
                foreach (RecognizedWord w in line.Words)
                {
                    words.Add(MakeWord(w, page, width, height, threshold, inkIsBright, ink, crop, cropWidth, includeChars));
                }

                read[index] = new OrderStrip
                {
                    Leaf = strips[index].Leaf,
                    Box = new OcrBox(ink.X, ink.Y, ink.Right, ink.Bottom),
                    Size = Math.Max(ink.Height, 0.85 * typical),
                    Words = words,
                };
            }
            finally
            {
                pixelPool.Return(cropPixels);
            }
        }

        ParallelRun.For(strips.Count, degree, ReadStrip, cancellationToken);

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

        List<OrderBlock> plan = ReadingOrder.Arrange(ordered, cancellationToken);
        return Assemble(plan, ordered, pageIndex, width, height, units, diagnostics);
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
        var chars = new List<OcrChar>();
        if (includeChars)
        {
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

                chars.Add(new OcrChar(c.Text, new OcrBox(left, box.Top, right, box.Bottom), c.Probability));
                left = right;
            }
        }

        return new OcrWord(w.Text, box, w.Confidence, w.RawConfidence, OcrSource.Ocr, chars);
    }

    private static string SkipNote(LineBox ink, OcrLimits limits) =>
        $"A line at x {ink.X}, y {ink.Y} ({ink.Width} by {ink.Height} pixels) was skipped: scaled to the model's input height it would be wider than OcrLimits.MaxLineWidth ({limits.MaxLineWidth}).";

    private static OcrPage Assemble(
        List<OrderBlock> plan, List<OrderStrip> strips, int pageIndex, int width, int height, double units, List<string> diagnostics)
    {
        var blocks = new List<OcrBlock>(plan.Count);
        var tables = new List<OcrTable>();
        int wordIndex = 0;
        foreach (OrderBlock block in plan)
        {
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

        return new OcrPage(pageIndex, width, height, units, blocks, null, tables, null, diagnostics);
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

        return new OcrLine(text, words, cellStarts, box, (float)(weighted / chars), OcrSource.Ocr);
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
}
