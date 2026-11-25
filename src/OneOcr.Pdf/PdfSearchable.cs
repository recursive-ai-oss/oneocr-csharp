using OnePdf;
using OnePdf.Text;

namespace OneOcr.Pdf;

/// <summary>Adds text layers to the pages of an existing PDF.</summary>
internal static class PdfSearchable
{
    public static SearchablePdfResult Run(ReadOnlyMemory<byte> pdf, Stream output, OcrEngine engine, SearchablePdfOptions options, CancellationToken cancellationToken)
    {
        PdfOcrOptions read = options.Pdf ?? new PdfOcrOptions();
        read.Validate();
        if (pdf.Length > read.MaxFileBytes)
        {
            throw new OcrLimitException(nameof(PdfOcrOptions.MaxFileBytes), read.MaxFileBytes, pdf.Length);
        }

        PdfDocument document;
        try
        {
            document = PdfDocument.Open(pdf, new PdfOpenOptions { Password = read.Password, AllowEditing = true });
        }
        catch (Exception e) when (PdfErrors.IsLibraryFailure(e))
        {
            throw PdfErrors.Wrap(e);
        }

        using (document)
        {
            var unreliable = new System.Collections.Concurrent.ConcurrentDictionary<int, List<OcrBox>>();
            IReadOnlyList<OcrPage> pages = PdfOcr.ReadPages(document, () => pdf, engine, read, cancellationToken, unreliable);
            var diagnostics = new List<string>();
            int changed = 0, words = 0;
            try
            {
                var writer = new PdfTextLayerWriter(document);
                foreach (OcrPage ocr in pages)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (!ocr.Words.Any(static w => w.Source == OcrSource.Ocr) || ocr.ToPdf is null)
                    {
                        continue;
                    }

                    unreliable.TryGetValue(ocr.Index, out List<OcrBox>? boxes);
                    int added = AddLayer(document, writer, ocr, read.TrustExistingOcrLayer, boxes, diagnostics, out int marked);
                    if (marked > 0)
                    {
                        diagnostics.Add($"Page {ocr.Index + 1}: {marked} visible text run(s) with an unreliable character map are marked with an empty /ActualText, so text extraction and copy return the recognized words instead; they paint as before.");
                    }
                    else if (ocr.Diagnostics.Any(static d => d.Message.Contains("unreadable characters", StringComparison.Ordinal) || d.Message.Contains("disagrees with the rendered page", StringComparison.Ordinal)))
                    {
                        diagnostics.Add($"Page {ocr.Index + 1}: visible text whose character map is unreliable stays in the file next to the new layer (it is part of what the page paints and could not be marked); readers that extract text will see both.");
                    }

                    if (added > 0)
                    {
                        changed++;
                        words += added;
                    }
                }

                if (changed == 0 && options.Conformance == SearchablePdfConformance.None)
                {
                    // Nothing to add: the file goes out exactly as it came in.
                    output.Write(pdf.Span);
                }
                else if (options.Conformance == SearchablePdfConformance.PdfA2b)
                {
                    // The converter reads font programs back, and a font added in this session only gets its program (the subset) when the document is
                    // saved: so save first, then convert the saved file.
                    using var saved = new MemoryStream();
                    document.Save(saved, new PdfSaveOptions(), cancellationToken);
                    saved.Position = 0;
                    PdfAConverter.Convert(saved, output, new PdfAConversionOptions { Part = PdfAPart.Part2, Conformance = PdfAConformance.B }, cancellationToken);
                }
                else
                {
                    document.Save(output, new PdfSaveOptions(), cancellationToken);
                }
            }
            catch (PdfAException e)
            {
                throw new OcrFormatException("The file cannot be written as PDF/A-2b: " + e.Message, e);
            }
            catch (Exception e) when (PdfErrors.IsLibraryFailure(e))
            {
                throw PdfErrors.Wrap(e);
            }

            return new SearchablePdfResult(pages, changed, words, diagnostics);
        }
    }

    private static int AddLayer(
        PdfDocument document, PdfTextLayerWriter writer, OcrPage ocr, bool trustExisting, List<OcrBox>? unreliable, List<string> diagnostics, out int marked)
    {
        marked = 0;
        PdfPage page = document.Pages[ocr.Index];
        string label = $"Page {ocr.Index + 1}: ";
        if (!trustExisting)
        {
            int removed = RemoveInvisibleText(page, label, diagnostics);
            if (removed >= 0 && HasInvisibleWords(page))
            {
                diagnostics.Add(label + "invisible text from another OCR step remains (it is drawn from a form XObject or a construct the rewrite does not follow); the page now carries both layers.");
            }
        }

        if (unreliable is { Count: > 0 } && ocr.ToPdf is { } toPdf)
        {
            marked = MarkUnreliable(page, ocr, unreliable, toPdf, label, diagnostics);
        }

        int added;
        using (PdfPageContentBuilder b = page.BeginContent())
        {
            added = writer.Draw(b, ocr, ocr.ToPdf, static w => w.Source == OcrSource.Ocr);
            b.Commit();
        }

        // The builder writes the new stream uncompressed; the layer is the only stream this call adds, so it is the last of /Contents.
        if (page.Dictionary.Get("Contents") is PdfArray contents && contents.Count > 0 && contents[contents.Count - 1] is PdfStream layer)
        {
            ReadOnlyMemory<byte> raw = layer.GetDecodedBytes();
            layer.SetData(raw.Span, compress: true);
        }

        return added;
    }

    // Wraps the visible shows of the lines the page reader set aside in an empty /ActualText. Only lines the recognizer produced words for
    // are marked: a line with no replacement would otherwise disappear from extraction. After the rewrite the page is extracted once; if a
    // word the reader kept from the text layer is gone, the rewrite is undone. Returns the number of shows marked.
    private static int MarkUnreliable(PdfPage page, OcrPage ocr, List<OcrBox> unreliable, OcrMatrix toPdf, string label, List<string> diagnostics)
    {
        var boxes = new List<(double, double, double, double)>();
        foreach (OcrBox line in unreliable)
        {
            if (!ocr.Words.Any(w => w.Source == OcrSource.Ocr && Inside(line, w.Box)))
            {
                continue;
            }

            double[] xs = new double[4], ys = new double[4];
            (double X, double Y)[] corners = [(line.Left, line.Top), (line.Right, line.Top), (line.Right, line.Bottom), (line.Left, line.Bottom)];
            for (int i = 0; i < 4; i++)
            {
                xs[i] = (toPdf.A * corners[i].X) + (toPdf.C * corners[i].Y) + toPdf.E;
                ys[i] = (toPdf.B * corners[i].X) + (toPdf.D * corners[i].Y) + toPdf.F;
            }

            boxes.Add((xs.Min(), ys.Min(), xs.Max(), ys.Max()));
        }

        if (boxes.Count == 0)
        {
            return 0;
        }

        try
        {
            IReadOnlyList<PdfStream> streams = page.ContentStreams;
            if (streams.Count == 0)
            {
                return 0;
            }

            using var joined = new MemoryStream();
            foreach (PdfStream s in streams)
            {
                joined.Write(s.GetDecodedBytes().Span);
                joined.WriteByte((byte)'\n');
            }

            byte[]? rewritten = ActualTextMarker.Mark(joined.GetBuffer().AsSpan(0, (int)joined.Length), boxes, out int count);
            if (rewritten is null)
            {
                return 0;
            }

            PdfObject? before = page.Dictionary.Get("Contents");
            page.Dictionary.SetEntry("Contents", page.Document.CreateStream(rewritten, compress: true));
            var extracted = new HashSet<string>(StringComparer.Ordinal);
            foreach (PdfTextWord w in page.ExtractText(new PdfTextOptions { Mode = PdfTextMode.ReadingOrder }).Words)
            {
                extracted.Add(w.Text);
            }

            if (ocr.Words.Any(w => w.Source == OcrSource.TextLayer && !extracted.Contains(w.Text)))
            {
                page.Dictionary.SetEntry("Contents", before);
                diagnostics.Add(label + "marking the unreliable text would have hidden reliable words from extraction, so the page content was left as it was.");
                return 0;
            }

            return count;
        }
        catch (Exception e) when (PdfErrors.IsLibraryFailure(e))
        {
            diagnostics.Add($"{label}the page content could not be read to mark unreliable text: {e.Message}");
            return 0;
        }
    }

    // A word belongs to a line when its centre lies inside the line's box.
    private static bool Inside(OcrBox line, OcrBox word)
    {
        double cx = (word.Left + word.Right) / 2, cy = (word.Top + word.Bottom) / 2;
        return cx >= line.Left && cx <= line.Right && cy >= line.Top && cy <= line.Bottom;
    }

    // Returns the number of text-showing operators cut out, or -1 when the page's content could not be rewritten.
    private static int RemoveInvisibleText(PdfPage page, string label, List<string> diagnostics)
    {
        try
        {
            IReadOnlyList<PdfStream> streams = page.ContentStreams;
            if (streams.Count == 0)
            {
                return 0;
            }

            using var joined = new MemoryStream();
            foreach (PdfStream s in streams)
            {
                joined.Write(s.GetDecodedBytes().Span);
                joined.WriteByte((byte)'\n');
            }

            byte[]? rewritten = InvisibleTextRemover.Remove(joined.GetBuffer().AsSpan(0, (int)joined.Length), out int removed);
            if (rewritten is null)
            {
                return 0;
            }

            page.Dictionary.SetEntry("Contents", page.Document.CreateStream(rewritten, compress: true));
            diagnostics.Add($"{label}removed {removed} invisible text operator(s) of an earlier OCR layer.");
            return removed;
        }
        catch (Exception e) when (PdfErrors.IsLibraryFailure(e))
        {
            diagnostics.Add($"{label}the page content could not be read to remove an earlier OCR layer: {e.Message}");
            return -1;
        }
    }

    private static bool HasInvisibleWords(PdfPage page)
    {
        try
        {
            foreach (PdfTextWord w in page.ExtractText().Words)
            {
                if (!w.IsVisible)
                {
                    return true;
                }
            }
        }
        catch (Exception e) when (PdfErrors.IsLibraryFailure(e))
        {
            return false;
        }

        return false;
    }
}
