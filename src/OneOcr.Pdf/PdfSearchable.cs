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
            IReadOnlyList<OcrPage> pages = PdfOcr.ReadPages(document, () => pdf, engine, read, cancellationToken);
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

                    int added = AddLayer(document, writer, ocr, read.TrustExistingOcrLayer, diagnostics);
                    if (ocr.Diagnostics.Any(static d => d.Message.Contains("unreadable characters", StringComparison.Ordinal) || d.Message.Contains("disagrees with the rendered page", StringComparison.Ordinal)))
                    {
                        diagnostics.Add($"Page {ocr.Index + 1}: visible text whose character map is unreliable stays in the file next to the new layer (it is part of what the page paints); readers that extract text will see both.");
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

    private static int AddLayer(PdfDocument document, PdfTextLayerWriter writer, OcrPage ocr, bool trustExisting, List<string> diagnostics)
    {
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
