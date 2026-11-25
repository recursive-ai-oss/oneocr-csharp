using System.Text;
using OnePdf;

namespace OneOcr.Pdf;

/// <summary>
/// Draws the words of an <see cref="OcrPage"/> as invisible text (render mode 3) so that a PDF reader can select, search and copy
/// them where they appear on the page. This is the low-level half of searchable-PDF output: it adds text to a page the caller
/// already has; <see cref="SearchablePdf"/> builds whole files on top of it.
/// </summary>
/// <remarks>
/// <para>
/// Every word is one text object. Its text matrix puts the baseline on the bottom edge of the word's frame (the word's left and
/// right edges and the line's top and bottom, in the line's own axes, so turned and deskewed lines lie along their own axis), scales
/// the glyphs horizontally so that the word is as wide as its frame, and sizes them so that the glyph cell (0.9 em above the
/// baseline, 0.2 em below) is as tall as the frame.
/// All words use one embedded font with empty glyphs and a <c>ToUnicode</c> map, so extraction returns the page model's text
/// exactly. The layout is specified in <c>docs/spec/searchable-pdf.md</c>.
/// </para>
/// <para>One writer is for one document; it embeds the font on first use and shares it between pages. It is not thread-safe.</para>
/// </remarks>
public sealed class PdfTextLayerWriter
{
    private const double CellEm = GlyphlessFont.Ascent + GlyphlessFont.Descent;
    private const double GlyphEm = GlyphlessFont.AdvanceUnits / 1000.0;

    private readonly PdfDocument _document;
    private PdfCreatedFont? _font;

    /// <summary>Creates a writer for <paramref name="document"/>.</summary>
    /// <param name="document">The document the text is added to; it must allow editing or be new.</param>
    /// <exception cref="ArgumentNullException"><paramref name="document"/> is null.</exception>
    public PdfTextLayerWriter(PdfDocument document)
    {
        ArgumentNullException.ThrowIfNull(document);
        _document = document;
    }

    /// <summary>The map from the page model's units (top-left origin, y down) to PDF user space for a page that did not come from a PDF: one unit is 1/<see cref="OcrPage.UnitsPerInch"/> inch across and 1/<see cref="OcrPage.VerticalUnitsPerInch"/> inch down.</summary>
    /// <param name="page">The page; its size and resolution are used.</param>
    /// <returns>The matrix that <see cref="Draw"/> uses when none is given and <see cref="OcrPage.ToPdf"/> is null.</returns>
    public static OcrMatrix ImageToPdf(OcrPage page)
    {
        ArgumentNullException.ThrowIfNull(page);
        double s = 72.0 / page.UnitsPerInch, sy = 72.0 / page.VerticalUnitsPerInch;
        return new OcrMatrix(s, 0, 0, -sy, 0, page.Height * sy);
    }

    /// <summary>
    /// Draws the words of <paramref name="page"/> into <paramref name="content"/> as invisible text, in the page's reading order.
    /// Words with no text, a degenerate or non-finite box, and words rejected by <paramref name="include"/> are skipped. Characters
    /// the font has no glyph for (controls, surrogates, anything beyond the Basic Multilingual Plane) are written as U+FFFD.
    /// </summary>
    /// <param name="content">The builder of the page the text goes on. Its graphics state is saved and restored around the layer.</param>
    /// <param name="page">The page model; its word <see cref="OcrWord.Quad"/> (or <see cref="OcrWord.Box"/> when there is none) is where the text goes.</param>
    /// <param name="toPdf">The map from page units to the user space of the page being drawn on. Null uses <see cref="OcrPage.ToPdf"/>, or, for a page without one, <see cref="ImageToPdf"/>.</param>
    /// <param name="include">Selects the words to draw; null draws all.</param>
    /// <returns>The number of words written.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="content"/> or <paramref name="page"/> is null.</exception>
    public int Draw(PdfContentBuilder content, OcrPage page, OcrMatrix? toPdf = null, Func<OcrWord, bool>? include = null)
    {
        ArgumentNullException.ThrowIfNull(content);
        ArgumentNullException.ThrowIfNull(page);
        OcrMatrix m = toPdf ?? page.ToPdf ?? ImageToPdf(page);
        int written = 0;
        bool open = false;
        foreach (OcrBlock block in page.Blocks)
        {
            foreach (OcrLine line in block.Lines)
            {
                foreach (OcrWord word in line.Words)
                {
                    if (include is not null && !include(word))
                    {
                        continue;
                    }

                    if (!Place(line, word, m, out string text, out double fontSize, out double a, out double b, out double c, out double d, out double e, out double f))
                    {
                        continue;
                    }

                    if (!open)
                    {
                        content.Save();
                        content.SetTextRenderMode(PdfTextRenderMode.Invisible);
                        open = true;
                    }

                    content.SetFont(_font ??= _document.AddFont(GlyphlessFont.Bytes, new PdfEmbedFontOptions { BaseFontName = "OneOcrText" }), fontSize);
                    content.DrawText(text, a, b, c, d, e, f);
                    written++;
                }
            }
        }

        if (open)
        {
            content.Restore();
        }

        return written;
    }

    /// <summary>
    /// The frame a word is drawn in, in page units: the corners (bottom-left, bottom-right, top-right, top-left) of the rectangle, in the
    /// line's own axes, that spans the word along the line and the whole line across it.
    /// </summary>
    /// <remarks>
    /// Word boxes are tight around ink, so their top and bottom change from word to word ("of" against "by"). Drawing each word on its own
    /// box would put every word on a different baseline and a reader that groups text into lines would take one line for several.
    /// All words of a line therefore share the line's top and bottom edges and keep their own left and right edges.
    /// </remarks>
    internal static (OcrPoint BottomLeft, OcrPoint BottomRight, OcrPoint TopRight, OcrPoint TopLeft) Frame(OcrLine line, OcrWord word)
    {
        OcrQuad lq = line.Quad ?? (word.Quad is null
            ? new OcrQuad(new OcrPoint(line.Box.Left, line.Box.Top), new OcrPoint(line.Box.Right, line.Box.Top), new OcrPoint(line.Box.Right, line.Box.Bottom), new OcrPoint(line.Box.Left, line.Box.Bottom))
            : word.Quad.Value);
        OcrQuad wq = word.Quad ?? new OcrQuad(new OcrPoint(word.Box.Left, word.Box.Top), new OcrPoint(word.Box.Right, word.Box.Top), new OcrPoint(word.Box.Right, word.Box.Bottom), new OcrPoint(word.Box.Left, word.Box.Bottom));
        if (line.Quad is null && word.Quad is not null)
        {
            // A word with its own axis on a line without one: no line frame to share, use the word's.
            lq = wq;
        }

        double bx = lq.P2.X - lq.P3.X, by = lq.P2.Y - lq.P3.Y;
        double length = Math.Sqrt((bx * bx) + (by * by));
        if (!(length > 0) || !double.IsFinite(length))
        {
            return (wq.P3, wq.P2, wq.P1, wq.P0);
        }

        bx /= length;
        by /= length;
        OcrPoint o = lq.P3;
        double lo = double.MaxValue, hi = double.MinValue;
        foreach (OcrPoint p in new[] { wq.P0, wq.P1, wq.P2, wq.P3 })
        {
            double t = ((p.X - o.X) * bx) + ((p.Y - o.Y) * by);
            lo = Math.Min(lo, t);
            hi = Math.Max(hi, t);
        }

        double ux = lq.P0.X - lq.P3.X, uy = lq.P0.Y - lq.P3.Y;
        OcrPoint bl = new(o.X + (bx * lo), o.Y + (by * lo)), br = new(o.X + (bx * hi), o.Y + (by * hi));
        return (bl, br, new OcrPoint(br.X + ux, br.Y + uy), new OcrPoint(bl.X + ux, bl.Y + uy));
    }

    /// <summary>The text matrix and font size for a word, or false when the word cannot be placed.</summary>
    internal static bool Place(
        OcrLine line, OcrWord word, OcrMatrix m, out string text, out double fontSize, out double a, out double b, out double c, out double d, out double e, out double f)
    {
        text = string.Empty;
        fontSize = a = b = c = d = e = f = 0;
        if (string.IsNullOrEmpty(word.Text))
        {
            return false;
        }

        var (p3, p2, _, p0) = Frame(line, word);
        (double x0, double y0) = Map(m, p0);
        (double x2, double y2) = Map(m, p2);
        (double x3, double y3) = Map(m, p3);

        // Baseline direction (bottom-left to bottom-right) and the way up (bottom-left to top-left), in user space.
        double bx = x2 - x3, by = y2 - y3, ux = x0 - x3, uy = y0 - y3;
        double width = Math.Sqrt(bx * bx + by * by), height = Math.Sqrt(ux * ux + uy * uy);
        if (!double.IsFinite(width) || !double.IsFinite(height) || width < 1e-6 || height < 1e-6)
        {
            return false;
        }

        var sb = new StringBuilder(word.Text.Length);
        int glyphs = 0;
        foreach (Rune r in word.Text.EnumerateRunes())
        {
            if (r.Value < 0x20 || (r.Value >= 0x7F && r.Value < 0xA0))
            {
                continue;
            }

            sb.Append(char.ConvertFromUtf32(GlyphlessFont.Map(r)));
            glyphs++;
        }

        if (glyphs == 0)
        {
            return false;
        }

        // Glyph cell: 0.9 em above the baseline, 0.2 em below. The baseline therefore sits 0.2/1.1 of the frame's height above its bottom edge.
        fontSize = height / CellEm;
        double scale = width / (glyphs * GlyphEm * fontSize);
        a = bx / width * scale;
        b = by / width * scale;
        c = ux / height;
        d = uy / height;
        double lift = GlyphlessFont.Descent / CellEm;
        e = x3 + ux * lift;
        f = y3 + uy * lift;
        text = sb.ToString();
        return true;
    }

    private static (double X, double Y) Map(OcrMatrix m, OcrPoint p) => (m.A * p.X + m.C * p.Y + m.E, m.B * p.X + m.D * p.Y + m.F);
}
