using OnePdf;

namespace OneOcr.Pdf;

/// <summary>
/// Records where a page draws images, in page units (top-left origin, y down). Everything else the interpreter reports is ignored.
/// Images inside glyph procedures (Type 3 fonts) are part of letters, not pictures, and are skipped.
/// </summary>
internal sealed class ImagePlacementCollector : IPdfDevice
{
    private readonly int _max;
    private int _type3Depth;

    public ImagePlacementCollector(int max) => _max = max;

    /// <summary>Boxes of the drawn images in the interpreter's output space (y down, points).</summary>
    public List<OcrBox> Images { get; } = [];

    /// <summary>True when more images were drawn than the collector keeps.</summary>
    public bool Truncated { get; private set; }

    public void DrawImage(in PdfImage image, in PdfGraphicsState state)
    {
        if (_type3Depth > 0)
        {
            return;
        }

        if (Images.Count >= _max)
        {
            Truncated = true;
            return;
        }

        PdfMatrix m = state.Ctm;
        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        for (int i = 0; i < 4; i++)
        {
            (double x, double y) = m.Transform(i & 1, i >> 1);
            minX = Math.Min(minX, x);
            maxX = Math.Max(maxX, x);
            minY = Math.Min(minY, y);
            maxY = Math.Max(maxY, y);
        }

        if (double.IsFinite(minX) && double.IsFinite(minY) && double.IsFinite(maxX) && double.IsFinite(maxY) && maxX > minX && maxY > minY)
        {
            Images.Add(new OcrBox(minX, minY, maxX, maxY));
        }
    }

    public void BeginType3Glyph(in PdfType3GlyphInfo info, in PdfGraphicsState state) => _type3Depth++;

    public void EndType3Glyph() => _type3Depth--;

    public void BeginPage(in PdfPageInfo info)
    {
    }

    public void EndPage()
    {
    }

    public void FillPath(in PdfPath path, bool evenOdd, in PdfGraphicsState state)
    {
    }

    public void StrokePath(in PdfPath path, in PdfGraphicsState state)
    {
    }

    public void FillStrokePath(in PdfPath path, bool evenOdd, in PdfGraphicsState state)
    {
    }

    public void PushClip(in PdfPath path, bool evenOdd, in PdfGraphicsState state)
    {
    }

    public void PushTextClip(in PdfGraphicsState state)
    {
    }

    public void PopClip()
    {
    }

    public void ShowGlyphRun(in PdfGlyphRun run, in PdfGraphicsState state)
    {
    }

    public void PaintShading(PdfShading shading, in PdfGraphicsState state)
    {
    }

    public void BeginGroup(in PdfGroupInfo info, in PdfGraphicsState state)
    {
    }

    public void EndGroup()
    {
    }

    public void BeginSoftMask(in PdfSoftMaskInfo info, in PdfGraphicsState state)
    {
    }

    public void EndSoftMask()
    {
    }

    public void BeginMarkedContent(string tag, PdfDictionary? properties)
    {
    }

    public void EndMarkedContent()
    {
    }

    public void BeginAnnotation(in PdfAnnotationInfo info)
    {
    }

    public void EndAnnotation()
    {
    }
}
