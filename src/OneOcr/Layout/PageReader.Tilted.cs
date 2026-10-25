using OneOcr.Recognition;

namespace OneOcr.Layout;

internal static partial class PageReader
{
    // A word of a line read along its own axis. Its extent along the line comes from the alignment, as for an upright line; its box is
    // the ink inside that extent and the line's rows, found in the line's frame, so the edges are tight on all sides. The box is
    // reported as a quad (the rectangle's corners in the page) and as the box around the quad.
    private static OcrWord MakeTiltedWord(
        RecognizedWord w, ReadOnlySpan<byte> page, int pageWidth, int pageHeight, int threshold, bool inkIsBright, TiltedLine tilt, LineBox crop,
        int cropWidth, bool includeChars)
    {
        LineBox ink = tilt.Ink;
        double scale = (double)crop.Width / cropWidth;
        double ua = Math.Clamp(crop.X + (w.X0 * scale), crop.X, crop.Right);
        double ub = Math.Clamp(crop.X + (w.X1 * scale), crop.X, crop.Right);
        int from = (int)Math.Floor(ua), to = Math.Max(from, (int)Math.Ceiling(ub));
        int minU = int.MaxValue, maxU = -1, minV = int.MaxValue, maxV = -1;
        for (int v = ink.Y; v < ink.Bottom; v++)
        {
            for (int u = from; u < to; u++)
            {
                (double px, double py) = tilt.ToPage(u + 0.5, v + 0.5);
                int x = (int)Math.Floor(px), y = (int)Math.Floor(py);
                if (x < 0 || y < 0 || x >= pageWidth || y >= pageHeight)
                {
                    continue;
                }

                if (inkIsBright ? page[(y * pageWidth) + x] > threshold : page[(y * pageWidth) + x] <= threshold)
                {
                    minU = Math.Min(minU, u);
                    maxU = Math.Max(maxU, u);
                    minV = Math.Min(minV, v);
                    maxV = Math.Max(maxV, v);
                }
            }
        }

        (double left, double top, double right, double bottom) = maxU < 0 ? (ua, ink.Y, ub, ink.Bottom) : (minU, minV, maxU + 1, maxV + 1);
        (OcrBox box, OcrQuad quad) = FrameRect(tilt, left, top, right, bottom);
        IReadOnlyList<OcrChar> chars = [];
        if (includeChars)
        {
            var list = new List<OcrChar>(w.Chars.Count);
            chars = list;
            double edgeLeft = left;
            for (int i = 0; i < w.Chars.Count; i++)
            {
                RecognizedChar c = w.Chars[i];
                double edgeRight = right;
                if (i + 1 < w.Chars.Count)
                {
                    edgeRight = Math.Clamp(crop.X + ((c.X1 + w.Chars[i + 1].X0) / 2 * scale), edgeLeft, right);
                }

                (OcrBox cb, OcrQuad cq) = FrameRect(tilt, edgeLeft, top, edgeRight, bottom);
                list.Add(new OcrChar(c.Text, cb, c.Probability, Quad: cq));
                edgeLeft = edgeRight;
            }
        }

        return new OcrWord(w.Text, box, w.Confidence, w.RawConfidence, OcrSource.Ocr, chars, quad);
    }

    // A rectangle of the line frame as page corners (top-left, top-right, bottom-right, bottom-left of the text's own frame) and the box around them.
    private static (OcrBox Box, OcrQuad Quad) FrameRect(TiltedLine tilt, double u0, double v0, double u1, double v1)
    {
        (double x0, double y0) = tilt.ToPage(u0, v0);
        (double x1, double y1) = tilt.ToPage(u1, v0);
        (double x2, double y2) = tilt.ToPage(u1, v1);
        (double x3, double y3) = tilt.ToPage(u0, v1);
        var box = new OcrBox(
            Math.Min(Math.Min(x0, x1), Math.Min(x2, x3)), Math.Min(Math.Min(y0, y1), Math.Min(y2, y3)),
            Math.Max(Math.Max(x0, x1), Math.Max(x2, x3)), Math.Max(Math.Max(y0, y1), Math.Max(y2, y3)));
        return (box, new OcrQuad(new OcrPoint(x0, y0), new OcrPoint(x1, y1), new OcrPoint(x2, y2), new OcrPoint(x3, y3)));
    }

    // The rectangle, in the frame of the first word's own direction, that holds every word's corners; null unless every word has a quad.
    // Words of one line lean alike, so the first word's direction is the line's.
    private static OcrQuad? EnclosingQuad(List<OcrWord> words)
    {
        foreach (OcrWord w in words)
        {
            if (w.Quad is null)
            {
                return null;
            }
        }

        OcrQuad first = words[0].Quad!.Value;
        double dx = first.P1.X - first.P0.X, dy = first.P1.Y - first.P0.Y;
        double length = Math.Sqrt((dx * dx) + (dy * dy));
        if (!(length > 0))
        {
            return null;
        }

        dx /= length;
        dy /= length;
        double s0 = double.MaxValue, s1 = double.MinValue, t0 = double.MaxValue, t1 = double.MinValue;
        foreach (OcrWord w in words)
        {
            OcrQuad q = w.Quad!.Value;
            foreach (OcrPoint p in (ReadOnlySpan<OcrPoint>)[q.P0, q.P1, q.P2, q.P3])
            {
                double s = (p.X * dx) + (p.Y * dy), t = (p.Y * dx) - (p.X * dy);
                s0 = Math.Min(s0, s);
                s1 = Math.Max(s1, s);
                t0 = Math.Min(t0, t);
                t1 = Math.Max(t1, t);
            }
        }

        // (s, t) back to the page: x = s*dx - t*dy, y = s*dy + t*dx.
        static OcrPoint At(double s, double t, double dx, double dy) => new((s * dx) - (t * dy), (s * dy) + (t * dx));
        return new OcrQuad(At(s0, t0, dx, dy), At(s1, t0, dx, dy), At(s1, t1, dx, dy), At(s0, t1, dx, dy));
    }
}
