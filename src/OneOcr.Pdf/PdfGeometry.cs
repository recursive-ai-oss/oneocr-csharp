using OnePdf;

namespace OneOcr.Pdf;

/// <summary>Page geometry: the map from page units (top-left origin, y down, points) back to PDF user space.</summary>
internal static class PdfGeometry
{
    /// <summary>
    /// Upright page space is the page box with /Rotate applied, origin at its bottom-left, y up (what OnePDF's extraction reports).
    /// Page units flip y inside it. The result maps a page-unit point to user space; <paramref name="userUnit"/> scales points of
    /// the viewer's page to user-space units (1 unless the page sets /UserUnit).
    /// </summary>
    public static OcrMatrix ToPdf(PdfRectangle box, int rotate, double userUnit)
    {
        double l = box.Left, b = box.Bottom, w0 = box.Width, h0 = box.Height;
        int r = ((rotate % 360) + 360) % 360;
        if (r % 90 != 0)
        {
            r = 0;
        }

        double u = userUnit > 0 && double.IsFinite(userUnit) ? userUnit : 1;
        // Page units are points of the viewed page; user space measures them in 1/u.
        double k = 1 / u;
        return r switch
        {
            90 => new OcrMatrix(0, k, k, 0, l, b),
            180 => new OcrMatrix(-k, 0, 0, k, l + w0, b),
            270 => new OcrMatrix(0, -k, -k, 0, l + w0, b + h0),
            _ => new OcrMatrix(k, 0, 0, -k, l, b + h0),
        };
    }

    /// <summary>Page size in page units: the box with /Rotate and /UserUnit applied.</summary>
    public static (double Width, double Height) Size(PdfRectangle box, int rotate, double userUnit)
    {
        int r = ((rotate % 360) + 360) % 360;
        double u = userUnit > 0 && double.IsFinite(userUnit) ? userUnit : 1;
        return r is 90 or 270 ? (box.Height * u, box.Width * u) : (box.Width * u, box.Height * u);
    }
}
