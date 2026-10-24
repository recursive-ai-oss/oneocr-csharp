using OneImage;

namespace OneOcr.Llm;

/// <summary>Shrinks, masks and encodes a page image for upload.</summary>
internal static class ImagePrep
{
    /// <summary>Size after shrinking to fit every bound, never enlarging, keeping the aspect ratio. Dimensions round down so no bound is passed.</summary>
    public static (int Width, int Height) TargetSize(int width, int height, LlmImageOptions o)
    {
        double scale = 1;
        int longSide = Math.Max(width, height), shortSide = Math.Min(width, height);
        if (o.MaxLongEdge is { } l && longSide > l)
        {
            scale = Math.Min(scale, (double)l / longSide);
        }

        if (o.MaxShortEdge is { } s && shortSide > s)
        {
            scale = Math.Min(scale, (double)s / shortSide);
        }

        if (o.MaxPixels is { } p && (long)width * height > p)
        {
            scale = Math.Min(scale, Math.Sqrt((double)p / ((long)width * height)));
        }

        return scale >= 1
            ? (width, height)
            : (Math.Max(1, (int)Math.Floor((width * scale) + 1e-6)), Math.Max(1, (int)Math.Floor((height * scale) + 1e-6)));
    }

    public static PreparedImage Prepare(
        Image image, string label, LlmImageOptions options, IReadOnlyList<OcrBox> masks, double pageWidth, double pageHeight, OcrLimits limits,
        CancellationToken cancellationToken)
    {
        long pixels = (long)image.Width * image.Height;
        if (pixels > limits.MaxImagePixels)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxImagePixels), limits.MaxImagePixels, pixels);
        }

        (int w, int h) = TargetSize(image.Width, image.Height, options);
        Image work = (w, h) == (image.Width, image.Height) ? image : image.Resize(w, h, Fit.Stretch);
        work = work.Flatten(new Rgba32(255, 255, 255, 255));

        if (masks.Count > 0 && pageWidth > 0 && pageHeight > 0)
        {
            Rgba32[] data = work.ToArray<Rgba32>(0, cancellationToken);
            var black = new Rgba32(0, 0, 0, 255);
            double sx = w / pageWidth, sy = h / pageHeight;
            foreach (OcrBox box in masks)
            {
                // Round outward and add a pixel so resampling blur at a box edge cannot leave readable ink.
                int x0 = Math.Clamp((int)Math.Floor(box.Left * sx) - 1, 0, w), x1 = Math.Clamp((int)Math.Ceiling(box.Right * sx) + 1, 0, w);
                int y0 = Math.Clamp((int)Math.Floor(box.Top * sy) - 1, 0, h), y1 = Math.Clamp((int)Math.Ceiling(box.Bottom * sy) + 1, 0, h);
                for (int y = y0; y < y1; y++)
                {
                    data.AsSpan((y * w) + x0, Math.Max(0, x1 - x0)).Fill(black);
                }
            }

            work = Image.FromPixels<Rgba32>(data, w, h);
        }

        long max = options.MaxBytes ?? long.MaxValue;
        byte[] png = Encode(work, PngFormat.Instance, null, cancellationToken);
        if (png.Length <= max)
        {
            return new PreparedImage(label, png, "image/png", w, h);
        }

        // Lossless is too big for the provider: fall back to JPEG at falling quality.
        for (int q = Math.Clamp(options.JpegQuality, 1, 100); ; q = Math.Max(30, q - 20))
        {
            byte[] jpeg = Encode(work, JpegFormat.Instance, new JpegEncoderOptions { Quality = q }, cancellationToken);
            if (jpeg.Length <= max)
            {
                return new PreparedImage(label, jpeg, "image/jpeg", w, h);
            }

            if (q <= 30)
            {
                throw new OcrLimitException(nameof(LlmImageOptions.MaxBytes), max, jpeg.Length);
            }
        }
    }

    private static byte[] Encode(Image image, IImageFormat format, ImageEncoderOptions? encoderOptions, CancellationToken cancellationToken)
    {
        using var ms = new MemoryStream();
        image.Save(ms, format, new SaveOptions { EncoderOptions = encoderOptions }, cancellationToken);
        return ms.ToArray();
    }
}
