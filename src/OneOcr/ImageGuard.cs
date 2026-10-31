using OneImage;

namespace OneOcr;

/// <summary>
/// Carries the image library's failures over to the library's own exception types, so that a single <c>catch (OcrException)</c>
/// covers untrusted image bytes, and maps <see cref="OcrLimits"/> onto the image library's decode limits.
/// </summary>
internal static class ImageGuard
{
    public static void Run(Action action)
    {
        try
        {
            action();
        }
        catch (ImageLimitException e)
        {
            throw Limit(e);
        }
        catch (ImageException e)
        {
            throw Format(e);
        }
    }

    public static T Run<T>(Func<T> function)
    {
        try
        {
            return function();
        }
        catch (ImageLimitException e)
        {
            throw Limit(e);
        }
        catch (ImageException e)
        {
            throw Format(e);
        }
    }

    /// <summary>The image library's decode limits for an engine with these limits: pixels per frame and frames follow <see cref="OcrLimits"/>.</summary>
    public static LoadOptions LoadOptionsFor(OcrLimits limits) => new()
    {
        Limits = new ImageLimits { MaxPixelsPerFrame = limits.MaxImagePixels, MaxFrames = limits.MaxPagesPerCall },
    };

    private static OcrLimitException Limit(ImageLimitException e) => new("ImageLimits", "The image exceeds a decoding limit: " + e.Message, e);

    private static OcrFormatException Format(ImageException e) => new("The image cannot be decoded: " + e.Message, e);
}
