using OneImage;

namespace OneOcr;

/// <summary>
/// The process-wide default engine used by the extension methods that take no engine. Call <see cref="Configure(OcrEngine)"/>
/// once at startup; everything else can pass an <see cref="OcrEngine"/> explicitly.
/// </summary>
public static class OneOcr
{
    private static OcrEngine? s_default;

    /// <summary>The configured default engine.</summary>
    /// <exception cref="OcrModelException">No engine has been configured.</exception>
    public static OcrEngine Default =>
        Volatile.Read(ref s_default) ?? throw new OcrModelException(
            "No OCR model is configured. Call OneOcr.Configure(OcrEngine.Load(modelPath)) once at startup, or pass an engine explicitly.");

    /// <summary>Sets <see cref="Default"/>. May be called once.</summary>
    /// <param name="engine">The engine to use by default.</param>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/> is null.</exception>
    /// <exception cref="InvalidOperationException">A default engine was already configured.</exception>
    public static void Configure(OcrEngine engine)
    {
        ArgumentNullException.ThrowIfNull(engine);
        if (Interlocked.CompareExchange(ref s_default, engine, null) is not null)
        {
            throw new InvalidOperationException("OneOcr.Default is already configured.");
        }
    }

    /// <summary>Loads a model file and sets it as <see cref="Default"/>. May be called once.</summary>
    /// <param name="modelPath">A model container file.</param>
    /// <exception cref="OcrModelException">The model is malformed.</exception>
    /// <exception cref="InvalidOperationException">A default engine was already configured.</exception>
    public static void Configure(string modelPath) => Configure(OcrEngine.Load(modelPath));
}

/// <summary>OCR as extension methods on images.</summary>
public static class OcrImageExtensions
{
    /// <summary>Reads the text of a page with <see cref="OneOcr.Default"/>. See <see cref="OcrEngine.ToText"/>.</summary>
    /// <param name="image">The page.</param>
    /// <returns>The text, lines joined with <c>\n</c>.</returns>
    /// <exception cref="OcrModelException">No default engine is configured.</exception>
    public static string ToText(this Image image) => OneOcr.Default.ToText(image);

    /// <summary>Reads the text of a page with the given engine. See <see cref="OcrEngine.ToText"/>.</summary>
    /// <param name="image">The page.</param>
    /// <param name="engine">The engine to use.</param>
    /// <returns>The text, lines joined with <c>\n</c>.</returns>
    public static string ToText(this Image image, OcrEngine engine)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return engine.ToText(image);
    }
}
