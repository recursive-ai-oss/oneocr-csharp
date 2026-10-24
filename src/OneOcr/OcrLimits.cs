namespace OneOcr;

/// <summary>
/// Bounds checked against declared or running values before the work they guard. The defaults suit the models this library
/// ships with and are far below what could exhaust memory; raise them deliberately for larger models.
/// </summary>
public sealed class OcrLimits
{
    /// <summary>The default limits.</summary>
    public static OcrLimits Default { get; } = new();

    /// <summary>Largest model file, in bytes, that will be read. Default 128 MiB.</summary>
    public long MaxModelFileBytes { get; init; } = 128L * 1024 * 1024;

    /// <summary>Largest manifest, in bytes. Default 1 MiB.</summary>
    public int MaxManifestBytes { get; init; } = 1024 * 1024;

    /// <summary>Deepest JSON nesting accepted in a manifest. Default 16.</summary>
    public int MaxJsonDepth { get; init; } = 16;

    /// <summary>Most tensors a model file may declare. Default 1024.</summary>
    public int MaxTensorCount { get; init; } = 1024;

    /// <summary>Most dimensions of one tensor. Default 8.</summary>
    public int MaxTensorRank { get; init; } = 8;

    /// <summary>Most elements of one tensor. Default 2^26.</summary>
    public long MaxTensorElements { get; init; } = 1L << 26;

    /// <summary>Longest tensor name in UTF-16 code units. Default 128.</summary>
    public int MaxTensorNameLength { get; init; } = 128;

    /// <summary>Most output classes (charset entries). Default 65536.</summary>
    public int MaxCharsetEntries { get; init; } = 65536;

    /// <summary>Longest charset entry in UTF-16 code units. Default 16.</summary>
    public int MaxCharsetEntryLength { get; init; } = 16;

    /// <summary>Most convolution blocks in a recognizer. Default 32.</summary>
    public int MaxConvBlocks { get; init; } = 32;

    /// <summary>Most stacked LSTM layers in a recognizer. Default 8.</summary>
    public int MaxLstmLayers { get; init; } = 8;

    /// <summary>Most channels or hidden units in any layer. Default 4096.</summary>
    public int MaxLayerWidth { get; init; } = 4096;

    /// <summary>Largest convolution kernel or pooling window side. Default 15.</summary>
    public int MaxKernelSize { get; init; } = 15;

    /// <summary>Tallest line image a model may declare as its input height. Default 256.</summary>
    public int MaxInputHeight { get; init; } = 256;

    /// <summary>Widest line image, in pixels, that will be recognized. Default 8192.</summary>
    public int MaxLineWidth { get; init; } = 8192;

    /// <summary>Most elements in any one activation buffer while recognizing a line. Default 2^27 (512 MiB of float32).</summary>
    public long MaxActivationElements { get; init; } = 1L << 27;

    /// <summary>Most pixels (width times height) of an image that will be read for recognition. Default 2^27 (a 11585 x 11585 image).</summary>
    public long MaxImagePixels { get; init; } = 1L << 27;

    /// <summary>
    /// Most bytes of scratch memory each engine keeps for reuse between lines and calls, separately for the recognizer's
    /// activation buffers and the page and line pixel buffers. Memory in use is not counted, so peak memory is the working set of
    /// the lines being read plus at most this much. Reuse keeps repeated large allocations, and the garbage they leave, out of the
    /// managed heap. Default 64 MiB; 0 keeps nothing.
    /// </summary>
    public long MaxPooledBufferBytes { get; init; } = 64L * 1024 * 1024;

    /// <summary>Largest character language model file, in bytes, that will be read. Default 16 MiB.</summary>
    public long MaxLanguageModelFileBytes { get; init; } = 16L * 1024 * 1024;

    /// <summary>Most script models in one language model file. Default 8.</summary>
    public int MaxLanguageModelScripts { get; init; } = 8;

    /// <summary>Highest n-gram order of a language model. Default 8.</summary>
    public int MaxLanguageModelOrder { get; init; } = 8;

    /// <summary>Most n-grams, over all orders and scripts, in one language model file. Default 4,000,000.</summary>
    public long MaxLanguageModelNgrams { get; init; } = 4_000_000;

    /// <summary>Most words in a lexicon. Default 2^20.</summary>
    public int MaxLexiconWords { get; init; } = 1 << 20;

    /// <summary>Longest lexicon word in UTF-16 code units. Default 64.</summary>
    public int MaxLexiconWordLength { get; init; } = 64;

    /// <summary>Largest lexicon file, in bytes, that will be read. Default 16 MiB.</summary>
    public long MaxLexiconFileBytes { get; init; } = 16L * 1024 * 1024;

    /// <summary>Widest beam beam-search decoding accepts. Default 64.</summary>
    public int MaxBeamWidth { get; init; } = 64;

    /// <summary>
    /// Most prefixes beam search may create for one line, bounded before the search starts from the line's length and the
    /// beam width (about 60 bytes each). Default 2^22.
    /// </summary>
    public long MaxBeamNodes { get; init; } = 1L << 22;

    /// <summary>Most text lines recognized on one page. Default 4096.</summary>
    public int MaxLinesPerPage { get; init; } = 4096;

    /// <summary>Most knots in one script's calibration curve in a model container. Default 64.</summary>
    public int MaxCalibrationKnots { get; init; } = 64;

    /// <summary>Most bytes of a page-model JSON document the reader will parse. Default 128 MiB.</summary>
    public long MaxPageJsonBytes { get; init; } = 128L * 1024 * 1024;

    /// <summary>Most pages in one call or one page-model JSON document. Default 1000.</summary>
    public int MaxPagesPerCall { get; init; } = 1000;

    /// <summary>Most blocks on one page. Default 4096.</summary>
    public int MaxBlocksPerPage { get; init; } = 4096;

    /// <summary>Most words on one page. Default 2^18.</summary>
    public int MaxWordsPerPage { get; init; } = 1 << 18;

    /// <summary>Most characters (the elements below words) on one page. Default 2^20.</summary>
    public int MaxCharsPerPage { get; init; } = 1 << 20;

    /// <summary>Most regions, tables, key–value pairs and table cells (each counted on its own) on one page. Default 2^16.</summary>
    public int MaxPageElements { get; init; } = 1 << 16;

    /// <summary>Longest text of one word, line or diagnostic in a page-model document, in UTF-16 code units. Default 2^16.</summary>
    public int MaxPageStringLength { get; init; } = 1 << 16;

    /// <summary>Most bytes of one language-model provider response that will be read. Checked against the declared length first, then while reading. Default 16 MiB.</summary>
    public long MaxLlmResponseBytes { get; init; } = 16L * 1024 * 1024;

    /// <summary>Deepest JSON nesting accepted in a provider response, in the model's answer and in a schema. Default 32.</summary>
    public int MaxLlmJsonDepth { get; init; } = 32;

    /// <summary>Most bytes of a JSON Schema handed to a provider. Default 256 KiB.</summary>
    public int MaxLlmSchemaBytes { get; init; } = 256 * 1024;

    /// <summary>Most bytes of one request to a provider (images as encoded, plus text). Checked before the request is built. Default 32 MiB.</summary>
    public long MaxLlmRequestBytes { get; init; } = 32L * 1024 * 1024;
}
