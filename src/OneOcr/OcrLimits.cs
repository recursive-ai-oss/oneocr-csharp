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
}
