using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Fuzz;

/// <summary>
/// Model container bytes through the parser, then (when they parse) through the recognizer builder and one forward pass on a
/// small image, under tight limits so that every size an input can declare is checked rather than allocated.
/// Only the typed exceptions may come out; anything else is a finding.
/// </summary>
internal static class ContainerTarget
{
    private static readonly OcrLimits Limits = new()
    {
        MaxModelFileBytes = 1 << 20,
        MaxManifestBytes = 1 << 16,
        MaxTensorCount = 64,
        MaxTensorElements = 1 << 18,
        MaxCharsetEntries = 512,
        MaxConvBlocks = 8,
        MaxLstmLayers = 3,
        MaxLayerWidth = 64,
        MaxInputHeight = 32,
        MaxLineWidth = 64,
        MaxActivationElements = 1 << 18,
    };

    /// <summary>Inputs that parsed as containers, and inputs that went on to build a recognizer (progress statistics for local runs).</summary>
    public static long Parsed, Built;

    public static void Run(ReadOnlySpan<byte> data)
    {
        try
        {
            ModelContainer container = ModelContainer.Parse(data, Limits);
            Parsed++;
            LineRecognizer recognizer = LineRecognizer.FromContainer(container, Limits);
            Built++;
            int width = 24;
            var pixels = new byte[recognizer.InputHeight * width];
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = (byte)(i * 37);
            }

            _ = recognizer.Recognize(pixels, width, 1);
            _ = recognizer.RecognizeWords(pixels, width, 1, System.Threading.CancellationToken.None);
        }
        catch (OcrModelException)
        {
        }
        catch (OcrLimitException)
        {
        }
    }
}
