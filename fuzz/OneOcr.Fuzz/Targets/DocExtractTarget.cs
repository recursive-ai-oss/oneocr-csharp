using OneOcr.Documents;
using OneOcr.Kernels;
using OneOcr.Model;

namespace OneOcr.Fuzz;

/// <summary>
/// Document-extraction model container bytes through the parser, then (when they describe a docextract-v0 model) through the model
/// builder and a forward pass and assembly on a fixed seven-word page, under tight limits so that every size an input can declare is
/// checked rather than allocated. Only the typed exceptions may come out.
/// </summary>
internal static class DocExtractTarget
{
    private static readonly OcrLimits Limits = new()
    {
        MaxModelFileBytes = 1 << 20,
        MaxManifestBytes = 1 << 16,
        MaxTensorCount = 128,
        MaxTensorElements = 1 << 18,
        MaxCharsetEntries = 512,
        MaxLayerWidth = 128,
        MaxActivationElements = 1 << 18,
        MaxWordsPerWindow = 32,
        MaxCharsPerWord = 16,
        MaxPositionBins = 64,
        MaxTransformerLayers = 4,
        MaxExtractionClasses = 32,
    };

    private static readonly string[] Texts = ["Invoice", "No.", "INV-0042", "03/10/2026", "Total", "1.234,50", "\u4e2d\u6587"];

    public static long Parsed, Built;

    public static void Run(ReadOnlySpan<byte> data)
    {
        try
        {
            ModelContainer container = ModelContainer.Parse(data, Limits);
            Parsed++;
            DocExtractModel model = DocExtractModel.FromContainer(container, Limits);
            Built++;
            var boxes = new OcrBox[Texts.Length];
            for (int i = 0; i < boxes.Length; i++)
            {
                boxes[i] = new OcrBox(10 + (i * 50), 10 + (i / 4 * 20), 55 + (i * 50), 25 + (i / 4 * 20));
            }

            DocFeatures features = DocFeatures.Build(model, Texts, boxes, 400, 100);
            DocOutputs outputs = DocForward.Run(model, features, Dispatch.Best, 1, System.Threading.CancellationToken.None);
            int[] index = [.. Enumerable.Range(0, Texts.Length)];
            OcrSource[] sources = [.. index.Select(_ => OcrSource.Ocr)];
            _ = DocAssembly.Assemble(model, outputs, Texts, boxes, index, sources, null, NumberStyle.Auto, DateOrder.Auto);
        }
        catch (OcrModelException)
        {
        }
        catch (OcrLimitException)
        {
        }
    }
}
