using OneImage;
using OneOcr.Models;

namespace OneOcr.Tests;

/// <summary>A tilted page of two unequal columns under a title, read with the packed models (skipped when they are not packed).</summary>
public class SkewedColumnsTests
{
    [Fact]
    public void TiltedTwoColumnPageIsReadColumnByColumn()
    {
        Assert.SkipUnless(BundledModels.HasRecognizer && BundledModels.HasDetector, "The recognizer and detector are not packed into this build.");
        string dir = Path.Combine(AppContext.BaseDirectory, "testdata", "pages");
        string expected = string.Join(' ', File.ReadAllText(Path.Combine(dir, "skewed-two-column.txt")).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        Image image = Image.Load(Path.Combine(dir, "skewed-two-column.png"));
        string text = OcrEngine.LoadBundled().ToText(image);
        Assert.Equal(expected, string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)));
    }
}
