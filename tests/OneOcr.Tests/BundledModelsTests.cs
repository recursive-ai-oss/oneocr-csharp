using System.Security.Cryptography;
using OneOcr.Model;
using OneOcr.Models;

namespace OneOcr.Tests;

public sealed class BundledModelsTests
{
    [Fact]
    public void ThePinNamesTheProductionRecognizer()
    {
        BundledModel pin = BundledModels.Recognizer;
        Assert.Equal("crnn-v0", pin.Name);
        Assert.Equal("0.4.0", pin.Version);
        Assert.Equal("61498b53a9d4e4119740411566d41804bac6fc831f128729feecb85ad56b7021", pin.Sha256);
        Assert.Equal(64, pin.Sha256.Length);
        Assert.Equal("crnn-v0-0.4.0.onm", pin.Asset);
    }

    [Fact]
    public void BundledFileMatchesItsPinAndLoadsWhenPacked()
    {
        if (!BundledModels.HasRecognizer)
        {
            // Development build: no weights embedded. Loading must say so as a model error, not crash.
            Assert.Throws<OcrModelException>(() => OcrEngine.LoadBundled());
            return;
        }

        Assert.Null(BundledModels.TryReadRecognizer(out byte[] data));
        Assert.Equal(BundledModels.Recognizer.Sha256, Convert.ToHexStringLower(SHA256.HashData(data)));
        ModelContainer container = ModelContainer.Parse(data);
        Assert.Equal("crnn-v0", container.Architecture);
        Assert.NotNull(container.Calibration);
        Assert.NotNull(OcrEngine.LoadBundled());
    }
}
