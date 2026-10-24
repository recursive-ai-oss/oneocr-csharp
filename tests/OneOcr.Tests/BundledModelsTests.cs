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
    public void ThePinNamesThePageDetector()
    {
        BundledModel pin = BundledModels.Detector;
        Assert.Equal("dbnet-lite-v0", pin.Name);
        Assert.Equal("detector", pin.Role);
        Assert.Equal("0.1.0", pin.Version);
        Assert.Equal("ac663d83a0e35c84dd38742f76b2c7cbe58eca2ac4941f27a6b489d517f1284f", pin.Sha256);
        Assert.Equal(629896, pin.Bytes);
    }

    [Fact]
    public void BundledDetectorMatchesItsPinAndLoadsWhenPacked()
    {
        if (!BundledModels.HasDetector || !BundledModels.HasRecognizer)
        {
            Assert.False(BundledModels.HasDetector && OcrEngine.LoadBundled().HasDetector is false);
            return;
        }

        Assert.Null(BundledModels.TryReadDetector(out byte[] data));
        Assert.Equal(BundledModels.Detector.Sha256, Convert.ToHexStringLower(SHA256.HashData(data)));
        Assert.True(OcrEngine.LoadBundled().HasDetector);
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
