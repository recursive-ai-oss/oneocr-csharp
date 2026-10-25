using System.Security.Cryptography;
using OneOcr.Decoding;
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
        Assert.Equal("0.6.0", pin.Version);
        Assert.Equal("347c208d2626ed88f9c7c57c0f6bec86dff0ce7e03bcea5c2e0bf9ef87342749", pin.Sha256);
        Assert.Equal(64, pin.Sha256.Length);
        Assert.Equal("crnn-v0-0.6.0.onm", pin.Asset);
    }

    [Fact]
    public void ThePinNamesThePageDetector()
    {
        BundledModel pin = BundledModels.Detector;
        Assert.Equal("dbnet-lite-v1", pin.Name);
        Assert.Equal("detector", pin.Role);
        Assert.Equal("0.2.1", pin.Version);
        Assert.Equal("af520cb6f990ccd1cf0ad3860de7801e0e8411f373550345764c6b6e15093974", pin.Sha256);
        Assert.Equal(1370568, pin.Bytes);
    }

    [Fact]
    public void ThePinNamesTheWordLanguageModel()
    {
        BundledModel pin = BundledModels.LanguageModel;
        Assert.Equal("charlm-v1", pin.Name);
        Assert.Equal("language-model", pin.Role);
        Assert.Equal("38c190e804f17f54244cfb16cc03cc7dba30d13aa4cb5b6540a71e068bdd68b5", pin.Sha256);
        Assert.Equal(1421911, pin.Bytes);
    }

    [Fact]
    public void BundledLanguageModelMatchesItsPinAndIsUsedWhenPacked()
    {
        if (!BundledModels.HasLanguageModel || !BundledModels.HasRecognizer)
        {
            return;
        }

        Assert.Null(BundledModels.TryReadLanguageModel(out byte[] data));
        Assert.Equal(BundledModels.LanguageModel.Sha256, Convert.ToHexStringLower(SHA256.HashData(data)));
        Assert.Equal(3, CharLanguageModel.Parse(data).Scripts.Count);
        Assert.True(OcrEngine.LoadBundled().HasWordModel);
        Assert.False(OcrEngine.LoadBundled().WithWordModel(null).HasWordModel);
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
