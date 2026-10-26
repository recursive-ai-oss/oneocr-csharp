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
        Assert.Equal("0.6.2", pin.Version);
        Assert.Equal("73ed99cb2dac118d8a217ad5f5bddc42a29f94d148f447869be91a4d658753f0", pin.Sha256);
        Assert.Equal(64, pin.Sha256.Length);
        Assert.Equal("crnn-v0-0.6.2.onm", pin.Asset);
    }

    [Fact]
    public void ThePinNamesThePageDetector()
    {
        BundledModel pin = BundledModels.Detector;
        Assert.Equal("dbnet-lite-v1", pin.Name);
        Assert.Equal("detector", pin.Role);
        Assert.Equal("0.2.2", pin.Version);
        Assert.Equal("19e1adb9e57863e0d59988b60938621cec48bc38ce57172fdab4a1cd1e0e2cae", pin.Sha256);
        Assert.Equal(1370632, pin.Bytes);
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
