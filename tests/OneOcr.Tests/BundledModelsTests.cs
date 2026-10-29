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
        Assert.Equal("0.7.1", pin.Version);
        Assert.Equal("489dffe11cfec7b1a7620c698312ac6d0b2bc1c4a36e37992310e70a6cdbb0ab", pin.Sha256);
        Assert.Equal(64, pin.Sha256.Length);
        Assert.Equal("crnn-v0-0.7.1.onm", pin.Asset);
    }

    [Fact]
    public void ThePinNamesThePageDetector()
    {
        BundledModel pin = BundledModels.Detector;
        Assert.Equal("dbnet-lite-v1", pin.Name);
        Assert.Equal("detector", pin.Role);
        Assert.Equal("0.3.0", pin.Version);
        Assert.Equal("110254fc2f9e60136370c418402ec2b9e329cb53f573d27205f7ed899ce85235", pin.Sha256);
        Assert.Equal(1371464, pin.Bytes);
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
    public void ThePinNamesTheWordFrequencyTable()
    {
        BundledModel pin = BundledModels.WordFrequencies;
        Assert.Equal("wordfreq-v2", pin.Name);
        Assert.Equal("word-frequencies", pin.Role);
        Assert.Equal("7121a7b9dbe12d109fd6cac4b6e0196ef036afb74a0f0e5504d06dd398fdf851", pin.Sha256);
        Assert.Equal(1316145, pin.Bytes);
    }

    [Fact]
    public void BundledWordFrequenciesMatchTheirPinAndAreUsedWhenPacked()
    {
        if (!BundledModels.HasWordFrequencies || !BundledModels.HasRecognizer)
        {
            return;
        }

        Assert.Null(BundledModels.TryReadWordFrequencies(out byte[] data));
        Assert.Equal(BundledModels.WordFrequencies.Sha256, Convert.ToHexStringLower(SHA256.HashData(data)));
        Assert.True(WordFrequencyModel.Parse(data).WordCount > 100_000);
        Assert.True(OcrEngine.LoadBundled().HasWordFrequencies);
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
