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
        Assert.Equal("0.7.3", pin.Version);
        Assert.Equal("5c767a5ea36771a50caca812f5bbff9c64301536079b54d0c42b2982d7958eeb", pin.Sha256);
        Assert.Equal(64, pin.Sha256.Length);
        Assert.Equal("crnn-v0-0.7.3.onm", pin.Asset);
    }

    [Fact]
    public void ThePinNamesThePageDetector()
    {
        BundledModel pin = BundledModels.Detector;
        Assert.Equal("dbnet-lite-v1", pin.Name);
        Assert.Equal("detector", pin.Role);
        Assert.Equal("0.3.2", pin.Version);
        Assert.Equal("45950e9e4c27abf6642d7e61364e92d96439534708c641dece2d35c54f696d66", pin.Sha256);
        Assert.Equal(1371528, pin.Bytes);
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

        Assert.True(BundledModels.TryReadLanguageModel(out byte[] data, out _));
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

        Assert.True(BundledModels.TryReadWordFrequencies(out byte[] data, out _));
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

        Assert.True(BundledModels.TryReadDetector(out byte[] data, out _));
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

        Assert.True(BundledModels.TryReadRecognizer(out byte[] data, out _));
        Assert.Equal(BundledModels.Recognizer.Sha256, Convert.ToHexStringLower(SHA256.HashData(data)));
        ModelContainer container = ModelContainer.Parse(data);
        Assert.Equal("crnn-v0", container.Architecture);
        Assert.NotNull(container.Calibration);
        Assert.NotNull(OcrEngine.LoadBundled());
    }
}
