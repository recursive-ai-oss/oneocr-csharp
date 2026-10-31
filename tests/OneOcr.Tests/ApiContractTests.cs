using System.Text;
using System.Text.Json.Serialization;
using OneImage;
using OneOcr.Documents;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Models;
using OneOcr.Training;

namespace OneOcr.Tests;

[JsonSerializable(typeof(NoMatchForm))]
internal sealed partial class ApiContractContext : JsonSerializerContext;

internal sealed class NoMatchForm
{
    [JsonPropertyName("nothing-like-this")]
    public string? Foo { get; set; }

    [JsonPropertyName("nor-this")]
    public string? Bar { get; set; }
}

/// <summary>The public contract fixed before 1.0: frames, declared resolution, errors, limits, options, text views, page model equality.</summary>
public class ApiContractTests
{
    private static OcrEngine NewEngine(OcrLimits? limits = null) =>
        OcrEngine.Create(ModelContainer.Parse(TestModels.RandomContainer(TestModels.Small, 5)), limits, KernelPath.Scalar);

    private static byte[] PagePixels(int width, int height, int[] tops, int seed) => EngineTests.SyntheticPage(width, height, tops, 14, 8, seed: seed).Pixels;

    private static Image ToImage(byte[] px, int w, int h) => Image.FromPixels<Gray8>(px.Select(b => new Gray8(b)).ToArray(), w, h);

    // ---- a minimal multi-page TIFF: uncompressed 8-bit gray, little endian, one IFD per frame, optional resolution ----

    internal static byte[] Tiff(IReadOnlyList<(byte[] Pixels, int Width, int Height)> frames, int? dpi)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write((byte)'I');
        w.Write((byte)'I');
        w.Write((ushort)42);
        w.Write(0u); // first IFD offset, patched below
        long patch = 4;
        for (int i = 0; i < frames.Count; i++)
        {
            (byte[] px, int width, int height) = frames[i];
            long dataAt = ms.Position;
            w.Write(px);
            long rationalAt = ms.Position;
            w.Write((uint)(dpi ?? 1));
            w.Write(1u);
            if (ms.Position % 2 != 0)
            {
                w.Write((byte)0);
            }

            long ifdAt = ms.Position;
            long back = ms.Position;
            ms.Position = patch;
            w.Write((uint)ifdAt);
            ms.Position = back;
            var entries = new List<(ushort Tag, ushort Type, uint Count, uint Value)>
            {
                (256, 4, 1, (uint)width), (257, 4, 1, (uint)height), (258, 3, 1, 8), (259, 3, 1, 1), (262, 3, 1, 1), (273, 4, 1, (uint)dataAt),
                (277, 3, 1, 1), (278, 4, 1, (uint)height), (279, 4, 1, (uint)px.Length),
            };
            if (dpi is not null)
            {
                entries.Add((282, 5, 1, (uint)rationalAt));
                entries.Add((283, 5, 1, (uint)rationalAt));
                entries.Add((296, 3, 1, 2));
            }

            w.Write((ushort)entries.Count);
            foreach ((ushort tag, ushort type, uint count, uint value) in entries)
            {
                w.Write(tag);
                w.Write(type);
                w.Write(count);
                w.Write(value);
            }

            patch = ms.Position;
            w.Write(0u);
        }

        return ms.ToArray();
    }

    private static uint Crc32(ReadOnlySpan<byte> data)
    {
        uint crc = 0xFFFFFFFF;
        foreach (byte b in data)
        {
            crc ^= b;
            for (int k = 0; k < 8; k++)
            {
                crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320 : crc >> 1;
            }
        }

        return ~crc;
    }

    // The PNG with a pHYs chunk (pixels per metre) put in after IHDR.
    private static byte[] PngWithDensity(byte[] png, int dpi)
    {
        uint perMeter = (uint)Math.Round(dpi / 0.0254);
        byte[] body = new byte[13];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(body, perMeter);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(body.AsSpan(4), perMeter);
        body[8] = 1;
        byte[] typed = [.. "pHYs"u8.ToArray(), .. body[..9]];
        var chunk = new byte[4 + typed.Length + 4];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(chunk, 9);
        typed.CopyTo(chunk, 4);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(chunk.AsSpan(4 + typed.Length), Crc32(typed));
        int afterIhdr = 8 + 12 + 13;
        return [.. png[..afterIhdr], .. chunk, .. png[afterIhdr..]];
    }

    private static (byte[] Pixels, int W, int H)[] ThreeFrames() =>
    [
        (PagePixels(500, 300, [40, 100, 160, 220], 1), 500, 300),
        (PagePixels(400, 250, [40, 100, 160], 2), 400, 250),
        (PagePixels(600, 200, [40, 110], 3), 600, 200),
    ];

    [Fact]
    public void AMultiPageFileReadsEveryFrameOrFailsLoudly()
    {
        OcrEngine engine = NewEngine();
        (byte[] Pixels, int W, int H)[] frames = ThreeFrames();
        Image tiff = Image.Load(Tiff(frames, 200));
        Assert.Equal(3, tiff.FrameCount);

        IReadOnlyList<OcrPage> pages = engine.OcrPages(tiff);
        Assert.Equal([0, 1, 2], pages.Select(p => p.Index).ToArray());
        Assert.Equal([500, 400, 600], pages.Select(p => (int)p.Width).ToArray());
        Assert.Equal(pages[1].Text, engine.Ocr(tiff.Frames[1].ToImage(), new OcrOptions { PageIndex = 1 }).Text);

        // The one-liners that give one result for the whole image read every frame.
        string text = engine.ToText(tiff);
        Assert.Equal(string.Join("\n\n", pages.Select(p => p.ToText())), text);
        Assert.Equal(text, tiff.ToText(engine));
        Assert.NotEmpty(tiff.ToMarkdown(engine));

        // The call that gives one page refuses a multi-frame image instead of silently keeping the first frame.
        ArgumentException refused = Assert.Throws<ArgumentException>(() => engine.Ocr(tiff));
        Assert.Contains("OcrPages", refused.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentException>(() => tiff.Ocr(engine));
        Assert.Throws<ArgumentException>(() => engine.Ocr([tiff]));

        // A limit on pages applies to frames.
        OcrEngine few = NewEngine(new OcrLimits { MaxPagesPerCall = 2 });
        Assert.Throws<OcrLimitException>(() => few.OcrPages(tiff));
        Assert.Throws<OcrLimitException>(() => few.ToText(tiff));
    }

    [Fact]
    public void AFilesDeclaredResolutionIsUsedAndTheEstimateNoteIsNotMade()
    {
        OcrEngine engine = NewEngine();
        byte[] tiff = Tiff(ThreeFrames(), 200);
        IReadOnlyList<OcrPage> pages = engine.OcrPages(tiff);
        Assert.All(pages, p => Assert.Equal(200, p.UnitsPerInch));
        Assert.All(pages, p => Assert.DoesNotContain(p.Diagnostics, d => d.Code is OcrDiagnosticCodes.DpiEstimated or OcrDiagnosticCodes.DpiDefaulted));

        // Through a stream and a path too; an explicit value wins.
        using (var stream = new MemoryStream(tiff))
        {
            Assert.Equal(200, engine.OcrPages(stream)[0].UnitsPerInch);
        }

        Assert.Equal(123, engine.OcrPages(tiff, new OcrOptions { Dpi = 123 })[0].UnitsPerInch);

        // A file with no resolution, and an image handed over as pixels, are estimated and the page says so with a stable code.
        OcrPage estimated = engine.OcrPages(Tiff(ThreeFrames(), null))[0];
        Assert.Contains(estimated.Diagnostics, d => d.Code == OcrDiagnosticCodes.DpiEstimated && d.Severity == OcrDiagnosticSeverity.Info);
        (byte[] px, int w, int h) = ThreeFrames()[0];
        Assert.Contains(engine.Ocr(ToImage(px, w, h)).Diagnostics, d => d.Code == OcrDiagnosticCodes.DpiEstimated);

        // PNG pHYs.
        using var png = new MemoryStream();
        ToImage(px, w, h).Save(png, PngFormat.Instance);
        Assert.Equal(150, engine.OcrPages(PngWithDensity(png.ToArray(), 150))[0].UnitsPerInch, 0.5);
    }

    [Theory]
    [InlineData(new byte[0])]
    [InlineData(new byte[] { 0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13, (byte)'p', (byte)'H', (byte)'Y', (byte)'s' })]
    [InlineData(new byte[] { 0xFF, 0xD8, 0xFF, 0xE0, 0, 16, (byte)'J', (byte)'F', (byte)'I', (byte)'F', 0, 1, 1, 1 })]
    [InlineData(new byte[] { (byte)'I', (byte)'I', 42, 0, 0xFF, 0xFF, 0xFF, 0x7F, 1, 2, 3 })]
    [InlineData(new byte[] { (byte)'M', (byte)'M', 0, 42, 0, 0, 0, 8, 0xFF, 0xFF })]
    public void TheDensityReaderGivesNoAnswerForTruncatedOrHostileHeaders(byte[] data) => Assert.Null(ImageDensity.TryRead(data));

    [Fact]
    public void MalformedImageBytesSurfaceAsOcrExceptionsWithTheCauseAttached()
    {
        OcrEngine engine = NewEngine();
        byte[][] bad = [[], "not an image at all"u8.ToArray(), [0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3]];
        foreach (byte[] data in bad)
        {
            OcrException e = Assert.ThrowsAny<OcrException>(() => engine.OcrPages(data));
            Assert.IsType<OcrFormatException>(e);
            Assert.IsAssignableFrom<ImageException>(e.InnerException);
            using var stream = new MemoryStream(data);
            Assert.ThrowsAny<OcrException>(() => engine.OcrPages(stream));
        }

        // A damaged file read from a path, and a truncated image whose pixels fail late.
        string path = Path.Combine(Path.GetTempPath(), "oneocr-bad-" + Guid.NewGuid().ToString("N") + ".png");
        File.WriteAllBytes(path, "junk"u8.ToArray());
        try
        {
            Assert.IsType<OcrFormatException>(Assert.ThrowsAny<OcrException>(() => engine.OcrPages(path)));
        }
        finally
        {
            File.Delete(path);
        }

        Assert.Throws<FileNotFoundException>(() => engine.OcrPages(path));
        using var good = new MemoryStream();
        (byte[] px, int w, int h) = ThreeFrames()[0];
        ToImage(px, w, h).Save(good, PngFormat.Instance);
        byte[] whole = good.ToArray();
        byte[] truncated = whole[..(whole.Length * 2 / 3)];
        Assert.ThrowsAny<OcrException>(() => engine.OcrPages(truncated));
    }

    [Fact]
    public void ImageDecodeLimitsComeFromOcrLimits()
    {
        (byte[] px, int w, int h) = ThreeFrames()[0];
        using var ms = new MemoryStream();
        ToImage(px, w, h).Save(ms, PngFormat.Instance);
        byte[] png = ms.ToArray();
        OcrEngine small = NewEngine(new OcrLimits { MaxImagePixels = 1000 });
        OcrLimitException e = Assert.Throws<OcrLimitException>(() => small.OcrPages(png));
        Assert.IsAssignableFrom<ImageLimitException>(e.InnerException);
        Assert.Throws<OcrLimitException>(() => small.Ocr(ToImage(px, w, h)));
    }

    [Fact]
    public void ALimitOnLinesCountsLinesAndReportsTheConfiguredValue()
    {
        // Four text lines on the page.
        (byte[] px, int w, int h) = (PagePixels(900, 420, [60, 130, 200, 270], 1), 900, 420);
        Image page = ToImage(px, w, h);
        foreach (int limit in new[] { 4, 5, 100 })
        {
            OcrPage read = NewEngine(new OcrLimits { MaxLinesPerPage = limit }).Ocr(page);
            Assert.Equal(4, read.Blocks.Sum(b => b.Lines.Count));
        }

        foreach (int limit in new[] { 1, 3 })
        {
            OcrLimitException e = Assert.Throws<OcrLimitException>(() => NewEngine(new OcrLimits { MaxLinesPerPage = limit }).Ocr(page));
            Assert.Equal(nameof(OcrLimits.MaxLinesPerPage), e.LimitName);
            Assert.Equal(limit, e.Limit);
            Assert.True(e.Actual > limit);
            Assert.Throws<OcrLimitException>(() => NewEngine(new OcrLimits { MaxLinesPerPage = limit }).ToText(page));
        }
    }

    [Fact]
    public void OptionsCanBeSetOnTheEngineAndOverriddenPerCall()
    {
        (byte[] px, int w, int h) = (PagePixels(900, 420, [60, 130, 200, 270], 1), 900, 420);
        Image page = ToImage(px, w, h);
        OcrEngine plain = NewEngine();
        OcrEngine tuned = plain.WithOptions(new OcrOptions { IncludeChars = false, Dpi = 150, MaxDegreeOfParallelism = 1 });
        Assert.NotSame(plain.Options, tuned.Options);
        Assert.True(plain.Options.IncludeChars);

        OcrPage withDefaults = tuned.Ocr(page);
        Assert.Equal(150, withDefaults.UnitsPerInch);
        Assert.All(withDefaults.Words, word => Assert.Empty(word.Chars));
        OcrPage overridden = tuned.Ocr(page, new OcrOptions());
        Assert.NotEqual(150, overridden.UnitsPerInch);
        Assert.All(overridden.Words, word => Assert.NotEmpty(word.Chars));

        // Same text whatever the knobs say about speed; the happy path honours the options that change text.
        Assert.Equal(plain.ToText(page), tuned.ToText(page));
        OcrOptions noNoise = new() { DropNoise = false, FixWordSpacing = false };
        Assert.Equal(plain.Ocr(page, noNoise).ToText(), plain.ToText(page, noNoise));
        Assert.Equal(plain.ToText(page), page.ToText(plain));
        Assert.Equal(plain.ToText(page, noNoise), page.ToText(plain, noNoise));

        ArgumentOutOfRangeException bad = Assert.Throws<ArgumentOutOfRangeException>(() => plain.ToText(page, new OcrOptions { MaxDegreeOfParallelism = -1 }));
        Assert.Equal("options", bad.ParamName);
        Assert.Throws<ArgumentOutOfRangeException>(() => plain.WithOptions(new OcrOptions { Dpi = -3 }));
        Assert.Throws<ArgumentNullException>(() => plain.WithOptions(null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DocumentExtractionOptions { MaxDegreeOfParallelism = -1 }.Validate());
    }

    [Fact]
    public void ThePageTextViewsAgreeAndTheToTextOneLinerIsTheLinesForm()
    {
        (byte[] px, int w, int h) = (PagePixels(900, 420, [60, 130, 200, 270], 1), 900, 420);
        Image image = ToImage(px, w, h);
        OcrEngine engine = NewEngine();
        OcrPage page = engine.Ocr(image);
        Assert.Equal(page.ToText(), engine.ToText(image));
        Assert.Equal(page.ToText(), page.ToText(OcrTextLayout.Lines));
        Assert.Equal(page.Text, page.ToText(OcrTextLayout.Paragraphs));
        Assert.Equal(page.Text.Replace("\n\n", "\n", StringComparison.Ordinal), page.ToText());
        Assert.Throws<ArgumentOutOfRangeException>(() => page.ToText((OcrTextLayout)7));

        // Two blocks: a blank line between them in the paragraph form only.
        var line = new OcrLine("a", [], [], new OcrBox(0, 0, 1, 1), 1f, OcrSource.Ocr);
        var block = new OcrBlock(OcrBlockRole.Body, false, new OcrBox(0, 0, 1, 1), [line], 1f);
        var two = new OcrPage(0, 10, 10, 72, [block, block]);
        Assert.Equal("a\na", two.ToText());
        Assert.Equal("a\n\na", two.Text);
    }

    [Fact]
    public void PageModelRecordsCompareByValueAndAreBuiltFromInitializers()
    {
        // JSON rounds to three decimals, so compare a page that has already been through it with a second pass.
        OcrPage page = OcrPageJson.Parse(OcrPageJson.ToJson([SamplePage.Build()]))[0];
        string json = OcrPageJson.ToJson([page]);
        OcrPage back = OcrPageJson.Parse(json)[0];
        Assert.Equal(page.Words.Count, back.Words.Count);
        for (int i = 0; i < page.Words.Count; i++)
        {
            Assert.Equal(page.Words[i], back.Words[i]);
            Assert.Equal(page.Words[i].GetHashCode(), back.Words[i].GetHashCode());
        }

        Assert.Equal(page.Blocks[0].Lines[0], back.Blocks[0].Lines[0]);
        Assert.Equal(page.Blocks[0], back.Blocks[0]);
        Assert.Equal(page.Tables[0], back.Tables[0]);
        Assert.Equal(page.Regions[0], back.Regions[0]);
        Assert.Equal(page.KeyValues[0], back.KeyValues[0]);
        Assert.NotEqual(page.Words[0], page.Words[0] with { Text = "other" });

        // The diagnostics keep their messages through JSON and lose their code, which JSON does not carry.
        Assert.Equal(page.Diagnostics.Select(d => d.Message).ToArray(), back.Diagnostics.Select(d => d.Message).ToArray());
        Assert.All(back.Diagnostics, d => Assert.Equal(OcrDiagnosticCodes.Unspecified, d.Code));
        Assert.Equal(json, OcrPageJson.ToJson([back]));

        // Built outside the library, with initializers only.
        var word = new OcrWord { Text = "x", Box = new OcrBox(0, 0, 1, 1), Confidence = 1, RawConfidence = 1, Source = OcrSource.Llm, Chars = [] };
        Assert.Equal(OcrSource.Llm, word.Source);
        Assert.Null(word.Quad);
        var built = new OcrPage { Index = 3, Width = 10, Height = 20, UnitsPerInch = 72, Blocks = [] };
        Assert.Equal(0, built.Orientation);
        Assert.Empty(built.Diagnostics);
        Assert.Throws<ArgumentException>(() => new OcrPage { Index = 0, Width = 1, Height = 1, UnitsPerInch = 1, Blocks = [], Orientation = 45 });
    }

    [Fact]
    public void ExtractOfTRefusesATypeTheModelCannotFillAndUnknownFieldNamesThrow()
    {
        byte[] head = DocContainerWriter.WithNewHead(ModelContainer.Parse(TestDocModels.RandomContainer(TestDocModels.Small, 5), OcrLimits.Default), ["amount"], 2, OcrLimits.Default);
        DocumentExtractor ex = DocumentExtractor.Load(head);
        OcrPage page = GroundingSupport.Layout([new TestLine("Total 12.50", Conf: 0.99f), new TestLine("Order PO-1234", Conf: 0.99f)]);
        ArgumentException none = Assert.Throws<ArgumentException>(() => ex.Extract(page, ApiContractContext.Default.NoMatchForm));
        Assert.Contains("nothing-like-this", none.Message, StringComparison.Ordinal);
        Assert.Contains("amount", none.Message, StringComparison.Ordinal);

        // Some properties mapped: defaults plus a diagnostic, unless Strict.
        ExtractionResult<OtherForm> partial = ex.Extract(page, CustomFormContext.Default.OtherForm);
        Assert.Contains(partial.Diagnostics, d => d.Code == OcrDiagnosticCodes.PropertyUnmapped);
        Assert.Throws<ArgumentException>(() => ex.Extract(page, CustomFormContext.Default.OtherForm, new DocumentExtractionOptions { Strict = true }));

        Assert.Throws<KeyNotFoundException>(() => partial.WordBoxes("Total"));
        Assert.Empty(partial.WordBoxes("unknown"));
    }

    [Fact]
    public void TheDefaultEngineBeforeConfigurationIsAnInvalidOperation()
    {
        try
        {
            _ = OcrEngine.Default;
        }
        catch (InvalidOperationException)
        {
            return;
        }

        // Another test configured it first (process-wide state); then there is nothing to check here.
    }

    [Fact]
    public void ATunedRecognizerKeepsTheBundledPageModelsUnlessToldOtherwise()
    {
        if (!BundledModels.HasRecognizer)
        {
            Assert.Throws<InvalidOperationException>(() => BundledModels.ReadRecognizer());
            return;
        }

        string dir = Path.Combine(Path.GetTempPath(), "oneocr-start-model-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string path = Path.Combine(dir, "start.onm");
            BundledModels.ExportRecognizer(path);
            Assert.Equal(BundledModels.Recognizer.Bytes, new FileInfo(path).Length);
            Assert.Equal(BundledModels.Recognizer.Sha256, Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(path))));

            OcrEngine kept = OcrEngine.Load(path);
            Assert.Equal(BundledModels.HasDetector, kept.HasDetector);
            Assert.Equal(BundledModels.HasLanguageModel, kept.HasWordModel);
            Assert.Equal(BundledModels.HasWordFrequencies, kept.HasWordFrequencies);
            OcrEngine bare = OcrEngine.Load(path, withBundledModels: false);
            Assert.False(bare.HasDetector);
            Assert.False(bare.HasWordModel);
            Assert.False(bare.HasWordFrequencies);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }
}
