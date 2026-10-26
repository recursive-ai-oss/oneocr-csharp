using System.Diagnostics;
using System.Threading.Tasks;
using OneImage;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Pdf;
using OnePdf;

namespace OneOcr.Tests;

/// <summary>The Async twins run the same code as the sync calls, so their results are identical, and cancellation is honoured.</summary>
public class AsyncTests
{
    private static readonly OcrEngine s_engine = OcrEngine.Create(ModelContainer.Parse(TestModels.RandomContainer(TestModels.Small, 5)), null, KernelPath.Scalar);

    private static Image Page(int seed, int lines = 8)
    {
        int[] tops = [.. Enumerable.Range(0, lines).Select(i => 60 + (60 * i))];
        (byte[] px, _) = EngineTests.SyntheticPage(1200, 100 + (60 * lines), tops, 14, 8, plain: false, seed: seed);
        return Image.FromPixels<Gray8>(px.Select(b => new Gray8(b)).ToArray(), 1200, 100 + (60 * lines));
    }

    private static string Json(IReadOnlyList<OcrPage> pages) => OcrPageJson.ToJson(pages);

    [Fact]
    public async Task AsyncPagesAndTextAreIdenticalToSync()
    {
        foreach (int seed in new[] { 1, 2, 3 })
        {
            Image image = Page(seed);
            OcrPage sync = image.Ocr(s_engine);
            Assert.NotEmpty(sync.Words);
            Assert.Equal(Json([sync]), Json([await image.OcrAsync(s_engine)]));
            Assert.Equal(Json([sync]), Json([await s_engine.OcrAsync(image)]));
            Assert.Equal(image.ToText(s_engine), await image.ToTextAsync(s_engine));
            Assert.Equal(s_engine.ToText(image), await s_engine.ToTextAsync(image));
            Assert.Equal(image.ToMarkdown(s_engine), await image.ToMarkdownAsync(s_engine));
        }
    }

    [Fact]
    public async Task ManyImagesGiveTheSamePagesAsOneAtATimeWithConsecutiveIndices()
    {
        Image[] images = [Page(1), Page(2), Page(3)];
        IReadOnlyList<OcrPage> sync = s_engine.Ocr(images, new OcrOptions { PageIndex = 4 });
        IReadOnlyList<OcrPage> async = await s_engine.OcrAsync(images, new OcrOptions { PageIndex = 4 });
        Assert.Equal([4, 5, 6], sync.Select(p => p.Index));
        Assert.Equal(Json(sync), Json(async));
        for (int i = 0; i < images.Length; i++)
        {
            Assert.Equal(Json([s_engine.Ocr(images[i], new OcrOptions { PageIndex = 4 + i })]), Json([sync[i]]));
        }

        Assert.Empty(s_engine.Ocr([]));
        Assert.Throws<ArgumentNullException>(() => s_engine.Ocr((IEnumerable<Image>)null!));
        Assert.Throws<ArgumentNullException>(() => s_engine.Ocr([images[0], null!]));
    }

    [Fact]
    public async Task ACancelledTokenThrowsBeforeAndBetweenPages()
    {
        Image image = Page(1);
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => s_engine.Ocr(image, null, cts.Token));
        Assert.ThrowsAny<OperationCanceledException>(() => s_engine.ToText(image, 0, cts.Token));
        Assert.ThrowsAny<OperationCanceledException>(() => image.ToMarkdown(s_engine, null, cts.Token));
        Assert.ThrowsAny<OperationCanceledException>(() => s_engine.Ocr([image, image], null, cts.Token));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => image.OcrAsync(s_engine, null, cts.Token));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => image.ToTextAsync(s_engine, cts.Token));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => image.ToMarkdownAsync(s_engine, null, cts.Token));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => s_engine.OcrAsync([image], null, cts.Token));
    }

    [Fact]
    public async Task CancellingWhileRunningStopsBetweenLinesAndPages()
    {
        Image[] images = [.. Enumerable.Range(0, 40).Select(i => Page(i, 12))];
        var sw = Stopwatch.StartNew();
        _ = s_engine.Ocr(images[0]);
        TimeSpan onePage = sw.Elapsed;

        using var cts = new CancellationTokenSource();
        sw.Restart();
        Task<IReadOnlyList<OcrPage>> task = s_engine.OcrAsync(images, null, cts.Token);
        await Task.Delay(onePage * 2);
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => task);
        // Far less than the 40 pages it was asked to read.
        Assert.True(sw.Elapsed < onePage * 12, $"stopped after {sw.Elapsed}, one page takes {onePage}");
    }

    [Fact]
    public async Task PdfAsyncTwinsGiveTheSameTextAndPages()
    {
        (byte[] pdf, _, _) = PdfFixtures.Sample();
        var options = new PdfOcrOptions { ReadImages = false, CrossCheckLines = 0 };
        using PdfDocument doc = PdfDocument.Open(pdf);
        IReadOnlyList<OcrPage> sync = doc.Ocr(s_engine, options);
        Assert.Equal(Json(sync), Json(await doc.OcrAsync(s_engine, options)));
        Assert.Equal(doc.ToText(s_engine, options), await doc.ToTextAsync(s_engine, options));
        Assert.Equal(doc.ToMarkdown(s_engine, null, options), await doc.ToMarkdownAsync(s_engine, null, options));
        PdfPage page = doc.Pages[0];
        Assert.Equal(page.ToText(s_engine, options), await page.ToTextAsync(s_engine, options));
        Assert.Equal(page.ToMarkdown(s_engine, null, options), await page.ToMarkdownAsync(s_engine, null, options));

        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => doc.OcrAsync(s_engine, options, cts.Token));
        Assert.ThrowsAny<OperationCanceledException>(() => doc.Ocr(s_engine, options, cts.Token));
    }

    /// <summary>With ONEOCR_MODEL and ONEOCR_ASYNC_PAGES (a folder of page images) set: real pages, real model, bundled detector when packed.</summary>
    [Fact]
    public async Task RealPagesReadTheSameAsyncAsSync()
    {
        string? model = Environment.GetEnvironmentVariable("ONEOCR_MODEL"), dir = Environment.GetEnvironmentVariable("ONEOCR_ASYNC_PAGES");
        if (model is null || dir is null || !File.Exists(model) || !Directory.Exists(dir))
        {
            return;
        }

        OcrEngine engine = OcrEngine.Load(model);
        foreach (string file in Directory.EnumerateFiles(dir, "*.png").Order().Take(3))
        {
            Image image = Image.Load(file);
            OcrPage sync = image.Ocr(engine);
            Assert.NotEmpty(sync.Words);
            Assert.Equal(Json([sync]), Json([await image.OcrAsync(engine)]));
            Assert.Equal(image.ToText(engine), await image.ToTextAsync(engine));
        }
    }
}
