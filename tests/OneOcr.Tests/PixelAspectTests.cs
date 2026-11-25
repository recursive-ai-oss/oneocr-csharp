using System.Buffers.Binary;
using OneImage;
using OneOcr.Kernels;
using OneOcr.Model;

namespace OneOcr.Tests;

/// <summary>Captures whose pixels are not square (a standard-mode fax is 200 by 100 dots per inch): the declared resolutions are read, the page is read stretched to square pixels, and the result is in the pixels of the image given.</summary>
public class PixelAspectTests
{
    private static OcrEngine NewEngine(OcrLimits? limits = null) =>
        OcrEngine.Create(ModelContainer.Parse(TestModels.RandomContainer(TestModels.Small, 5)), limits, KernelPath.Scalar);

    private static Image Page(int width, int height, int seed) =>
        Image.FromPixels<Gray8>(EngineTests.SyntheticPage(width, height, [.. Enumerable.Range(0, (height - 20) / 40).Select(i => 20 + (40 * i))], 14, 8, seed: seed).Pixels.Select(b => new Gray8(b)).ToArray(), width, height);

    // A one-frame uncompressed 8-bit TIFF with XResolution, YResolution and a unit (little endian, resolutions as whole numbers).
    private static byte[] Tiff(byte[] pixels, int width, int height, int? xdpi, int? ydpi, int unit = 2)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write((byte)'I');
        w.Write((byte)'I');
        w.Write((ushort)42);
        w.Write(8u + (uint)pixels.Length + 16);
        w.Write(pixels);
        long xAt = ms.Position;
        w.Write((uint)(xdpi ?? 1));
        w.Write(1u);
        long yAt = ms.Position;
        w.Write((uint)(ydpi ?? 1));
        w.Write(1u);
        var entries = new List<(ushort Tag, ushort Type, uint Count, uint Value)>
        {
            (256, 4, 1, (uint)width), (257, 4, 1, (uint)height), (258, 3, 1, 8), (259, 3, 1, 1), (262, 3, 1, 1), (273, 4, 1, 8), (277, 3, 1, 1), (278, 4, 1, (uint)height), (279, 4, 1, (uint)pixels.Length),
        };
        if (xdpi is not null)
        {
            entries.Add((282, 5, 1, (uint)xAt));
        }

        if (ydpi is not null)
        {
            entries.Add((283, 5, 1, (uint)yAt));
        }

        entries.Add((296, 3, 1, (uint)unit));
        w.Write((ushort)entries.Count);
        foreach ((ushort tag, ushort type, uint count, uint value) in entries)
        {
            w.Write(tag);
            w.Write(type);
            w.Write(count);
            w.Write(value);
        }

        w.Write(0u);
        return ms.ToArray();
    }

    [Fact]
    public void TheDensityReaderGivesBothResolutionsOfTiffPngAndJpeg()
    {
        byte[] px = new byte[64 * 32];
        Assert.Equal((200.0, 100.0), ImageDensity.TryReadPair(Tiff(px, 64, 32, 200, 100)));
        Assert.Equal(200.0, ImageDensity.TryRead(Tiff(px, 64, 32, 200, 100)));

        // No YResolution: square pixels, as TIFF readers take it. Centimetres are converted. No unit: no answer.
        Assert.Equal((200.0, 200.0), ImageDensity.TryReadPair(Tiff(px, 64, 32, 200, null)));
        (double X, double Y) cm = ImageDensity.TryReadPair(Tiff(px, 64, 32, 80, 40, unit: 3))!.Value;
        Assert.Equal(80 * 2.54, cm.X, 6);
        Assert.Equal(40 * 2.54, cm.Y, 6);
        Assert.Null(ImageDensity.TryReadPair(Tiff(px, 64, 32, 200, 100, unit: 1)));

        // A resolution that is out of range for one axis falls back to the other axis instead of dropping the file's answer.
        Assert.Equal((200.0, 200.0), ImageDensity.TryReadPair(Tiff(px, 64, 32, 200, 5)));

        // PNG pHYs: pixels per metre on each axis.
        byte[] phys = new byte[9];
        BinaryPrimitives.WriteUInt32BigEndian(phys, 7874); // 200 per inch
        BinaryPrimitives.WriteUInt32BigEndian(phys.AsSpan(4), 3937); // 100 per inch
        phys[8] = 1;
        (double X, double Y) png = ImageDensity.TryReadPair(PngWith(phys))!.Value;
        Assert.Equal(200, png.X, 0);
        Assert.Equal(100, png.Y, 0);

        // JFIF: units 1 (inch), 200 by 100.
        byte[] jpeg = [0xFF, 0xD8, 0xFF, 0xE0, 0, 16, (byte)'J', (byte)'F', (byte)'I', (byte)'F', 0, 1, 1, 1, 0, 200, 0, 100, 0, 0, 0xFF, 0xDA];
        Assert.Equal((200.0, 100.0), ImageDensity.TryReadPair(jpeg));
    }

    private static byte[] PngWith(byte[] physBody)
    {
        using var ms = new MemoryStream();
        ms.Write([0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        ms.Write([0, 0, 0, 9, (byte)'p', (byte)'H', (byte)'Y', (byte)'s']);
        ms.Write(physBody);
        ms.Write([0, 0, 0, 0]);
        ms.Write([0, 0, 0, 0, (byte)'I', (byte)'D', (byte)'A', (byte)'T']);
        return ms.ToArray();
    }

    [Fact]
    public void ARectangularCaptureIsReadStretchedAndTheBoxesStayInTheImageGiven()
    {
        OcrEngine engine = NewEngine();
        Image page = Page(400, 100, 3);
        OcrPage square = engine.Ocr(page, new OcrOptions { Dpi = 200 });
        OcrPage stretched = engine.Ocr(page, new OcrOptions { Dpi = 200, VerticalDpi = 100 });

        Assert.Equal(400, stretched.Width);
        Assert.Equal(100, stretched.Height);
        Assert.Equal(200, stretched.UnitsPerInch);
        Assert.Equal(100, stretched.VerticalUnitsPerInch);
        Assert.Equal(200, square.VerticalUnitsPerInch);
        Assert.Contains(stretched.Diagnostics, d => d.Code == OcrDiagnosticCodes.PixelsStretched);
        Assert.DoesNotContain(square.Diagnostics, d => d.Code == OcrDiagnosticCodes.PixelsStretched);

        // Every box is inside the image given, not the stretched page (which is twice as tall).
        foreach (OcrWord word in stretched.Words)
        {
            Assert.InRange(word.Box.Left, -1e-6, 400 + 1e-6);
            Assert.InRange(word.Box.Right, -1e-6, 400 + 1e-6);
            Assert.InRange(word.Box.Top, -1e-6, 100 + 1e-6);
            Assert.InRange(word.Box.Bottom, -1e-6, 100 + 1e-6);
        }

        // Identical at degree 1 and 4, and through the async twin.
        OcrPage again = engine.Ocr(page, new OcrOptions { Dpi = 200, VerticalDpi = 100, MaxDegreeOfParallelism = 4 });
        Assert.Equal(OcrPageJson.ToJson([stretched]), OcrPageJson.ToJson([again]));
    }

    [Fact]
    public void SquarePixelsAndNearSquarePixelsAreReadAsTheyAre()
    {
        OcrEngine engine = NewEngine();
        Image page = Page(300, 100, 4);
        string plain = OcrPageJson.ToJson([engine.Ocr(page, new OcrOptions { Dpi = 300 })]);
        // Scanners declare 300 by 298: within the 5% tolerance nothing is stretched and the text, boxes and notes are the same.
        OcrPage near = engine.Ocr(page, new OcrOptions { Dpi = 300, VerticalDpi = 298 });
        Assert.DoesNotContain(near.Diagnostics, d => d.Code is OcrDiagnosticCodes.PixelsStretched or OcrDiagnosticCodes.PixelsNotStretched);
        Assert.Equal(plain, OcrPageJson.ToJson([near]));
        // Without Dpi there is nothing to compare the vertical resolution with.
        Assert.DoesNotContain(engine.Ocr(page, new OcrOptions { VerticalDpi = 100 }).Diagnostics, d => d.Code == OcrDiagnosticCodes.PixelsStretched);
    }

    [Fact]
    public void ABadHeaderIsIgnoredWithANote()
    {
        OcrEngine engine = NewEngine();
        OcrPage page = engine.Ocr(Page(300, 100, 5), new OcrOptions { Dpi = 300, VerticalDpi = 30 });
        Assert.Contains(page.Diagnostics, d => d.Code == OcrDiagnosticCodes.PixelsNotStretched);
        Assert.Equal(100, page.Height);
        Assert.Equal(page.UnitsPerInch, page.VerticalUnitsPerInch);

        // A stretched page that would pass the pixel limit is read as it is.
        OcrEngine tight = NewEngine(new OcrLimits { MaxImagePixels = 300 * 100 + 10 });
        Assert.Contains(tight.Ocr(Page(300, 100, 5), new OcrOptions { Dpi = 200, VerticalDpi = 100 }).Diagnostics, d => d.Code == OcrDiagnosticCodes.PixelsNotStretched);
    }

    [Fact]
    public void ADeclaredResolutionOfAFileReachesThePage()
    {
        OcrEngine engine = NewEngine();
        byte[] px = EngineTests.SyntheticPage(400, 100, [20, 60], 14, 8, seed: 6).Pixels;
        OcrPage page = engine.OcrPages(Tiff(px, 400, 100, 200, 100))[0];
        Assert.Equal(200, page.UnitsPerInch);
        Assert.Equal(100, page.VerticalUnitsPerInch);
        Assert.Contains(page.Diagnostics, d => d.Code == OcrDiagnosticCodes.PixelsStretched);
        Assert.Equal(400, page.Width);
        Assert.Equal(100, page.Height);

        // The caller's own Dpi wins and the file's vertical resolution is not applied to it.
        OcrPage given = engine.OcrPages(Tiff(px, 400, 100, 200, 100), new OcrOptions { Dpi = 123 })[0];
        Assert.Equal(123, given.UnitsPerInch);
        Assert.Equal(123, given.VerticalUnitsPerInch);
    }

    [Fact]
    public void TheVerticalResolutionSurvivesPageJsonAndOnlyShowsWhenItDiffers()
    {
        OcrEngine engine = NewEngine();
        OcrPage stretched = engine.Ocr(Page(400, 100, 7), new OcrOptions { Dpi = 200, VerticalDpi = 100 });
        string json = OcrPageJson.ToJson([stretched]);
        Assert.Contains("\"unitsPerInchY\":", json, StringComparison.Ordinal);
        OcrPage back = OcrPageJson.Parse(json)[0];
        Assert.Equal(100, back.VerticalUnitsPerInch);
        Assert.Equal(200, back.UnitsPerInch);
        Assert.DoesNotContain("unitsPerInchY", OcrPageJson.ToJson([engine.Ocr(Page(400, 100, 7), new OcrOptions { Dpi = 200 })]), StringComparison.Ordinal);
        Assert.Throws<ArgumentOutOfRangeException>(() => engine.Ocr(Page(40, 20, 1), new OcrOptions { Dpi = 200, VerticalDpi = 0 }));
    }

    [Fact]
    public void SmoothingIsTheIntegerBinomialFilterAndLeavesFlatPagesAlone()
    {
        byte[] flat = Enumerable.Repeat((byte)200, 9 * 7).ToArray();
        Layout.PageStretch.Smooth(flat, 9, 7);
        Assert.All(flat, b => Assert.Equal(200, b));

        // One bright pixel on black spreads symmetrically, stays inside its 5 x 5 neighbourhood (two passes of a 3-tap filter in each direction)
        // and loses height; the same input gives the same output.
        byte[] dot = new byte[9 * 9];
        dot[(4 * 9) + 4] = 240;
        byte[] a = (byte[])dot.Clone(), b = (byte[])dot.Clone();
        Layout.PageStretch.Smooth(a, 9, 9);
        Layout.PageStretch.Smooth(b, 9, 9);
        Assert.Equal(a, b);
        Assert.InRange(a[(4 * 9) + 4], 1, 239);
        Assert.Equal(a[(4 * 9) + 3], a[(4 * 9) + 5]);
        Assert.Equal(a[(3 * 9) + 4], a[(5 * 9) + 4]);
        Assert.Equal(a[(4 * 9) + 3], a[(3 * 9) + 4]);
        Assert.Equal(0, a[(4 * 9) + 0]);
        Assert.Equal(0, a[(0 * 9) + 4]);
    }

    [Fact]
    public void OnlyOneBitPagesAreCalledBilevel()
    {
        Assert.True(Layout.PageStretch.IsBilevel(Enumerable.Range(0, 1000).Select(i => i % 3 == 0 ? (byte)0 : (byte)255).ToArray()));
        Assert.False(Layout.PageStretch.IsBilevel(Enumerable.Range(0, 1000).Select(i => (byte)(i % 256)).ToArray()));
        Assert.False(Layout.PageStretch.IsBilevel([]));
    }

    /// <summary>A generated standard-mode fax (CCITT Group 3, 1-bit, 200 x 100 dpi) of a few statement lines; needs ONEOCR_MODEL.</summary>
    [Fact]
    public void AStandardModeGroup3FaxIsReadAtItsRealShape()
    {
        string? model = Environment.GetEnvironmentVariable("ONEOCR_MODEL");
        Assert.SkipWhen(model is null || !File.Exists(model), "ONEOCR_MODEL (a trained model container) is not set.");
        byte[] tiff = File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "testdata", "fax", "std-g3.tif"));
        Assert.Equal((200.0, 100.0), ImageDensity.TryReadPair(tiff));

        OcrPage page = OcrEngine.Load(model!).OcrPages(tiff)[0];
        Assert.Equal(1728, page.Width);
        Assert.Equal(100, page.VerticalUnitsPerInch);
        Assert.Contains(page.Diagnostics, d => d.Code == OcrDiagnosticCodes.PixelsStretched);
        Assert.All(page.Words, w => Assert.InRange(w.Box.Bottom, 0, page.Height + 1e-6));
        string text = page.ToText();
        foreach (string amount in new[] { "1,234.56", "2,500.00", "2,421.81" })
        {
            Assert.Contains(amount, text, StringComparison.Ordinal);
        }
    }
}
