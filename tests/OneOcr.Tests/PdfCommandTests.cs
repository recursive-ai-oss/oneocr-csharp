using OneImage;
using OnePdf;
using OnePdf.Text;
using OneOcr.Pdf;
using OneOcr.Tool;

namespace OneOcr.Tests;

/// <summary>The <c>oneocr pdf</c> command driven in process: inputs in, a searchable PDF out, one-line errors with exit codes. Needs ONEOCR_MODEL.</summary>
public class PdfCommandTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "oneocr-pdfcmd-" + Guid.NewGuid().ToString("N"));

    public PdfCommandTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, true);

    private static string Model()
    {
        string? path = Environment.GetEnvironmentVariable("ONEOCR_MODEL");
        Assert.SkipWhen(path is null || !File.Exists(path), "ONEOCR_MODEL (a trained model container) is not set.");
        return path!;
    }

    private string Path_(string name) => Path.Combine(_dir, name);

    private static (int Code, string Out, string Err) Run(params string[] args)
    {
        var o = new StringWriter();
        var e = new StringWriter();
        int code = Program.RunAsync(args, o, e, CancellationToken.None).GetAwaiter().GetResult();
        return (code, o.ToString(), e.ToString());
    }

    private static string[] Words(PdfPageText t) => [.. t.Words.Select(static w => w.Text)];

    [Fact]
    public void AJpegBecomesASearchablePage()
    {
        string model = Model();
        string jpeg = Path_("scan.jpg");
        File.WriteAllBytes(jpeg, SearchablePdfTests.Encode(SearchablePdfTests.ScanImage(PdfFixtures.BodyA), JpegFormat.Instance));
        string output = Path_("scan.pdf");
        (int code, string stdout, string stderr) = Run("pdf", jpeg, "--out", output, "--model", model, "--dpi", "300", "--degree", "1");
        Assert.Equal(0, code);
        Assert.Equal(string.Empty, stderr);
        Assert.Matches(@"scan\.pdf: 1 page, 1 given a text layer, \d+ words, ", stdout);
        Assert.Single(stdout.Trim().Split('\n'));
        using var doc = PdfDocument.Open(File.ReadAllBytes(output));
        string[] got = Words(doc.Pages[0].ExtractText());
        Assert.Contains("Quarterly", got);
        Assert.Contains("harbor", got);

        // The same command again gives the same bytes.
        string again = Path_("again.pdf");
        Assert.Equal(0, Run("pdf", jpeg, "--out", again, "--model", model, "--dpi", "300", "--degree", "4").Code);
        Assert.Equal(File.ReadAllBytes(output), File.ReadAllBytes(again));
    }

    [Fact]
    public void ABilevelTiffAndAFolderMakeOnePdf()
    {
        string model = Model();
        var frames = new List<(byte[], int, int)>();
        foreach (string[] lines in new[] { PdfFixtures.BodyA, PdfFixtures.BodyB, PdfFixtures.Garbled })
        {
            (byte[] gray, int w, int h) = PdfFixtures.RenderLines(lines, 612, 792, 72, 12, 22, 92);
            frames.Add((gray.Select(static b => b >= 128 ? (byte)255 : (byte)0).ToArray(), w, h));
        }

        string tiff = Path_("fax.tif");
        File.WriteAllBytes(tiff, SearchablePdfTests.Tiff1(frames, 300));
        string output = Path_("fax.pdf");
        (int code, string stdout, string stderr) = Run("pdf", tiff, "--out", output, "--model", model);
        Assert.Equal(0, code);
        Assert.Equal(string.Empty, stderr);
        Assert.Matches(@"fax\.pdf: 3 pages, 3 given a text layer, \d+ words", stdout);
        using (var doc = PdfDocument.Open(File.ReadAllBytes(output)))
        {
            Assert.Equal(3, doc.PageCount);
            Assert.Contains("Cargo", Words(doc.Pages[0].ExtractText()));
        }

        // A folder and a file: images by file name, then the three TIFF frames.
        string folder = Path_("pages");
        Directory.CreateDirectory(folder);
        File.WriteAllBytes(Path.Combine(folder, "b.png"), SearchablePdfTests.Encode(SearchablePdfTests.ScanImage(PdfFixtures.BodyB), PngFormat.Instance));
        File.WriteAllBytes(Path.Combine(folder, "a.png"), SearchablePdfTests.Encode(SearchablePdfTests.ScanImage(PdfFixtures.BodyA), PngFormat.Instance));
        File.WriteAllText(Path.Combine(folder, "notes.txt"), "not an image");
        string combined = Path_("all.pdf");
        Assert.Equal(0, Run("pdf", folder, tiff, "--out", combined, "--model", model, "--pdfa", "--title", "Harbor").Code);
        using var all = PdfDocument.Open(File.ReadAllBytes(combined));
        Assert.Equal(5, all.PageCount);
        Assert.Contains("Quarterly", Words(all.Pages[0].ExtractText()));
        Assert.Contains("Meeting", Words(all.Pages[1].ExtractText()));
    }

    [Fact]
    public void AMixedPdfIsMadeSearchableAndNotesGoToStandardError()
    {
        string model = Model();
        (byte[] sample, PageTruth[] _, _) = PdfFixtures.Sample();
        string input = Path_("mixed.pdf");
        File.WriteAllBytes(input, sample);
        string output = Path_("mixed-out.pdf");
        (int code, string stdout, string stderr) = Run("pdf", input, "--out", output, "--model", model);
        Assert.Equal(0, code);
        Assert.Matches(@"mixed-out\.pdf: 6 pages, 5 given a text layer, \d+ words", stdout);
        Assert.Contains("oneocr: Page 4: removed", stderr, StringComparison.Ordinal);
        Assert.Contains("/ActualText", stderr, StringComparison.Ordinal);
        Assert.All(stderr.TrimEnd().Split('\n'), static l => Assert.StartsWith("oneocr: ", l, StringComparison.Ordinal));

        // The output may be the input.
        File.Copy(input, Path_("inplace.pdf"));
        Assert.Equal(0, Run("pdf", Path_("inplace.pdf"), "--out", Path_("inplace.pdf"), "--model", model).Code);
        Assert.Equal(File.ReadAllBytes(output), File.ReadAllBytes(Path_("inplace.pdf")));
    }

    [Fact]
    public void BadInputIsOneLineAndANonZeroExit()
    {
        string model = Model();
        string text = Path_("readme.txt");
        File.WriteAllText(text, "hello");
        string good = Path_("good.png");
        File.WriteAllBytes(good, SearchablePdfTests.Encode(SearchablePdfTests.ScanImage(PdfFixtures.BodyA), PngFormat.Instance));
        string pdf = Path_("x.pdf");
        File.WriteAllBytes(pdf, PdfFixtures.Sample().Pdf);
        string output = Path_("never.pdf");

        (string Name, string[] Args, int Code, string Message)[] cases =
        [
            ("no inputs", ["pdf", "--out", output], 2, "Name at least one input"),
            ("no out", ["pdf", good], 2, "--out is required"),
            ("missing input", ["pdf", Path_("nope.png"), "--out", output], 2, "neither a file nor a folder"),
            ("unknown option", ["pdf", good, "--out", output, "--bogus"], 2, "Unknown option --bogus"),
            ("images and a pdf", ["pdf", good, pdf, "--out", output, "--model", model], 2, "not both"),
            ("two pdfs", ["pdf", pdf, pdf, "--out", output, "--model", model], 2, "one PDF at a time"),
            ("output is an input", ["pdf", good, "--out", good, "--model", model], 2, "is also an input"),
            ("not an image", ["pdf", text, "--out", output, "--model", model], 1, string.Empty),
        ];
        foreach ((string name, string[] args, int expected, string message) in cases)
        {
            (int code, string stdout, string stderr) = Run(args);
            Assert.True(code == expected, $"{name}: exit {code}, stderr {stderr}");
            Assert.Equal(string.Empty, stdout);
            Assert.Single(stderr.TrimEnd().Split('\n'));
            Assert.StartsWith("oneocr: ", stderr, StringComparison.Ordinal);
            Assert.Contains(message, stderr, StringComparison.Ordinal);
        }

        Assert.False(File.Exists(output));
    }

    [Fact]
    public void HelpListsThePdfCommand()
    {
        (int code, string stdout, _) = Run("help");
        Assert.Equal(0, code);
        Assert.Contains("oneocr pdf ", stdout, StringComparison.Ordinal);
        Assert.Contains("--pdfa", stdout, StringComparison.Ordinal);
    }
}
