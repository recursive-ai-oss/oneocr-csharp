using System.Diagnostics;
using System.Globalization;
using OneImage;
using OneOcr.Decoding;
using OneOcr.Pdf;

namespace OneOcr.Tool;

/// <summary>
/// <c>oneocr pdf</c>: makes a searchable PDF. Images (PNG, JPEG, TIFF with any number of frames and the other formats OneImage reads) become
/// the pages of one PDF with an invisible text layer; a PDF gets a layer on the pages that need one and keeps everything else. Nothing here
/// touches the network.
/// </summary>
internal static class PdfCommand
{
    /// <summary>Runs the command and returns the exit code.</summary>
    /// <param name="args">The arguments after the command name: input files and folders first, then the options.</param>
    /// <param name="stdout">Receives the summary line.</param>
    /// <param name="stderr">Receives warnings and notes about the pages.</param>
    /// <param name="cancellationToken">Cancels between pages.</param>
    public static int Run(string[] args, TextWriter stdout, TextWriter stderr, CancellationToken cancellationToken)
    {
        int firstOption = 0;
        while (firstOption < args.Length && !args[firstOption].StartsWith("--", StringComparison.Ordinal))
        {
            firstOption++;
        }

        string[] inputs = args[..firstOption];
        var f = new Flags(args.AsSpan(firstOption));
        string outFile = f.Required("out");
        string? modelFile = f.Optional("model"), detectorFile = f.Optional("detector"), wordModelFile = f.Optional("wordlm"), wordFreqFile = f.Optional("wordfreq"), title = f.Optional("title");
        int degree = f.Int("degree", 0);
        double dpi = f.Double("dpi", 0);
        bool pdfa = f.Switch("pdfa"), keepLayer = f.Switch("keep-ocr-layer");
        f.Done();

        if (inputs.Length == 0)
        {
            throw new UsageException("Name at least one input: image files, folders of images, or one PDF.");
        }

        if (detectorFile is not null && modelFile is null)
        {
            throw new UsageException("--detector goes with --model; without --model the bundled models are used.");
        }

        if (dpi < 0 || !double.IsFinite(dpi))
        {
            throw new UsageException("--dpi must be a positive number.");
        }

        List<string> files = Expand(inputs);
        if (files.Count == 0)
        {
            throw new UsageException("The inputs hold no image or PDF files.");
        }

        List<string> pdfs = [.. files.Where(IsPdf)];
        if (pdfs.Count > 0 && files.Count > 1)
        {
            throw new UsageException(pdfs.Count == files.Count
                ? "Give one PDF at a time: a PDF is made searchable page for page, several PDFs are not merged."
                : "Give images or one PDF, not both: a PDF keeps its own pages, images become new ones.");
        }

        string outFull = Path.GetFullPath(outFile);
        if (pdfs.Count == 0 && files.Any(p => string.Equals(Path.GetFullPath(p), outFull, StringComparison.Ordinal)))
        {
            throw new UsageException("--out '" + outFile + "' is also an input.");
        }

        OcrEngine engine = modelFile is null ? OcrEngine.LoadBundled() : OcrEngine.Load(modelFile);
        if (detectorFile is not null)
        {
            engine = engine.WithDetector(detectorFile);
        }

        if (wordModelFile is not null)
        {
            engine = engine.WithWordModel(CharLanguageModel.Load(wordModelFile));
        }

        if (wordFreqFile is not null)
        {
            engine = engine.WithWordFrequencies(WordFrequencyModel.Load(wordFreqFile));
        }

        if (engine.PerformanceNotice is { } notice)
        {
            stderr.WriteLine("oneocr: warning: " + notice);
        }

        var options = new SearchablePdfOptions
        {
            Ocr = new OcrOptions { MaxDegreeOfParallelism = degree, Dpi = dpi > 0 ? dpi : null },
            Pdf = new PdfOcrOptions { MaxDegreeOfParallelism = degree, TrustExistingOcrLayer = keepLayer },
            Conformance = pdfa ? SearchablePdfConformance.PdfA2b : SearchablePdfConformance.None,
            Title = title,
        };

        var clock = Stopwatch.StartNew();
        SearchablePdfResult result = pdfs.Count == 1
            ? SearchablePdf.MakeSearchable(files[0], outFile, engine, options, cancellationToken)
            : SearchablePdf.FromFiles(files, outFile, engine, options, cancellationToken);

        foreach (string d in result.Diagnostics)
        {
            stderr.WriteLine("oneocr: " + d);
        }

        stdout.WriteLine(string.Create(
            CultureInfo.InvariantCulture,
            $"{outFile}: {result.Pages.Count} {(result.Pages.Count == 1 ? "page" : "pages")}, {result.PagesWithTextLayer} given a text layer, {result.WordsWritten} words, {clock.Elapsed.TotalSeconds:0.0} s"));
        return 0;
    }

    // Files in the order given; a folder contributes its images and PDFs by file name. A missing input is a usage error, not an empty result.
    private static List<string> Expand(string[] inputs)
    {
        var files = new List<string>();
        foreach (string input in inputs)
        {
            if (Directory.Exists(input))
            {
                files.AddRange(Directory.GetFiles(input).Where(static p => IsPdfName(p) || IsImageName(p)).Order(StringComparer.Ordinal));
            }
            else if (File.Exists(input))
            {
                files.Add(input);
            }
            else
            {
                throw new UsageException("Input '" + input + "' is neither a file nor a folder.");
            }
        }

        return files;
    }

    private static bool IsPdfName(string file) => file.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase);

    private static bool IsImageName(string file) =>
        file.EndsWith(".png", StringComparison.OrdinalIgnoreCase) || file.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) || file.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
        file.EndsWith(".tif", StringComparison.OrdinalIgnoreCase) || file.EndsWith(".tiff", StringComparison.OrdinalIgnoreCase);

    // A PDF by its signature, not only its name: a renamed file is still one.
    private static bool IsPdf(string file)
    {
        Span<byte> head = stackalloc byte[5];
        using FileStream s = File.OpenRead(file);
        return s.Read(head) == 5 && head.SequenceEqual("%PDF-"u8);
    }
}
