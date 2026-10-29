using System.Diagnostics;
using System.Globalization;
using OneImage;
using OneOcr.Pdf;
using OneOcr.Training;

namespace OneOcr.Fuzz;

/// <summary>
/// The no-crash, no-hang sweep: the shipped engine (bundled models) over every file under the given paths, with a time limit per
/// file. Images go through the page path, PDFs through the PDF path, JSON files through the page-model and review readers (a typed
/// error is the right answer for the reproducers in testdata). Typed errors are counted, anything else is a failure.
///
///   OneOcr.Fuzz sweep --degree N --timeout SECONDS [--out FILE] PATH...
/// </summary>
internal static class Sweep
{
    private static readonly string[] ImageExtensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"];

    public static int Run(string[] args)
    {
        int degree = 1;
        double timeout = 60;
        string? outFile = null;
        var roots = new List<string>();
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--degree":
                    degree = int.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--timeout":
                    timeout = double.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--out":
                    outFile = args[++i];
                    break;
                default:
                    roots.Add(args[i]);
                    break;
            }
        }

        OcrEngine engine = OcrEngine.LoadBundled();
        var options = new OcrOptions { MaxDegreeOfParallelism = degree };
        var pdfOptions = new PdfOcrOptions();
        var log = new StringWriter();
        long files = 0, ok = 0, typed = 0, failures = 0, timeouts = 0, skipped = 0;
        var slowest = new List<(double Seconds, string File)>();
        var clock = Stopwatch.StartNew();

        foreach (string file in roots.SelectMany(Expand))
        {
            string ext = Path.GetExtension(file).ToLowerInvariant();
            Action<CancellationToken>? work = Plan(file, ext, engine, options, pdfOptions);
            if (work is null)
            {
                skipped++;
                continue;
            }

            files++;
            var sw = Stopwatch.StartNew();
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(timeout));
            Exception? error = null;
            var task = Task.Run(() =>
            {
                try
                {
                    work(cts.Token);
                }
                catch (Exception e)
                {
                    error = e;
                }
            });
            bool finished = task.Wait(TimeSpan.FromSeconds(timeout + 30));
            double seconds = sw.Elapsed.TotalSeconds;
            slowest.Add((seconds, file));
            if (!finished)
            {
                failures++;
                log.WriteLine($"HANG {file} (no end {timeout + 30:F0} s after the limit)");
                Report(log, outFile, files, ok, typed, failures, timeouts, skipped, slowest, clock.Elapsed);
                return 3;
            }

            switch (error)
            {
                case null:
                    ok++;
                    break;
                case OcrFormatException or OcrLimitException or ImageException:
                    typed++;
                    break;
                case OperationCanceledException:
                    timeouts++;
                    log.WriteLine($"TIMEOUT {file} ({seconds:F1} s)");
                    break;
                default:
                    failures++;
                    log.WriteLine($"FAIL {file}: {error}");
                    break;
            }

            if (seconds > timeout)
            {
                log.WriteLine($"SLOW {file} ({seconds:F1} s)");
            }
        }

        Report(log, outFile, files, ok, typed, failures, timeouts, skipped, slowest, clock.Elapsed);
        return failures + timeouts == 0 ? 0 : 1;
    }

    private static IEnumerable<string> Expand(string root) =>
        Directory.Exists(root) ? Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories).Order(StringComparer.Ordinal) : [root];

    private static Action<CancellationToken>? Plan(string file, string ext, OcrEngine engine, OcrOptions options, PdfOcrOptions pdfOptions)
    {
        // Corpus files carry no useful extension; their target directory says what they are.
        string[] parts = file.Split('/');
        bool pdfCorpus = parts.Contains("pdf") && (parts.Contains("corpus") || parts.Contains("min"));
        bool imageCorpus = (parts.Contains("image") || parts.Contains("image-real")) && (parts.Contains("corpus") || parts.Contains("min"));
        if (ext == ".pdf" || pdfCorpus)
        {
            return ct => RoundTrip(PdfOcr.Read(File.ReadAllBytes(file), engine, pdfOptions, ct));
        }

        if (ext == ".json" && !imageCorpus)
        {
            bool review = Path.GetFileName(file).StartsWith("review", StringComparison.Ordinal);
            return _ =>
            {
                byte[] bytes = File.ReadAllBytes(file);
                if (review)
                {
                    ReviewImport.Parse(bytes, OcrLimits.Default);
                }
                else
                {
                    OcrPageJson.Read(bytes);
                }
            };
        }

        if (imageCorpus || (ImageExtensions.Contains(ext) && ext != ".seed"))
        {
            return ct => RoundTrip([engine.Ocr(Image.Load(File.ReadAllBytes(file)), options, ct)]);
        }

        return null;
    }

    private static void RoundTrip(IReadOnlyList<OcrPage> pages)
    {
        string first = OcrPageJson.ToJson(pages);
        string second = OcrPageJson.ToJson(OcrPageJson.Read(System.Text.Encoding.UTF8.GetBytes(first)));
        if (first != second)
        {
            throw new InvalidOperationException("page JSON does not write back to the same bytes");
        }
    }

    private static void Report(StringWriter log, string? outFile, long files, long ok, long typed, long failures, long timeouts, long skipped, List<(double Seconds, string File)> slowest, TimeSpan elapsed)
    {
        long rssKb = 0;
        foreach (string line in File.ReadLines("/proc/self/status"))
        {
            if (line.StartsWith("VmHWM:", StringComparison.Ordinal))
            {
                rssKb = long.Parse(line.Split(' ', StringSplitOptions.RemoveEmptyEntries)[1], CultureInfo.InvariantCulture);
            }
        }

        log.WriteLine($"files={files} ok={ok} typedErrors={typed} untypedFailures={failures} timeouts={timeouts} skippedNonInputs={skipped} " +
            $"elapsed={elapsed.TotalSeconds:F0}s peakRssMB={rssKb / 1024}");
        foreach ((double s, string f) in slowest.OrderByDescending(static x => x.Seconds).Take(3))
        {
            log.WriteLine($"slowest {s:F2}s {f}");
        }

        Console.Write(log.ToString());
        if (outFile is not null)
        {
            File.WriteAllText(outFile, log.ToString());
        }
    }
}
