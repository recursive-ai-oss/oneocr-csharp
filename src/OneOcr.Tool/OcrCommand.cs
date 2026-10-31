using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using OneImage;
using OneOcr.Decoding;

namespace OneOcr.Tool;

/// <summary>
/// <c>oneocr ocr</c>: reads page images with the local engine and writes the text (and optionally the page model) next to a
/// name per page. Nothing here touches the network.
/// </summary>
internal static class OcrCommand
{
    public static int Run(Flags f)
    {
        string input = f.Required("in");
        string? outDir = f.Optional("out"), modelFile = f.Optional("model"), detectorFile = f.Optional("detector"), wordModelFile = f.Optional("wordlm"), wordFreqFile = f.Optional("wordfreq"), statsFile = f.Optional("stats");
        int degree = f.Int("degree", 0);
        bool json = f.Switch("json");
        f.Done();

        string[] pages = Directory.Exists(input)
            ? [.. Directory.GetFiles(input).Where(static f => IsImage(f) || IsPdfName(f)).Order(StringComparer.Ordinal)]
            : File.Exists(input) ? [input] : throw new UsageException("--in '" + input + "' is neither a file nor a directory.");
        if (outDir is null && (pages.Length != 1 || json))
        {
            throw new UsageException("--out <dir> is needed for several pages or for --json.");
        }

        if (detectorFile is not null && modelFile is null)
        {
            throw new UsageException("--detector goes with --model; without --model the bundled models are used.");
        }

        var clock = Stopwatch.StartNew();
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

        double loadMs = clock.Elapsed.TotalMilliseconds;
        if (engine.PerformanceNotice is { } notice)
        {
            Console.Error.WriteLine("oneocr: warning: " + notice);
        }

        if (outDir is not null)
        {
            Directory.CreateDirectory(outDir);
        }

        var times = new List<(string Id, double Ms)>();
        int failed = 0;
        foreach (string path in pages)
        {
            string id = Path.GetFileNameWithoutExtension(path);
            clock.Restart();
            try
            {
                if (IsPdfName(path) || LooksLikePdf(path))
                {
                    throw new UsageException("PDF files are not read by this tool. Render the pages to PNG or JPEG, or call RecursiveAI.OneOcr.Pdf from code.");
                }

                Image image = Image.Load(path);
                string text;
                string? pageJson = null;
                if (json)
                {
                    OcrPage page = engine.Ocr(image, new OcrOptions { MaxDegreeOfParallelism = degree });
                    text = page.ToText();
                    pageJson = OcrPageJson.ToJson([page]);
                }
                else
                {
                    text = engine.ToText(image, new OcrOptions { MaxDegreeOfParallelism = degree });
                }

                times.Add((id, clock.Elapsed.TotalMilliseconds));
                if (outDir is null)
                {
                    Console.Out.Write(text);
                    continue;
                }

                File.WriteAllText(Path.Combine(outDir, id + ".txt"), text, new UTF8Encoding(false));
                if (pageJson is not null)
                {
                    File.WriteAllText(Path.Combine(outDir, id + ".page.json"), pageJson, new UTF8Encoding(false));
                }
            }
            catch (Exception ex) when (ex is UsageException or OcrException or ImageException or ArgumentException or IOException or UnauthorizedAccessException or InvalidOperationException)
            {
                // One unreadable page is reported and the rest are still read; the exit code says there were failures.
                failed++;
                Console.Error.WriteLine("oneocr: " + Path.GetFileName(path) + ": " + ex.Message);
            }
        }

        if (statsFile is not null)
        {
            WriteStats(statsFile, degree, loadMs, times);
        }

        return failed > 0 ? 3 : 0;
    }

    private static bool IsPdfName(string file) => file.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase);

    private static bool LooksLikePdf(string file)
    {
        Span<byte> head = stackalloc byte[5];
        using FileStream s = File.OpenRead(file);
        return s.Read(head) == 5 && head.SequenceEqual("%PDF-"u8);
    }

    private static bool IsImage(string file) =>
        file.EndsWith(".png", StringComparison.OrdinalIgnoreCase) || file.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
        file.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase);

    // Model load time, per-page time and peak memory, for timing runs.
    private static void WriteStats(string path, int degree, double loadMs, List<(string Id, double Ms)> times)
    {
        using var stream = File.Create(path);
        using var w = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });
        w.WriteStartObject();
        w.WriteNumber("degree", degree);
        w.WriteNumber("model_load_ms", Math.Round(loadMs, 1));
        using (var process = Process.GetCurrentProcess())
        {
            w.WriteNumber("peak_rss_mb", Math.Round(process.PeakWorkingSet64 / 1048576.0, 1));
        }

        w.WriteNumber("mean_page_ms", times.Count == 0 ? 0 : Math.Round(times.Average(t => t.Ms), 1));
        w.WriteStartArray("pages");
        foreach ((string id, double ms) in times)
        {
            w.WriteStartObject();
            w.WriteString("id", id);
            w.WriteNumber("ocr_ms", Math.Round(ms, 1));
            w.WriteEndObject();
        }

        w.WriteEndArray();
        w.WriteEndObject();
    }
}
