using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using OneImage;
using OneOcr;
using OneOcr.Layout;

// Runs the OCR engine over a directory of page images for the scorer in bench/oneocr_bench, and checks line finding.
//
//   OneOcr.Bench run   --model FILE --pages DIR --out DIR [--degree N] [--repeat K]
//       writes <page-id>.txt for every <page-id>.png and timing.json: the one-time model load, per page the image
//       decode time and the median-of-K recognition time, and the process's peak working set. --degree 0 is the library default.
//   OneOcr.Bench lines --pages DIR --truth DIR [--repeat K]
//       compares found text lines with lines derived from the truth word boxes (precision and recall at IoU >= 0.5) and times
//       grayscale conversion, line finding and line preprocessing per page.
//   OneOcr.Bench kernels --model FILE [--width 1000] [--degree 1] [--repeat 20]
//       median wall time of each stage (conv blocks, LSTM layers, classifier) for one line of the given width, on a
//       synthetic image, at the given degree of parallelism.
//   OneOcr.Bench randommodel --out FILE
//       writes a random-weight model of the intended size, to exercise the pipeline before trained weights exist (the text is garbage).
//
// Run each degree as its own process so that peak memory is per configuration.
internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.Error.WriteLine("usage: OneOcr.Bench run|lines ...");
            return 2;
        }

        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 1; i + 1 < args.Length; i += 2)
        {
            options[args[i].TrimStart('-')] = args[i + 1];
        }

        return args[0] switch
        {
            "run" => Run(options),
            "lines" => Lines(options),
            "randommodel" => RandomModel(options),
            "kernels" => Kernels(options),
            "decode" => DecodeBench.Run(options),
            _ => 2,
        };
    }

    private static string Need(Dictionary<string, string> o, string key) =>
        o.TryGetValue(key, out string? v) ? v : throw new ArgumentException($"missing --{key}");

    private static int Int(Dictionary<string, string> o, string key, int fallback) =>
        o.TryGetValue(key, out string? v) ? int.Parse(v, CultureInfo.InvariantCulture) : fallback;

    private static double Median(List<double> values)
    {
        values.Sort();
        return values[values.Count / 2];
    }

    private static int Kernels(Dictionary<string, string> o)
    {
        int width = Int(o, "width", 1000), degree = Int(o, "degree", 1), repeat = Math.Max(1, Int(o, "repeat", 20));
        var recognizer = OneOcr.Recognition.LineRecognizer.Load(Need(o, "model"));
        byte[] image = OneOcr.Tests.TestModels.RandomImage(recognizer.InputHeight, width, 1);
        var samples = new Dictionary<string, List<double>>(StringComparer.Ordinal);
        var total = new List<double>();
        for (int r = -2; r < repeat; r++)
        {
            var trace = new OneOcr.Recognition.ForwardTrace();
            var clock = Stopwatch.StartNew();
            recognizer.Forward(image, width, degree, trace, System.Threading.CancellationToken.None, out _);
            double all = clock.Elapsed.TotalMilliseconds;
            if (r < 0)
            {
                continue;
            }

            total.Add(all);
            foreach (var (name, ms) in trace.Milliseconds)
            {
                if (!samples.TryGetValue(name, out List<double>? list))
                {
                    samples[name] = list = [];
                }

                list.Add(ms);
            }
        }

        Console.WriteLine($"line 32x{width}, degree {degree}, median of {repeat}:");
        foreach (var (name, list) in samples)
        {
            Console.WriteLine($"  {name,-8} {Median(list),8:F2} ms");
        }

        Console.WriteLine($"  {"total",-8} {Median(total),8:F2} ms");
        return 0;
    }

    private static int RandomModel(Dictionary<string, string> o)
    {
        File.WriteAllBytes(Need(o, "out"), OneOcr.Tests.TestModels.RandomContainer(OneOcr.Tests.TestModels.Full, 1));
        return 0;
    }

    private static int Run(Dictionary<string, string> o)
    {
        string pages = Need(o, "pages"), outDir = Need(o, "out");
        int degree = Int(o, "degree", 0), repeat = Math.Max(1, Int(o, "repeat", 1));
        Directory.CreateDirectory(outDir);

        var clock = Stopwatch.StartNew();
        OcrEngine engine = OcrEngine.Load(Need(o, "model"));
        double loadMs = clock.Elapsed.TotalMilliseconds;

        var results = new List<(string Id, double DecodeMs, double OcrMs, int Chars)>();
        foreach (string path in Directory.GetFiles(pages, "*.png").Order(StringComparer.Ordinal))
        {
            string id = Path.GetFileNameWithoutExtension(path);
            clock.Restart();
            Image image = Image.Load(path);
            // Decoding is lazy; touching the first pixel row is not enough to pay for all of it, so the page is decoded by the
            // engine's own pixel read and this number is only the header work.
            double decodeMs = clock.Elapsed.TotalMilliseconds;
            var times = new List<double>();
            string text = string.Empty;
            for (int r = 0; r < repeat; r++)
            {
                clock.Restart();
                text = engine.ToText(image, degree);
                times.Add(clock.Elapsed.TotalMilliseconds);
            }

            File.WriteAllText(Path.Combine(outDir, id + ".txt"), text, new UTF8Encoding(false));
            results.Add((id, decodeMs, Median(times), text.Length));
            Console.WriteLine($"{id}: {Median(times):F0} ms, {text.Length} chars");
        }

        using var stream = File.Create(Path.Combine(outDir, "timing.json"));
        using (var w = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            w.WriteStartObject();
            w.WriteNumber("degree", degree);
            w.WriteNumber("repeat", repeat);
            w.WriteNumber("model_load_ms", Math.Round(loadMs, 1));
            w.WriteNumber("peak_rss_mb", Math.Round(Process.GetCurrentProcess().PeakWorkingSet64 / 1048576.0, 1));
            w.WriteString("runtime", System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription);
            w.WriteNumber("mean_page_ms", Math.Round(results.Average(r => r.OcrMs), 1));
            w.WriteStartArray("pages");
            foreach (var r in results)
            {
                w.WriteStartObject();
                w.WriteString("id", r.Id);
                w.WriteNumber("open_ms", Math.Round(r.DecodeMs, 2));
                w.WriteNumber("ocr_ms", Math.Round(r.OcrMs, 1));
                w.WriteNumber("chars", r.Chars);
                w.WriteEndObject();
            }

            w.WriteEndArray();
            w.WriteEndObject();
        }

        Console.WriteLine($"model load {loadMs:F0} ms, mean page {results.Average(r => r.OcrMs):F0} ms, peak RSS {Process.GetCurrentProcess().PeakWorkingSet64 / 1048576.0:F0} MB");
        return 0;
    }

    private readonly record struct Box(double X0, double Y0, double X1, double Y1)
    {
        public double Area => Math.Max(0, X1 - X0) * Math.Max(0, Y1 - Y0);

        public static double Iou(Box a, Box b)
        {
            double iw = Math.Min(a.X1, b.X1) - Math.Max(a.X0, b.X0), ih = Math.Min(a.Y1, b.Y1) - Math.Max(a.Y0, b.Y0);
            if (iw <= 0 || ih <= 0)
            {
                return 0;
            }

            double inter = iw * ih;
            return inter / (a.Area + b.Area - inter);
        }
    }

    // Truth lines from word boxes: words in document order join the current line while their vertical centre lies inside the line's
    // box and they start to its right. A word that wraps at a hyphen has one box spanning two lines (much taller than the other
    // words of its block); it is left out, since it belongs to neither line alone.
    private static List<Box> TruthLines(string truthJson)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(truthJson));
        var words = new List<(Box Box, int Block)>();
        foreach (JsonElement w in doc.RootElement.GetProperty("words").EnumerateArray())
        {
            JsonElement b = w.GetProperty("box");
            words.Add((new Box(b[0].GetDouble(), b[1].GetDouble(), b[2].GetDouble(), b[3].GetDouble()), w.GetProperty("block").GetInt32()));
        }

        var blockHeight = words.GroupBy(w => w.Block)
            .ToDictionary(g => g.Key, g => g.Select(w => w.Box.Y1 - w.Box.Y0).Order().ElementAt((g.Count() - 1) / 2));
        var lines = new List<Box>();
        Box? current = null;
        foreach (var (w, block) in words)
        {
            if (w.Y1 - w.Y0 > 1.7 * blockHeight[block])
            {
                continue;
            }

            double cy = (w.Y0 + w.Y1) / 2;
            if (current is Box c && cy >= c.Y0 && cy <= c.Y1 && w.X0 >= c.X0)
            {
                current = new Box(c.X0, Math.Min(c.Y0, w.Y0), Math.Max(c.X1, w.X1), Math.Max(c.Y1, w.Y1));
            }
            else
            {
                if (current is Box done)
                {
                    lines.Add(done);
                }

                current = w;
            }
        }

        if (current is Box last)
        {
            lines.Add(last);
        }

        return lines;
    }

    private static int Lines(Dictionary<string, string> o)
    {
        string pages = Need(o, "pages"), truthDir = Need(o, "truth");
        int repeat = Math.Max(1, Int(o, "repeat", 3));
        double iouThreshold = 0.5;
        long truthTotal = 0, foundTotal = 0, matched = 0;
        var ious = new List<double>();
        var findMs = new List<double>();
        var prepMs = new List<double>();
        var grayMs = new List<double>();
        var bad = new List<string>();
        var clock = new Stopwatch();

        foreach (string path in Directory.GetFiles(pages, "*.png").Order(StringComparer.Ordinal))
        {
            string id = Path.GetFileNameWithoutExtension(path);
            Image image = Image.Load(path);
            int w = image.Width, h = image.Height;
            byte[] gray = new byte[0];
            Gray8[] px = [];
            var g = new List<double>();
            var f = new List<double>();
            var p = new List<double>();
            List<LineBox> found = [];
            for (int r = 0; r < repeat; r++)
            {
                clock.Restart();
                px = image.ConvertTo(PixelFormat.Gray8).ToArray<Gray8>();
                gray = new byte[px.Length];
                for (int i = 0; i < gray.Length; i++)
                {
                    gray[i] = px[i].V;
                }

                g.Add(clock.Elapsed.TotalMilliseconds);
                clock.Restart();
                found = LineFinder.Find(gray, w, h, 100000);
                f.Add(clock.Elapsed.TotalMilliseconds);
                clock.Restart();
                var padPool = new OneOcr.BufferPool<Gray8>(1L << 28, 1);
                int typical = LinePreprocessor.TypicalHeight(found);
                foreach (LineBox line in found)
                {
                    padPool.Return(LinePreprocessor.Prepare(gray, w, LinePreprocessor.CropBox(line, typical, w, h), 32, padPool, out _));
                }

                p.Add(clock.Elapsed.TotalMilliseconds);
            }

            grayMs.Add(Median(g));
            findMs.Add(Median(f));
            prepMs.Add(Median(p));

            List<Box> truth = TruthLines(Path.Combine(truthDir, id + ".json"));
            var used = new bool[found.Count];
            int pageMatched = 0;
            foreach (Box t in truth)
            {
                int best = -1;
                double bestIou = 0;
                for (int i = 0; i < found.Count; i++)
                {
                    if (used[i])
                    {
                        continue;
                    }

                    LineBox l = found[i];
                    double iou = Box.Iou(t, new Box(l.X, l.Y, l.Right, l.Bottom));
                    if (iou > bestIou)
                    {
                        bestIou = iou;
                        best = i;
                    }
                }

                if (best >= 0 && bestIou >= iouThreshold)
                {
                    used[best] = true;
                    pageMatched++;
                    ious.Add(bestIou);
                }
            }

            truthTotal += truth.Count;
            foundTotal += found.Count;
            matched += pageMatched;
            if (pageMatched != truth.Count || pageMatched != found.Count)
            {
                for (int i = 0; i < found.Count; i++)
                {
                    if (!used[i])
                    {
                        bad.Add($"  {id}: unmatched found line x {found[i].X} y {found[i].Y} w {found[i].Width} h {found[i].Height}");
                    }
                }

                foreach (Box t in truth)
                {
                    if (!found.Any(l => Box.Iou(t, new Box(l.X, l.Y, l.Right, l.Bottom)) >= iouThreshold))
                    {
                        bad.Add($"  {id}: unmatched truth line x {t.X0:F0} y {t.Y0:F0} w {t.X1 - t.X0:F0} h {t.Y1 - t.Y0:F0}");
                    }
                }

                bad.Add($"{id}: truth {truth.Count}, found {found.Count}, matched {pageMatched}");
            }
        }

        Console.WriteLine($"line recall    {(double)matched / truthTotal:P2}  ({matched}/{truthTotal})");
        Console.WriteLine($"line precision {(double)matched / foundTotal:P2}  ({matched}/{foundTotal})");
        Console.WriteLine($"mean IoU of matched lines {ious.Average():F3} (truth boxes are the font's content area, taller than the ink)");
        Console.WriteLine($"per page ms (median of {repeat}, mean over pages): gray {grayMs.Average():F1}, line finding {findMs.Average():F1}, crops+scaling {prepMs.Average():F1}");
        foreach (string line in bad)
        {
            Console.WriteLine("  mismatch " + line);
        }

        return 0;
    }
}
