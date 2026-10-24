using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using OneImage;
using OneOcr;
using OneOcr.Layout;

// Runs the OCR engine over a directory of page images for the scorer in bench/oneocr_bench, and checks line finding.
//
//   OneOcr.Bench run   --model FILE [--detector FILE] --pages DIR --out DIR [--degree N] [--repeat K] [--beam W [--lm FILE]] [--words 1] [--pagejson 1]
//       writes <page-id>.txt for every <page-id>.png and timing.json: the one-time model load, per page the image
//       decode time and the median-of-K recognition time, and the process's peak working set. --degree 0 is the library default.
//       --words 1 also writes <page-id>.json, the scorer's word sidecar {"words":[{"box":[x0,y0,x1,y1],"text":..,"conf":..,"conf_raw":..}]}
//       from OcrEngine.Ocr (greedy; word boxes are ink extents; calibrated and raw word confidence). It runs after the timed loop, so timings are unaffected.
//       --noorient 1 reads with orientation detection and deskew off (the text of a page that needs neither is the same either way).
//       --metaonly 1 skips the timed text run and its .txt (with --meta 1: orientation and skew only).
//       --meta 1 also writes <page-id>.meta.json {"orientation":..,"skew":..} as the engine reports them.
//   OneOcr.Bench lines --pages DIR --truth DIR [--repeat K]
//       compares found text lines with lines derived from the truth word boxes (precision and recall at IoU >= 0.5) and times
//       grayscale conversion, line finding and line preprocessing per page.
//   OneOcr.Bench kernels --model FILE [--width 1000] [--degree 1] [--repeat 20]
//       median wall time of each stage (conv blocks, LSTM layers, classifier) for one line of the given width, on a
//       synthetic image, at the given degree of parallelism.
//   OneOcr.Bench trainstep --model FILE [--batch 16] [--width 400] [--chars 40] [--degree 1] [--repeat 5]
//       median time of one training step (see TrainBench.cs).
//   OneOcr.Bench detect [--model FILE] [--page 2480x3508] [--degree 1] [--repeat 15] [--path avx2]
//       median time of the page detector (scaling, network, decoding) and of the network alone; see DetectBench.cs.
//   OneOcr.Bench docextract --model FILE [--words 150,300] [--degree 1] [--repeat 15] [--path avx2]
//       median time of the document-extraction forward pass and assembly for pages of the given word counts; see DocExtractBench.cs.
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
            "trainstep" => TrainBench.Run(options),
            "detect" => DetectBench.Run(options),
            "docextract" => DocExtractBench.Run(options),
            "strips" => Strips(options),
            "detlines" => DetLines(options),
            "grids" => Grids(options),
            "skew" => Skew(options),
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

    // Prints the strips (leaf region, box) the page finder cuts from one page: OneOcr.Bench strips --image FILE
    private static int Strips(Dictionary<string, string> o)
    {
        Image image = Image.Load(Need(o, "image"));
        var gray = new Gray8[image.Width * image.Height];
        image.CopyPixelsTo<Gray8>(gray, new ImageRect(0, 0, image.Width, image.Height), 1);
        var masks = new OneOcr.BufferPool<byte>(1L << 28, 1);
        List<StripLine> strips = StripFinder.Find(gray, image.Width, image.Height, 100000, masks, 1, out bool used, out _, out _);
        Console.WriteLine($"{strips.Count} strips, regions used: {used}");
        foreach (StripLine s in strips)
        {
            Console.WriteLine($"leaf {s.Leaf,3}  x {s.Box.X,5} y {s.Box.Y,5} w {s.Box.Width,5} h {s.Box.Height,3}");
        }

        return 0;
    }

    // Prints what the page detector finds on one page and the ink strips made from it: OneOcr.Bench detlines --image FILE --detector FILE
    private static int DetLines(Dictionary<string, string> o)
    {
        Image image = Image.Load(Need(o, "image"));
        var gray = new Gray8[image.Width * image.Height];
        image.CopyPixelsTo<Gray8>(gray, new ImageRect(0, 0, image.Width, image.Height), 1);
        var detector = new OneOcr.Detection.PageDetector(
            OneOcr.Detection.DbNetModel.FromContainer(OneOcr.Model.ModelContainer.Load(Need(o, "detector"), null), OneOcr.OcrLimits.Default),
            OneOcr.OcrLimits.Default, OneOcr.Kernels.Dispatch.Best);
        var pixels = new OneOcr.BufferPool<Gray8>(1L << 28, 1);
        var resizers = new ResizerPool();
        OneOcr.Detection.PageDetection d = detector.Detect(gray, image.Width, image.Height, pixels, resizers, 1, System.Threading.CancellationToken.None);
        Console.WriteLine($"{d.Lines.Count} lines, {d.Regions.Count} regions, quarter {d.Quarter} ({d.OrientationConfidence:F2})");
        foreach (var l in d.Lines)
        {
            Console.WriteLine($"line x {l.Left,7:F1} y {l.Top,7:F1} w {l.Right - l.Left,6:F1} h {l.Bottom - l.Top,5:F1} score {l.Score:F2}");
        }

        foreach (var r in d.Regions)
        {
            Console.WriteLine($"region {r.Name,-13} x {r.Left,7:F1} y {r.Top,7:F1} w {r.Right - r.Left,6:F1} h {r.Bottom - r.Top,6:F1} score {r.Score:F2}");
        }

        LocatedPage located = DetectorLines.Locate(detector, gray, image.Width, image.Height, pixels, resizers, 1, System.Threading.CancellationToken.None, d, false);
        foreach (StripLine s in located.Strips)
        {
            Console.WriteLine($"strip leaf {s.Leaf,7}  x {s.Box.X,5} y {s.Box.Y,5} w {s.Box.Width,5} h {s.Box.Height,3}");
        }

        return 0;
    }

    // Prints the ruled grids of one page: OneOcr.Bench grids --image FILE
    private static int Grids(Dictionary<string, string> o)
    {
        Image image = Image.Load(Need(o, "image"));
        var gray = new Gray8[image.Width * image.Height];
        image.CopyPixelsTo<Gray8>(gray, new ImageRect(0, 0, image.Width, image.Height), 1);
        ReadOnlySpan<byte> bytes = OneOcr.Kernels.PixelSpans.AsBytes(gray);
        (int threshold, bool inkIsBright) = LineFinder.Binarize(bytes);
        var clock = Stopwatch.StartNew();
        List<RuleGrid> grids = TableRules.Find(bytes, image.Width, image.Height, threshold, inkIsBright);
        Console.WriteLine($"{grids.Count} grids in {clock.Elapsed.TotalMilliseconds:F1} ms (threshold {threshold}, ink bright {inkIsBright})");
        foreach (RuleGrid g in grids)
        {
            Console.WriteLine($"rows {g.Rows} columns {g.Columns} cells {g.Cells.Count} box {g.Left},{g.Top}-{g.Right},{g.Bottom} spans {g.Cells.Count(c => c.RowSpan > 1 || c.ColumnSpan > 1)} xs {string.Join(",", g.Xs)} ys {string.Join(",", g.Ys.Take(5))}");
        }

        if (grids.Count > 0)
        {
            var masks = new OneOcr.BufferPool<byte>(1L << 28, 1);
            List<StripLine> strips = StripFinder.Find(gray, image.Width, image.Height, 100000, masks, 1, out _, out _, out _, eraseRules: false);
            List<StripLine> cut = TableStructure.SplitStrips(strips, grids, bytes, image.Width, image.Height, threshold, inkIsBright);
            Console.WriteLine($"{strips.Count} strips, {cut.Count} after cutting at the rules");
        }

        return 0;
    }

    // Prints the skew search of one page, and with --model the orientation evidence: OneOcr.Bench skew --image FILE [--model FILE]
    private static int Skew(Dictionary<string, string> o)
    {
        Image image = Image.Load(Need(o, "image"));
        var gray = new Gray8[image.Width * image.Height];
        image.CopyPixelsTo<Gray8>(gray, new ImageRect(0, 0, image.Width, image.Height), 1);
        ReadOnlySpan<byte> bytes = OneOcr.Kernels.PixelSpans.AsBytes(gray);
        (int threshold, bool inkIsBright) = LineFinder.Binarize(bytes);
        InkGrid grid = InkGrid.Build(bytes, image.Width, image.Height, threshold, inkIsBright);
        SkewEstimate flat = SkewEstimator.Estimate(grid), sideways = SkewEstimator.Estimate(grid.TurnClockwise());
        Console.WriteLine($"threshold {threshold} inkIsBright {inkIsBright} grid {grid.Width}x{grid.Height} factor {grid.Factor}");
        Console.WriteLine($"flat     slope {flat.Slope,4} gain {flat.Gain:F3} contrast {flat.Contrast:F2}");
        Console.WriteLine($"sideways slope {sideways.Slope,4} gain {sideways.Gain:F3} contrast {sideways.Contrast:F2}");
        if (o.TryGetValue("model", out string? model))
        {
            var recognizer = OneOcr.Recognition.LineRecognizer.Load(model);
            var detector = new ClassicalOrientation(recognizer, new OneOcr.BufferPool<Gray8>(1L << 28, 1), new OneOcr.BufferPool<byte>(1L << 28, 1), new ResizerPool(), OneOcr.OcrLimits.Default);
            var trace = new List<string>();
            int degrees = detector.Detect(new OrientationInput(gray, image.Width, image.Height, 4), trace, System.Threading.CancellationToken.None);
            foreach (string line in trace)
            {
                Console.WriteLine(line);
            }

            Console.WriteLine($"turn {degrees}");
        }

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
        if (o.TryGetValue("beam", out string? beamWidth))
        {
            // --beam W [--lm FILE]: beam search of width W, with the character model when one is named.
            engine = engine.WithDecoding(new OneOcr.Decoding.DecodeOptions
            {
                Mode = OneOcr.Decoding.CtcDecoding.Beam,
                BeamWidth = int.Parse(beamWidth, CultureInfo.InvariantCulture),
                LanguageModel = o.TryGetValue("lm", out string? lmPath) ? OneOcr.Decoding.CharLanguageModel.Load(lmPath) : null,
            });
        }
        if (o.TryGetValue("detector", out string? detectorFile))
        {
            engine = engine.WithDetector(detectorFile);
        }

        double loadMs = clock.Elapsed.TotalMilliseconds;

        bool noOrient = Int(o, "noorient", 0) != 0, metaOnly = Int(o, "metaonly", 0) != 0, writeWords = Int(o, "words", 0) != 0, writePageJson = Int(o, "pagejson", 0) != 0;
        var results = new List<(string Id, double DecodeMs, double OcrMs, int Chars)>();
        var errors = new List<(string Id, string Message)>();
        foreach (string path in Directory.GetFiles(pages).Where(f => f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase)).Order(StringComparer.Ordinal))
        {
            string id = Path.GetFileNameWithoutExtension(path);
            clock.Restart();
            try
            {
                Image image = Image.Load(path);
                // Decoding is lazy; touching the first pixel row is not enough to pay for all of it, so the page is decoded by the
                // engine's own pixel read and this number is only the header work.
                double decodeMs = clock.Elapsed.TotalMilliseconds;
                var times = new List<double>();
                string text = string.Empty;
                for (int r = 0; r < (metaOnly ? 0 : repeat); r++)
                {
                    clock.Restart();
                    text = noOrient
                        ? engine.Ocr(image, new OcrOptions { MaxDegreeOfParallelism = degree, DetectOrientation = false, Deskew = false, IncludeChars = false }).ToText()
                        : engine.ToText(image, degree);
                    times.Add(clock.Elapsed.TotalMilliseconds);
                }

                if (metaOnly)
                {
                    times.Add(0);
                }
                else
                {
                    File.WriteAllText(Path.Combine(outDir, id + ".txt"), text, new UTF8Encoding(false));
                }

                if (writeWords || writePageJson)
                {
                    OcrPage page = engine.Ocr(image, new OcrOptions { MaxDegreeOfParallelism = degree });
                    WriteWordSidecar(Path.Combine(outDir, id + ".json"), page);
                    if (writePageJson)
                    {
                        File.WriteAllText(Path.Combine(outDir, id + ".page.json"), OcrPageJson.ToJson([page]), new UTF8Encoding(false));
                    }
                }

                if (Int(o, "meta", 0) != 0)
                {
                    // The orientation and skew the page reports, for checking against a known rotation.
                    OcrPage found = engine.Ocr(image, new OcrOptions { MaxDegreeOfParallelism = degree });
                    File.WriteAllText(
                        Path.Combine(outDir, id + ".meta.json"),
                        string.Create(CultureInfo.InvariantCulture, $"{{\"orientation\":{found.Orientation},\"skew\":{found.SkewDegrees:0.####}}}"));
                }

                results.Add((id, decodeMs, Median(times), text.Length));
                Console.WriteLine($"{id}: {Median(times):F0} ms, {text.Length} chars");
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                // One unreadable or failing page is recorded and the run goes on; the exit code says there were errors.
                errors.Add((id, e.GetType().Name + ": " + e.Message));
                Console.Error.WriteLine($"{id}: ERROR {e.GetType().Name}: {e.Message}");
                File.WriteAllText(Path.Combine(outDir, id + ".error.txt"), e + "\n", new UTF8Encoding(false));
            }
        }

        using var stream = File.Create(Path.Combine(outDir, "timing.json"));
        using (var w = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            w.WriteStartObject();
            w.WriteNumber("degree", degree);
            w.WriteNumber("repeat", repeat);
            w.WriteNumber("model_load_ms", Math.Round(loadMs, 1));
            w.WriteNumber("peak_rss_mb", Math.Round(Process.GetCurrentProcess().PeakWorkingSet64 / 1048576.0, 1));
            GCMemoryInfo gc = GC.GetGCMemoryInfo();
            w.WriteNumber("gc_committed_mb", Math.Round(gc.TotalCommittedBytes / 1048576.0, 1));
            w.WriteNumber("gc_heap_mb", Math.Round(gc.HeapSizeBytes / 1048576.0, 1));
            w.WriteNumber("gc_fragmented_mb", Math.Round(gc.FragmentedBytes / 1048576.0, 1));
            w.WriteNumber("gc_count", GC.CollectionCount(2));
            w.WriteNumber("allocated_mb", Math.Round(GC.GetTotalAllocatedBytes() / 1048576.0, 0));
            w.WriteString("runtime", System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription);
            w.WriteNumber("mean_page_ms", results.Count == 0 ? 0 : Math.Round(results.Average(r => r.OcrMs), 1));
            w.WriteNumber("allocated_mb_per_page", results.Count == 0 ? 0 : Math.Round(GC.GetTotalAllocatedBytes() / 1048576.0 / results.Count, 1));
            w.WriteNumber("gen2_collections", GC.CollectionCount(2));
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
            w.WriteStartArray("errors");
            foreach (var (errId, message) in errors)
            {
                w.WriteStartObject();
                w.WriteString("id", errId);
                w.WriteString("message", message);
                w.WriteEndObject();
            }

            w.WriteEndArray();
            w.WriteEndObject();
        }

        Console.WriteLine($"model load {loadMs:F0} ms, mean page {(results.Count == 0 ? 0 : results.Average(r => r.OcrMs)):F0} ms, peak RSS {Process.GetCurrentProcess().PeakWorkingSet64 / 1048576.0:F0} MB");
        if (errors.Count > 0)
        {
            Console.Error.WriteLine($"{errors.Count} page(s) failed; see *.error.txt and timing.json");
        }

        return errors.Count > 0 ? 3 : 0;
    }

    private static void WriteWordSidecar(string path, OcrPage page)
    {
        using var stream = File.Create(path);
        using var w = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = false });
        w.WriteStartObject();
        w.WriteStartArray("words");
        foreach (OcrWord word in page.Words)
        {
            w.WriteStartObject();
            w.WriteStartArray("box");
            foreach (double v in new[] { word.Box.Left, word.Box.Top, word.Box.Right, word.Box.Bottom })
            {
                w.WriteNumberValue(Math.Round(v, 2));
            }

            w.WriteEndArray();
            w.WriteString("text", word.Text);
            // Round-trip precision: the scorer's calibration table is built from these.
            w.WriteNumber("conf", word.Confidence);
            w.WriteNumber("conf_raw", word.RawConfidence);
            w.WriteEndObject();
        }

        w.WriteEndArray();
        w.WriteEndObject();
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

        foreach (string path in Directory.GetFiles(pages).Where(f => f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase)).Order(StringComparer.Ordinal))
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
                var resizers = new OneOcr.Layout.ResizerPool();
                int typical = LinePreprocessor.TypicalHeight(found);
                foreach (LineBox line in found)
                {
                    padPool.Return(LinePreprocessor.Prepare(gray, w, LinePreprocessor.CropBox(line, typical, w, h), 32, padPool, resizers, out _));
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
