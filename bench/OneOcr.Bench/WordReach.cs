using System.Globalization;
using System.Text;
using OneImage;
using OneOcr;
using OneOcr.Kernels;
using OneOcr.Layout;
using OneOcr.Recognition;


/// <summary>
/// Measures how much a second look at uncertain words could gain. For every word of a page that matches a truth word, the word is
/// cut out again with several margins and read again; the report says how often some re-reading equals the truth when the first
/// reading did not (the reachable share), and how often a re-reading differs from a first reading that was right (the risk).
///
///   OneOcr.Bench reach --model FILE [--detector FILE] [--wordlm FILE] --pages DIR --truth DIR [--out FILE] [--maxconf 0.995]
/// </summary>
internal static class WordReach
{
    private static readonly (string Name, double H, double Top, double Bottom)[] Variants =
    [
        ("base", 0.5, 0.25, 0.25),
        ("h.25", 0.25, 0.25, 0.25),
        ("h1.0", 1.0, 0.25, 0.25),
        ("v.4", 0.5, 0.4, 0.4),
        ("v.1", 0.5, 0.1, 0.1),
        ("tight", 0.15, 0.15, 0.15),
        ("wide", 0.75, 0.5, 0.5),
        ("h.35v.3", 0.35, 0.3, 0.3),
    ];

    public static int Run(Dictionary<string, string> o)
    {
        string pages = o["pages"], truthDir = o["truth"];
        double maxConf = o.TryGetValue("maxconf", out string? mc) ? double.Parse(mc, CultureInfo.InvariantCulture) : 0.995;
        OcrEngine engine = OcrEngine.Load(o["model"]);
        if (o.TryGetValue("wordlm", out string? wl))
        {
            engine = engine.WithWordModel(OneOcr.Decoding.CharLanguageModel.Load(wl));
        }

        if (o.TryGetValue("detector", out string? det))
        {
            engine = engine.WithDetector(det);
        }

        LineRecognizer recognizer = LineRecognizer.Load(o["model"]);
        var plausibility = new OneOcr.Decoding.WordPlausibility(OneOcr.Decoding.CharLanguageModel.Load(o["wordlm"]));
        var pool = new BufferPool<Gray8>(1L << 28, 1);
        var resizers = new ResizerPool();
        var rows = new List<Row>();
        foreach (string path in Directory.GetFiles(pages, "*.*").Where(f => f.EndsWith(".png") || f.EndsWith(".jpg")).Order(StringComparer.Ordinal))
        {
            string id = Path.GetFileNameWithoutExtension(path);
            Image image = Image.Load(path);
            OcrPage page = engine.Ocr(image, new OcrOptions { MaxDegreeOfParallelism = 4 });
            var gray = new Gray8[image.Width * image.Height];
            image.CopyPixelsTo<Gray8>(gray, new ImageRect(0, 0, image.Width, image.Height), 1);
            byte[] bytes = PixelSpans.AsBytes(gray).ToArray();

            var truth = ReadTruth(Path.Combine(truthDir, id + ".json"));
            var lines = new List<(OcrLine Line, int Index)>();
            foreach (OcrBlock b in page.Blocks)
            {
                foreach (OcrLine l in b.Lines)
                {
                    lines.Add((l, lines.Count));
                }
            }

            int typical = (int)Median(lines.Select(l => (int)l.Line.Box.Height).ToList());
            var used = new HashSet<OcrWord>();
            foreach (OcrLine line in lines.Select(l => l.Line))
            {
                for (int wi = 0; wi < line.Words.Count; wi++)
                {
                    OcrWord w = line.Words[wi];
                    // Truth word with the largest overlap.
                    string? want = null;
                    double bestIou = 0.3;
                    foreach ((string text, OcrBox box) in truth)
                    {
                        double v = Iou(w.Box, box);
                        if (v > bestIou)
                        {
                            bestIou = v;
                            want = text;
                        }
                    }

                    if (want is null)
                    {
                        continue;
                    }

                    bool right = Norm(w.Text) == Norm(want);
                    var row = new Row { Page = id, Truth = want, Base = w.Text, Conf = w.Confidence, Right = right, Cands = new(), BaseLm = Lm(plausibility, w.Text) };
                    rows.Add(row);
                    if (w.Confidence >= maxConf && right)
                    {
                        continue;
                    }

                    double eh = Math.Max(line.Box.Height, 0.85 * typical);
                    double cy = (line.Box.Top + line.Box.Bottom) / 2;
                    foreach ((string name, double h, double top, double bottom) in Variants)
                    {
                        double mx = eh * h;
                        int x0 = Math.Max(0, (int)Math.Floor(w.Box.Left - mx)), x1 = Math.Min(image.Width, (int)Math.Ceiling(w.Box.Right + mx));
                        int y0 = Math.Max(0, (int)Math.Floor(cy - (eh * (0.5 + top)))), y1 = Math.Min(image.Height, (int)Math.Ceiling(cy + (eh * (0.5 + bottom))));
                        if (x1 - x0 < 2 || y1 - y0 < 2)
                        {
                            continue;
                        }

                        var box = new LineBox(x0, y0, x1 - x0, y1 - y0);
                        Gray8[] px = LinePreprocessor.Prepare(bytes, image.Width, box, recognizer.InputHeight, pool, resizers, out int cw);
                        try
                        {
                            LineWords lw = recognizer.RecognizeWords(PixelSpans.AsBytes(px.AsSpan(0, cw * recognizer.InputHeight)), cw, 1, default, false);
                            double scale = (double)box.Width / cw;
                            var sb = new StringBuilder();
                            float conf = 1;
                            foreach (RecognizedWord rw in lw.Words)
                            {
                                double a = x0 + (rw.X0 * scale), b = x0 + (rw.X1 * scale);
                                double cx = (a + b) / 2;
                                if (cx >= w.Box.Left - 0.2 * eh && cx <= w.Box.Right + 0.2 * eh)
                                {
                                    sb.Append(rw.Text);
                                    conf = Math.Min(conf, rw.Confidence);
                                }
                            }

                            row.Cands[name] = (sb.ToString(), conf);
                            row.Lm[name] = Lm(plausibility, sb.ToString());
                        }
                        finally
                        {
                            pool.Return(px);
                        }
                    }
                }
            }
        }

        Report(rows, o.TryGetValue("out", out string? outFile) ? outFile : null);
        return 0;
    }

    private static void Report(List<Row> rows, string? outFile)
    {
        var sb = new StringBuilder();
        foreach (Row r in rows.Where(r => r.Cands.Count > 0))
        {
            sb.Append(CultureInfo.InvariantCulture, $"{r.Page}\t{r.Truth}\t{r.Base}\t{r.Conf:0.###}\t{(r.Right ? 1 : 0)}\t{r.BaseLm:0.###}");
            foreach ((string name, (string text, float conf) c) in r.Cands)
            {
                sb.Append(CultureInfo.InvariantCulture, $"\t{name}={c.text}:{c.conf:0.###}:{r.Lm[name]:0.###}");
            }

            sb.Append('\n');
        }

        if (outFile is not null)
        {
            File.WriteAllText(outFile, sb.ToString());
        }

        int total = rows.Count, wrong = rows.Count(r => !r.Right);
        Console.WriteLine($"matched words {total}, wrong {wrong} ({100.0 * wrong / Math.Max(1, total):F2}%)");
        Console.WriteLine("conf<  wrong  reach  | right  differs(any)");
        foreach (double th in new[] { 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01 })
        {
            var w = rows.Where(r => !r.Right && r.Conf < th).ToList();
            var ok = rows.Where(r => r.Right && r.Conf < th && r.Cands.Count > 0).ToList();
            int reach = w.Count(r => r.Cands.Values.Any(c => Norm(c.Text) == Norm(r.Truth)));
            int differs = ok.Count(r => r.Cands.Values.Any(c => Norm(c.Text) != Norm(r.Truth)));
            Console.WriteLine($"{th,5}  {w.Count,5}  {reach,5}  | {ok.Count,5}  {differs,5}");
        }

        Console.WriteLine("per variant, wrong words fixed (all confs):");
        foreach ((string name, _, _, _) in Variants)
        {
            int fixedCount = rows.Count(r => !r.Right && r.Cands.TryGetValue(name, out var c) && Norm(c.Text) == Norm(r.Truth));
            Console.WriteLine($"  {name,-8} {fixedCount}");
        }
    }

    // Score of the letters of a word with surrounding punctuation removed; a word that has no score reads as 0.
    private static double Lm(OneOcr.Decoding.WordPlausibility p, string text)
    {
        string core = text.Trim(' ', '.', ',', ';', ':', '!', '?', '"', '\'', '(', ')', '\u201C', '\u201D', '\u2019', '\u2018', '-', '\u2014');
        return p.Score(core) ?? 0;
    }

    private static string Norm(string s) => s.Normalize(NormalizationForm.FormC).Replace('\u2019', '\'').Replace('\u2018', '\'').Replace('\u201C', '"').Replace('\u201D', '"');

    private static double Median(List<int> v)
    {
        if (v.Count == 0)
        {
            return 0;
        }

        v.Sort();
        return v[v.Count / 2];
    }

    private static double Iou(OcrBox a, OcrBox b)
    {
        double x0 = Math.Max(a.Left, b.Left), y0 = Math.Max(a.Top, b.Top), x1 = Math.Min(a.Right, b.Right), y1 = Math.Min(a.Bottom, b.Bottom);
        double i = Math.Max(0, x1 - x0) * Math.Max(0, y1 - y0);
        double u = (a.Width * a.Height) + (b.Width * b.Height) - i;
        return u > 0 ? i / u : 0;
    }

    private static List<(string Text, OcrBox Box)> ReadTruth(string path)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(path));
        var list = new List<(string, OcrBox)>();
        foreach (var w in doc.RootElement.GetProperty("words").EnumerateArray())
        {
            var b = (w.TryGetProperty("ink", out var ink) ? ink : w.GetProperty("box")).EnumerateArray().Select(e => e.GetDouble()).ToArray();
            list.Add((w.GetProperty("text").GetString()!, new OcrBox(b[0], b[1], b[2], b[3])));
        }

        return list;
    }

    private sealed class Row
    {
        public string Page = "", Truth = "", Base = "";
        public float Conf;
        public bool Right;
        public Dictionary<string, (string Text, float Conf)> Cands = new();
        public Dictionary<string, double> Lm = new();
        public double BaseLm;
    }
}
