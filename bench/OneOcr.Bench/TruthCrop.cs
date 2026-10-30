using System.Text;
using OneImage;
using OneOcr;
using OneOcr.Kernels;
using OneOcr.Layout;
using OneOcr.Recognition;

/// <summary>
/// Reads the recognizer on crops cut from the truth's word boxes, so that crop geometry (what the page stage hands the recognizer)
/// can be told apart from recognition (what the recognizer does with a good crop). Truth words are grouped into lines, each line is
/// cut with the page reader's rule (ink extent, margins of half a line height sideways and a quarter above and below, the line
/// height raised to 0.85 of the page's median), read by the network alone (no page-stage space or digit rules), and the text is
/// written as one page hypothesis, so the scorer's own comparison applies to it.
///
///   OneOcr.Bench truthcrop --model FILE --pages DIR --truth DIR --out DIR
/// </summary>
internal static class TruthCrop
{
    public static int Run(Dictionary<string, string> o)
    {
        string pages = o["pages"], truthDir = o["truth"], outDir = o["out"];
        Directory.CreateDirectory(outDir);
        LineRecognizer recognizer = LineRecognizer.Load(o["model"]);
        var pool = new BufferPool<Gray8>(1L << 28, 1);
        var resizers = new ResizerPool();
        foreach (string path in Directory.GetFiles(pages, "*.*").Where(f => f.EndsWith(".png") || f.EndsWith(".jpg")).Order(StringComparer.Ordinal))
        {
            string id = Path.GetFileNameWithoutExtension(path);
            Image image = Image.Load(path);
            var gray = new Gray8[image.Width * image.Height];
            image.CopyPixelsTo<Gray8>(gray, new ImageRect(0, 0, image.Width, image.Height), 1);
            byte[] bytes = PixelSpans.AsBytes(gray).ToArray();
            List<(string Text, double[] Box)> truth = ReadTruth(Path.Combine(truthDir, id + ".json"));
            List<List<int>> lines = Group(truth);
            var heights = lines.Select(l => LineBounds(truth, l)).Select(b => b[3] - b[1]).Order().ToList();
            double typical = heights.Count == 0 ? 0 : heights[heights.Count / 2];
            var text = new StringBuilder();
            foreach (List<int> line in lines)
            {
                double[] b = LineBounds(truth, line);
                double eh = Math.Max(b[3] - b[1], 0.85 * typical), cy = (b[1] + b[3]) / 2;
                int x0 = Math.Max(0, (int)Math.Floor(b[0] - (eh * 0.5))), x1 = Math.Min(image.Width, (int)Math.Ceiling(b[2] + (eh * 0.5)));
                int y0 = Math.Max(0, (int)Math.Floor(cy - (eh * 0.75))), y1 = Math.Min(image.Height, (int)Math.Ceiling(cy + (eh * 0.75)));
                if (x1 - x0 < 2 || y1 - y0 < 2)
                {
                    continue;
                }

                var box = new LineBox(x0, y0, x1 - x0, y1 - y0);
                Gray8[] px = LinePreprocessor.Prepare(bytes, image.Width, box, recognizer.InputHeight, pool, resizers, out int cw);
                try
                {
                    LineWords lw = recognizer.RecognizeWords(PixelSpans.AsBytes(px.AsSpan(0, cw * recognizer.InputHeight)), cw, 1, default, false);
                    text.Append(string.Join(' ', lw.Words.Select(w => w.Text))).Append('\n');
                }
                finally
                {
                    pool.Return(px);
                }
            }

            File.WriteAllText(Path.Combine(outDir, id + ".txt"), text.ToString());
        }

        return 0;
    }

    private static double[] LineBounds(List<(string Text, double[] Box)> truth, List<int> line)
    {
        double[] r = [double.MaxValue, double.MaxValue, double.MinValue, double.MinValue];
        foreach (int i in line)
        {
            double[] b = truth[i].Box;
            r[0] = Math.Min(r[0], b[0]);
            r[1] = Math.Min(r[1], b[1]);
            r[2] = Math.Max(r[2], b[2]);
            r[3] = Math.Max(r[3], b[3]);
        }

        return r;
    }

    // Truth words of one printed line: left to right, overlapping vertically by half of the smaller height, no wider a gap than
    // two and a half line heights (a column gutter is wider).
    private static List<List<int>> Group(List<(string Text, double[] Box)> truth)
    {
        var order = Enumerable.Range(0, truth.Count).OrderBy(i => (truth[i].Box[1] + truth[i].Box[3]) / 2).ToList();
        var lines = new List<List<int>>();
        foreach (int i in order)
        {
            double[] b = truth[i].Box;
            List<int>? home = null;
            foreach (List<int> l in lines)
            {
                double[] lb = LineBounds(truth, l);
                double h = Math.Min(b[3] - b[1], lb[3] - lb[1]);
                double overlap = Math.Min(b[3], lb[3]) - Math.Max(b[1], lb[1]);
                if (overlap < 0.5 * h)
                {
                    continue;
                }

                double gap = Math.Max(b[0] - lb[2], lb[0] - b[2]);
                // Compare with the nearest member, not the line's extent.
                double nearest = l.Min(j => Math.Max(b[0] - truth[j].Box[2], truth[j].Box[0] - b[2]));
                if (nearest <= 2.5 * (b[3] - b[1]) || gap < 0)
                {
                    home = l;
                    break;
                }
            }

            if (home is null)
            {
                lines.Add([i]);
            }
            else
            {
                home.Add(i);
            }
        }

        foreach (List<int> l in lines)
        {
            l.Sort((a, c) => truth[a].Box[0].CompareTo(truth[c].Box[0]));
        }

        return lines;
    }

    private static List<(string Text, double[] Box)> ReadTruth(string path)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(path));
        var list = new List<(string, double[])>();
        foreach (var w in doc.RootElement.GetProperty("words").EnumerateArray())
        {
            double[] b = (w.TryGetProperty("ink", out var ink) ? ink : w.GetProperty("box")).EnumerateArray().Select(e => e.GetDouble()).ToArray();
            list.Add((w.GetProperty("text").GetString()!, b));
        }

        return list;
    }
}
