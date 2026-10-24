using System.Diagnostics;
using System.Globalization;
using OneOcr;
using OneOcr.Documents;
using OneOcr.Kernels;
using OneOcr.Model;

/// <summary>
/// <c>OneOcr.Bench docextract --model FILE [--words 150,300] [--degree N] [--repeat K] [--path scalar|vector128|avx2|avx512]</c>:
/// median wall time of the document-extraction forward pass (featurization and the network, and the assembly separately) for pages of
/// the given word counts laid out in rows, and the process's peak working set. Words are random strings over the model's charset.
/// </summary>
internal static class DocExtractBench
{
    public static int Run(Dictionary<string, string> o)
    {
        if (!o.TryGetValue("model", out string? file))
        {
            Console.Error.WriteLine("docextract needs --model FILE");
            return 2;
        }

        int degree = o.TryGetValue("degree", out string? d) ? int.Parse(d, CultureInfo.InvariantCulture) : 1;
        int repeat = o.TryGetValue("repeat", out string? r) ? Math.Max(1, int.Parse(r, CultureInfo.InvariantCulture)) : 15;
        KernelPath path = o.TryGetValue("path", out string? pathName) ? Enum.Parse<KernelPath>(pathName, ignoreCase: true) : Dispatch.Best;
        int[] counts = [.. (o.TryGetValue("words", out string? w) ? w : "150,300").Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture))];

        var load = Stopwatch.StartNew();
        ModelContainer container = ModelContainer.Load(file);
        DocExtractModel model = DocExtractModel.FromContainer(container, OcrLimits.Default);
        double loadMs = load.Elapsed.TotalMilliseconds;
        Console.WriteLine($"path {path}, degree {degree}, model load {loadMs:F0} ms");

        foreach (int n in counts)
        {
            var rng = new Random(n);
            var texts = new string[n];
            var boxes = new OcrBox[n];
            double x = 20, y = 20;
            for (int i = 0; i < n; i++)
            {
                int len = rng.Next(2, 12);
                texts[i] = string.Concat(Enumerable.Range(0, len).Select(_ => (char)('a' + rng.Next(26))));
                double width = 8.5 * len;
                if (x + width > 760)
                {
                    x = 20;
                    y += 22;
                }

                boxes[i] = new OcrBox(x, y, x + width, y + 18);
                x += width + 8;
            }

            var forward = new List<double>();
            var assembly = new List<double>();
            int[] index = [.. Enumerable.Range(0, n)];
            OcrSource[] sources = [.. index.Select(_ => OcrSource.Ocr)];
            for (int rep = -2; rep < repeat; rep++)
            {
                var clock = Stopwatch.StartNew();
                DocFeatures features = DocFeatures.Build(model, texts, boxes, 800, y + 40);
                DocOutputs outputs = DocForward.Run(model, features, path, degree, System.Threading.CancellationToken.None);
                double t1 = clock.Elapsed.TotalMilliseconds;
                clock.Restart();
                _ = DocAssembly.Assemble(model, outputs, texts, boxes, index, sources, null, NumberStyle.PointDecimal, DateOrder.Auto);
                double t2 = clock.Elapsed.TotalMilliseconds;
                if (rep >= 0)
                {
                    forward.Add(t1);
                    assembly.Add(t2);
                }
            }

            forward.Sort();
            assembly.Sort();
            Console.WriteLine($"words {n}: featurize+forward median {forward[forward.Count / 2]:F1} ms (min {forward[0]:F1}), assembly median {assembly[assembly.Count / 2]:F2} ms");
        }

        Console.WriteLine($"peak RSS {Process.GetCurrentProcess().PeakWorkingSet64 / 1048576.0:F0} MB");
        return 0;
    }
}
