using System.Diagnostics;
using System.Globalization;
using OneOcr.Kernels;

// OneOcr.Bench gemm [--path avx512|avx2|vector128|scalar] [--m 2048] [--repeat 20]
//   GMAC/s of Gemm.NT for the shapes the networks use (k = reduction length, n = outputs per row), on one thread.
internal static class GemmBench
{
    public static int Run(Dictionary<string, string> o)
    {
        KernelPath path = o.TryGetValue("path", out string? p) ? Enum.Parse<KernelPath>(p, ignoreCase: true) : Dispatch.Best;
        int m = o.TryGetValue("m", out string? ms) ? int.Parse(ms, CultureInfo.InvariantCulture) : 2048;
        int repeat = o.TryGetValue("repeat", out string? rs) ? int.Parse(rs, CultureInfo.InvariantCulture) : 20;
        var rng = new Random(1);
        Console.WriteLine($"path {path}, m {m}");
        foreach ((int n, int k) in new[] { (32, 16), (32, 32), (64, 32), (64, 64), (96, 64), (128, 64), (128, 128), (64, 128), (192, 128), (192, 192), (256, 256), (622, 512), (1024, 256), (1024, 512) })
        {
            float[] a = new float[(long)m * k], b = new float[(long)n * k], bias = new float[n], c = new float[(long)m * n];
            foreach (float[] arr in new[] { a, b, bias })
            {
                for (int i = 0; i < arr.Length; i++)
                {
                    arr[i] = (float)(rng.NextDouble() - 0.5);
                }
            }

            var samples = new List<double>();
            for (int r = -2; r < repeat; r++)
            {
                long t0 = Stopwatch.GetTimestamp();
                Gemm.NT(path, a, k, b, k, m, n, k, bias, c, n, 1);
                double ms2 = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                if (r >= 0)
                {
                    samples.Add(ms2);
                }
            }

            samples.Sort();
            double best = samples[0], med = samples[samples.Count / 2];
            Console.WriteLine($"  n={n,5} k={k,4}: min {best,7:F3} ms  {(double)m * n * k / best / 1e6,6:F1} GMAC/s (median {(double)m * n * k / med / 1e6:F1})");
        }

        return 0;
    }
}
