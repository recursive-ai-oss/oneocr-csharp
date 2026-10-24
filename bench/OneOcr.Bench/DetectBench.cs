using System.Diagnostics;
using System.Globalization;
using OneImage;
using OneOcr;
using OneOcr.Detection;
using OneOcr.Kernels;
using OneOcr.Layout;
using OneOcr.Model;



/// <summary>
/// <c>OneOcr.Bench detect [--model FILE] [--page 2480x3508] [--degree N] [--repeat K] [--path scalar|vector128|avx2|avx512]</c>:
/// median wall time of the page detector on a synthetic gray page (scaling to 1024 pixels, the network, and decoding the maps), and the
/// network alone on the padded input, with the process's peak working set. Without <c>--model</c> a random-weight detector of the
/// shipped size is used; its boxes are meaningless but the work done is the same.
/// </summary>
internal static class DetectBench
{
    public static int Run(Dictionary<string, string> o)
    {
        int degree = Int(o, "degree", 1), repeat = Math.Max(1, Int(o, "repeat", 15));
        string[] size = (o.TryGetValue("page", out string? p) ? p : "2480x3508").Split('x');
        int width = int.Parse(size[0], CultureInfo.InvariantCulture), height = int.Parse(size[1], CultureInfo.InvariantCulture);
        KernelPath path = o.TryGetValue("path", out string? pathName) ? Enum.Parse<KernelPath>(pathName, ignoreCase: true) : Dispatch.Best;

        var load = Stopwatch.StartNew();
        ModelContainer container = o.TryGetValue("model", out string? file)
            ? ModelContainer.Load(file)
            : ModelContainer.Parse(global::OneOcr.Tests.TestDetectorModels.Random(global::OneOcr.Tests.TestDetectorModels.Default, 1));
        DbNetModel model = DbNetModel.FromContainer(container, OcrLimits.Default);
        var detector = new PageDetector(model, OcrLimits.Default, path);
        double loadMs = load.Elapsed.TotalMilliseconds;

        var page = new Gray8[width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Text-like rows: dark bars on white, so the maps are not degenerate.
                bool ink = ((y / 12) % 3 == 0) && ((x / 7) % 5 != 0) && x > 150 && x < width - 150;
                page[(y * width) + x] = new Gray8(ink ? (byte)30 : (byte)250);
            }
        }

        var pixels = new BufferPool<Gray8>(64L << 20, 1);
        var resizers = new ResizerPool();
        var whole = new List<double>();
        for (int r = -2; r < repeat; r++)
        {
            var clock = Stopwatch.StartNew();
            PageDetection d = detector.Detect(page, width, height, pixels, resizers, degree, System.Threading.CancellationToken.None);
            if (r >= 0)
            {
                whole.Add(clock.Elapsed.TotalMilliseconds);
            }

            GC.KeepAlive(d);
        }

        // The network alone on the padded 1024 x 736 (A4) input.
        double scale = PageDetector.Scale(width, height);
        int ph = ((int)Math.Round(height * scale) + 31) / 32 * 32, pw = ((int)Math.Round(width * scale) + 31) / 32 * 32;
        var input = new float[ph * pw];
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (((i * 37) & 255) / 127.5f) - 1f;
        }

        var floats = new BufferPool<float>(64L << 20, sizeof(float));
        var network = new List<double>();
        for (int r = -2; r < repeat; r++)
        {
            var clock = Stopwatch.StartNew();
            DetectorMaps maps = DbNetForward.Run(model, path, degree, floats, input, ph, pw, OcrLimits.Default, null, System.Threading.CancellationToken.None);
            if (r >= 0)
            {
                network.Add(clock.Elapsed.TotalMilliseconds);
            }

            GC.KeepAlive(maps);
        }

        whole.Sort();
        network.Sort();
        double peak = Process.GetCurrentProcess().PeakWorkingSet64 / 1048576.0;
        Console.WriteLine(
            $"detect path={path} degree={degree} page={width}x{height} input={pw}x{ph} load={loadMs:F1} ms  " +
            $"whole(median/min)={whole[whole.Count / 2]:F1}/{whole[0]:F1} ms  network(median/min)={network[network.Count / 2]:F1}/{network[0]:F1} ms  peak={peak:F0} MB  " +
            $"allocated={GC.GetTotalAllocatedBytes() / 1048576.0:F0} MB total");
        return 0;
    }

    private static int Int(Dictionary<string, string> o, string key, int fallback) =>
        o.TryGetValue(key, out string? v) ? int.Parse(v, CultureInfo.InvariantCulture) : fallback;
}
