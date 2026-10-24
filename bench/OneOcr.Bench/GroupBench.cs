using System.Diagnostics;
using System.Globalization;
using System.Threading;
using OneImage;
using OneOcr.Recognition;

// OneOcr.Bench group --model FILE [--width 1000] [--lines 12] [--repeat 10]
//   median time to read the same synthetic lines one at a time (RecognizeWords) and as one group (RecognizeWordsGroup), on one thread.
internal static class GroupBench
{
    public static int Run(Dictionary<string, string> o)
    {
        int width = o.TryGetValue("width", out string? w) ? int.Parse(w, CultureInfo.InvariantCulture) : 1000;
        int lines = o.TryGetValue("lines", out string? l) ? int.Parse(l, CultureInfo.InvariantCulture) : 12;
        int repeat = o.TryGetValue("repeat", out string? r) ? int.Parse(r, CultureInfo.InvariantCulture) : 10;
        LineRecognizer recognizer = LineRecognizer.Load(o["model"]);
        int h = recognizer.InputHeight;
        var images = new LineImage[lines];
        for (int i = 0; i < lines; i++)
        {
            int wi = width - (i * 7);
            byte[] bytes = OneOcr.Tests.TestModels.RandomImage(h, wi, i + 1);
            var px = new Gray8[bytes.Length];
            for (int k = 0; k < px.Length; k++)
            {
                px[k] = new Gray8(bytes[k]);
            }

            images[i] = new LineImage(px, wi);
        }

        var single = new List<double>();
        var group = new List<double>();
        for (int rep = -2; rep < repeat; rep++)
        {
            long t0 = Stopwatch.GetTimestamp();
            foreach (LineImage im in images)
            {
                recognizer.RecognizeWords(OneOcr.Kernels.PixelSpans.AsBytes(im.Pixels.AsSpan(0, im.Width * h)), im.Width, 1, CancellationToken.None, false);
            }

            double a = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
            t0 = Stopwatch.GetTimestamp();
            recognizer.RecognizeWordsGroup(images, CancellationToken.None, false);
            double b = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
            if (rep >= 0)
            {
                single.Add(a);
                group.Add(b);
            }
        }

        single.Sort();
        group.Sort();
        Console.WriteLine($"{lines} lines of ~{width}: one by one {single[single.Count / 2]:F1} ms, as a group {group[group.Count / 2]:F1} ms");
        return 0;
    }
}
