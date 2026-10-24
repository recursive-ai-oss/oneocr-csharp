using System.Diagnostics;
using System.Globalization;
using OneOcr;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Recognition;
using OneOcr.Tests;
using OneOcr.Training;

// OneOcr.Bench trainstep --model FILE [--batch 16] [--width 400] [--chars 40] [--degree 1] [--repeat 5]
//
// Median wall time of one training step (forward, CTC loss, backward, AdamW) on a batch of synthetic lines of the given width with
// random targets, starting from the weights in --model, plus the time of the forward and backward part alone and the peak working
// set. Information only: it measures speed, not learning. Run each degree as its own process so that peak memory is per configuration.
internal static class TrainBench
{
    public static int Run(Dictionary<string, string> o)
    {
        int batchSize = o.TryGetValue("batch", out string? b) ? int.Parse(b, CultureInfo.InvariantCulture) : 16;
        int width = o.TryGetValue("width", out string? w) ? int.Parse(w, CultureInfo.InvariantCulture) : 400;
        int chars = o.TryGetValue("chars", out string? c) ? int.Parse(c, CultureInfo.InvariantCulture) : 40;
        int degree = o.TryGetValue("degree", out string? d) ? int.Parse(d, CultureInfo.InvariantCulture) : 1;
        int repeat = o.TryGetValue("repeat", out string? r) ? int.Parse(r, CultureInfo.InvariantCulture) : 5;
        if (degree < 1)
        {
            degree = Math.Min(4, Environment.ProcessorCount);
        }

        long t0 = Stopwatch.GetTimestamp();
        var model = CrnnModel.FromContainer(ModelContainer.Load(o["model"]), OcrLimits.Default);
        double loadMs = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        var trainer = new CrnnTrainer(model, Dispatch.Best, new AdamWOptions { LearningRate = 1e-5 }, maxGradNorm: 5);
        var rng = new Random(1);
        var batch = new List<TrainLine>();
        for (int i = 0; i < batchSize; i++)
        {
            byte[] image = TestModels.RandomImage(model.Input.Height, width, seed: i);
            batch.Add(new TrainLine(image, width, Enumerable.Range(0, chars).Select(_ => rng.Next(1, model.Classes)).ToArray()));
        }

        var step = new List<double>();
        var grads = new List<double>();
        double loss = 0;
        for (int i = 0; i < repeat + 1; i++)
        {
            long s = Stopwatch.GetTimestamp();
            loss = trainer.Step(batch, degree);
            double total = Stopwatch.GetElapsedTime(s).TotalMilliseconds;
            s = Stopwatch.GetTimestamp();
            trainer.ComputeGradients(batch, degree);
            double g = Stopwatch.GetElapsedTime(s).TotalMilliseconds;
            if (i > 0)
            {
                step.Add(total);
                grads.Add(g);
            }
        }

        step.Sort();
        grads.Sort();
        Console.WriteLine(
            $"path {Dispatch.Best} degree {degree} batch {batchSize} width {width}: step median {step[step.Count / 2]:F0} ms, " +
            $"forward+backward median {grads[grads.Count / 2]:F0} ms, model load {loadMs:F0} ms, loss {loss:F2}, " +
            $"peak working set {Process.GetCurrentProcess().PeakWorkingSet64 / (1024 * 1024)} MB");
        return 0;
    }
}
