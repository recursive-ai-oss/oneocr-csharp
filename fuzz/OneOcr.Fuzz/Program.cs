using OneOcr;
using System.Diagnostics;
using OneOcr.Fuzz;
using OneOcr.Tests;
using SharpFuzz;

// One executable for the fuzz targets.
//
//   OneOcr.Fuzz                      run the target under libFuzzer (the host starts the program without arguments; the
//                                    target comes from ONEOCR_FUZZ_TARGET, default "container")
//   OneOcr.Fuzz replay <file>...     run saved inputs once each, to reproduce a finding
//   OneOcr.Fuzz mutate <seconds> <corpusdir> [seed]
//                                    a plain in-process mutation loop with no coverage feedback, for a quick local run
//   OneOcr.Fuzz seeds <dir>          write the seed corpus (small valid containers)
internal static class Program
{
    private delegate void Target(ReadOnlySpan<byte> data);

    private static Target ChooseTarget()
    {
        string name = Environment.GetEnvironmentVariable("ONEOCR_FUZZ_TARGET") ?? "container";
        return name switch
        {
            "container" => ContainerTarget.Run,
            "charlm" => CharLmTarget.Run,
            "lexicon" => LexiconTarget.Run,
            "pagejson" => PageJsonTarget.Run,
            _ => throw new ArgumentException($"Unknown target '{name}'."),
        };
    }

    private static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            Target target = ChooseTarget();
            Fuzzer.LibFuzzer.Run(span => target(span));
            return 0;
        }

        switch (args[0])
        {
            case "replay":
                foreach (string file in args.Skip(1))
                {
                    ChooseTarget()(File.ReadAllBytes(file));
                    Console.WriteLine($"ok {file}");
                }

                return 0;
            case "mutate" when args.Length >= 3:
                return Mutate(double.Parse(args[1], System.Globalization.CultureInfo.InvariantCulture), args[2], args.Length > 3 ? int.Parse(args[3], System.Globalization.CultureInfo.InvariantCulture) : 1);
            case "seeds" when args.Length == 2:
                WriteSeeds(args[1]);
                return 0;
            default:
                Console.Error.WriteLine("usage: OneOcr.Fuzz [replay <file>... | mutate <seconds> <corpusdir> [seed] | seeds <dir>]");
                return 2;
        }
    }

    private static int Mutate(double seconds, string corpusDir, int seed)
    {
        Target target = ChooseTarget();
        var corpus = Directory.GetFiles(corpusDir).Select(File.ReadAllBytes).ToList();
        if (corpus.Count == 0)
        {
            Console.Error.WriteLine("empty corpus");
            return 2;
        }

        var rng = new Random(seed);
        // Truncated inputs are not kept for further mutation: they would shrink the corpus toward inputs that fail at once.
        int keepAtLeast = corpus.Max(c => c.Length) / 2;
        var clock = Stopwatch.StartNew();
        long runs = 0;
        while (clock.Elapsed.TotalSeconds < seconds)
        {
            byte[] input = (byte[])corpus[rng.Next(corpus.Count)].Clone();
            int edits = 1 + rng.Next(6);
            for (int e = 0; e < edits && input.Length > 0; e++)
            {
                switch (rng.Next(5))
                {
                    case 0:
                        input[rng.Next(input.Length)] = (byte)rng.Next(256);
                        break;
                    case 1:
                        input[rng.Next(input.Length)] ^= (byte)(1 << rng.Next(8));
                        break;
                    case 2:
                        // Interesting values at 4-byte positions: sizes, lengths and offsets live there.
                        int at = rng.Next(Math.Max(1, input.Length - 3));
                        uint v = new uint[] { 0, 1, 63, 64, 65, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 1 << 20 }[rng.Next(9)];
                        for (int b = 0; b < 4 && at + b < input.Length; b++)
                        {
                            input[at + b] = (byte)(v >> (8 * b));
                        }

                        break;
                    case 3:
                        if (rng.Next(10) == 0)
                        {
                            input = input[..rng.Next(input.Length + 1)];
                        }

                        break;
                    default:
                        // Digit changes inside the JSON manifest change sizes without breaking the syntax.
                        int p = rng.Next(input.Length);
                        if (input[p] is >= (byte)'0' and <= (byte)'9')
                        {
                            input[p] = (byte)('0' + rng.Next(10));
                        }

                        break;
                }
            }

            target(input);
            runs++;
            if (runs % 64 == 0 && input.Length >= Math.Max(16, keepAtLeast))
            {
                if (corpus.Count < 256)
                {
                    corpus.Add(input);
                }
                else
                {
                    corpus[rng.Next(3, corpus.Count)] = input;
                }
            }
        }

        Console.WriteLine($"mutate: {runs} executions in {clock.Elapsed.TotalSeconds:F0} s ({ContainerTarget.Parsed + CharLmTarget.Parsed} parsed, {ContainerTarget.Built} built a recognizer), no findings");
        return 0;
    }

    private static void WriteSeeds(string dir)
    {
        Directory.CreateDirectory(dir);
        string target = Environment.GetEnvironmentVariable("ONEOCR_FUZZ_TARGET") ?? "container";
        if (target == "charlm")
        {
            string[] alphabet = ["a", "b", "c", "d", "o", "\u00e9"];
            File.WriteAllBytes(Path.Combine(dir, "order3.seed"), TestLm.Build("latin", alphabet, ["cab", "cab", "dab", "cod", "b\u00e9"], 3));
            File.WriteAllBytes(Path.Combine(dir, "order2.seed"), TestLm.Build("latin", alphabet, ["cab", "dab", "cod"], 2));
            File.WriteAllBytes(Path.Combine(dir, "two-scripts.seed"), TestLm.Assemble([("latin", alphabet, ["cab", "dab"]), ("other", ["x", "y"], ["xy", "yx"])], 3));
            return;
        }

        if (target == "pagejson")
        {
            OcrPage sample = SamplePage.Build();
            File.WriteAllText(Path.Combine(dir, "full.seed"), OcrPageJson.ToJson([sample, new OcrPage(1, 100, 50, 300, [])]));
            File.WriteAllText(Path.Combine(dir, "indented.seed"), OcrPageJson.ToJson([sample], new OcrPageJsonOptions { Indented = true, IncludeChars = false }));
            File.WriteAllText(Path.Combine(dir, "empty.seed"), OcrPageJson.ToJson([]));
            return;
        }

        if (target == "lexicon")
        {
            File.WriteAllText(Path.Combine(dir, "words.seed"), "alpha\nbeta\r\n\ngamma  \nINV-0042\nZo\u00eb\n");
            return;
        }

        var variants = new (string Name, TestModels.Shape Shape)[]
        {
            ("tiny", new TestModels.Shape(8, [(4, 2, 2), (4, 2, 2), (8, 2, 1)], 8, 1, 12)),
            ("two-layer", new TestModels.Shape(8, [(4, 2, 2), (6, 2, 1), (8, 2, 1)], 8, 2, 20)),
            ("height16", new TestModels.Shape(16, [(4, 2, 2), (4, 2, 2), (6, 2, 1), (8, 2, 1)], 12, 1, 30)),
        };
        foreach (var (name, shape) in variants)
        {
            File.WriteAllBytes(Path.Combine(dir, $"{name}.seed"), TestModels.RandomContainer(shape, 1));
        }

        // Version 1: with calibration curves, with one curve and one-knot curves, and with a charset that has space entries.
        var tinyShape = variants[0].Shape;
        File.WriteAllBytes(Path.Combine(dir, "v1-all-scripts.seed"), TestModels.RandomContainerV1(tinyShape, 1,
            "{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Latin\":{\"x\":[0,0.5,1],\"y\":[0,0.25,1]},\"Greek\":{\"x\":[0.2,0.8],\"y\":[0.1,0.9]},\"Cyrillic\":{\"x\":[0.5],\"y\":[0.5]}}}"));
        File.WriteAllBytes(Path.Combine(dir, "v1-empty.seed"), TestModels.RandomContainerV1(variants[1].Shape, 2,
            "{\"raw\":\"word-mean-max-run\",\"scripts\":{}}"));
        string[] spaced = TestModels.Charset(tinyShape.Classes);
        spaced[1] = " ";
        spaced[2] = "a";
        File.WriteAllBytes(Path.Combine(dir, "v1-spaces.seed"), TestContainerWriter.Write(
            "crnn-v0", tinyShape.Height, spaced, TestModels.ConfigJson(tinyShape), TestModels.RandomTensors(tinyShape, 3),
            calibrationJson: "{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Latin\":{\"x\":[0,1],\"y\":[0.1,0.9]}}}"));
    }
}
