using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using OneImage;
using OneOcr;
using OneOcr.Decoding;
using OneOcr.Recognition;

// OneOcr.Bench decode --model FILE --lines DIR [--lm FILE] [--modes greedy,beam:8,lm:8:0.5:0] [--max N] [--lexicon FILE]
//
// Reads truth-line crops (DIR/labels.jsonl with id and text, DIR/images/<id>.png at the model's input height), runs the network once
// per line and decodes the same logits in every requested mode. Prints per mode: CER and WER over all lines, the same for
// words that contain a digit in the truth ("numbers and IDs") and for the others, and the mean decode time per line.
//   greedy                  arg-max per step
//   beam:W                  prefix beam search, width W, no language model
//   lm:W:ALPHA:BETA         beam search with the language model (weight ALPHA, per-character bonus BETA)
//   lexlm:W:ALPHA:BETA:B    as lm, plus the lexicon from --lexicon with boost B
// --max keeps every k-th line so that at most N are read (default all).
internal static class DecodeBench
{
    private sealed record Mode(string Name, DecodeOptions Options);

    public static int Run(Dictionary<string, string> o)
    {
        string dir = o["lines"];
        int max = o.TryGetValue("max", out string? m) ? int.Parse(m, CultureInfo.InvariantCulture) : int.MaxValue;
        var recognizer = LineRecognizer.Load(o["model"]);
        CharLanguageModel? lm = o.TryGetValue("lm", out string? lmPath) ? CharLanguageModel.Load(lmPath) : null;
        Lexicon? lexicon = o.TryGetValue("lexicon", out string? lexPath) ? Lexicon.Load(lexPath) : null;
        List<Mode> modes = ParseModes(o.GetValueOrDefault("modes", "greedy,beam:8,lm:8:0.5:0"), lm, lexicon);

        var rows = new List<(string Id, string Text)>();
        foreach (string line in File.ReadLines(Path.Combine(dir, "labels.jsonl")))
        {
            using JsonDocument doc = JsonDocument.Parse(line);
            rows.Add((doc.RootElement.GetProperty("id").GetString()!, doc.RootElement.GetProperty("text").GetString()!));
        }

        int stride = Math.Max(1, (rows.Count + max - 1) / Math.Max(1, max));
        var stats = modes.ToDictionary(x => x.Name, _ => new Stats());
        double forwardMs = 0;
        int lines = 0, skipped = 0;
        var clock = new Stopwatch();
        for (int r = 0; r < rows.Count; r += stride)
        {
            (string id, string truth) = rows[r];
            Image image;
            Gray8[] px;
            try
            {
                image = Image.Load(Path.Combine(dir, "images", id + ".png"));
                px = image.ConvertTo(PixelFormat.Gray8).ToArray<Gray8>();
            }
            catch (ImageException)
            {
                skipped++;
                continue;
            }

            var gray = new byte[px.Length];
            for (int i = 0; i < gray.Length; i++)
            {
                gray[i] = px[i].V;
            }

            clock.Restart();
            float[] logits = recognizer.Forward(gray, image.Width, 1, null, System.Threading.CancellationToken.None, out int steps);
            forwardMs += clock.Elapsed.TotalMilliseconds;
            lines++;
            foreach (Mode mode in modes)
            {
                clock.Restart();
                string text = mode.Options.Mode == CtcDecoding.Beam
                    ? recognizer.DecodeBeam(logits, steps, mode.Options).Text
                    : recognizer.Decode(logits, steps).Text;
                stats[mode.Name].Add(truth, text, clock.Elapsed.TotalMilliseconds);
            }
        }

        Console.WriteLine($"{lines} lines ({skipped} unreadable images skipped), forward {forwardMs / Math.Max(1, lines):F2} ms/line");
        Console.WriteLine("mode                          CER%    WER%   numCER%  numTokErr%  numTokens  otherTokErr%  decode ms/line");
        foreach (Mode mode in modes)
        {
            Console.WriteLine(stats[mode.Name].Format(mode.Name, lines));
        }

        return 0;
    }

    private static List<Mode> ParseModes(string spec, CharLanguageModel? lm, Lexicon? lexicon)
    {
        var list = new List<Mode>();
        foreach (string item in spec.Split(','))
        {
            string[] p = item.Split(':');
            double D(int i, double fallback) => p.Length > i ? double.Parse(p[i], CultureInfo.InvariantCulture) : fallback;
            list.Add(p[0] switch
            {
                "greedy" => new Mode(item, DecodeOptions.Greedy),
                "beam" => new Mode(item, new DecodeOptions { Mode = CtcDecoding.Beam, BeamWidth = (int)D(1, 8) }),
                "lm" => new Mode(item, new DecodeOptions
                {
                    Mode = CtcDecoding.Beam, BeamWidth = (int)D(1, 8), LanguageModel = lm ?? throw new ArgumentException("--lm needed"),
                    LanguageModelWeight = D(2, 0.5), InsertionBonus = D(3, 0),
                }),
                "lexlm" => new Mode(item, new DecodeOptions
                {
                    Mode = CtcDecoding.Beam, BeamWidth = (int)D(1, 8), LanguageModel = lm, Lexicon = lexicon ?? throw new ArgumentException("--lexicon needed"),
                    LanguageModelWeight = D(2, 0.5), InsertionBonus = D(3, 0), LexiconBoost = D(4, 2),
                }),
                _ => throw new ArgumentException($"unknown mode {item}"),
            });
        }

        return list;
    }

    private sealed class Stats
    {
        private long _chars, _charErrors, _words, _wordErrors;
        private long _numChars, _numCharErrors, _numTokens, _numTokenErrors, _otherTokens, _otherTokenErrors;
        private double _ms;

        public void Add(string truth, string hyp, double ms)
        {
            _ms += ms;
            truth = Normalize(truth);
            hyp = Normalize(hyp);
            _chars += truth.Length;
            _charErrors += Edit(truth.ToCharArray(), hyp.ToCharArray());
            string[] t = truth.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            string[] h = hyp.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            _words += t.Length;
            // Word alignment: each truth word is matched, replaced or deleted; digit words are scored separately.
            bool[] wrong = AlignWords(t, h);
            for (int i = 0; i < t.Length; i++)
            {
                bool digits = t[i].Any(char.IsDigit);
                if (wrong[i])
                {
                    _wordErrors++;
                }

                if (digits)
                {
                    _numTokens++;
                    _numChars += t[i].Length;
                    _numTokenErrors += wrong[i] ? 1 : 0;
                }
                else
                {
                    _otherTokens++;
                    _otherTokenErrors += wrong[i] ? 1 : 0;
                }
            }

            _numCharErrors += DigitWordCharErrors(t, h);
        }

        public string Format(string name, int lines) => string.Create(CultureInfo.InvariantCulture,
            $"{name,-28} {100.0 * _charErrors / Math.Max(1, _chars),6:F2} {100.0 * _wordErrors / Math.Max(1, _words),7:F2} " +
            $"{100.0 * _numCharErrors / Math.Max(1, _numChars),9:F2} {100.0 * _numTokenErrors / Math.Max(1, _numTokens),11:F2} {_numTokens,10} " +
            $"{100.0 * _otherTokenErrors / Math.Max(1, _otherTokens),13:F2} {_ms / Math.Max(1, lines),15:F3}");

        private static string Normalize(string s) => string.Join(' ', s.Split(' ', StringSplitOptions.RemoveEmptyEntries));

        /// <summary>Levenshtein alignment of words; returns for each truth word whether it was not read exactly.</summary>
        private static bool[] AlignWords(string[] t, string[] h)
        {
            int[,] d = new int[t.Length + 1, h.Length + 1];
            for (int i = 0; i <= t.Length; i++)
            {
                d[i, 0] = i;
            }

            for (int j = 0; j <= h.Length; j++)
            {
                d[0, j] = j;
            }

            for (int i = 1; i <= t.Length; i++)
            {
                for (int j = 1; j <= h.Length; j++)
                {
                    d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1), d[i - 1, j - 1] + (t[i - 1] == h[j - 1] ? 0 : 1));
                }
            }

            var wrong = new bool[t.Length];
            int a = t.Length, b = h.Length;
            while (a > 0)
            {
                if (b > 0 && d[a, b] == d[a - 1, b - 1] + (t[a - 1] == h[b - 1] ? 0 : 1))
                {
                    wrong[a - 1] = t[a - 1] != h[b - 1];
                    a--;
                    b--;
                }
                else if (d[a, b] == d[a - 1, b] + 1)
                {
                    wrong[a - 1] = true;
                    a--;
                }
                else
                {
                    b--;
                }
            }

            return wrong;
        }

        /// <summary>Character errors on truth words that contain a digit, against the hypothesis word they align to.</summary>
        private static long DigitWordCharErrors(string[] t, string[] h)
        {
            int[,] d = new int[t.Length + 1, h.Length + 1];
            for (int i = 0; i <= t.Length; i++)
            {
                d[i, 0] = i;
            }

            for (int j = 0; j <= h.Length; j++)
            {
                d[0, j] = j;
            }

            for (int i = 1; i <= t.Length; i++)
            {
                for (int j = 1; j <= h.Length; j++)
                {
                    d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1), d[i - 1, j - 1] + (t[i - 1] == h[j - 1] ? 0 : 1));
                }
            }

            long errors = 0;
            int a = t.Length, b = h.Length;
            while (a > 0)
            {
                if (b > 0 && d[a, b] == d[a - 1, b - 1] + (t[a - 1] == h[b - 1] ? 0 : 1))
                {
                    if (t[a - 1].Any(char.IsDigit))
                    {
                        errors += Edit(t[a - 1].ToCharArray(), h[b - 1].ToCharArray());
                    }

                    a--;
                    b--;
                }
                else if (d[a, b] == d[a - 1, b] + 1)
                {
                    errors += t[a - 1].Any(char.IsDigit) ? t[a - 1].Length : 0;
                    a--;
                }
                else
                {
                    b--;
                }
            }

            return errors;
        }

        private static int Edit(char[] a, char[] b)
        {
            var prev = new int[b.Length + 1];
            var cur = new int[b.Length + 1];
            for (int j = 0; j <= b.Length; j++)
            {
                prev[j] = j;
            }

            for (int i = 1; i <= a.Length; i++)
            {
                cur[0] = i;
                for (int j = 1; j <= b.Length; j++)
                {
                    cur[j] = Math.Min(Math.Min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
                }

                (prev, cur) = (cur, prev);
            }

            return prev[b.Length];
        }
    }
}
