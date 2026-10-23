using OneOcr.Decoding;
using OneOcr.Kernels;
using OneOcr.Recognition;

namespace OneOcr.Fuzz;

/// <summary>
/// Character language model bytes through the parser and, when they parse, through beam search over pseudo-random class
/// probabilities with the model attached (so every table lookup path runs), under tight limits.
/// Only the typed exceptions may come out; anything else is a finding.
/// </summary>
internal static class CharLmTarget
{
    private static readonly OcrLimits Limits = new()
    {
        MaxLanguageModelFileBytes = 1 << 20,
        MaxLanguageModelNgrams = 1 << 16,
        MaxLanguageModelScripts = 4,
        MaxLanguageModelOrder = 6,
    };

    /// <summary>Inputs that parsed (progress statistic for local runs).</summary>
    public static long Parsed;

    public static void Run(ReadOnlySpan<byte> data)
    {
        try
        {
            CharLanguageModel lm = CharLanguageModel.Parse(data, Limits);
            Parsed++;
            // A charset made of the first script's alphabet, so that the model's symbols are reachable.
            ScriptModel script = lm.Scripts[0];
            var charset = new List<string> { "<blank>", " " };
            charset.AddRange(script.Alphabet);
            charset.Add("7");
            int classes = charset.Count;
            int steps = 24;
            var logits = new float[steps * classes];
            var rng = new Random(data.Length);
            for (int i = 0; i < logits.Length; i++)
            {
                logits[i] = (float)((rng.NextDouble() * 8) - 3);
            }

            var options = new DecodeOptions
            {
                Mode = CtcDecoding.Beam, BeamWidth = 4, LanguageModel = lm, Lexicon = Lexicon.FromWords(["ab", "a7"]),
            };
            Prepared prepared = options.Prepare(charset);
            _ = BeamDecoder.Decode(options, prepared, logits, steps, classes, KernelPath.Scalar, new RecognizedLine(string.Empty, 0f));
        }
        catch (OcrModelException)
        {
        }
        catch (OcrLimitException)
        {
        }
    }
}

/// <summary>Lexicon file bytes through the parser. Only the typed exceptions may come out.</summary>
internal static class LexiconTarget
{
    private static readonly OcrLimits Limits = new() { MaxLexiconFileBytes = 1 << 16, MaxLexiconWords = 1000, MaxLexiconWordLength = 32 };

    public static void Run(ReadOnlySpan<byte> data)
    {
        try
        {
            _ = Lexicon.Parse(data, Limits);
        }
        catch (OcrModelException)
        {
        }
        catch (OcrLimitException)
        {
        }
    }
}
