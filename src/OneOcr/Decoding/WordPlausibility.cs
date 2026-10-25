using System.Text;

namespace OneOcr.Decoding;

/// <summary>
/// How word-like a string of letters is under the character language model (<c>docs/spec/char-lm-v1.md</c>): the sum, over its
/// letters and its end, of the model's log probability centred on the model's average and floored at the decoder's floor, so that
/// a typical word scores about zero and each odd letter costs at most one nat. Used to compare readings of the same letters
/// that differ only in where the spaces are. Words with digits, code symbols or letters outside the script have no score.
/// </summary>
internal sealed class WordPlausibility
{
    private readonly ScriptModel[] _scripts;
    private readonly Dictionary<string, int>[] _symbolOf;

    public WordPlausibility(CharLanguageModel model)
    {
        _scripts = [.. model.Scripts];
        _symbolOf = new Dictionary<string, int>[_scripts.Length];
        for (int s = 0; s < _scripts.Length; s++)
        {
            var map = new Dictionary<string, int>(StringComparer.Ordinal);
            string[] alphabet = _scripts[s].Alphabet;
            for (int a = 0; a < alphabet.Length; a++)
            {
                map[alphabet[a]] = a + ScriptModel.FirstLetter;
            }

            _symbolOf[s] = map;
        }
    }

    /// <summary>
    /// The score of <paramref name="letters"/>, which must be letters only, or null when no script of the model covers all of them.
    /// </summary>
    public double? Score(string letters)
    {
        if (letters.Length == 0)
        {
            return null;
        }

        var folded = new List<string>(letters.Length);
        foreach (Rune r in letters.EnumerateRunes())
        {
            if (!Rune.IsLetter(r))
            {
                return null;
            }

            folded.Add(Prepared.Fold(r.ToString()));
        }

        // The script that covers the most letters; the first on a tie. A word the script does not fully cover has no score.
        int best = -1, bestCovered = 0;
        for (int s = 0; s < _scripts.Length; s++)
        {
            int covered = 0;
            foreach (string f in folded)
            {
                covered += _symbolOf[s].ContainsKey(f) ? 1 : 0;
            }

            if (covered > bestCovered)
            {
                bestCovered = covered;
                best = s;
            }
        }

        if (best < 0 || bestCovered != folded.Count)
        {
            return null;
        }

        ScriptModel script = _scripts[best];
        var context = new byte[folded.Count + 1];
        context[0] = ScriptModel.Start;
        double score = 0;
        for (int i = 0; i < folded.Count; i++)
        {
            byte symbol = (byte)_symbolOf[best][folded[i]];
            float lp = script.LogProb(context.AsSpan(0, i + 1), symbol);
            score += Math.Max(lp - script.MeanLogProb, BeamDecoder.Floor);
            context[i + 1] = symbol;
        }

        float end = script.LogProb(context, ScriptModel.End);
        return score + Math.Max(end - script.MeanLogProb, BeamDecoder.Floor);
    }
}
