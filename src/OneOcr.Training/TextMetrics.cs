using System.Text;
using OneOcr.Unicode;

namespace OneOcr.Training;

/// <summary>Text comparison used for agreement, alignment and the evaluation report. Everything works on code points, after the same normalization.</summary>
internal static class TextMetrics
{
    /// <summary>NFC, whitespace runs collapsed to one space, trimmed. Case and punctuation are kept.</summary>
    public static string Normalize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        string nfc = TextNormalization.Nfc(text);
        var sb = new StringBuilder(nfc.Length);
        bool space = false;
        foreach (char c in nfc)
        {
            if (char.IsWhiteSpace(c))
            {
                space = sb.Length > 0;
                continue;
            }

            if (space)
            {
                sb.Append(' ');
                space = false;
            }

            sb.Append(c);
        }

        return sb.ToString();
    }

    public static int[] ToCodePoints(string s)
    {
        var list = new List<int>(s.Length);
        foreach (Rune r in s.EnumerateRunes())
        {
            list.Add(r.Value);
        }

        return [.. list];
    }

    /// <summary>Levenshtein distance between two sequences.</summary>
    public static int EditDistance<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : IEquatable<T>
    {
        if (a.Length == 0)
        {
            return b.Length;
        }

        if (b.Length == 0)
        {
            return a.Length;
        }

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
                int sub = prev[j - 1] + (a[i - 1].Equals(b[j - 1]) ? 0 : 1);
                cur[j] = Math.Min(sub, Math.Min(prev[j] + 1, cur[j - 1] + 1));
            }

            (prev, cur) = (cur, prev);
        }

        return prev[b.Length];
    }

    /// <summary>Edit distance over the length of the reference, in code points; a hypothesis with text against an empty reference is 1.</summary>
    public static double Cer(string reference, string hypothesis)
    {
        int[] r = ToCodePoints(reference);
        int[] h = ToCodePoints(hypothesis);
        if (r.Length == 0)
        {
            return h.Length == 0 ? 0 : 1;
        }

        return EditDistance<int>(r, h) / (double)r.Length;
    }

    public static string[] Words(string s) => s.Split(' ', StringSplitOptions.RemoveEmptyEntries);
}

/// <summary>Running sums for a character and a word error rate over many lines (micro-averaged).</summary>
internal sealed class ErrorTally
{
    private long _charErrors, _chars, _wordErrors, _words;
    private int _lines, _exact;

    public void Add(string reference, string hypothesis)
    {
        reference = TextMetrics.Normalize(reference);
        hypothesis = TextMetrics.Normalize(hypothesis);
        int[] r = TextMetrics.ToCodePoints(reference), h = TextMetrics.ToCodePoints(hypothesis);
        string[] rw = TextMetrics.Words(reference), hw = TextMetrics.Words(hypothesis);
        _charErrors += TextMetrics.EditDistance<int>(r, h);
        _chars += r.Length;
        _wordErrors += TextMetrics.EditDistance<string>(rw, hw);
        _words += rw.Length;
        _lines++;
        if (reference == hypothesis)
        {
            _exact++;
        }
    }

    public TextErrorRates Rates() => new(_chars == 0 ? 0 : _charErrors / (double)_chars, _words == 0 ? 0 : _wordErrors / (double)_words, _lines, _exact);
}

/// <summary>Character and word error rate of a model's reading against reference text, summed over lines.</summary>
/// <param name="Cer">Character edit distance over reference characters, in code points.</param>
/// <param name="Wer">Word edit distance over reference words (words split at spaces).</param>
/// <param name="Lines">Lines compared.</param>
/// <param name="ExactLines">Lines read exactly as the reference (after whitespace normalization).</param>
public readonly record struct TextErrorRates(double Cer, double Wer, int Lines, int ExactLines);
