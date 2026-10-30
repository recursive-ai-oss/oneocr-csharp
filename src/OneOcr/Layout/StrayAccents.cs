using OneOcr.Decoding;
using OneOcr.Unicode;

namespace OneOcr.Layout;

/// <summary>
/// Takes accents and dots off letters that were read with them from grain. On a degraded scan a speck above or below a letter makes
/// the network write <c>thé</c> for <c>the</c> or <c>dọwn</c> for <c>down</c>, usually with middling confidence. The rule works on the
/// whole page, after recognition and before the noise filter (<c>docs/spec/oneocr-page-v1.md</c>, "Stray accents"):
/// a word changes only when it is unsure, made of Latin letters and edge punctuation, and its plain-letter form is a word the
/// frequency table knows and is far more frequent than the accented spelling. On a page where accented words are common, which is
/// a page in a language that uses them, nothing changes. The word keeps its box and confidence; only the letters change.
/// </summary>
internal static class StrayAccents
{
    /// <summary>
    /// Returns how many words changed. Entries of <paramref name="strips"/> are replaced by copies with the new words.
    /// </summary>
    public static int Apply(OrderStrip?[] strips, JunkParameters p, WordFrequencyModel frequencies)
    {
        // Accented words are counted for the whole page first: a page in French or Vietnamese has many, a page of English text
        // that was scanned badly has a few.
        int letterWords = 0, accented = 0;
        foreach (OrderStrip? s in strips)
        {
            if (s is null)
            {
                continue;
            }

            foreach (OcrWord w in s.Words)
            {
                bool any = false, mark = false;
                foreach (char c in w.Text)
                {
                    any |= char.IsLetter(c);
                    mark |= TextNormalization.TryStripMarks(c, out _);
                }

                letterWords += any ? 1 : 0;
                accented += mark ? 1 : 0;
            }
        }

        if (accented == 0 || accented > p.AccentPageShare * letterWords)
        {
            return 0;
        }

        int changed = 0;
        for (int i = 0; i < strips.Length; i++)
        {
            if (strips[i] is not { } s)
            {
                continue;
            }

            List<OcrWord>? words = null;
            for (int k = 0; k < s.Words.Count; k++)
            {
                OcrWord w = s.Words[k];
                OcrWord? fixedWord = w.Confidence < p.AccentConfidence ? Fix(w, p.AccentMargin, frequencies) : null;
                if (fixedWord is not null)
                {
                    words ??= new List<OcrWord>(s.Words.Take(k));
                    words.Add(fixedWord);
                    changed++;
                }
                else
                {
                    words?.Add(w);
                }
            }

            if (words is not null)
            {
                strips[i] = new OrderStrip { Leaf = s.Leaf, Box = s.Box, Size = s.Size, Words = words };
            }
        }

        return changed;
    }

    // The word with its marks taken off, or null when the rule does not apply to it.
    private static OcrWord? Fix(OcrWord w, int margin, WordFrequencyModel frequencies)
    {
        string text = w.Text;
        int from = 0, to = text.Length;
        while (from < to && !char.IsLetterOrDigit(text[from]))
        {
            from++;
        }

        while (to > from && !char.IsLetterOrDigit(text[to - 1]))
        {
            to--;
        }

        // Two letters at least, and only letters between the edge marks: digits, hyphens and apostrophes make codes, names and
        // compounds, which the table does not judge.
        if (to - from < 2)
        {
            return null;
        }

        var plain = new char[text.Length];
        var accentedLower = new char[to - from];
        var plainLower = new char[to - from];
        bool marked = false;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (TextNormalization.TryStripMarks(c, out char bare))
            {
                marked = true;
                plain[i] = bare;
            }
            else
            {
                plain[i] = c;
                if (i >= from && i < to && !char.IsAsciiLetter(c))
                {
                    // Inside the edge marks only plain and strippable letters are allowed: a letter outside Latin (Cyrillic,
                    // Greek) or with no plain form (sharp s, ae, oe), a digit or an inner mark makes the word not ours to change.
                    return null;
                }
            }

            if (i >= from && i < to)
            {
                accentedLower[i - from] = char.ToLowerInvariant(c);
                plainLower[i - from] = char.ToLowerInvariant(plain[i]);
            }
        }

        if (!marked)
        {
            return null;
        }

        string plainWord = new(plainLower), accentedWord = new(accentedLower);
        if (!frequencies.Contains(plainWord) || frequencies.Quarter(plainWord) < frequencies.Quarter(accentedWord) + margin)
        {
            return null;
        }

        string result = new(plain);
        if (w.Chars.Count == 0)
        {
            return w with { Text = result };
        }

        if (w.Chars.Count != text.Length)
        {
            return null;
        }

        var chars = new List<OcrChar>(w.Chars.Count);
        for (int i = 0; i < w.Chars.Count; i++)
        {
            if (w.Chars[i].Text.Length != 1 || w.Chars[i].Text[0] != text[i])
            {
                return null;
            }

            chars.Add(w.Chars[i] with { Text = plain[i].ToString() });
        }

        return w with { Text = result, Chars = chars };
    }
}
