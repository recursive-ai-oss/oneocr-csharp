namespace OneOcr.Recognition;

/// <summary>
/// Reads the letter <c>O</c> as the digit <c>0</c> where the characters around it are digits (<c>docs/spec/oneocr-page-v1.md</c>,
/// "Digits and the letter O"). The recognizer confuses the two in identifiers, order numbers and dates (<c>ACC-71O3O5</c>,
/// <c>2020-04-O7</c>), and the word's neighbours settle which it is. This is one of two places (with <see cref="SeparatorGlyphs"/>) after recognition that change a
/// character; it never adds or removes one, and it leaves every word that is not mostly a number alone.
/// </summary>
internal static class DigitContext
{
    /// <summary>A word needs at least this many digits before an <c>O</c> in it may change: shorter strings (<c>O2</c>, <c>No.7</c>) are words and units.</summary>
    internal const int MinDigits = 3;

    /// <summary>Returns <paramref name="text"/> with the <c>O</c> of every run of letters and digits that holds only digits and <c>O</c>s (and at least one of each) read as <c>0</c>, or <paramref name="text"/> itself.</summary>
    internal static string Fix(string text)
    {
        if (!text.Contains('O'))
        {
            return text;
        }

        int digits = 0;
        foreach (char c in text)
        {
            digits += char.IsAsciiDigit(c) ? 1 : 0;
        }

        if (digits < MinDigits)
        {
            return text;
        }

        char[]? result = null;
        int i = 0;
        while (i < text.Length)
        {
            if (!char.IsAsciiLetterOrDigit(text[i]))
            {
                i++;
                continue;
            }

            int end = i;
            bool onlyDigitsAndO = true, hasO = false, hasDigit = false;
            while (end < text.Length && char.IsAsciiLetterOrDigit(text[end]))
            {
                char c = text[end];
                hasO |= c == 'O';
                hasDigit |= char.IsAsciiDigit(c);
                onlyDigitsAndO &= c == 'O' || char.IsAsciiDigit(c);
                end++;
            }

            if (onlyDigitsAndO && hasO && hasDigit)
            {
                result ??= text.ToCharArray();
                for (int k = i; k < end; k++)
                {
                    if (result[k] == 'O')
                    {
                        result[k] = '0';
                    }
                }
            }

            i = end;
        }

        return result is null ? text : new string(result);
    }

    /// <summary>Applies <see cref="Fix"/> to every word of <paramref name="line"/> (and to its characters); returns <paramref name="line"/> itself when nothing changes.</summary>
    internal static LineWords Apply(LineWords line)
    {
        List<RecognizedWord>? words = null;
        for (int k = 0; k < line.Words.Count; k++)
        {
            RecognizedWord w = line.Words[k];
            string fixedText = Fix(w.Text);
            if (ReferenceEquals(fixedText, w.Text) || !CharsMatch(w))
            {
                words?.Add(w);
                continue;
            }

            words ??= [.. line.Words.Take(k)];
            var chars = new List<RecognizedChar>(w.Chars.Count);
            for (int c = 0; c < w.Chars.Count; c++)
            {
                RecognizedChar old = w.Chars[c];
                chars.Add(old with { Text = fixedText[c].ToString() });
            }

            words.Add(w with { Text = fixedText, Chars = chars });
        }

        return words is null ? line : new LineWords(line.Text, line.RawConfidence, words);
    }

    // The characters of a word are one text character each and spell the word; a word read without its characters is left alone.
    private static bool CharsMatch(RecognizedWord w)
    {
        if (w.Chars.Count != w.Text.Length)
        {
            return false;
        }

        for (int i = 0; i < w.Chars.Count; i++)
        {
            if (w.Chars[i].Text.Length != 1 || w.Chars[i].Text[0] != w.Text[i])
            {
                return false;
            }
        }

        return true;
    }
}
