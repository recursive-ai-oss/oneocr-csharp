namespace OneOcr;

/// <summary>
/// A check for JSON text read from untrusted bytes. <c>System.Text.Json</c> accepts <c>\ud800</c>-style escapes that do not form a
/// surrogate pair while parsing, and only fails with a plain <see cref="InvalidOperationException"/> when the string is later read
/// or compared (property names included, even inside <c>JsonDocument.Parse</c>). Rejecting them up front keeps every later string
/// read safe.
/// </summary>
internal static class JsonEscapes
{
    /// <summary>Throws <see cref="OcrFormatException"/> when the UTF-8 JSON text contains a <c>\uXXXX</c> escape for a lone surrogate.</summary>
    /// <param name="utf8">The document; already known to be valid UTF-8.</param>
    /// <param name="what">What the document is, for the message.</param>
    public static void RejectLoneSurrogates(ReadOnlySpan<byte> utf8, string what)
    {
        int i = 0;
        while (i < utf8.Length)
        {
            if (utf8[i] != (byte)'\\')
            {
                i++;
                continue;
            }

            // An escape is two bytes, or six for \uXXXX; skipping by whole escapes keeps an escaped backslash from hiding the next byte.
            if (i + 1 < utf8.Length && utf8[i + 1] == (byte)'u' && TryHex4(utf8, i + 2, out int unit))
            {
                if (unit is >= 0xD800 and <= 0xDBFF)
                {
                    if (i + 11 < utf8.Length && utf8[i + 6] == (byte)'\\' && utf8[i + 7] == (byte)'u'
                        && TryHex4(utf8, i + 8, out int low) && low is >= 0xDC00 and <= 0xDFFF)
                    {
                        i += 12;
                        continue;
                    }

                    throw new OcrFormatException($"{what} has a high surrogate escape without a low surrogate.");
                }

                if (unit is >= 0xDC00 and <= 0xDFFF)
                {
                    throw new OcrFormatException($"{what} has a low surrogate escape without a high surrogate.");
                }

                i += 6;
                continue;
            }

            i += 2;
        }
    }

    private static bool TryHex4(ReadOnlySpan<byte> s, int at, out int value)
    {
        value = 0;
        if (at + 4 > s.Length)
        {
            return false;
        }

        for (int k = 0; k < 4; k++)
        {
            int c = s[at + k];
            int d = c is >= '0' and <= '9' ? c - '0' : c is >= 'a' and <= 'f' ? c - 'a' + 10 : c is >= 'A' and <= 'F' ? c - 'A' + 10 : -1;
            if (d < 0)
            {
                return false;
            }

            value = (value << 4) | d;
        }

        return true;
    }
}
