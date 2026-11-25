using System.Text;

namespace OneOcr.Pdf;

/// <summary>
/// Removes the invisible text a third-party OCR step left in a page's content, without touching anything the page paints.
/// </summary>
/// <remarks>
/// <para>
/// The text-showing operators (<c>Tj</c>, <c>TJ</c>, <c>'</c>, <c>"</c>) that run while the text render mode is 3 (neither fill nor
/// stroke) are cut out of the content stream; every other byte is copied as it was, so what the page paints cannot change. The
/// render mode is part of the graphics state, so it is tracked through <c>q</c>/<c>Q</c> and <c>Tr</c>. A cut <c>'</c> or <c>"</c>
/// keeps its line move (<c>T*</c>, and the word and character spacing of <c>"</c>). Cutting a show also drops its advance of the
/// text matrix, which only matters when visible text is then shown on the same line without a positioning operator in between;
/// that case is detected and the page is left as it was (the result is null).
/// </para>
/// <para>
/// The input is untrusted: the scan is one pass over the bytes, allocates only for the result, never recurses, and gives up (null)
/// on anything it cannot place with certainty (an inline image it cannot delimit, a graphics state nested deeper than
/// <see cref="MaxStateDepth"/>, a malformed <c>"</c>). Form XObjects are not entered.
/// </para>
/// </remarks>
internal static class InvisibleTextRemover
{
    /// <summary>Deepest <c>q</c> nesting that is followed.</summary>
    public const int MaxStateDepth = 256;

    /// <summary>Removes invisible text shows; returns the new content, or null when nothing was removed or the content could not be handled safely.</summary>
    /// <param name="content">The decoded content stream (several streams joined by a line feed).</param>
    /// <param name="removed">The number of text-showing operators removed.</param>
    /// <returns>The rewritten content, or null.</returns>
    public static byte[]? Remove(ReadOnlySpan<byte> content, out int removed)
    {
        removed = 0;
        var output = new List<(int Start, int End, byte[] Replacement)>();
        var modes = new Stack<int>();
        int mode = 0;
        bool dirty = false; // a cut show moved the text matrix that the next show on this line would have continued from
        int pos = 0;
        int operandStart = -1;
        var spans = new List<(int Start, int End)>(4);
        while (true)
        {
            if (!Next(content, ref pos, out Token t))
            {
                break;
            }

            if (t.Kind != TokenKind.Operator)
            {
                if (operandStart < 0)
                {
                    operandStart = t.Start;
                }

                if (spans.Count < 4)
                {
                    spans.Add((t.Start, t.End));
                }

                continue;
            }

            ReadOnlySpan<byte> op = content[t.Start..t.End];
            int from = operandStart < 0 ? t.Start : operandStart;
            if (op.SequenceEqual("q"u8))
            {
                if (modes.Count >= MaxStateDepth)
                {
                    return null;
                }

                modes.Push(mode);
            }
            else if (op.SequenceEqual("Q"u8))
            {
                if (modes.Count > 0)
                {
                    mode = modes.Pop();
                }
            }
            else if (op.SequenceEqual("Tr"u8))
            {
                if (spans.Count == 1 && TryInt(content[spans[0].Start..spans[0].End], out int m))
                {
                    mode = m;
                }
            }
            else if (op.SequenceEqual("BT"u8) || op.SequenceEqual("Tm"u8) || op.SequenceEqual("Td"u8) || op.SequenceEqual("TD"u8) || op.SequenceEqual("T*"u8))
            {
                dirty = false;
            }
            else if (op.SequenceEqual("Tj"u8) || op.SequenceEqual("TJ"u8))
            {
                if (mode == 3)
                {
                    output.Add((from, t.End, []));
                    dirty = true;
                }
                else if (dirty)
                {
                    return null;
                }
            }
            else if (op.SequenceEqual("'"u8))
            {
                if (mode == 3)
                {
                    output.Add((from, t.End, "T*"u8.ToArray()));
                    dirty = false;
                }
            }
            else if (op.SequenceEqual("\""u8))
            {
                if (mode == 3)
                {
                    if (spans.Count != 3 || operandStart < 0)
                    {
                        return null;
                    }

                    string aw = Encoding.ASCII.GetString(content[spans[0].Start..spans[0].End]);
                    string ac = Encoding.ASCII.GetString(content[spans[1].Start..spans[1].End]);
                    if (!IsNumber(Encoding.ASCII.GetBytes(aw)) || !IsNumber(Encoding.ASCII.GetBytes(ac)))
                    {
                        return null;
                    }

                    output.Add((from, t.End, Encoding.ASCII.GetBytes($"{aw} Tw {ac} Tc T*")));
                    dirty = false;
                }
            }
            else if (op.SequenceEqual("BI"u8))
            {
                if (!SkipInlineImage(content, ref pos))
                {
                    return null;
                }
            }

            operandStart = -1;
            spans.Clear();
        }

        if (output.Count == 0)
        {
            return null;
        }

        var result = new MemoryStream(content.Length);
        int copied = 0;
        foreach ((int start, int end, byte[] replacement) in output)
        {
            result.Write(content[copied..start]);
            result.WriteByte((byte)' ');
            result.Write(replacement);
            result.WriteByte((byte)' ');
            copied = end;
        }

        result.Write(content[copied..]);
        removed = output.Count;
        return result.ToArray();
    }

    private enum TokenKind
    {
        Operand,
        Operator,
    }

    private readonly record struct Token(TokenKind Kind, int Start, int End);

    private static bool IsWhite(byte b) => b is 0 or 9 or 10 or 12 or 13 or 32;

    private static bool IsDelimiter(byte b) => b is (byte)'(' or (byte)')' or (byte)'<' or (byte)'>' or (byte)'[' or (byte)']' or (byte)'{' or (byte)'}' or (byte)'/' or (byte)'%';

    private static bool Next(ReadOnlySpan<byte> d, ref int pos, out Token token)
    {
        token = default;
        while (pos < d.Length)
        {
            byte b = d[pos];
            if (IsWhite(b))
            {
                pos++;
            }
            else if (b == (byte)'%')
            {
                while (pos < d.Length && d[pos] is not (10 or 13))
                {
                    pos++;
                }
            }
            else
            {
                break;
            }
        }

        if (pos >= d.Length)
        {
            return false;
        }

        int start = pos;
        byte c = d[pos];
        switch (c)
        {
            case (byte)'(':
            {
                int depth = 0;
                while (pos < d.Length)
                {
                    byte x = d[pos++];
                    if (x == (byte)'\\')
                    {
                        pos++;
                    }
                    else if (x == (byte)'(')
                    {
                        depth++;
                    }
                    else if (x == (byte)')' && --depth == 0)
                    {
                        break;
                    }
                }

                pos = Math.Min(pos, d.Length);
                token = new Token(TokenKind.Operand, start, pos);
                return true;
            }

            case (byte)'<':
                if (pos + 1 < d.Length && d[pos + 1] == (byte)'<')
                {
                    pos += 2;
                }
                else
                {
                    while (pos < d.Length && d[pos] != (byte)'>')
                    {
                        pos++;
                    }

                    pos = Math.Min(pos + 1, d.Length);
                }

                token = new Token(TokenKind.Operand, start, pos);
                return true;
            case (byte)'>':
                pos += pos + 1 < d.Length && d[pos + 1] == (byte)'>' ? 2 : 1;
                token = new Token(TokenKind.Operand, start, pos);
                return true;
            case (byte)'[' or (byte)']' or (byte)'{' or (byte)'}' or (byte)')':
                pos++;
                token = new Token(TokenKind.Operand, start, pos);
                return true;
            case (byte)'/':
                pos++;
                while (pos < d.Length && !IsWhite(d[pos]) && !IsDelimiter(d[pos]))
                {
                    pos++;
                }

                token = new Token(TokenKind.Operand, start, pos);
                return true;
        }

        while (pos < d.Length && !IsWhite(d[pos]) && !IsDelimiter(d[pos]))
        {
            pos++;
        }

        ReadOnlySpan<byte> word = d[start..pos];
        bool operand = IsNumber(word) || word.SequenceEqual("true"u8) || word.SequenceEqual("false"u8) || word.SequenceEqual("null"u8);
        token = new Token(operand ? TokenKind.Operand : TokenKind.Operator, start, pos);
        return true;
    }

    private static bool IsNumber(ReadOnlySpan<byte> s)
    {
        if (s.Length == 0)
        {
            return false;
        }

        bool digit = false;
        foreach (byte ch in s)
        {
            if (ch is >= (byte)'0' and <= (byte)'9')
            {
                digit = true;
            }
            else if (ch is not ((byte)'+' or (byte)'-' or (byte)'.'))
            {
                return false;
            }
        }

        return digit;
    }

    private static bool TryInt(ReadOnlySpan<byte> s, out int value) => int.TryParse(Encoding.ASCII.GetString(s), System.Globalization.NumberStyles.AllowLeadingSign, System.Globalization.CultureInfo.InvariantCulture, out value);

    // After BI: key/value tokens up to ID, one whitespace byte, the data, then EI. Without a reliable length the end is the first
    // whitespace + "EI" + (whitespace or end) whose following bytes look like content-stream text.
    private static bool SkipInlineImage(ReadOnlySpan<byte> d, ref int pos)
    {
        int p = pos;
        while (true)
        {
            if (!Next(d, ref p, out Token t))
            {
                return false;
            }

            if (t.Kind == TokenKind.Operator && d[t.Start..t.End].SequenceEqual("ID"u8))
            {
                break;
            }
        }

        int dataStart = p < d.Length && IsWhite(d[p]) ? p + 1 : p;
        for (int i = dataStart; i + 1 < d.Length; i++)
        {
            if (d[i] != (byte)'E' || d[i + 1] != (byte)'I')
            {
                continue;
            }

            bool before = i == dataStart || IsWhite(d[i - 1]);
            bool after = i + 2 >= d.Length || IsWhite(d[i + 2]) || IsDelimiter(d[i + 2]);
            if (!before || !after)
            {
                continue;
            }

            int check = Math.Min(d.Length, i + 2 + 12);
            bool text = true;
            for (int k = i + 2; k < check; k++)
            {
                if (d[k] > 126 || (d[k] < 32 && !IsWhite(d[k])))
                {
                    text = false;
                    break;
                }
            }

            if (text)
            {
                pos = i + 2;
                return true;
            }
        }

        return false;
    }
}
