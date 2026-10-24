using System.Diagnostics;
using System.Text;

namespace OneOcr.Llm;

/// <summary>
/// Holds an API key so that it cannot leak by accident: it prints as <c>[redacted]</c> from <c>ToString()</c> and in the
/// debugger, and <see cref="Scrub"/> removes it from any text that came from a server.
/// </summary>
[DebuggerDisplay("[redacted]")]
internal sealed class Secret(string value)
{
    public const string Mask = "[redacted]";
    private const int MinFragment = 8;
    private const int MaxEchoed = 400;

    [DebuggerBrowsable(DebuggerBrowsableState.Never)]
    private readonly string _value = value;

    [DebuggerBrowsable(DebuggerBrowsableState.Never)]
    public string Value => _value;

    public override string ToString() => Mask;

    /// <summary>
    /// Replaces the key (also trimmed of surrounding whitespace) and every run of at least 8 characters that occurs in both the
    /// key and the text with <c>[redacted]</c>, then cuts the result to a few hundred characters.
    /// </summary>
    public string Scrub(string text) => Cap(ScrubAll(text));

    /// <summary>As <see cref="Scrub"/> without the length cut.</summary>
    public string ScrubAll(string text)
    {
        string key = _value.Trim();
        if (key.Length == 0 || text.Length == 0)
        {
            return text;
        }

        var hidden = new bool[text.Length];
        int window = Math.Min(MinFragment, key.Length);
        for (int i = 0; i + window <= key.Length; i++)
        {
            ReadOnlySpan<char> piece = key.AsSpan(i, window);
            int from = 0;
            while (from <= text.Length - window)
            {
                int at = text.AsSpan(from).IndexOf(piece, StringComparison.Ordinal);
                if (at < 0)
                {
                    break;
                }

                at += from;
                hidden.AsSpan(at, window).Fill(true);
                from = at + 1;
            }
        }

        var sb = new StringBuilder(text.Length);
        for (int i = 0; i < text.Length; i++)
        {
            if (!hidden[i])
            {
                sb.Append(text[i]);
            }
            else if (i == 0 || !hidden[i - 1])
            {
                sb.Append(Mask);
            }
        }

        return sb.ToString();
    }

    private static string Cap(string s) => s.Length <= MaxEchoed ? s : s[..MaxEchoed];
}
