// Clause labels in a hanging indent. Contracts, statements of terms and numbered notes set a label ("(a)", "1.2", "§ 4", a bullet) in the
// margin and indent the text of the clause, including the lines it wraps to, past the label. The label and the first line of its text
// stand on one row, the wrapped lines under it have no label, so the column cut finds two columns and reads every label before any text,
// or reads the text of one clause and then the label of the next. A clause is read as a unit instead: its label, then its lines.
//
// The cut is only made when the left column holds nothing but labels, each with text on its own row, and the text column is left-aligned
// (a hanging indent), so a table, a form or a menu with short entries is not taken for it.

namespace OneOcr.Layout;

internal static partial class ReadingOrder
{
    /// <summary>Most words of a label fragment ("§ 4", "(a)").</summary>
    private const int LabelMaxWords = 2;

    /// <summary>Longest label in characters ("1.2.3.4.").</summary>
    private const int LabelMaxChars = 10;

    /// <summary>At least this many labels make a column of clause labels.</summary>
    private const int LabelMinCount = 2;

    /// <summary>The text column is left-aligned: this share of its fragments start within <see cref="HangingAlignEm"/> of its left edge.</summary>
    private const double HangingAlignShare = 0.7;

    /// <summary>Left edges of the lines of one hanging indent differ by at most this many em.</summary>
    private const double HangingAlignEm = 1;

    // A region cut into two columns (labels, text): the clauses as groups, each the label with the lines of its text, in order. Null when the
    // columns are not a column of clause labels and a hanging text column.
    private static List<int[]>? HangingClauses(List<OrderFragment> f, IReadOnlyList<OrderStrip> strips, int[] reg, List<int[]> columns, double ms)
    {
        if (columns.Count != 2)
        {
            return null;
        }

        int[] labels = columns[0], text = columns[1];
        if (labels.Length < LabelMinCount || text.Length < labels.Length)
        {
            return null;
        }

        var labelLines = new SortedSet<int>();
        foreach (int id in labels)
        {
            if (!IsClauseLabel(strips, f[id]) || !labelLines.Add(f[id].Line))
            {
                return null;
            }
        }

        // Each label has text on its own row.
        var textLines = new HashSet<int>();
        double left = double.MaxValue;
        foreach (int id in text)
        {
            textLines.Add(f[id].Line);
            left = Math.Min(left, f[id].X0);
        }

        foreach (int line in labelLines)
        {
            if (!textLines.Contains(line))
            {
                return null;
            }
        }

        // The text column hangs: its lines start at one left edge.
        int aligned = 0;
        foreach (int id in text)
        {
            aligned += f[id].X0 - left <= HangingAlignEm * ms ? 1 : 0;
        }

        if (aligned < HangingAlignShare * text.Length)
        {
            return null;
        }

        // Lines are numbered in the order of their vertical position, so a clause runs from its label's line to the line before the next label.
        int[] starts = [.. labelLines];
        bool lead = reg.Any(id => f[id].Line < starts[0]);
        int offset = lead ? 1 : 0;
        var groups = new List<List<int>>();
        for (int i = 0; i < starts.Length + offset; i++)
        {
            groups.Add([]);
        }

        foreach (int id in reg)
        {
            int k = Array.BinarySearch(starts, f[id].Line);
            groups[(k < 0 ? ~k - 1 : k) + offset].Add(id);
        }

        return [.. groups.Where(g => g.Count > 0).Select(g => g.ToArray())];
    }

    // The words of a fragment as text.
    private static string FragmentText(IReadOnlyList<OrderStrip> strips, OrderFragment frag)
    {
        IReadOnlyList<OcrWord> words = strips[frag.Strip].Words;
        return frag.WordCount == 1 ? words[frag.WordStart].Text : string.Join(' ', Enumerable.Range(frag.WordStart, frag.WordCount).Select(i => words[i].Text));
    }

    // A clause label: "(a)", "(12)", "(iv)", "a)", "1)", "1.", "1.2", "1.2.3.", "A.", "iv.", "§ 4", "§4.1", or a bullet.
    internal static bool IsClauseLabel(IReadOnlyList<OrderStrip> strips, OrderFragment frag) =>
        frag.WordCount <= LabelMaxWords && IsClauseLabel(FragmentText(strips, frag));

    internal static bool IsClauseLabel(string text)
    {
        string t = text.Trim();
        if (t.Length == 0 || t.Length > LabelMaxChars)
        {
            return false;
        }

        if (t is "\u2022" or "\u25CF" or "\u25CB" or "\u25E6" or "\u25AA" or "\u2023" or "\u2043" or "\u00B7" or "-" or "\u2013" or "\u2014" or "*")
        {
            return true;
        }

        if (t[0] == '\u00A7')
        {
            string rest = t[1..].Trim();
            return rest.Length > 0 && IsNumber(rest.TrimEnd('.'));
        }

        if (t[0] == '(' && t[^1] == ')' && t.Length >= 3)
        {
            return IsEnumerator(t[1..^1]);
        }

        if (t[^1] == ')' && t.Length >= 2)
        {
            return IsEnumerator(t[..^1]);
        }

        string body = t[^1] == '.' ? t[..^1] : t;
        if (body.Length == 0)
        {
            return false;
        }

        // "1.2.3": dotted numbers are labels with or without a final dot; a lone number or letter needs the final dot.
        return body.Contains('.', StringComparison.Ordinal) ? IsNumber(body) : t[^1] == '.' && IsEnumerator(body);
    }

    // Digits, a single letter, or a roman numeral.
    private static bool IsEnumerator(string s)
    {
        if (s.Length == 0 || s.Length > 4)
        {
            return false;
        }

        bool digits = true, roman = true;
        foreach (char c in s)
        {
            digits &= char.IsAsciiDigit(c);
            roman &= c is 'i' or 'v' or 'x' or 'l' or 'I' or 'V' or 'X' or 'L';
        }

        return digits || roman || (s.Length == 1 && char.IsAsciiLetter(s[0]));
    }

    // "1", "1.2", "1.2.3": digit groups separated by single points.
    private static bool IsNumber(string s)
    {
        int groupLength = 0;
        foreach (char c in s)
        {
            if (char.IsAsciiDigit(c))
            {
                groupLength++;
            }
            else if (c == '.' && groupLength > 0)
            {
                groupLength = 0;
            }
            else
            {
                return false;
            }
        }

        return groupLength > 0;
    }
}
