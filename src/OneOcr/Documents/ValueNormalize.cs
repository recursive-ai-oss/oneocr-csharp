using System.Globalization;
using System.Text;

namespace OneOcr.Documents;

/// <summary>How a number with a single separator and three digits after it (<c>1,234</c>) is read, and which character is the decimal mark.</summary>
public enum NumberStyle
{
    /// <summary>Decided from the page: the style the page's unambiguous numbers use; point when it has none.</summary>
    Auto = 0,

    /// <summary><c>.</c> is the decimal mark, <c>,</c> groups thousands (1,234.50).</summary>
    PointDecimal = 1,

    /// <summary><c>,</c> is the decimal mark, <c>.</c> groups thousands (1.234,50).</summary>
    CommaDecimal = 2,
}

/// <summary>How a numeric date whose first two parts are both 12 or less (<c>03/10/2026</c>) is read.</summary>
public enum DateOrder
{
    /// <summary>Decided from the page: the order the page's unambiguous dates use; when it has none, both readings are accepted and the field's confidence is lowered.</summary>
    Auto = 0,

    /// <summary>Day, month, year (03/10/2026 is 3 October).</summary>
    DayFirst = 1,

    /// <summary>Month, day, year (03/10/2026 is 10 March).</summary>
    MonthFirst = 2,
}

internal readonly record struct ParsedAmount(decimal Value, string[]? Currencies);

internal readonly record struct ParsedDate(int? Year, int Month, int Day, bool Ambiguous);

/// <summary>
/// The normalizations behind grounding, specified in <c>docs/spec/grounding.md</c>: text keys, amounts, dates and currency.
/// Pure functions with invariant culture; nothing depends on the machine's locale.
/// </summary>
internal static class ValueNormalize
{
    // ---- text ----

    /// <summary>Folds a string for comparison: NFKC, typographic quotes and dashes to ASCII, lower case, no whitespace, format or control characters.</summary>
    public static string Fold(string s)
    {
        string n;
        try
        {
            n = s.Normalize(NormalizationForm.FormKC);
        }
        catch (ArgumentException)
        {
            n = s; // lone surrogates: compared as they are
        }

        var sb = new StringBuilder(n.Length);
        foreach (char c in n)
        {
            if (char.IsWhiteSpace(c))
            {
                continue;
            }

            UnicodeCategory cat = char.GetUnicodeCategory(c);
            if (cat is UnicodeCategory.Format or UnicodeCategory.Control)
            {
                continue;
            }

            sb.Append(c switch
            {
                '\u2018' or '\u2019' or '\u201A' or '\u201B' or '\u2032' => '\'',
                '\u201C' or '\u201D' or '\u201E' or '\u201F' or '\u2033' => '"',
                '\u2010' or '\u2011' or '\u2012' or '\u2013' or '\u2014' or '\u2212' => '-',
                _ => char.ToLowerInvariant(c),
            });
        }

        return sb.ToString();
    }

    /// <summary>Removes <c>. , ; :</c> from both ends of an already folded string.</summary>
    public static string TrimPunct(string folded)
    {
        int a = 0, b = folded.Length;
        while (a < b && folded[a] is '.' or ',' or ';' or ':')
        {
            a++;
        }

        while (b > a && folded[b - 1] is '.' or ',' or ';' or ':')
        {
            b--;
        }

        return folded[a..b];
    }

    public static string RemoveWhitespace(string s)
    {
        var sb = new StringBuilder(s.Length);
        foreach (char c in s)
        {
            if (!char.IsWhiteSpace(c) && char.GetUnicodeCategory(c) != UnicodeCategory.Format)
            {
                sb.Append(c);
            }
        }

        return sb.ToString();
    }

    // ---- currency ----

    private static readonly Dictionary<string, string[]> Symbols = new(StringComparer.Ordinal)
    {
        ["€"] = ["EUR"],
        ["$"] = ["USD", "CAD", "AUD", "NZD", "MXN", "SGD", "HKD", "ARS", "CLP", "COP"],
        ["US$"] = ["USD"],
        ["£"] = ["GBP", "EGP"],
        ["¥"] = ["JPY", "CNY"],
        ["₹"] = ["INR"],
        ["₽"] = ["RUB"],
        ["₺"] = ["TRY"],
        ["₩"] = ["KRW"],
        ["₫"] = ["VND"],
        ["₪"] = ["ILS"],
        ["฿"] = ["THB"],
        ["₴"] = ["UAH"],
        ["₱"] = ["PHP"],
        ["R$"] = ["BRL"],
        ["zł"] = ["PLN"],
        ["kč"] = ["CZK"],
        ["kr"] = ["SEK", "NOK", "DKK", "ISK"],
        ["ft"] = ["HUF"],
        ["lei"] = ["RON"],
    };

    private static readonly HashSet<string> Codes = new(StringComparer.Ordinal)
    {
        "USD", "EUR", "GBP", "JPY", "CNY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "RON", "BGN", "TRY", "RUB",
        "UAH", "INR", "KRW", "VND", "THB", "SGD", "HKD", "MXN", "BRL", "ARS", "CLP", "COP", "ZAR", "ILS", "AED", "SAR", "EGP", "PHP", "IDR",
        "MYR", "TWD", "ISK", "RSD",
    };

    public static bool IsCurrencyCode(string s) => s.Length == 3 && Codes.Contains(s);

    /// <summary>The ISO codes a word stands for when it is a currency symbol or an ISO code; null when it is neither.</summary>
    public static string[]? CurrencyOf(string word)
    {
        string t = word.Trim().TrimEnd('.', ',', ':');
        if (Codes.Contains(t))
        {
            return [t];
        }

        return Symbols.TryGetValue(t.ToLowerInvariant(), out string[]? c) || Symbols.TryGetValue(t, out c) ? c : null;
    }

    // ---- amounts ----

    /// <summary>Reads <c>€ 1.234,50</c>, <c>(12.50)</c>, <c>12.50-</c>, <c>USD 3</c> and similar; false when the text is not entirely an amount.</summary>
    public static bool TryParseAmount(string text, NumberStyle style, out ParsedAmount amount)
    {
        amount = default;
        string s = text.Trim();
        if (s.Length == 0 || s.Length > 48)
        {
            return false;
        }

        bool negative = false;
        if (s.StartsWith('(') && s.EndsWith(')'))
        {
            negative = true;
            s = s[1..^1].Trim();
        }

        string[]? currencies = null;
        // Strip a currency symbol or code from either end (possibly twice: "-€" is handled by sign below).
        for (int pass = 0; pass < 2; pass++)
        {
            s = s.Trim();
            foreach (string sym in SymbolsByLength)
            {
                if (s.StartsWith(sym, StringComparison.OrdinalIgnoreCase) && IsBoundary(s, sym.Length, true))
                {
                    currencies ??= Symbols[sym];
                    s = s[sym.Length..];
                    break;
                }

                if (s.EndsWith(sym, StringComparison.OrdinalIgnoreCase) && IsBoundary(s, s.Length - sym.Length, false))
                {
                    currencies ??= Symbols[sym];
                    s = s[..^sym.Length];
                    break;
                }
            }

            if (s.Length >= 3)
            {
                string head = s[..3].ToUpperInvariant();
                string tail = s[^3..].ToUpperInvariant();
                if (Codes.Contains(head) && char.IsLetter(s[0]) && (s.Length == 3 || !char.IsLetter(s[3])))
                {
                    currencies ??= [head];
                    s = s[3..];
                }
                else if (Codes.Contains(tail) && char.IsLetter(s[^1]) && (s.Length == 3 || !char.IsLetter(s[^4])))
                {
                    currencies ??= [tail];
                    s = s[..^3];
                }
            }
        }

        s = s.Trim();
        if (s.StartsWith('-') || s.StartsWith('\u2212') || s.StartsWith('+'))
        {
            negative ^= s[0] != '+';
            s = s[1..].Trim();
        }

        if (s.EndsWith('-') || s.EndsWith('\u2212'))
        {
            negative = true;
            s = s[..^1].Trim();
        }

        if (s.Length == 0 || !char.IsAsciiDigit(s[0]) || !char.IsAsciiDigit(s[^1]) || s.Length > 32)
        {
            return false;
        }

        foreach (char c in s)
        {
            if (!(char.IsAsciiDigit(c) || c is '.' or ',' or ' ' or '\'' or '\u2019' or '\u00A0' or '\u202F' or '\u2009'))
            {
                return false;
            }
        }

        s = s.Replace(" ", "", StringComparison.Ordinal).Replace("'", "", StringComparison.Ordinal).Replace("\u2019", "", StringComparison.Ordinal)
            .Replace("\u00A0", "", StringComparison.Ordinal).Replace("\u202F", "", StringComparison.Ordinal).Replace("\u2009", "", StringComparison.Ordinal);

        char? decimalMark = null;
        int lastDot = s.LastIndexOf('.'), lastComma = s.LastIndexOf(',');
        if (lastDot >= 0 && lastComma >= 0)
        {
            decimalMark = lastDot > lastComma ? '.' : ',';
        }
        else if (lastDot >= 0 || lastComma >= 0)
        {
            char sep = lastDot >= 0 ? '.' : ',';
            int last = Math.Max(lastDot, lastComma);
            int after = s.Length - last - 1;
            if (s.Count(ch => ch == sep) == 1)
            {
                bool groupLike = after == 3 && last is >= 1 and <= 3 && s[0] != '0';
                decimalMark = !groupLike || (style == NumberStyle.CommaDecimal ? sep == ',' : sep == '.') ? sep : null;
            }
        }

        string intPart = s, frac = "";
        if (decimalMark is { } dm)
        {
            int idx = s.LastIndexOf(dm);
            intPart = s[..idx];
            frac = s[(idx + 1)..];
            if (frac.Length == 0 || !frac.All(char.IsAsciiDigit))
            {
                return false;
            }
        }

        string[] groups = intPart.Split(['.', ',']);
        for (int g = 0; g < groups.Length; g++)
        {
            bool ok = groups.Length == 1 || (g == 0 ? groups[g].Length is >= 1 and <= 3 : groups[g].Length == 3);
            if (!ok || groups[g].Length == 0 || !groups[g].All(char.IsAsciiDigit))
            {
                return false;
            }
        }

        string digits = string.Concat(groups) + (frac.Length > 0 ? "." + frac : "");
        if (!decimal.TryParse(digits, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out decimal v))
        {
            return false;
        }

        amount = new ParsedAmount(negative ? -v : v, currencies);
        return true;
    }

    private static readonly string[] SymbolsByLength = [.. Symbols.Keys.OrderByDescending(static k => k.Length).ThenBy(static k => k, StringComparer.Ordinal)];

    // A letter-based symbol ("kr", "ft", "lei") counts only when not glued to other letters.
    private static bool IsBoundary(string s, int index, bool start)
    {
        if (start)
        {
            return index >= s.Length || !char.IsLetter(s[index]) || !char.IsLetter(s[index - 1]);
        }

        return index == 0 || !char.IsLetter(s[index - 1]) || !char.IsLetter(s[index]);
    }

    /// <summary>The canonical text of an amount: no grouping, <c>.</c> as the decimal mark, no trailing zeros beyond two decimals; 12.5 is <c>12.50</c>, 12 is <c>12</c>.</summary>
    public static string CanonicalAmount(decimal v)
    {
        string s = v.ToString(CultureInfo.InvariantCulture);
        int dot = s.IndexOf('.');
        if (dot < 0)
        {
            return s;
        }

        s = s.TrimEnd('0');
        if (s.EndsWith('.'))
        {
            return s[..^1];
        }

        return s.Length - s.IndexOf('.') - 1 < 2 ? s + "0" : s;
    }

    /// <summary>The style used by the page's unambiguous numbers: majority of "both marks present", "one mark followed by one, two or four or more digits".</summary>
    public static NumberStyle DetectStyle(IEnumerable<string> words)
    {
        int point = 0, comma = 0;
        foreach (string w in words)
        {
            string t = w.Trim().TrimEnd('.', ',', ':', ';');
            if (t.Length < 3 || !char.IsAsciiDigit(t[^1]))
            {
                continue;
            }

            int a = t.LastIndexOf('.'), b = t.LastIndexOf(',');
            if (a < 0 && b < 0)
            {
                continue;
            }

            bool numeric = true;
            foreach (char c in t)
            {
                if (!(char.IsAsciiDigit(c) || c is '.' or ',' or '-' or '$' or '€' or '£'))
                {
                    numeric = false;
                    break;
                }
            }

            if (!numeric)
            {
                continue;
            }

            if (a >= 0 && b >= 0)
            {
                if (a > b)
                {
                    point++;
                }
                else
                {
                    comma++;
                }

                continue;
            }

            char sep = a >= 0 ? '.' : ',';
            int count = t.Count(ch => ch == sep);
            int after = t.Length - Math.Max(a, b) - 1;
            if (count > 1)
            {
                if (sep == '.')
                {
                    comma++;
                }
                else
                {
                    point++;
                }
            }
            else if (after is 1 or 2 || after >= 4)
            {
                if (sep == '.')
                {
                    point++;
                }
                else
                {
                    comma++;
                }
            }
        }

        return comma > point ? NumberStyle.CommaDecimal : NumberStyle.PointDecimal;
    }

    // ---- dates ----

    private static readonly string[][] Months =
    [
        ["jan", "january", "januar", "janvier", "enero", "gennaio", "janeiro", "januari", "ocak", "janv"],
        ["feb", "february", "februar", "février", "fevrier", "febrero", "febbraio", "fevereiro", "februari", "şubat", "subat", "févr", "fevr"],
        ["mar", "march", "märz", "marz", "mars", "marzo", "março", "marco", "maart", "mart", "mär"],
        ["apr", "april", "avril", "abril", "aprile", "nisan", "avr"],
        ["may", "mai", "mayo", "maggio", "maio", "mei", "mayıs", "mayis"],
        ["jun", "june", "juni", "juin", "junio", "giugno", "junho", "haziran"],
        ["jul", "july", "juli", "juillet", "julio", "luglio", "julho", "temmuz", "juil"],
        ["aug", "august", "août", "aout", "agosto", "augustus", "ağustos", "agustos"],
        ["sep", "sept", "september", "septembre", "septiembre", "settembre", "setembro", "eylül", "eylul"],
        ["oct", "october", "oktober", "octobre", "octubre", "ottobre", "outubro", "okt", "ekim", "okt"],
        ["nov", "november", "novembre", "noviembre", "novembro", "kasım", "kasim"],
        ["dec", "december", "dezember", "décembre", "decembre", "diciembre", "dicembre", "dezembro", "aralık", "aralik", "déc", "dez", "dic"],
    ];

    private static readonly Dictionary<string, int> MonthByName = BuildMonths();

    private static Dictionary<string, int> BuildMonths()
    {
        var d = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int m = 0; m < 12; m++)
        {
            foreach (string name in Months[m])
            {
                d[name] = m + 1;
            }
        }

        return d;
    }

    public static bool IsMonthName(string word) => MonthByName.ContainsKey(word.Trim().TrimEnd('.', ',').ToLowerInvariant());

    private static readonly HashSet<string> DateFiller = new(StringComparer.Ordinal) { "of", "de", "del", "der", "the", "le", "den", "am", "on" };

    /// <summary>
    /// Reads a date: ISO and year-first numeric, day/month/year numeric with <c>. / - space</c>, month names in English, German,
    /// French, Spanish, Italian, Portuguese, Dutch and Turkish, ordinal suffixes, two-digit years (00–69 are 2000–2069), and
    /// day-and-month without a year. Returns every reading (two when day and month cannot be told apart and <paramref name="order"/> is Auto).
    /// </summary>
    public static List<ParsedDate> ParseDate(string text, DateOrder order)
    {
        var result = new List<ParsedDate>();
        string s = text.Trim().ToLowerInvariant();
        if (s.Length is < 3 or > 40)
        {
            return result;
        }

        // Tokens: digit runs and letter runs; anything else only separates.
        var tokens = new List<(bool Num, string Text)>();
        int i = 0;
        while (i < s.Length)
        {
            char c = s[i];
            if (char.IsAsciiDigit(c))
            {
                int j = i;
                while (j < s.Length && char.IsAsciiDigit(s[j]))
                {
                    j++;
                }

                tokens.Add((true, s[i..j]));
                i = j;
                // ordinal suffix: 1st 2nd 3rd 4th 1er
                foreach (string suf in new[] { "st", "nd", "rd", "th", "er", "º", "." })
                {
                    if (suf != "." && string.CompareOrdinal(s, i, suf, 0, suf.Length) == 0 && (i + suf.Length >= s.Length || !char.IsLetter(s[i + suf.Length])))
                    {
                        i += suf.Length;
                        break;
                    }
                }
            }
            else if (char.IsLetter(c))
            {
                int j = i;
                while (j < s.Length && char.IsLetter(s[j]))
                {
                    j++;
                }

                tokens.Add((false, s[i..j]));
                i = j;
            }
            else
            {
                if (c is not (' ' or '.' or ',' or '/' or '-' or '\u2010' or '\u2011' or '\u2012' or '\u2013' or '\u00A0'))
                {
                    return result;
                }

                i++;
            }
        }

        tokens.RemoveAll(static t => !t.Num && DateFiller.Contains(t.Text));
        if (tokens.Count is < 1 or > 3)
        {
            return result;
        }

        int monthWords = tokens.Count(static t => !t.Num);
        if (monthWords > 1 || (monthWords == 1 && !MonthByName.ContainsKey(tokens.First(static t => !t.Num).Text)))
        {
            return result;
        }

        void Add(int? y, int m, int d, bool amb)
        {
            if (m is < 1 or > 12 || d < 1)
            {
                return;
            }

            int max = DateTime.DaysInMonth(y ?? 2000, m);
            if (d <= max)
            {
                result.Add(new ParsedDate(y, m, d, amb));
            }
        }

        static int? Year(string t) => t.Length switch
        {
            4 => int.Parse(t, CultureInfo.InvariantCulture),
            2 => int.Parse(t, CultureInfo.InvariantCulture) is var v && v <= 69 ? 2000 + v : 1900 + v,
            _ => null,
        };

        static bool Small(string t) => t.Length is 1 or 2;

        if (monthWords == 0)
        {
            if (tokens.Count == 1 && tokens[0].Text.Length == 8)
            {
                string t = tokens[0].Text;
                Add(int.Parse(t[..4], CultureInfo.InvariantCulture), int.Parse(t[4..6], CultureInfo.InvariantCulture), int.Parse(t[6..], CultureInfo.InvariantCulture), false);
                return result;
            }

            if (tokens.Count != 3)
            {
                return result;
            }

            string a = tokens[0].Text, b = tokens[1].Text, c3 = tokens[2].Text;
            if (a.Length == 4 && Small(b) && Small(c3))
            {
                Add(int.Parse(a, CultureInfo.InvariantCulture), int.Parse(b, CultureInfo.InvariantCulture), int.Parse(c3, CultureInfo.InvariantCulture), false);
            }
            else if (Small(a) && Small(b) && (c3.Length is 2 or 4))
            {
                int x = int.Parse(a, CultureInfo.InvariantCulture), y = int.Parse(b, CultureInfo.InvariantCulture);
                int? year = Year(c3);
                if (x > 12)
                {
                    Add(year, y, x, false);
                }
                else if (y > 12)
                {
                    Add(year, x, y, false);
                }
                else if (x == y)
                {
                    Add(year, x, y, false);
                }
                else if (order == DateOrder.DayFirst)
                {
                    Add(year, y, x, false);
                }
                else if (order == DateOrder.MonthFirst)
                {
                    Add(year, x, y, false);
                }
                else
                {
                    Add(year, y, x, true);
                    Add(year, x, y, true);
                }
            }

            return result;
        }

        int mi = tokens.FindIndex(static t => !t.Num);
        int month = MonthByName[tokens[mi].Text];
        var nums = tokens.Where(static t => t.Num).Select(static t => t.Text).ToList();
        if (nums.Count == 1)
        {
            if (Small(nums[0]))
            {
                Add(null, month, int.Parse(nums[0], CultureInfo.InvariantCulture), false);
            }

            return result;
        }

        if (nums.Count == 2)
        {
            string first = nums[0], second = nums[1];
            if (first.Length == 4 && Small(second))
            {
                Add(int.Parse(first, CultureInfo.InvariantCulture), month, int.Parse(second, CultureInfo.InvariantCulture), false);
            }
            else if (Small(first) && (second.Length is 2 or 4))
            {
                Add(Year(second), month, int.Parse(first, CultureInfo.InvariantCulture), false);
            }
        }

        return result;
    }

    /// <summary>The order used by the page's unambiguous numeric dates; Auto when there is no evidence either way.</summary>
    public static DateOrder DetectOrder(IEnumerable<string> words)
    {
        int dayFirst = 0, monthFirst = 0;
        foreach (string w in words)
        {
            string t = w.Trim().TrimEnd('.', ',', ';', ':');
            if (t.Length is < 6 or > 10 || !char.IsAsciiDigit(t[0]) || !char.IsAsciiDigit(t[^1]))
            {
                continue;
            }

            string[] parts = t.Split(['.', '/', '-']);
            if (parts.Length != 3 || parts[0].Length > 2 || parts[1].Length > 2 || !int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out int x)
                || !int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out int y))
            {
                continue;
            }

            if (x > 12 && y <= 12)
            {
                dayFirst++;
            }
            else if (y > 12 && x <= 12)
            {
                monthFirst++;
            }
        }

        return dayFirst > monthFirst ? DateOrder.DayFirst : monthFirst > dayFirst ? DateOrder.MonthFirst : DateOrder.Auto;
    }

    public static string CanonicalDate(ParsedDate d) =>
        d.Year is { } y
            ? string.Create(CultureInfo.InvariantCulture, $"{y:0000}-{d.Month:00}-{d.Day:00}")
            : string.Create(CultureInfo.InvariantCulture, $"--{d.Month:00}-{d.Day:00}");

    // ---- edit distance ----

    /// <summary>Levenshtein distance, or <paramref name="bound"/> + 1 as soon as it is known to exceed the bound.</summary>
    public static int BoundedDistance(string a, string b, int bound)
    {
        if (Math.Abs(a.Length - b.Length) > bound)
        {
            return bound + 1;
        }

        int[] prev = new int[b.Length + 1], cur = new int[b.Length + 1];
        for (int j = 0; j <= b.Length; j++)
        {
            prev[j] = j;
        }

        for (int i = 1; i <= a.Length; i++)
        {
            cur[0] = i;
            int rowMin = cur[0];
            for (int j = 1; j <= b.Length; j++)
            {
                int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                cur[j] = Math.Min(Math.Min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + cost);
                rowMin = Math.Min(rowMin, cur[j]);
            }

            if (rowMin > bound)
            {
                return bound + 1;
            }

            (prev, cur) = (cur, prev);
        }

        return prev[b.Length];
    }
}
