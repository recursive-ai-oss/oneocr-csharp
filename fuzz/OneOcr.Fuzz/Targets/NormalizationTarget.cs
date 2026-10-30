using OneOcr.Documents;
using OneOcr.Training;
using OneOcr.Unicode;

namespace OneOcr.Fuzz;

/// <summary>
/// Arbitrary text through the Unicode normalization and the value normalizers that grounding and document extraction apply to page
/// and model text: NFC, NFKC, comparison folding, amounts, dates, currency and the bounded edit distance. The input is read as UTF-16
/// code units, so lone surrogates and every other ill-formed sequence reach the code. Nothing may throw, and the normalization forms
/// must keep their algebra: idempotent, and NFKC(NFC(x)) = NFKC(x).
/// </summary>
internal static class NormalizationTarget
{
    private const int MaxChars = 2048;

    public static void Run(ReadOnlySpan<byte> data)
    {
        int n = Math.Min(data.Length / 2, MaxChars);
        var chars = new char[n];
        for (int i = 0; i < n; i++)
        {
            chars[i] = (char)(data[2 * i] | (data[(2 * i) + 1] << 8));
        }

        string text = new(chars);

        string nfc = TextNormalization.Nfc(text);
        string nfkc = TextNormalization.Nfkc(text);
        Check(TextNormalization.Nfc(nfc) == nfc, "NFC is not idempotent");
        Check(TextNormalization.Nfkc(nfkc) == nfkc, "NFKC is not idempotent");
        Check(TextNormalization.Nfkc(nfc) == nfkc, "NFKC(NFC(x)) differs from NFKC(x)");
        Check(TextNormalization.Nfc(nfkc) == nfkc, "NFC(NFKC(x)) differs from NFKC(x)");

        _ = ValueNormalize.Fold(text);
        _ = ValueNormalize.TrimPunct(text);
        _ = ValueNormalize.RemoveWhitespace(text);
        _ = ValueNormalize.CurrencyOf(text);
        _ = ValueNormalize.IsCurrencyCode(text);
        _ = ValueNormalize.IsMonthName(text);
        _ = TextMetrics.Normalize(text);

        foreach (NumberStyle style in new[] { NumberStyle.Auto, NumberStyle.PointDecimal, NumberStyle.CommaDecimal })
        {
            if (ValueNormalize.TryParseAmount(text, style, out ParsedAmount amount))
            {
                _ = ValueNormalize.CanonicalAmount(amount.Value);
            }
        }

        foreach (DateOrder order in new[] { DateOrder.Auto, DateOrder.DayFirst, DateOrder.MonthFirst })
        {
            foreach (ParsedDate d in ValueNormalize.ParseDate(text, order))
            {
                _ = ValueNormalize.CanonicalDate(d);
            }
        }

        string[] words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        _ = ValueNormalize.DetectStyle(words);
        _ = ValueNormalize.DetectOrder(words);

        string a = text[..(text.Length / 2)];
        string b = text[(text.Length / 2)..];
        int bound = data.Length > 0 ? data[0] % 40 : 0;
        int dist = ValueNormalize.BoundedDistance(a, b, bound);
        Check(dist >= 0 && dist <= bound + 1, "bounded distance outside [0, bound + 1]");
        Check(ValueNormalize.BoundedDistance(a, a, bound) == 0, "distance to itself is not 0");
    }

    private static void Check(bool ok, string what)
    {
        if (!ok)
        {
            throw new InvalidOperationException(what);
        }
    }
}
