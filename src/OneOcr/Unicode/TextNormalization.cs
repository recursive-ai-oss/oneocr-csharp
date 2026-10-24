using System.Text;

namespace OneOcr.Unicode;

/// <summary>
/// Unicode normalization forms C and KC (UAX #15), computed from tables generated from the Unicode Character Database
/// (<see cref="NormalizationTables"/>). The BCL's <c>string.Normalize</c> does nothing for non-ASCII text when the runtime runs
/// with invariant globalization and composes under ICU, so output that depends on it would change with the configuration.
/// This implementation reads no data at run time and gives the same answer everywhere. Specified in
/// <c>docs/spec/text-normalization-v1.md</c>.
/// </summary>
/// <remarks>
/// Code points the tables do not know (unassigned, or newer than the pinned UCD version) have combining class 0, no
/// decomposition and no composition, so they pass through unchanged. An unpaired surrogate is such a code point: it is
/// copied as it is and never throws.
/// </remarks>
internal static class TextNormalization
{
    // Hangul syllable arithmetic, Unicode Standard section 3.12.
    private const int SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7;
    private const int LCount = 19, VCount = 21, TCount = 28, NCount = VCount * TCount, SCount = LCount * NCount;

    /// <summary>Normalization Form C: canonical decomposition, canonical ordering, canonical composition.</summary>
    public static string Nfc(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        // Nothing below U+0300 is a combining mark or decomposes, and no pair of such characters composes.
        return Normalize(text, compat: false, quickLimit: 0x300);
    }

    /// <summary>Normalization Form KC: compatibility decomposition, canonical ordering, canonical composition.</summary>
    public static string Nfkc(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        // U+00A0 is the first character with a compatibility mapping.
        return Normalize(text, compat: true, quickLimit: 0xA0);
    }

    private static string Normalize(string text, bool compat, int quickLimit)
    {
        bool any = false;
        foreach (char c in text)
        {
            if (c >= quickLimit)
            {
                any = true;
                break;
            }
        }

        if (!any)
        {
            return text;
        }

        // Decompose.
        var buf = new List<int>(text.Length + 8);
        for (int i = 0; i < text.Length; i++)
        {
            int cp = text[i];
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                cp = char.ConvertToUtf32(text[i], text[i + 1]);
                i++;
            }

            Decompose(cp, compat, buf);
        }

        // Canonical ordering: stable sort of each run of non-starters by combining class.
        for (int i = 1; i < buf.Count; i++)
        {
            int cc = CombiningClass(buf[i]);
            if (cc == 0)
            {
                continue;
            }

            int j = i;
            while (j > 0)
            {
                int prev = CombiningClass(buf[j - 1]);
                if (prev <= cc)
                {
                    break;
                }

                (buf[j - 1], buf[j]) = (buf[j], buf[j - 1]);
                j--;
            }
        }

        // Compose: each character combines with the last starter unless something of equal or higher class sits between them.
        var outp = new List<int>(buf.Count);
        int starter = -1;
        int lastClass = 0; // class of the last character after the starter; 0 when it is adjacent
        foreach (int cp in buf)
        {
            int cc = CombiningClass(cp);
            if (starter >= 0)
            {
                bool adjacent = outp.Count - 1 == starter;
                if (adjacent || lastClass < cc)
                {
                    int composite = Compose(outp[starter], cp);
                    if (composite >= 0)
                    {
                        outp[starter] = composite;
                        continue;
                    }
                }
            }

            if (cc == 0)
            {
                starter = outp.Count;
                lastClass = 0;
            }
            else
            {
                lastClass = cc;
            }

            outp.Add(cp);
        }

        var sb = new StringBuilder(outp.Count);
        foreach (int cp in outp)
        {
            if (cp >= 0x10000)
            {
                sb.Append(char.ConvertFromUtf32(cp));
            }
            else
            {
                sb.Append((char)cp);
            }
        }

        string result = sb.ToString();
        return string.Equals(result, text, StringComparison.Ordinal) ? text : result;
    }

    internal static int CombiningClass(int cp)
    {
        if (cp < 0x300)
        {
            return 0;
        }

        ReadOnlySpan<int> r = NormalizationTables.CccRanges;
        int lo = 0, hi = (r.Length / 3) - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >> 1;
            int start = r[mid * 3];
            if (cp < start)
            {
                hi = mid - 1;
            }
            else if (cp > r[(mid * 3) + 1])
            {
                lo = mid + 1;
            }
            else
            {
                return r[(mid * 3) + 2];
            }
        }

        return 0;
    }

    private static void Decompose(int cp, bool compat, List<int> output)
    {
        if (cp < 0xC0 && !(compat && cp >= 0xA0))
        {
            output.Add(cp);
            return;
        }

        int s = cp - SBase;
        if ((uint)s < SCount)
        {
            output.Add(LBase + (s / NCount));
            output.Add(VBase + (s % NCount / TCount));
            int t = s % TCount;
            if (t != 0)
            {
                output.Add(TBase + t);
            }

            return;
        }

        if (compat && Lookup(NormalizationTables.CompatKeys, NormalizationTables.CompatOffsets, NormalizationTables.CompatPool, cp, output))
        {
            return;
        }

        if (!Lookup(NormalizationTables.CanonicalKeys, NormalizationTables.CanonicalOffsets, NormalizationTables.CanonicalPool, cp, output))
        {
            output.Add(cp);
        }
    }

    private static bool Lookup(ReadOnlySpan<int> keys, ReadOnlySpan<int> offsets, ReadOnlySpan<int> pool, int cp, List<int> output)
    {
        int i = keys.BinarySearch(cp);
        if (i < 0)
        {
            return false;
        }

        for (int k = offsets[i]; k < offsets[i + 1]; k++)
        {
            output.Add(pool[k]);
        }

        return true;
    }

    /// <summary>The primary composite of the pair, or -1.</summary>
    private static int Compose(int first, int second)
    {
        // Hangul: L + V, then LV + T.
        int l = first - LBase;
        if ((uint)l < LCount)
        {
            int v = second - VBase;
            if ((uint)v < VCount)
            {
                return SBase + (((l * VCount) + v) * TCount);
            }
        }

        int s = first - SBase;
        if ((uint)s < SCount && s % TCount == 0)
        {
            int t = second - TBase;
            if (t > 0 && t < TCount)
            {
                return first + t;
            }
        }

        ReadOnlySpan<int> firsts = NormalizationTables.CompositionFirst;
        ReadOnlySpan<int> seconds = NormalizationTables.CompositionSecond;
        int lo = 0, hi = firsts.Length - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >> 1;
            int cmp = firsts[mid] != first ? firsts[mid].CompareTo(first) : seconds[mid].CompareTo(second);
            if (cmp == 0)
            {
                return NormalizationTables.CompositionResult[mid];
            }

            if (cmp < 0)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }

        return -1;
    }
}
