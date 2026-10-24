using System.Text;
using OneOcr.Documents;
using OneOcr.Unicode;

namespace OneOcr.Tests;

public class TextNormalizationTests
{
    private static string S(params int[] cps) => string.Concat(cps.Select(char.ConvertFromUtf32));

    public static TheoryData<string, string> NfcCases => new()
    {
        { "a\u0304", "\u0101" },                          // decomposed Latin
        { "e\u0323\u0302", "\u1EC7" },                    // Vietnamese: dot below then circumflex
        { "e\u0302\u0323", "\u1EC7" },                    // same marks in the other order (canonical ordering)
        { "o\u031B\u0301", "\u1EDB" },                    // horn + acute
        { "A\u0306\u0301", "\u1EAE" },                    // breve + acute
        { "\u03B1\u0313\u0301\u0345", "\u1F84" },         // polytonic Greek: alpha, psili, oxia, ypogegrammeni
        { "\u03B7\u0342\u0345", "\u1FC7" }, // eta, perispomeni, ypogegrammeni
        { "\u0438\u0306", "\u0439" },                     // Cyrillic i + breve
        { "\u0418\u0306", "\u0419" },
        { "\u0435\u0308", "\u0451" },
        { "\u1100\u1161\u11A8", "\uAC01" },               // Hangul jamo
        { "\u212B", "\u00C5" },                           // singleton: angstrom sign
        { "\u2126", "\u03A9" },                           // ohm sign
        { "\u00C5", "\u00C5" },                           // already NFC
        { "Caf\u00E9 \u0391\u03B2 \u0416", "Caf\u00E9 \u0391\u03B2 \u0416" },
        { "", "" },
        { "plain ASCII 123", "plain ASCII 123" },
        { "\uFB01 \uFF11 \u00A0\u00B2", "\uFB01 \uFF11 \u00A0\u00B2" }, // compatibility characters stay in NFC
        { "\u0304", "\u0304" },                           // lone mark
        { "a\uD800b", "a\uD800b" },                       // unpaired surrogates pass through
        { "\uDC00", "\uDC00" },
        { "\U0001D157\U0001D165", "\U0001D157\U0001D165" }, // composition-excluded pair stays decomposed
    };

    [Theory]
    [MemberData(nameof(NfcCases))]
    public void Nfc_matches_the_standard(string input, string expected)
    {
        Assert.Equal(expected, TextNormalization.Nfc(input));
        Assert.Equal(expected, TextNormalization.Nfc(expected));
    }

    public static TheoryData<string, string> NfkcCases => new()
    {
        { "a\u0304", "\u0101" },
        { "\uFB01nd", "find" },                           // ligature
        { "\uFB00 \uFB02", "ff fl" },
        { "\uFF11\uFF12\uFF13", "123" },                  // full-width digits
        { "\uFF21\uFF42", "Ab" },
        { "x\u00B2 \u00B9\u00B3", "x2 13" },              // superscripts
        { "a\u00A0b", "a b" },                            // no-break space
        { "\u2003", " " },
        { "\u00BD", "1\u20442" },                         // vulgar fraction
        { "\u2160", "I" },                                // roman numeral
        { "\u00B5", "\u03BC" },                           // micro sign to Greek mu
        { "\u2126", "\u03A9" },
        { "\u1E9B\u0323", "\u1E69" },                     // long s with dot above + dot below
        { "\u00C5\u212B", "\u00C5\u00C5" },
        { "plain", "plain" },
        { "\uD83D", "\uD83D" },
    };

    [Theory]
    [MemberData(nameof(NfkcCases))]
    public void Nfkc_matches_the_standard(string input, string expected)
    {
        Assert.Equal(expected, TextNormalization.Nfkc(input));
        Assert.Equal(expected, TextNormalization.Nfkc(expected));
    }

    [Fact]
    public void Composed_text_is_returned_as_the_same_instance()
    {
        string s = "xxx" + "\u00E9\u0101\u1EC7\u03CE\u0439";
        Assert.Same(s, TextNormalization.Nfc(s));
        Assert.Same(s, TextNormalization.Nfkc(s));
    }

    [Fact]
    public void Result_does_not_depend_on_the_globalization_mode()
    {
        // The BCL leaves non-ASCII text alone under invariant globalization; ours must compose either way.
        Assert.Equal("\u0101", TextNormalization.Nfc("a\u0304"));
        Assert.Equal("fi", TextNormalization.Nfkc("\uFB01"));
    }

    /// <summary>Sweep candidates: every scalar value, plus a base and one or two marks over Latin, Greek and Cyrillic.</summary>
    private static IEnumerable<string> Sweep()
    {
        for (int cp = 0; cp <= 0x10FFFF; cp++)
        {
            if (cp is >= 0xD800 and <= 0xDFFF)
            {
                continue;
            }

            yield return char.ConvertFromUtf32(cp);
        }

        var bases = new List<int>();
        bases.AddRange(Enumerable.Range(0x20, 0x250 - 0x20));
        bases.AddRange(Enumerable.Range(0x370, 0x400 - 0x370));
        bases.AddRange(Enumerable.Range(0x400, 0x530 - 0x400));
        bases.AddRange(Enumerable.Range(0x1E00, 0x2000 - 0x1E00));
        var marks = new List<int>();
        marks.AddRange(Enumerable.Range(0x300, 0x70));
        marks.AddRange(Enumerable.Range(0x483, 5));
        marks.AddRange([0x3099, 0x309A, 0x1AB0, 0x1DC4, 0x20D0]);

        foreach (int b in bases)
        {
            foreach (int m in marks)
            {
                yield return S(b, m);
            }
        }

        // Stacked diacritics (Vietnamese and polytonic Greek): bases x the marks those scripts stack.
        int[] stack = [0x300, 0x301, 0x303, 0x309, 0x323, 0x302, 0x306, 0x31B, 0x308, 0x313, 0x314, 0x342, 0x345, 0x304, 0x30C];
        foreach (int b in bases)
        {
            foreach (int m1 in stack)
            {
                foreach (int m2 in stack)
                {
                    yield return S(b, m1, m2);
                }
            }
        }
    }

    [Fact]
    public void Matches_the_BCL_under_ICU_on_the_sweep()
    {
        // Only meaningful when the runtime really normalizes (ICU). Under invariant globalization the BCL is a no-op.
        if ("a\u0304".Normalize(NormalizationForm.FormC) != "\u0101")
        {
            Assert.Skip("The runtime is in invariant globalization mode; run with DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=0.");
        }

        int checkedCount = 0;
        var diffs = new List<string>();
        foreach (string s in Sweep())
        {
            string c = TextNormalization.Nfc(s);
            string k = TextNormalization.Nfkc(s);
            string bclC, bclK;
            try
            {
                bclC = s.Normalize(NormalizationForm.FormC);
                bclK = s.Normalize(NormalizationForm.FormKC);
            }
            catch (ArgumentException)
            {
                // The BCL rejects noncharacters; ours passes them through.
                Assert.Equal(s, c);
                Assert.Equal(s, k);
                continue;
            }

            if (c != bclC)
            {
                diffs.Add("NFC " + string.Join(" ", s.EnumerateRunes().Select(r => r.Value.ToString("X4"))));
            }

            if (k != bclK)
            {
                diffs.Add("NFKC " + string.Join(" ", s.EnumerateRunes().Select(r => r.Value.ToString("X4"))));
            }

            checkedCount++;
        }

        Assert.True(checkedCount > 1_000_000);
        Assert.True(diffs.Count == 0, $"{diffs.Count} differences, first: {string.Join("; ", diffs.Take(8))}");
    }

    [Fact]
    public void Output_is_normalized_and_idempotent_on_the_sweep()
    {
        foreach (string s in Sweep().Where(static (_, i) => i % 7 == 0))
        {
            string c = TextNormalization.Nfc(s);
            Assert.Equal(c, TextNormalization.Nfc(c));
            string k = TextNormalization.Nfkc(s);
            Assert.Equal(k, TextNormalization.Nfkc(k));
        }
    }

    [Fact]
    public void BoundedDistanceNeverExceedsBoundPlusOne()
    {
        // Found by the normalization fuzz target: the corner cell can be far above the bound while the last row still has a cell inside it.
        Assert.Equal(3, ValueNormalize.BoundedDistance("aa", "bbbb", 2));
        Assert.Equal(2, ValueNormalize.BoundedDistance("ab", "ba", 5));
        Assert.Equal(0, ValueNormalize.BoundedDistance("same", "same", 0));
    }
}
