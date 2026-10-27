using System.Text;
using OneOcr.Decoding;

namespace OneOcr.Recognition;

/// <summary>
/// Decides, for each pair of neighbouring characters of a recognized line, whether a space belongs between them
/// (<c>docs/spec/oneocr-page-v1.md</c>, "Word segmentation"). The network emits a space class where it believes there is a gap,
/// and it is sometimes wrong in both directions: it cuts words ("immedi ately", "$40, 181.65") where the ink is continuous and
/// fuses words ("Iagree") across an empty stretch of the line. The line image settles part of it: the empty stretches between
/// ink in the scaled line show where a gap really is and how wide the line's letter gaps and word gaps are. A cut inside a
/// number is taken back when the pieces are the halves of a number or date and the stretch is narrow; a cut inside a word is
/// taken back when the character model finds the joined word much more plausible than its two pieces and the stretch is
/// narrow; a word is cut where an empty stretch is far wider than the line's letter gaps. Only spaces are added or removed:
/// no character is ever read again, so numbers, amounts and addresses keep exactly the characters the network gave them.
/// </summary>
internal static class WordSegmenter
{
    /// <summary>A stretch of at least this many pixels of the scaled line image that no network space accounts for may become a space.</summary>
    internal const double SplitGap = 9.0;

    /// <summary>... and it has to be at least this many times the line's median stretch between letters (letter-spaced print has wide letter gaps).</summary>
    internal const double SplitRatio = 3.5;

    /// <summary>A network space inside a number or date goes when the empty stretch there is at most this many pixels and the pieces are halves of a number or date.</summary>
    internal const double NumberGap = 6.0;

    /// <summary>A network space between two letters is a candidate for removal by the character model when the stretch there is at most this many pixels.</summary>
    internal const double ModelGap = 5.0;

    /// <summary>The character model removes a space when the joined word scores at least this many nats above the two pieces together.</summary>
    internal const double ModelMargin = 2.0;

    /// <summary>A weaker gain suffices, at a stretch of up to <see cref="ModelGapWeak"/> pixels, when one of the two pieces is itself no word (see <see cref="WeakPieceMax"/>).</summary>
    internal const double ModelMarginWeak = 1.0;

    /// <summary>The widest stretch at which the weaker gain is enough.</summary>
    internal const double ModelGapWeak = 6.0;

    /// <summary>A piece counts as no word when its own score is at most this (a typical word scores about zero or more; a rare real word such as "tax" scores about -1.7).</summary>
    internal const double WeakPieceMax = -2.0;

    /// <summary>The widest stretch, in pixels, at which the word frequency table may take a space back.</summary>
    internal const int FrequencyGap = 9;

    /// <summary>... and, on a line with a usable word gap, the stretch is at most this fraction of it.</summary>
    internal const double FrequencyRatio = 0.85;

    /// <summary>The joined word has to be at least this frequent in the table (quarter-bits of its count).</summary>
    internal const int FrequencyMinQuarter = 10;

    /// <summary>How many quarter-bits more common the joined word is than its pieces together (28 is 7 bits, about 4.9 nats of log-odds).</summary>
    internal const int FrequencyMargin = 28;

    /// <summary>Added to the margin when one of the pieces is a single letter.</summary>
    internal const int FrequencySingleExtra = 24;

    /// <summary>A piece in the table with at least this value (quarter-bits of its count) is a common word.</summary>
    internal const int FrequencyWordQuarter = 24;

    /// <summary>Added to the margin when both pieces are common words.</summary>
    internal const int FrequencyBothExtra = 16;

    /// <summary>The widest stretch, in pixels, at which a space inside an address, web address or amount may be taken back (see <see cref="JoinsCode"/>).</summary>
    internal const int CodeGap = 6;

    /// <summary>... and, when the line has at least <see cref="MinSpacesForGap"/> spaces, the stretch has to be at most this fraction of the line's median word gap.</summary>
    internal const double ModelRatio = 0.7;

    /// <summary>A line needs this many network spaces before its median word gap is trusted.</summary>
    internal const int MinSpacesForGap = 4;

    /// <summary>Lines whose strongest ink is fainter than this (distance from the background, 0..255) have no usable column profile and are left as read.</summary>
    internal const int MinContrast = 64;

    // Characters that mark a word as a code, as in the decoder.
    private const string CodeSymbols = "@#_/\\=%+*~^|<>[]{}";

    private const string Closing = ".,;:!?)]}%\u2019\u201D'\"\u00BB\u2026";
    private const string Opening = "$\u00A3\u20AC([{\u2018\u201C\u00AB";

    /// <summary>
    /// Returns the word breaks to use: entry <c>p</c> is true when a space belongs between character <c>p</c> and <c>p + 1</c> of
    /// the line's characters in reading order. Null when the pixels give no evidence or nothing needs to change.
    /// </summary>
    /// <param name="chars">The characters of the line in order, with their extents in pixels of the scaled line image.</param>
    /// <param name="wasSpace">Entry <c>p</c>: the network put a space between character <c>p</c> and <c>p + 1</c>.</param>
    /// <param name="gray">The scaled line image, row-major.</param>
    /// <param name="width">Its width in pixels.</param>
    /// <param name="height">Its height in pixels.</param>
    /// <param name="words">The character model that judges whether letters form one word, or null.</param>
    /// <param name="frequencies">The word frequencies that judge whether a cut word was one, or null.</param>
    public static bool[]? Plan(IReadOnlyList<RecognizedChar> chars, bool[] wasSpace, ReadOnlySpan<byte> gray, int width, int height, WordPlausibility? words, WordFrequencyModel? frequencies = null)
    {
        int n = chars.Count;
        if (n < 2 || width < 1 || gray.Length < width * height)
        {
            return null;
        }

        bool[]? ink = InkColumns(gray, width, height);
        if (ink is null)
        {
            return null;
        }

        // Empty stretches between ink; the margins of the line are not gaps.
        var stretches = new List<(int From, int To)>();
        for (int x = 0; x < ink.Length; x++)
        {
            if (ink[x])
            {
                continue;
            }

            int e = x;
            while (e + 1 < ink.Length && !ink[e + 1])
            {
                e++;
            }

            if (x > 0 && e < ink.Length - 1)
            {
                stretches.Add((x, e));
            }

            x = e;
        }

        // A character's run of steps is not always inside its glyph, so a stretch belongs to the pairs whose span between the two
        // character centres it touches. widest[p] is the widest stretch touching pair p.
        var widest = new int[n - 1];
        var cuts = new List<(int Pair, int Width)>();
        foreach ((int from, int to) in stretches)
        {
            int span = to - from + 1;
            int owner = -1;
            double ownerOverlap = -1;
            bool claimed = false;
            for (int p = 0; p < n - 1; p++)
            {
                double centerA = (chars[p].X0 + chars[p].X1) / 2, centerB = (chars[p + 1].X0 + chars[p + 1].X1) / 2;
                double overlap = Math.Min(centerB, to + 1) - Math.Max(centerA, from);
                if (overlap < 0)
                {
                    continue;
                }

                widest[p] = Math.Max(widest[p], span);
                if (wasSpace[p])
                {
                    claimed = true;
                }
                else if (overlap > ownerOverlap)
                {
                    ownerOverlap = overlap;
                    owner = p;
                }
            }

            // A stretch that a network space accounts for is not evidence of a missing one; the network's space class sits in the gap.
            if (!claimed && owner >= 0)
            {
                cuts.Add((owner, span));
            }
        }

        var result = (bool[])wasSpace.Clone();
        double letterGap = MedianLetterGap(widest, wasSpace);
        double needed = Math.Max(SplitGap, SplitRatio * letterGap);
        foreach ((int pair, int span) in cuts)
        {
            if (span >= needed && MaySplit(chars, wasSpace, pair))
            {
                result[pair] = true;
            }
        }

        // The line's usual word gap, when it has enough spaces to tell.
        double wordGap = MedianWordGap(widest, wasSpace);
        bool changed = false;
        for (int p = 0; p < n - 1; p++)
        {
            if (wasSpace[p] && widest[p] <= Math.Max(NumberGap, Math.Max(ModelGap, Math.Max(ModelGapWeak, frequencies is null ? CodeGap : Math.Max(CodeGap, FrequencyGap))))
                && ShouldJoin(chars, wasSpace, p, widest[p], wordGap, words, frequencies))
            {
                result[p] = false;
            }

            changed |= result[p] != wasSpace[p];
        }

        return changed ? result : null;
    }

    // The median of the empty stretches at breaks inside words: the line's usual spacing between letters.
    private static double MedianLetterGap(int[] widest, bool[] wasSpace)
    {
        var gaps = new List<int>(widest.Length);
        for (int p = 0; p < widest.Length; p++)
        {
            if (!wasSpace[p])
            {
                gaps.Add(widest[p]);
            }
        }

        if (gaps.Count == 0)
        {
            return 0;
        }

        gaps.Sort();
        return gaps[gaps.Count / 2];
    }

    // The median stretch at the network's own spaces, or 0 when the line has fewer than MinSpacesForGap of them.
    private static double MedianWordGap(int[] widest, bool[] wasSpace)
    {
        var gaps = new List<int>();
        for (int p = 0; p < widest.Length; p++)
        {
            if (wasSpace[p])
            {
                gaps.Add(widest[p]);
            }
        }

        if (gaps.Count < MinSpacesForGap)
        {
            return 0;
        }

        gaps.Sort();
        return gaps[gaps.Count / 2];
    }

    // Whether a space may be inserted between characters p and p + 1. Both sides must be able to start or end a word, numbers
    // never lose digits to a space between them, and a mark that closes a sentence stays with the word before it.
    private static bool MaySplit(IReadOnlyList<RecognizedChar> chars, bool[] wasSpace, int p)
    {
        string a = chars[p].Text, b = chars[p + 1].Text;
        if (a.Length != 1 || b.Length != 1)
        {
            return false;
        }

        char l = a[0], r = b[0];
        if (char.IsDigit(l) && char.IsDigit(r))
        {
            return false;
        }

        // A point or comma after a digit may be part of the number, and a faint one does not show in the column profile, which
        // makes the stretch beside it look wider than it is.
        if (l is '.' or ',' && p > 0 && char.IsDigit(chars[p - 1].Text[^1]))
        {
            return false;
        }

        if (!(char.IsLetterOrDigit(r) || Opening.Contains(r)) || !(char.IsLetterOrDigit(l) || Closing.Contains(l)))
        {
            return false;
        }

        // A word with an address in it is read as one piece.
        (int from, int to) = WordSpan(wasSpace, p, chars.Count);
        for (int i = from; i <= to; i++)
        {
            if (chars[i].Text is "@" || (chars[i].Text is "/" && i > from && chars[i - 1].Text is "/"))
            {
                return false;
            }
        }

        return true;
    }

    // The first and last character of the network's word that holds characters p and p + 1 (the span of the two words when a space
    // lies between them).
    private static (int From, int To) WordSpan(bool[] wasSpace, int p, int count)
    {
        int from = p;
        while (from > 0 && !wasSpace[from - 1])
        {
            from--;
        }

        int to = p + 1;
        while (to < count - 1 && !wasSpace[to])
        {
            to++;
        }

        return (from, to);
    }

    // Whether the network's space after character p is taken back: the words on its two sides are the halves of one number or date,
    // or one word that the character model finds much more plausible whole.
    private static bool ShouldJoin(IReadOnlyList<RecognizedChar> chars, bool[] wasSpace, int p, int gap, double wordGap, WordPlausibility? words, WordFrequencyModel? frequencies)
    {
        (int from, int to) = WordSpan(wasSpace, p, chars.Count);
        var left = new StringBuilder();
        for (int i = from; i <= p; i++)
        {
            left.Append(chars[i].Text);
        }

        var right = new StringBuilder();
        for (int i = p + 1; i <= to; i++)
        {
            right.Append(chars[i].Text);
        }

        string l = left.ToString(), r = right.ToString();
        if (gap <= NumberGap && JoinsNumber(l, r))
        {
            return true;
        }

        // An address, web address or amount cut in two: the stretch is held to the same share of the line's word gap as for words.
        if (gap <= CodeGap && !(wordGap != 0 && gap > FrequencyRatio * wordGap) && JoinsCode(l, r))
        {
            return true;
        }

        if (words is not null && gap <= ModelGapWeak && !(wordGap != 0 && gap > ModelRatio * wordGap)
            && JoinEvidence(words, l, r) is { } evidence)
        {
            // Plain evidence: the joined word is much more plausible than the pieces. Weak evidence, enough when one piece could not
            // stand as a word by itself ("val" + "uable", "in" + "debted"): a smaller gain. Two pieces that are both words ("tax" "at")
            // need the plain evidence.
            if ((gap <= ModelGap && evidence.Gain >= ModelMargin) || (evidence.Gain >= ModelMarginWeak && evidence.MinPiece <= WeakPieceMax))
            {
                return true;
            }
        }

        return frequencies is not null && JoinsByFrequency(frequencies, l, r, gap, wordGap);
    }

    // How much more word-like the two words are as one word than as two (null when the model has no score for them): the pieces
    // must end and begin with letters, and only their outer marks (quotes, a full stop) are left out of the score.
    private static (double Gain, double MinPiece)? JoinEvidence(WordPlausibility words, string left, string right)
    {
        if (!Cores(left, right, out string leftCore, out string rightCore))
        {
            return null;
        }

        double? joined = words.Score(leftCore + rightCore);
        double? one = words.Score(leftCore);
        double? two = words.Score(rightCore);
        if (joined is null || one is null || two is null)
        {
            return null;
        }

        return (joined.Value - (one.Value + two.Value), Math.Min(one.Value, two.Value));
    }

    // The letters of the two pieces without the marks around them (quotes, a full stop). False when either piece does not end or begin
    // with a letter, or holds a capital the models cannot judge: they are trained on running prose, and a capital inside a piece
    // marks a code, an abbreviation or a heading ("TehES", "DI"); only a capital that starts the left piece is prose.
    private static bool Cores(string left, string right, out string leftCore, out string rightCore)
    {
        leftCore = rightCore = string.Empty;
        if (left.Length == 0 || right.Length == 0 || !char.IsLetter(left[^1]) || !char.IsLetter(right[0]))
        {
            return false;
        }

        int a = 0;
        while (a < left.Length && !char.IsLetter(left[a]))
        {
            a++;
        }

        int b = right.Length;
        while (b > 0 && !char.IsLetter(right[b - 1]))
        {
            b--;
        }

        leftCore = left[a..];
        rightCore = right[..b];
        for (int i = 1; i < leftCore.Length; i++)
        {
            if (char.IsUpper(leftCore[i]))
            {
                return false;
            }
        }

        foreach (char c in rightCore)
        {
            if (char.IsUpper(c))
            {
                return false;
            }
        }

        return true;
    }

    // Whether the frequency table finds the joined word far more common than its two pieces together: the joined word is in the table
    // with at least FrequencyMinQuarter, and its count over the product of the pieces' counts (a word the table lacks counts as
    // seen about half as often as the rarest word it holds) is at least FrequencyMargin quarter-bits above what chance puts
    // together. A one-letter piece needs FrequencySingleExtra more. Integer arithmetic only, so every platform decides alike.
    private static bool JoinsByFrequency(WordFrequencyModel frequencies, string left, string right, int gap, double wordGap)
    {
        if (gap > FrequencyGap || (wordGap != 0 && gap > FrequencyRatio * wordGap) || !Cores(left, right, out string l, out string r))
        {
            return false;
        }

        // A digit or a code symbol anywhere in either word, also among the marks around the letters, makes it a code: never judged.
        foreach (char c in left + right)
        {
            if (char.IsDigit(c) || CodeSymbols.Contains(c, StringComparison.Ordinal))
            {
                return false;
            }
        }

        string a = Lower(l), b = Lower(r);
        if (a.Length == 0 || b.Length == 0)
        {
            return false;
        }

        foreach (char c in a + b)
        {
            if (!char.IsLetter(c))
            {
                return false;
            }
        }

        if (!TryJoined(frequencies, a + b, out int joined) || joined < FrequencyMinQuarter)
        {
            return false;
        }

        int qa = frequencies.Quarter(a), qb = frequencies.Quarter(b);
        int gain = joined + frequencies.TotalQuarterBits - qa - qb;
        int needed = FrequencyMargin + (a.Length == 1 || b.Length == 1 ? FrequencySingleExtra : 0);

        // Two pieces that are both common words ("dare" "say") are most likely two words, whatever the joined word's count: they need more.
        if (qa >= FrequencyWordQuarter && qb >= FrequencyWordQuarter)
        {
            needed += FrequencyBothExtra;
        }

        return gain >= needed;
    }

    private static bool TryJoined(WordFrequencyModel frequencies, string word, out int quarter)
    {
        quarter = 0;
        if (word.Length > 40)
        {
            return false;
        }

        Span<byte> bytes = stackalloc byte[160];
        int n = Encoding.UTF8.GetBytes(word, bytes);
        return frequencies.TryFind(bytes[..n], out quarter);
    }

    // The table's folding: the simple lowercase of each code point, as the character model does it.
    private static string Lower(string s)
    {
        var sb = new StringBuilder(s.Length);
        foreach (Rune r in s.EnumerateRunes())
        {
            sb.Append(Prepared.Fold(r.ToString()));
        }

        return sb.ToString();
    }

    // Whether the words on either side of a break are the two halves of one number or date: a thousands group that the break cut
    // ("$96," "221.95", "16,0" "80.65") or a date separator that the break cut ("2014-" "02-15", "24/07" "/2010").
    // Only the break is taken out; the digits and marks stay as read.
    internal static bool JoinsNumber(string left, string right)
    {
        if (left.Length == 0 || right.Length == 0 || (!char.IsAsciiDigit(right[0]) && right[0] != '/' && right[0] != '-'))
        {
            return false;
        }

        // The number the left word ends with: its longest suffix of digits, commas, points, slashes, colons and hyphens.
        int start = left.Length;
        while (start > 0 && (char.IsAsciiDigit(left[start - 1]) || ",./:-".Contains(left[start - 1])))
        {
            start--;
        }

        string tail = left[start..];
        while (tail.Length > 0 && !char.IsAsciiDigit(tail[0]))
        {
            tail = tail[1..];
        }

        if (tail.Length == 0)
        {
            return false;
        }

        int rightDigits = 0;
        while (rightDigits < right.Length && char.IsAsciiDigit(right[rightDigits]))
        {
            rightDigits++;
        }

        // A thousands group is cut after its comma: the next three digits complete it.
        if (tail[^1] == ',' && char.IsAsciiDigit(right[0]) && rightDigits == 3 && IsGrouped(tail[..^1]))
        {
            return true;
        }

        // ... or inside it: one or two digits are there, and the rest of the three follow.
        int comma = tail.LastIndexOf(',');
        if (comma > 0 && char.IsAsciiDigit(right[0]) && IsGrouped(tail[..comma]))
        {
            string group = tail[(comma + 1)..];
            if (group.Length is 1 or 2 && group.All(char.IsAsciiDigit) && rightDigits == 3 - group.Length)
            {
                return true;
            }
        }

        // A date or time separator is cut off its number: the left word is made of digits and separators, and either ends in a
        // hyphen or slash with digits following, or already holds a date separator and the right word continues with one.
        bool numeric = tail.All(c => char.IsAsciiDigit(c) || "/-:.".Contains(c)) && tail.Count(char.IsAsciiDigit) >= 2;
        if (numeric && tail[^1] is '/' or '-' && char.IsAsciiDigit(right[0]))
        {
            return true;
        }

        return numeric && tail.Any(c => c is '/' or '-') && right[0] is '/' or '-' && right.Length > 1 && char.IsAsciiDigit(right[1]);
    }

    // Top-level domains that follow a full stop inside a web or mail address; words that also read as plain English (us, co) are left out.
    private static readonly string[] Domains = ["com", "gov", "org", "net", "edu", "io", "uk", "mil", "info", "biz", "app", "dev", "example"];

    private const string Currency = "$£€";

    /// <summary>
    /// Whether the words on either side of a break are the two halves of one address, web address or amount. None of those has a space
    /// in it, so a break that sits where the pieces only make sense together goes: (a) the right word starts with <c>@</c> and carries a
    /// dotted domain, and the left word is made of the characters of a mail name; (b) the left word is a mail address whose domain
    /// ends in a full stop, and the right word is letters (the rest of the domain); (c) the left word is a name made of letters, digits,
    /// points and hyphens that ends in a full stop, and the right word is a top-level domain (<c>gov</c>, <c>com</c> ...) in one case
    /// (a sentence would continue with a capital); (d) the left word is a scheme and <c>:</c> and the right word starts with <c>//</c>;
    /// (e) the left word is a web address (it holds <c>//</c> or starts with <c>www.</c>) that ends in <c>/</c> and the right word starts
    /// with a letter or digit; (f) the left word ends in a currency sign that no digit precedes and the right word starts with two digits or a digit and a point or comma.
    /// Only the break is taken out.
    /// </summary>
    internal static bool JoinsCode(string left, string right)
    {
        if (left.Length == 0 || right.Length == 0)
        {
            return false;
        }

        // (f) "$ 17,906.36"
        if (Currency.Contains(left[^1], StringComparison.Ordinal) && right.Length >= 2 && char.IsAsciiDigit(right[0]) && (char.IsAsciiDigit(right[1]) || right[1] is ',' or '.')
            && (left.Length == 1 || !char.IsAsciiDigit(left[^2])))
        {
            return true;
        }

        // (d) "https: //example.org"
        if (right.StartsWith("//", StringComparison.Ordinal) && left.Length > 1 && left[^1] == ':' && left[..^1].All(char.IsAsciiLetter))
        {
            return true;
        }

        // (e) "https://example.org/ about"
        if (left[^1] == '/' && (left.Contains("//", StringComparison.Ordinal) || left.StartsWith("www.", StringComparison.OrdinalIgnoreCase)) && char.IsAsciiLetterOrDigit(right[0]))
        {
            return true;
        }

        // (a) "elena.nair @example.com"; the mail name is not empty, the domain has a point inside it.
        if (right[0] == '@' && right.Length > 3 && left.All(c => char.IsAsciiLetterOrDigit(c) || "._%+-".Contains(c)) && char.IsAsciiLetterOrDigit(left[^1]))
        {
            string domain = right[1..].TrimEnd('.', ',', ';', ':', '!', '?', ')');
            int dot = domain.IndexOf('.');
            return dot > 0 && dot < domain.Length - 1 && domain.All(c => char.IsAsciiLetterOrDigit(c) || ".-".Contains(c));
        }

        // (b) "olga@corp.example. net"
        int at = left.IndexOf('@');
        if (at > 0 && left[^1] == '.' && char.IsAsciiLetter(left[^2]) && left.IndexOf('.', at) > 0 && char.IsAsciiLetter(right[0]))
        {
            string tail = right.TrimEnd('.', ',', ';', ':', '!', '?', ')');
            return tail.Length >= 2 && tail.All(char.IsAsciiLetter) && tail.All(char.IsAsciiLetterLower);
        }

        // (c) "IRS. gov"
        if (left.Length > 2 && left[^1] == '.' && char.IsAsciiLetterOrDigit(left[^2]) && char.IsAsciiLetterOrDigit(left[0])
            && left.All(c => char.IsAsciiLetterOrDigit(c) || ".-".Contains(c)))
        {
            string tail = right.TrimEnd('.', ',', ';', ':', '!', '?', ')');
            foreach (string d in Domains)
            {
                if (string.Equals(tail, d, StringComparison.OrdinalIgnoreCase) && (tail.All(char.IsAsciiLetterLower) || tail.All(char.IsAsciiLetterUpper)))
                {
                    return true;
                }
            }
        }

        return false;
    }

    // Digits in groups of three after one to three: "1", "96", "1,234", "12,345,678".
    private static bool IsGrouped(string number)
    {
        string[] parts = number.Split(',');
        if (parts[0].Length is < 1 or > 3 || !parts[0].All(char.IsAsciiDigit))
        {
            return false;
        }

        for (int i = 1; i < parts.Length; i++)
        {
            if (parts[i].Length != 3 || !parts[i].All(char.IsAsciiDigit))
            {
                return false;
            }
        }

        return true;
    }

    // The columns of the line image that hold ink: pixels far enough from the line's background (its median) toward the line's
    // strongest contrast. Null when the line is too faint to tell.
    private static bool[]? InkColumns(ReadOnlySpan<byte> gray, int width, int height)
    {
        if (InkPixels(gray, width, height) is not { } pixels)
        {
            return null;
        }

        var columns = new bool[width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                columns[x] |= pixels[(y * width) + x];
            }
        }

        return columns;
    }

    // The pixels of the line image that hold ink (the rule above), row-major; null when the line is too faint to tell.
    internal static bool[]? InkPixels(ReadOnlySpan<byte> gray, int width, int height)
    {
        Span<int> histogram = stackalloc int[256];
        histogram.Clear();
        int total = width * height;
        for (int i = 0; i < total; i++)
        {
            histogram[gray[i]]++;
        }

        int median = 0;
        for (int sum = 0; median < 255; median++)
        {
            sum += histogram[median];
            if (sum * 2 >= total)
            {
                break;
            }
        }

        // The 99.5th percentile of the distance from the background; a few specks do not set the scale.
        Span<int> distances = stackalloc int[256];
        distances.Clear();
        for (int v = 0; v < 256; v++)
        {
            distances[Math.Abs(v - median)] += histogram[v];
        }

        int allowed = total / 200;
        int top = 255;
        for (int seen = 0; top > 0; top--)
        {
            seen += distances[top];
            if (seen > allowed)
            {
                break;
            }
        }

        if (top < MinContrast)
        {
            return null;
        }

        int limit = (top + 1) / 2;
        var pixels = new bool[width * height];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = Math.Abs(gray[i] - median) > limit;
        }

        return pixels;
    }
}
