using System.Text.Json;
using OneOcr.Documents;

namespace OneOcr.Llm;

internal enum MatchKind
{
    Exact = 0,
    Normalized = 1,
    Partial = 2,
    Gapped = 3,
}

/// <summary>A run of words in a scope that holds the value. <paramref name="Start"/> and <paramref name="Length"/> index the scope array.</summary>
internal sealed record Hit(int Start, int Length, MatchKind Kind, string? Normalized, bool YearInferred, bool DateAmbiguous, int[]? Picks = null);

/// <summary>One value as the model gave it: its text, JSON kind, cited word ids and whether it called the value derived.</summary>
internal sealed record LeafAnswer(string Path, string Value, JsonValueKind Kind, IReadOnlyList<string> WordIds, bool Derived);

/// <summary>The words of a call as the grounder sees them, with folded text and the page's number and date habits worked out once.</summary>
internal sealed class GroundingContext
{
    private readonly IReadOnlyList<LlmPageInput> _pages;
    private readonly Dictionary<string, int> _ids;

    public GroundingContext(IReadOnlyList<LlmWordRef> words, IReadOnlyList<LlmPageInput> pages, NumberStyle style, DateOrder order)
    {
        Words = words;
        _pages = pages;
        _ids = new Dictionary<string, int>(words.Count, StringComparer.Ordinal);
        Folded = new string[words.Count];
        for (int i = 0; i < words.Count; i++)
        {
            _ids[words[i].Id] = i;
            Folded[i] = ValueNormalize.Fold(words[i].Text);
        }

        Style = style == NumberStyle.Auto ? ValueNormalize.DetectStyle(words.Select(static w => w.Text)) : style;
        Order = order == DateOrder.Auto ? ValueNormalize.DetectOrder(words.Select(static w => w.Text)) : order;
        All = new int[words.Count];
        for (int i = 0; i < All.Length; i++)
        {
            All[i] = i;
        }
    }

    public IReadOnlyList<LlmWordRef> Words { get; }

    public string[] Folded { get; }

    public int[] All { get; }

    public NumberStyle Style { get; }

    public DateOrder Order { get; }

    public bool TryIndex(string id, out int index) => _ids.TryGetValue(id, out index);

    public float Confidence(int index)
    {
        LlmWordRef w = Words[index];
        IReadOnlyList<OcrWord> page = _pages[w.PageIndex].Page.Words;
        return w.WordIndex < page.Count ? page[w.WordIndex].Confidence : 0f;
    }
}

/// <summary>The value being grounded, parsed every way it can be compared.</summary>
internal sealed class ValueForm
{
    public ValueForm(string raw, JsonValueKind kind, GroundingContext c)
    {
        Raw = raw;
        WsKey = ValueNormalize.RemoveWhitespace(raw);
        string folded = ValueNormalize.Fold(raw);
        Key = ValueNormalize.TrimPunct(folded);
        if (Key.Length == 0 && folded.Length > 0)
        {
            // A value that is only punctuation (":") is compared as it is.
            Key = folded;
            NoTrim = true;
        }

        if (kind == JsonValueKind.Number)
        {
            if (decimal.TryParse(raw, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out decimal d))
            {
                Amount = new ParsedAmount(d, null);
            }
        }
        else if (ValueNormalize.TryParseAmount(raw, c.Style, out ParsedAmount a))
        {
            Amount = a;
        }

        if (kind != JsonValueKind.Number)
        {
            Dates = ValueNormalize.ParseDate(raw, c.Order);
            string up = raw.Trim().ToUpperInvariant();
            if (ValueNormalize.IsCurrencyCode(up))
            {
                Currency = up;
            }
        }
    }

    public string Raw { get; }

    public string WsKey { get; }

    public string Key { get; }

    public bool NoTrim { get; private set; }

    public ParsedAmount? Amount { get; }

    public List<ParsedDate> Dates { get; } = [];

    public string? Currency { get; }
}

/// <summary>Grounds one value against the page: citation first, then a search of the whole page; see <c>docs/spec/grounding.md</c>.</summary>
internal static class Grounder
{
    private const int MaxRunWords = 64;
    private const int MaxDerivedTerms = 16;

    public static GroundedField Ground(GroundingContext c, LeafAnswer leaf, bool reasked)
    {
        if (leaf.Kind is JsonValueKind.True or JsonValueKind.False)
        {
            return new GroundedField(leaf.Path, leaf.Value, GroundingState.Derived, 0.5f, null, [], [], Reasked: reasked, Note: "a boolean is inferred from the page, never printed");
        }

        var v = new ValueForm(leaf.Value, leaf.Kind, c);
        var cited = new SortedSet<int>();
        int unknown = 0;
        foreach (string id in leaf.WordIds)
        {
            if (c.TryIndex(id, out int idx))
            {
                cited.Add(idx);
            }
            else
            {
                unknown++;
            }
        }

        int[] citedScope = [.. cited];
        Hit? weakCited = null;
        string? note = unknown > 0 ? $"{unknown} cited id(s) are not in the word list" : null;

        if (citedScope.Length > 0)
        {
            Hit? hit = Best(Find(c, citedScope, v));
            if (hit is { Kind: MatchKind.Exact or MatchKind.Normalized })
            {
                return Build(c, leaf, v, citedScope, hit, corrected: false, ambiguous: 1, reasked, note);
            }

            weakCited = hit;
        }

        List<Hit> all = Find(c, c.All, v);
        if (weakCited is not null && !all.Exists(static h => h.Kind is MatchKind.Exact or MatchKind.Normalized))
        {
            // Only a part-of-a-word or interleaved match exists, and the model cited it: keep the citation.
            return Build(c, leaf, v, citedScope, weakCited, corrected: false, ambiguous: 1, reasked, note);
        }

        if (all.Count > 0)
        {
            Hit best = BestNear(all, citedScope);
            int distinct = CountDistinct(all, best.Kind);
            string why = citedScope.Length == 0 ? "no words were cited" : "the cited words do not hold the value";
            note = (note is null ? "" : note + "; ") + why + (distinct > 1 ? $"; {distinct} places on the page match, the nearest to the citation was taken" : "; found elsewhere on the page");
            return Build(c, leaf, v, c.All, best, corrected: true, distinct, reasked, note);
        }

        if (leaf.Derived)
        {
            bool verified = v.Amount is { } amount && DerivedFromCited(c, citedScope, amount.Value);
            return new GroundedField(
                leaf.Path, leaf.Value, GroundingState.Derived, verified ? 0.9f : 0.3f, null, [], [], Reasked: reasked,
                Note: verified ? "computed from the cited amounts" : "claimed as computed or inferred; nothing on the page reproduces it");
        }

        string? near = Nearest(c, v);
        return new GroundedField(
            leaf.Path, leaf.Value, GroundingState.Ungrounded, 0f, null, [], [], Reasked: reasked,
            Note: (note is null ? "" : note + "; ") + "not found on the page" + (near is null ? "" : "; closest page text: " + Clip(near)));
    }

    private static string Clip(string s) => s.Length <= 80 ? s : s[..80] + "…";

    private static GroundedField Build(
        GroundingContext c, LeafAnswer leaf, ValueForm v, int[] scope, Hit hit, bool corrected, int ambiguous, bool reasked, string? note)
    {
        var words = new List<LlmWordRef>(hit.Length);
        double conf = 0;
        int[] positions = hit.Picks ?? [.. Enumerable.Range(hit.Start, hit.Length)];
        foreach (int pos in positions)
        {
            int idx = scope[pos];
            words.Add(c.Words[idx]);
            conf += c.Confidence(idx);
        }

        conf /= Math.Max(1, positions.Length);
        double factor = hit.Kind switch { MatchKind.Exact => 1.0, MatchKind.Normalized => 0.95, MatchKind.Partial => 0.9, _ => 0.8 };
        if (corrected)
        {
            factor *= 0.85;
        }

        if (corrected && ambiguous > 1)
        {
            factor *= 0.8;
        }

        if (hit.YearInferred)
        {
            factor *= 0.7;
            note = (note is null ? "" : note + "; ") + "the year is not printed";
        }

        if (hit.DateAmbiguous)
        {
            factor *= 0.8;
            note = (note is null ? "" : note + "; ") + "day and month are ambiguous on this page";
        }

        if (hit.Kind == MatchKind.Partial)
        {
            note = (note is null ? "" : note + "; ") + "the value is part of a longer word";
        }

        if (hit.Kind == MatchKind.Gapped)
        {
            note = (note is null ? "" : note + "; ") + "the words are interleaved with other text";
        }

        GroundingState state = hit.Kind == MatchKind.Exact ? GroundingState.Grounded : GroundingState.GroundedNormalized;
        return new GroundedField(
            leaf.Path, leaf.Value, state, (float)Math.Clamp(factor * conf, 0, 1), state == GroundingState.GroundedNormalized ? hit.Normalized : null,
            words, Boxes(words), corrected, reasked, note);
    }

    private static Hit? Best(List<Hit> hits) => hits.Count == 0 ? null : hits.OrderBy(static h => h.Kind).ThenBy(static h => h.Start).First();

    private static Hit BestNear(List<Hit> hits, int[] cited)
    {
        // Exact beats normalized beats partial; among equals the one nearest a cited word, then the first on the page.
        return hits.OrderBy(static h => h.Kind)
            .ThenBy(h => cited.Length == 0 ? 0 : cited.Min(i => Math.Abs(i - h.Start)))
            .ThenBy(static h => h.Start).First();
    }

    private static int CountDistinct(List<Hit> hits, MatchKind kind)
    {
        int count = 0, end = -1;
        foreach (Hit h in hits.Where(h => h.Kind == kind).OrderBy(static h => h.Start))
        {
            if (h.Start >= end)
            {
                count++;
                end = h.Start + h.Length;
            }
        }

        return count;
    }

    /// <summary>Every run of scope words that holds the value, by exact text, folded text, amount, date, currency, or as part of one word.</summary>
    public static List<Hit> Find(GroundingContext c, int[] scope, ValueForm v)
    {
        var hits = new List<Hit>();
        bool Has(int start, int len) => hits.Exists(h => h.Start == start && h.Length == len);

        if (v.Key.Length > 0)
        {
            for (int s = 0; s < scope.Length; s++)
            {
                var acc = new System.Text.StringBuilder();
                var raw = new System.Text.StringBuilder();
                for (int k = 0; k < MaxRunWords && s + k < scope.Length; k++)
                {
                    acc.Append(c.Folded[scope[s + k]]);
                    raw.Append(ValueNormalize.RemoveWhitespace(c.Words[scope[s + k]].Text));
                    string a = v.NoTrim ? acc.ToString() : ValueNormalize.TrimPunct(acc.ToString());
                    if (a == v.Key)
                    {
                        hits.Add(new Hit(s, k + 1, raw.ToString() == v.WsKey ? MatchKind.Exact : MatchKind.Normalized, null, false, false));
                        break;
                    }

                    if (a.Length >= v.Key.Length || !v.Key.StartsWith(a, StringComparison.Ordinal))
                    {
                        break;
                    }
                }
            }
        }

        if (v.Amount is { } amount)
        {
            for (int s = 0; s < scope.Length; s++)
            {
                string w = c.Words[scope[s]].Text;
                if (!w.Any(char.IsAsciiDigit) && ValueNormalize.CurrencyOf(w) is null)
                {
                    continue;
                }

                for (int len = 1; len <= 2 && s + len <= scope.Length; len++)
                {
                    if (Has(s, len))
                    {
                        continue;
                    }

                    string text = len == 1 ? w : w + " " + c.Words[scope[s + 1]].Text;
                    if (ValueNormalize.TryParseAmount(text, c.Style, out ParsedAmount p) && p.Value == amount.Value && CurrencyCompatible(amount.Currencies, p.Currencies))
                    {
                        hits.Add(new Hit(s, len, MatchKind.Normalized, ValueNormalize.CanonicalAmount(p.Value), false, false));
                        break;
                    }
                }
            }
        }

        if (v.Dates.Count > 0)
        {
            for (int s = 0; s < scope.Length; s++)
            {
                string w = c.Words[scope[s]].Text;
                if (!w.Any(char.IsAsciiDigit) && !ValueNormalize.IsMonthName(w))
                {
                    continue;
                }

                for (int len = 1; len <= 4 && s + len <= scope.Length; len++)
                {
                    if (Has(s, len))
                    {
                        continue;
                    }

                    var text = new System.Text.StringBuilder(w);
                    for (int k = 1; k < len; k++)
                    {
                        text.Append(' ').Append(c.Words[scope[s + k]].Text);
                    }

                    Hit? dh = null;
                    foreach (ParsedDate pd in ValueNormalize.ParseDate(text.ToString(), c.Order))
                    {
                        // A date without a year is not a match when the words go on to print one ("24 October" inside "24 October 2024").
                        if (pd.Year is null && s + len < scope.Length && ValueNormalize.ParseDate(text + " " + c.Words[scope[s + len]].Text, c.Order).Count > 0)
                        {
                            continue;
                        }

                        foreach (ParsedDate vd in v.Dates)
                        {
                            if (pd.Month == vd.Month && pd.Day == vd.Day && (pd.Year is null || vd.Year is null || pd.Year == vd.Year))
                            {
                                dh = new Hit(s, len, MatchKind.Normalized, ValueNormalize.CanonicalDate(vd.Year is null && pd.Year is not null ? pd : vd), pd.Year is null && vd.Year is not null, pd.Ambiguous);
                            }
                        }
                    }

                    if (dh is not null)
                    {
                        hits.Add(dh);
                        break;
                    }
                }
            }
        }

        if (v.Currency is { } code)
        {
            for (int s = 0; s < scope.Length; s++)
            {
                if (!Has(s, 1) && ValueNormalize.CurrencyOf(c.Words[scope[s]].Text) is { } codes && codes.Contains(code))
                {
                    hits.Add(new Hit(s, 1, MatchKind.Normalized, code, false, false));
                }
            }
        }

        if (hits.Count == 0 && v.Raw.Any(char.IsWhiteSpace))
        {
            FindGapped(c, scope, v, hits);
        }

        if (hits.Count == 0 && v.Key.Length >= 4)
        {
            for (int s = 0; s < scope.Length; s++)
            {
                string f = c.Folded[scope[s]];
                int at = f.IndexOf(v.Key, StringComparison.Ordinal);
                while (at >= 0)
                {
                    bool before = at == 0 || !char.IsLetterOrDigit(f[at - 1]);
                    int end = at + v.Key.Length;
                    bool after = end == f.Length || !char.IsLetterOrDigit(f[end]);
                    if (before && after)
                    {
                        hits.Add(new Hit(s, 1, MatchKind.Partial, null, false, false));
                        break;
                    }

                    at = f.IndexOf(v.Key, at + 1, StringComparison.Ordinal);
                }
            }
        }

        return hits;
    }

    /// <summary>
    /// A multi-word value whose words all appear in order with at most <see cref="MaxGap"/> other words between neighbours (a name wrapped
    /// around a quantity column). Only the matching words are boxed.
    /// </summary>
    private static void FindGapped(GroundingContext c, int[] scope, ValueForm v, List<Hit> hits)
    {
        string[] tokens = [.. v.Raw.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Select(static t => ValueNormalize.TrimPunct(ValueNormalize.Fold(t)))];
        if (tokens.Length < 2 || tokens.Any(static t => t.Length == 0))
        {
            return;
        }

        for (int s = 0; s < scope.Length; s++)
        {
            if (ValueNormalize.TrimPunct(c.Folded[scope[s]]) != tokens[0])
            {
                continue;
            }

            int pos = s;
            bool ok = true;
            var picks = new List<int> { s };
            for (int t = 1; t < tokens.Length && ok; t++)
            {
                int next = -1;
                for (int g = 1; g <= MaxGap + 1 && pos + g < scope.Length; g++)
                {
                    if (ValueNormalize.TrimPunct(c.Folded[scope[pos + g]]) == tokens[t])
                    {
                        next = pos + g;
                        break;
                    }
                }

                ok = next >= 0;
                pos = next;
                if (ok)
                {
                    picks.Add(next);
                }
            }

            if (ok)
            {
                // Report the span; Build boxes every word in it, so mark the run as gapped through its length.
                hits.Add(new Hit(s, pos - s + 1, MatchKind.Gapped, null, false, false, [.. picks]));
                return;
            }
        }
    }

    private const int MaxGap = 3;

    private static bool CurrencyCompatible(string[]? a, string[]? b) => a is null || b is null || a.Intersect(b).Any();

    private static bool DerivedFromCited(GroundingContext c, int[] cited, decimal target)
    {
        var terms = new List<decimal>();
        foreach (int i in cited)
        {
            if (terms.Count < MaxDerivedTerms && ValueNormalize.TryParseAmount(c.Words[i].Text, c.Style, out ParsedAmount p))
            {
                terms.Add(p.Value);
            }
        }

        int n = terms.Count;
        for (long mask = 3; mask < (1L << n); mask++)
        {
            if ((mask & (mask - 1)) == 0)
            {
                continue;
            }

            decimal sum = 0;
            for (int k = 0; k < n; k++)
            {
                if ((mask & (1L << k)) != 0)
                {
                    sum += terms[k];
                }
            }

            if (sum == target)
            {
                return true;
            }
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    continue;
                }

                decimal a = terms[i], b = terms[j];
                if (a - b == target || Math.Round(a * b, 2, MidpointRounding.ToEven) == target || Math.Round(a * b / 100m, 2, MidpointRounding.ToEven) == target)
                {
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>The page text closest to the value (within 40% edit distance), for the re-ask and the note; null when nothing is close or the page is too big to scan.</summary>
    public static string? Nearest(GroundingContext c, ValueForm v)
    {
        if (v.Key.Length is < 2 or > 64 || c.Words.Count > 5000)
        {
            return null;
        }

        int bound = Math.Max(1, v.Key.Length * 2 / 5);
        int best = bound + 1;
        string? bestText = null;
        for (int s = 0; s < c.Words.Count; s++)
        {
            var acc = new System.Text.StringBuilder();
            var shown = new System.Text.StringBuilder();
            for (int k = 0; k < 8 && s + k < c.Words.Count; k++)
            {
                acc.Append(c.Folded[s + k]);
                if (k > 0)
                {
                    shown.Append(' ');
                }

                shown.Append(c.Words[s + k].Text);
                if (acc.Length > v.Key.Length + bound)
                {
                    break;
                }

                if (acc.Length + bound < v.Key.Length)
                {
                    continue;
                }

                int d = ValueNormalize.BoundedDistance(ValueNormalize.TrimPunct(acc.ToString()), v.Key, Math.Min(bound, best - 1));
                if (d < best)
                {
                    best = d;
                    bestText = shown.ToString();
                }
            }
        }

        return bestText;
    }

    private static List<GroundedBox> Boxes(List<LlmWordRef> words)
    {
        var boxes = new List<GroundedBox>();
        foreach (LlmWordRef w in words)
        {
            if (boxes.Count > 0 && boxes[^1].PageIndex == w.PageIndex)
            {
                OcrBox last = boxes[^1].Box;
                double overlap = Math.Min(last.Bottom, w.Box.Bottom) - Math.Max(last.Top, w.Box.Top);
                if (overlap >= 0.5 * Math.Min(last.Height, w.Box.Height))
                {
                    boxes[^1] = new GroundedBox(w.PageIndex, new OcrBox(
                        Math.Min(last.Left, w.Box.Left), Math.Min(last.Top, w.Box.Top), Math.Max(last.Right, w.Box.Right), Math.Max(last.Bottom, w.Box.Bottom)));
                    continue;
                }
            }

            boxes.Add(new GroundedBox(w.PageIndex, w.Box));
        }

        return boxes;
    }
}
