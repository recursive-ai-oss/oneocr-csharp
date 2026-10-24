using System.Text;
using OneOcr.Documents;

namespace OneOcr.Llm;

/// <summary>One whitespace-delimited piece of the model's text. <c>Core</c> is the part compared with OCR words; in Markdown, leading and trailing syntax characters are left out of it.</summary>
internal sealed record Tok(string Core, int CoreStart, string SepAfter);

/// <summary>One word of the merged text.</summary>
internal sealed class MergeEntry
{
    public required string Text { get; init; }

    public string SepAfter { get; set; } = " ";

    /// <summary>Index into the page's word sequence, or -1 for a word only the model has.</summary>
    public int OcrIdx { get; init; } = -1;

    public OcrSource Source { get; init; }

    public float Confidence { get; init; }

    public int Anchor { get; init; } = -1;

    public int DisagreementIndex { get; init; } = -1;
}

/// <summary>
/// Aligns the model's words with the local reader's and merges them. Specified in <c>docs/spec/grounding.md</c> (text merge):
/// global alignment with unit gap cost and substitution cost growing with edit distance; ties prefer a pairing over a gap.
/// </summary>
internal static class TextMerge
{
    public const float LlmWordConfidence = 0.5f;
    private const int Gap = 1000;
    private const int FullCells = 1 << 22;
    private const long MaxBandCells = 64L * 1024 * 1024;

    public static List<Tok> Tokenize(string text, bool markdown)
    {
        var toks = new List<Tok>();
        int i = 0;
        while (i < text.Length)
        {
            while (i < text.Length && char.IsWhiteSpace(text[i]))
            {
                i++;
            }

            if (i >= text.Length)
            {
                break;
            }

            int start = i;
            while (i < text.Length && !char.IsWhiteSpace(text[i]))
            {
                i++;
            }

            int end = i;
            int sepStart = i;
            while (i < text.Length && char.IsWhiteSpace(text[i]))
            {
                i++;
            }

            string sep = text[sepStart..i];
            int cs = start, ce = end;
            if (markdown)
            {
                string seg = text[start..end];
                if (seg.All(static ch => ch is '-' or '+' or '|' or ':' or '#' or '>' or '*' or '_' or '~' or '`' or '='))
                {
                    cs = ce = start;
                }
                else
                {
                    while (cs < ce && text[cs] is '#' or '>' or '*' or '_' or '|' or '~' or '`')
                    {
                        cs++;
                    }

                    while (ce > cs && text[ce - 1] is '*' or '_' or '|' or '~' or '`')
                    {
                        ce--;
                    }
                }
            }

            toks.Add(new Tok(text[cs..ce], cs, sep));
        }

        return toks;
    }

    private static string Key(string s) => ValueNormalize.TrimPunct(ValueNormalize.Fold(s));

    private static int SubCost(string a, string b)
    {
        if (a == b)
        {
            return 0;
        }

        int max = Math.Max(a.Length, b.Length);
        if (max == 0)
        {
            return 0;
        }

        int d = ValueNormalize.BoundedDistance(a, b, max);
        return 400 + (int)(1600L * d / max);
    }

    /// <summary>Pairs of (ocr index, llm index); -1 on one side is a gap. Null when the band would be too large.</summary>
    public static List<(int O, int L)>? Align(string[] ocr, string[] llm)
    {
        int n = ocr.Length, m = llm.Length;
        if (n == 0 || m == 0)
        {
            var trivial = new List<(int, int)>(n + m);
            for (int i = 0; i < n; i++)
            {
                trivial.Add((i, -1));
            }

            for (int j = 0; j < m; j++)
            {
                trivial.Add((-1, j));
            }

            return trivial;
        }

        int band = (long)n * m <= FullCells ? n + m + 2 : Math.Max(256, Math.Abs(n - m) + 64);
        int Lo(int i) => (int)Math.Max(0, ((long)i * m / n) - band);
        int Hi(int i) => (int)Math.Min(m, ((long)i * m / n) + band);
        long cells = 0;
        for (int i = 0; i <= n; i++)
        {
            cells += Hi(i) - Lo(i) + 1;
            if (cells > MaxBandCells)
            {
                return null;
            }
        }

        const int Inf = int.MaxValue / 2;
        var dir = new byte[n + 1][];
        int[] prev = new int[m + 1], cur = new int[m + 1];
        Array.Fill(prev, Inf);
        Array.Fill(cur, Inf);
        dir[0] = new byte[Hi(0) - Lo(0) + 1];
        prev[0] = 0;
        for (int j = 1; j <= Hi(0); j++)
        {
            prev[j] = j * Gap;
            dir[0][j - Lo(0)] = 3;
        }

        for (int i = 1; i <= n; i++)
        {
            int lo = Lo(i), hi = Hi(i);
            Array.Fill(cur, Inf);
            dir[i] = new byte[hi - lo + 1];
            for (int j = lo; j <= hi; j++)
            {
                int best = Inf;
                byte d = 0;
                if (j > 0 && prev[j - 1] < Inf)
                {
                    best = prev[j - 1] + SubCost(ocr[i - 1], llm[j - 1]);
                    d = 1;
                }

                if (prev[j] < Inf && prev[j] + Gap < best)
                {
                    best = prev[j] + Gap;
                    d = 2;
                }

                if (j > lo && cur[j - 1] < Inf && cur[j - 1] + Gap < best)
                {
                    best = cur[j - 1] + Gap;
                    d = 3;
                }

                if (j == 0 && i > 0 && prev[0] < Inf && d == 0)
                {
                    best = prev[0] + Gap;
                    d = 2;
                }

                cur[j] = best;
                dir[i][j - lo] = d;
            }

            (prev, cur) = (cur, prev);
        }

        var path = new List<(int, int)>(n + m);
        int ti = n, tj = m;
        while (ti > 0 || tj > 0)
        {
            byte d = ti == 0 ? (byte)3 : dir[ti][tj - Lo(ti)];
            if (d == 1)
            {
                path.Add((ti - 1, tj - 1));
                ti--;
                tj--;
            }
            else if (d == 2)
            {
                path.Add((ti - 1, -1));
                ti--;
            }
            else if (d == 3)
            {
                path.Add((-1, tj - 1));
                tj--;
            }
            else
            {
                return null; // unreachable cell: the band lost the path
            }
        }

        path.Reverse();
        return path;
    }

    /// <summary>The merged words of one page, with the disagreements found.</summary>
    public static List<MergeEntry>? MergeText(
        int pageIndex, IReadOnlyList<OcrWord> ocr, List<Tok> toks, LlmReadOptions options, List<TextDisagreement> dis)
    {
        string[] ok = new string[ocr.Count], lk = new string[toks.Count];
        for (int i = 0; i < ok.Length; i++)
        {
            ok[i] = Key(ocr[i].Text);
        }

        for (int j = 0; j < lk.Length; j++)
        {
            lk[j] = Key(toks[j].Core);
        }

        List<(int O, int L)>? path = Align(ok, lk);
        if (path is null)
        {
            return null;
        }

        var entries = new List<MergeEntry>(path.Count);
        int lastOcr = -1;
        float thr = options.ConfidenceThreshold;
        foreach ((int o, int l) in path)
        {
            if (o >= 0 && l >= 0)
            {
                OcrWord w = ocr[o];
                string lt = toks[l].Core;
                if (string.Equals(w.Text, lt, StringComparison.Ordinal))
                {
                    entries.Add(new MergeEntry { Text = w.Text, SepAfter = toks[l].SepAfter, OcrIdx = o, Source = w.Source, Confidence = w.Confidence });
                }
                else
                {
                    bool keepOcr = w.Confidence >= thr;
                    dis.Add(new TextDisagreement(pageIndex, DisagreementKind.Substitution, w.Text, lt, w.Confidence, keepOcr ? w.Source : OcrSource.Llm, w.Box, -1, 0));
                    entries.Add(new MergeEntry
                    {
                        Text = keepOcr ? w.Text : lt, SepAfter = toks[l].SepAfter, OcrIdx = o, Source = keepOcr ? w.Source : OcrSource.Llm,
                        Confidence = keepOcr ? w.Confidence : LlmWordConfidence, DisagreementIndex = dis.Count - 1,
                    });
                }

                lastOcr = o;
            }
            else if (o >= 0)
            {
                OcrWord w = ocr[o];
                bool keep = w.Confidence >= thr;
                dis.Add(new TextDisagreement(pageIndex, DisagreementKind.OcrOnly, w.Text, null, w.Confidence, keep ? w.Source : null, w.Box, -1, 0));
                if (keep)
                {
                    MergeEntry? prev = entries.LastOrDefault(static x => x.OcrIdx != -2);
                    var e = new MergeEntry { Text = w.Text, SepAfter = prev?.SepAfter ?? " ", OcrIdx = o, Source = w.Source, Confidence = w.Confidence, DisagreementIndex = dis.Count - 1 };
                    if (prev is not null)
                    {
                        prev.SepAfter = " ";
                    }

                    entries.Add(e);
                }
                else
                {
                    // Dropped: the span sits where the next entry starts.
                    entries.Add(new MergeEntry { Text = "", SepAfter = "", OcrIdx = -2, DisagreementIndex = dis.Count - 1 });
                }

                lastOcr = o;
            }
            else
            {
                string lt = toks[l].Core;
                bool keep = options.KeepLlmOnlyWords;
                dis.Add(new TextDisagreement(pageIndex, DisagreementKind.LlmOnly, null, lt, 0f, keep ? OcrSource.Llm : null, null, -1, 0));
                if (keep)
                {
                    entries.Add(new MergeEntry
                    {
                        Text = lt, SepAfter = toks[l].SepAfter, Source = OcrSource.Llm, Confidence = LlmWordConfidence, Anchor = lastOcr, DisagreementIndex = dis.Count - 1,
                    });
                }
            }
        }

        return entries;
    }

    /// <summary>Joins entries into the page's text and fills each disagreement's span; <paramref name="baseOffset"/> is where the page starts in the whole text.</summary>
    public static string Assemble(List<MergeEntry> entries, List<TextDisagreement> dis, int baseOffset)
    {
        var sb = new StringBuilder();
        foreach (MergeEntry e in entries)
        {
            if (e.DisagreementIndex >= 0)
            {
                TextDisagreement d = dis[e.DisagreementIndex];
                dis[e.DisagreementIndex] = d with { Start = baseOffset + sb.Length, Length = e.OcrIdx == -2 ? 0 : e.Text.Length };
            }

            sb.Append(e.Text).Append(e.SepAfter);
        }

        while (sb.Length > 0 && char.IsWhiteSpace(sb[^1]))
        {
            sb.Length--;
        }

        return sb.ToString();
    }

    /// <summary>Flags for Markdown: where the model's words differ from the local reading (keys, so case-only and edge-punctuation differences are not flagged).</summary>
    public static bool FlagMarkdown(
        int pageIndex, IReadOnlyList<OcrWord> ocr, List<Tok> toks, LlmReadOptions options, int baseOffset, List<TextDisagreement> flags)
    {
        var words = toks.Where(static t => t.Core.Length > 0).ToList();
        string[] ok = new string[ocr.Count], lk = new string[words.Count];
        for (int i = 0; i < ok.Length; i++)
        {
            ok[i] = Key(ocr[i].Text);
        }

        for (int j = 0; j < lk.Length; j++)
        {
            lk[j] = Key(words[j].Core);
        }

        List<(int O, int L)>? path = Align(ok, lk);
        if (path is null)
        {
            return false;
        }

        int nextStart = 0;
        for (int p = 0; p < path.Count; p++)
        {
            (int o, int l) = path[p];
            if (l >= 0)
            {
                nextStart = words[l].CoreStart;
            }

            if (o >= 0 && l >= 0)
            {
                if (ok[o] != lk[l])
                {
                    flags.Add(new TextDisagreement(pageIndex, DisagreementKind.Substitution, ocr[o].Text, words[l].Core, ocr[o].Confidence, null, ocr[o].Box, baseOffset + words[l].CoreStart, words[l].Core.Length));
                }
            }
            else if (l >= 0)
            {
                flags.Add(new TextDisagreement(pageIndex, DisagreementKind.LlmOnly, null, words[l].Core, 0f, null, null, baseOffset + words[l].CoreStart, words[l].Core.Length));
            }
            else if (ocr[o].Confidence >= options.ConfidenceThreshold)
            {
                int at = p + 1 < path.Count && path[p + 1].L >= 0 ? words[path[p + 1].L].CoreStart : nextStart;
                flags.Add(new TextDisagreement(pageIndex, DisagreementKind.OcrOnly, ocr[o].Text, null, ocr[o].Confidence, null, ocr[o].Box, baseOffset + at, 0));
            }
        }

        return true;
    }

    // ---- the merged page model ----

    public static OcrPage BuildPage(OcrPage orig, List<MergeEntry> entries, string diagnostic)
    {
        IReadOnlyList<OcrWord> old = orig.Words;
        var kept = new Dictionary<int, MergeEntry>();
        var after = new Dictionary<int, List<MergeEntry>>();
        var lead = new List<MergeEntry>();
        foreach (MergeEntry e in entries)
        {
            if (e.OcrIdx >= 0)
            {
                kept[e.OcrIdx] = e;
            }
            else if (e.OcrIdx == -1)
            {
                // Anchor: the nearest earlier entry that is an OCR word still in the page.
                int a = e.Anchor;
                while (a >= 0 && !kept.ContainsKey(a))
                {
                    a--;
                }

                if (a < 0)
                {
                    lead.Add(e);
                }
                else
                {
                    if (!after.TryGetValue(a, out List<MergeEntry>? list))
                    {
                        after[a] = list = [];
                    }

                    list.Add(e);
                }
            }
        }

        OcrWord Llm(MergeEntry e, OcrBox box) => new(e.Text, box, e.Confidence, e.Confidence, OcrSource.Llm, []);

        var newIndex = new int[old.Count];
        Array.Fill(newIndex, -1);
        int flat = 0;
        int oldPos = 0;
        var blocks = new List<OcrBlock>(orig.Blocks.Count);
        bool leadPlaced = lead.Count == 0;
        foreach (OcrBlock block in orig.Blocks)
        {
            var lines = new List<OcrLine>();
            foreach (OcrLine line in block.Lines)
            {
                var words = new List<OcrWord>();
                var cellStartsOld = new List<int>(line.Cells);
                var cellOfNew = new List<int>(); // old cell number of each new word
                for (int k = 0; k < line.Words.Count; k++)
                {
                    int cell = 0;
                    while (cell + 1 < cellStartsOld.Count && cellStartsOld[cell + 1] <= k)
                    {
                        cell++;
                    }

                    if (!leadPlaced)
                    {
                        leadPlaced = true;
                        foreach (MergeEntry e in lead)
                        {
                            words.Add(Llm(e, new OcrBox(line.Words[k].Box.Left, line.Words[k].Box.Top, line.Words[k].Box.Left, line.Words[k].Box.Bottom)));
                            cellOfNew.Add(cell);
                            flat++;
                        }
                    }

                    int oi = oldPos++;
                    if (kept.TryGetValue(oi, out MergeEntry? me))
                    {
                        OcrWord ow = line.Words[k];
                        bool same = me.Source == ow.Source && string.Equals(me.Text, ow.Text, StringComparison.Ordinal);
                        words.Add(same ? ow : ow with
                        {
                            Text = me.Text, Confidence = me.Confidence, RawConfidence = me.Source == OcrSource.Llm ? me.Confidence : ow.RawConfidence, Source = me.Source,
                            Chars = string.Equals(me.Text, ow.Text, StringComparison.Ordinal) ? ow.Chars : [],
                        });
                        newIndex[oi] = flat++;
                        cellOfNew.Add(cell);
                        if (after.TryGetValue(oi, out List<MergeEntry>? extra))
                        {
                            foreach (MergeEntry e in extra)
                            {
                                words.Add(Llm(e, new OcrBox(ow.Box.Right, ow.Box.Top, ow.Box.Right, ow.Box.Bottom)));
                                cellOfNew.Add(cell);
                                flat++;
                            }
                        }
                    }
                }

                if (words.Count == 0)
                {
                    continue;
                }

                var cells = new List<int> { 0 };
                for (int w = 1; w < words.Count; w++)
                {
                    if (cellOfNew[w] != cellOfNew[w - 1])
                    {
                        cells.Add(w);
                    }
                }

                var sb = new StringBuilder();
                for (int w = 0; w < words.Count; w++)
                {
                    if (w > 0)
                    {
                        sb.Append(cells.Contains(w) ? '\t' : ' ');
                    }

                    sb.Append(words[w].Text);
                }

                lines.Add(new OcrLine(sb.ToString(), words, cells, line.Box, Weighted(words), words.Any(static x => x.Source != OcrSource.Llm) ? line.Source : OcrSource.Llm, line.IsRightToLeft, line.IsVertical, line.Quad));
            }

            blocks.Add(block with { Lines = lines });
        }

        if (!leadPlaced)
        {
            // The page had no words at all: the model's words form a single line with empty boxes.
            var words = lead.Select(e => Llm(e, new OcrBox(0, 0, 0, 0))).ToList();
            string text = string.Join(' ', words.Select(static w => w.Text));
            var line = new OcrLine(text, words, [0], new OcrBox(0, 0, 0, 0), Weighted(words), OcrSource.Llm);
            blocks.Add(new OcrBlock(OcrBlockRole.Body, false, line.Box, [line], line.Confidence, OcrSource.Llm));
        }

        List<int> Map(IReadOnlyList<int> idx) => [.. idx.Where(i => i >= 0 && i < newIndex.Length && newIndex[i] >= 0).Select(i => newIndex[i])];

        var tables = orig.Tables.Select(t => t with { Cells = [.. t.Cells.Select(c => c with { WordIndices = Map(c.WordIndices) })] }).ToList();
        var kvs = orig.KeyValues.Select(k => k with { KeyWordIndices = Map(k.KeyWordIndices), ValueWordIndices = Map(k.ValueWordIndices) })
            .Where(static k => k.KeyWordIndices.Count > 0 && k.ValueWordIndices.Count > 0).ToList();
        return new OcrPage(orig.Index, orig.Width, orig.Height, orig.UnitsPerInch, blocks, orig.Regions, tables, kvs, [.. orig.Diagnostics, diagnostic], orig.ToPdf);
    }

    private static float Weighted(List<OcrWord> words)
    {
        double sum = 0, n = 0;
        foreach (OcrWord w in words)
        {
            sum += w.Confidence * w.Text.Length;
            n += w.Text.Length;
        }

        return n == 0 ? 0f : (float)(sum / n);
    }
}
