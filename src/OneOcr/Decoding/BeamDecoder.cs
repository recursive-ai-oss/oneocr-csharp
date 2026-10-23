using OneOcr.Kernels;
using OneOcr.Recognition;

namespace OneOcr.Decoding;

/// <summary>Everything about a decoding configuration that depends on the model's charset, computed once.</summary>
internal sealed class Prepared
{
    // Characters that mark a word as a code: such words are never ranked by the language model.
    private const string CodeSymbols = "@#_/\\=%+*~^|<>[]{}";

    public Prepared(DecodeOptions options, IReadOnlyList<string> charset)
    {
        Charset = charset;
        int n = charset.Count;
        IsSpace = new bool[n];
        Protected = new bool[n];
        for (int i = 1; i < n; i++)
        {
            string s = charset[i];
            IsSpace[i] = s == " ";
            foreach (TextElement r in Runes(s))
            {
                if (r.Digit || CodeSymbols.Contains(r.Text, StringComparison.Ordinal))
                {
                    Protected[i] = true;
                }
            }
        }

        if (options.LanguageModel is { } lm)
        {
            var index = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int i = 1; i < n; i++)
            {
                index.TryAdd(charset[i], i);
            }

            Scripts = [.. lm.Scripts];
            Maps = new int[Scripts.Length][];
            for (int s = 0; s < Scripts.Length; s++)
            {
                var map = new int[n];
                Array.Fill(map, -1);
                string[] alphabet = Scripts[s].Alphabet;
                var symbolOf = new Dictionary<string, int>(StringComparer.Ordinal);
                for (int a = 0; a < alphabet.Length; a++)
                {
                    symbolOf[alphabet[a]] = a + ScriptModel.FirstLetter;
                }

                // The model is case-folded: a class is looked up by its simple lowercase form.
                for (int cls = 1; cls < n; cls++)
                {
                    if (symbolOf.TryGetValue(Fold(charset[cls]), out int sym))
                    {
                        map[cls] = sym;
                    }
                }

                Maps[s] = map;
            }
        }
        else
        {
            Scripts = [];
            Maps = [];
        }

        if (options.Lexicon is { Count: > 0 } lexicon)
        {
            var index = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int i = 1; i < n; i++)
            {
                index.TryAdd(charset[i], i);
            }

            var children = new Dictionary<long, int>();
            var words = new List<bool> { false };
            foreach (string word in lexicon.Words)
            {
                var path = new List<int>();
                bool known = true;
                foreach (TextElement r in Runes(word))
                {
                    if (!index.TryGetValue(r.Text, out int cls))
                    {
                        known = false;
                        break;
                    }

                    path.Add(cls);
                }

                if (!known)
                {
                    continue;
                }

                int node = 0;
                foreach (int cls in path)
                {
                    long key = ((long)node << 20) | (uint)cls;
                    if (!children.TryGetValue(key, out int next))
                    {
                        next = words.Count;
                        words.Add(false);
                        children[key] = next;
                    }

                    node = next;
                }

                words[node] = true;
            }

            LexiconWords = [.. lexicon.Words];
            TrieChildren = children;
            TrieIsWord = [.. words];
        }
    }

    public IReadOnlyList<string> Charset { get; }

    public HashSet<string>? LexiconWords { get; private set; }

    /// <summary>The words of <paramref name="text"/> that contain a digit or a code symbol, in order.</summary>
    public static List<string> CodeWords(string text)
    {
        var words = new List<string>();
        foreach (string word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            foreach (System.Text.Rune c in word.EnumerateRunes())
            {
                if (System.Text.Rune.IsDigit(c) || (c.IsBmp && CodeSymbols.Contains((char)c.Value, StringComparison.Ordinal)))
                {
                    words.Add(word);
                    break;
                }
            }
        }

        return words;
    }

    public bool[] IsSpace { get; }

    public bool[] Protected { get; }

    public ScriptModel[] Scripts { get; }

    /// <summary>Per script: charset class → symbol id in that script's tables, or -1.</summary>
    public int[][] Maps { get; }

    public Dictionary<long, int>? TrieChildren { get; }

    public bool[]? TrieIsWord { get; }

    /// <summary>The simple lowercase form of a one-code-point string (other strings are returned as they are).</summary>
    public static string Fold(string s)
    {
        if (s.Length is 0 or > 2 || !System.Text.Rune.TryGetRuneAt(s, 0, out System.Text.Rune r) || r.Utf16SequenceLength != s.Length)
        {
            return s;
        }

        // U+0130 has no one-to-one lowercase in the full mapping the builder uses, so it stays as it is.
        return r.Value == 0x130 ? s : System.Text.Rune.ToLowerInvariant(r).ToString();
    }

    private readonly record struct TextElement(string Text, bool Digit);

    private static IEnumerable<TextElement> Runes(string s)
    {
        // Code points, as the charset's entries are; Unicode normalization is left to the caller (platform-independent behaviour).
        foreach (System.Text.Rune r in s.EnumerateRunes())
        {
            yield return new TextElement(r.ToString(), System.Text.Rune.IsDigit(r));
        }
    }
}

/// <summary>
/// Prefix beam search for CTC outputs (Graves et al. 2006; Hannun et al. 2014). Each prefix carries the log probability of
/// the steps so far ending in a blank and ending in its last character, kept apart so repeated characters collapse correctly.
/// The language model and lexicon enter as a score that depends only on the prefix, added when ranking prefixes and when
/// choosing the result. All arithmetic is double precision from + - * / and <see cref="DetMath"/>, in an order fixed by
/// <c>docs/spec/decoding.md</c>, so the text does not depend on the CPU path or the degree of parallelism.
/// </summary>
internal sealed class BeamDecoder
{
    // A class is a candidate for a step only when its logit is within this many nats of the step's largest (probability >= ~1e-4
    // of the largest, so at most that of the whole step), and at most this many classes are tried per step.
    private const double PruneDelta = -9.2;
    public const int MaxCandidates = 8;

    // No symbol (or word end) may cost more than this many nats below the model's average, so that a rare name or an unusual
    // spelling is a bounded disadvantage rather than a veto. Found on held-out synthetic lines; see docs/spec/decoding.md.
    internal const double Floor = -1.0;

    private struct Node
    {
        public int Parent;
        public int Cls;
        public int Length;
        public ulong Context;
        public int ContextLength;
        public int TokenLength;
        public bool Protected;
        public double LmToken;
        public int Lex;
        public double Committed;
    }

    private readonly DecodeOptions _options;
    private readonly Prepared _prep;
    private readonly ScriptModel? _script;
    private readonly int[]? _map;
    private readonly double _alpha, _boost;
    private readonly List<Node> _nodes = new(1024);
    private readonly Dictionary<long, int> _child = new();

    private BeamDecoder(DecodeOptions options, Prepared prep, ScriptModel? script, int[]? map)
    {
        _options = options;
        _prep = prep;
        _script = script;
        _map = map;
        _alpha = options.LanguageModelWeight;
        _boost = options.LexiconBoost;
        _nodes.Add(Fresh(-1, 0, 0, 0.0, prep.TrieChildren is null ? -1 : 0));
    }

    /// <summary>The most nodes a line of <paramref name="steps"/> steps can create.</summary>
    public static long MaxNodes(DecodeOptions options, int steps) => 1 + ((long)steps * options.BeamWidth * (MaxCandidates + 1));

    // greedy is the greedy reading of the same steps. With a language model, words that contain a digit or code symbol must come
    // out exactly as in it (except where the user's lexicon supplies the word), or the greedy reading is returned instead.
    public static RecognizedLine Decode(
        DecodeOptions options, Prepared prep, ReadOnlySpan<float> logits, int steps, int classes, KernelPath path, RecognizedLine greedy)
    {
        if (steps == 0)
        {
            return new RecognizedLine(string.Empty, 0f);
        }

        (ScriptModel? script, int[]? map) = ChooseScript(prep, logits, steps, classes);
        return new BeamDecoder(options, prep, script, map).Run(logits, steps, classes, path, greedy);
    }

    private static (ScriptModel?, int[]?) ChooseScript(Prepared prep, ReadOnlySpan<float> logits, int steps, int classes)
    {
        if (prep.Scripts.Length == 0)
        {
            return (null, null);
        }

        var counts = new int[prep.Scripts.Length];
        int last = 0;
        for (int t = 0; t < steps; t++)
        {
            ReadOnlySpan<float> frame = logits.Slice(t * classes, classes);
            int best = 0;
            float value = frame[0];
            for (int j = 1; j < frame.Length; j++)
            {
                if (frame[j] > value)
                {
                    value = frame[j];
                    best = j;
                }
            }

            if (best != 0 && best != last && char.IsLetter(prep.Charset[best], 0))
            {
                for (int s = 0; s < counts.Length; s++)
                {
                    if (prep.Maps[s][best] >= 0)
                    {
                        counts[s]++;
                    }
                }
            }

            last = best;
        }

        int pick = 0;
        for (int s = 1; s < counts.Length; s++)
        {
            if (counts[s] > counts[pick])
            {
                pick = s;
            }
        }

        return counts[pick] > 0 ? (prep.Scripts[pick], prep.Maps[pick]) : (null, null);
    }

    private static Node Fresh(int parent, int cls, int length, double committed, int lex) => new()
    {
        Parent = parent,
        Cls = cls,
        Length = length,
        Context = 0, // the word-start symbol is 0
        ContextLength = 1,
        TokenLength = 0,
        Protected = false,
        LmToken = 0.0,
        Lex = lex,
        Committed = committed,
    };

    private double TokenScore(in Node n)
    {
        double score = 0.0;
        if (_script is not null && !n.Protected)
        {
            score = _alpha * n.LmToken;
        }

        if (n.Lex >= 0 && _prep.TrieIsWord is { } words && n.TokenLength > 0)
        {
            if (words[n.Lex])
            {
                score += _boost;
            }
            else if (n.TokenLength >= 2)
            {
                score += 0.5 * _boost;
            }
        }

        return score;
    }

    private double FinalTokenScore(in Node n)
    {
        if (n.TokenLength == 0)
        {
            return 0.0;
        }

        double score = TokenScore(n);
        if (_script is not null && !n.Protected)
        {
            score += _alpha * Math.Max(EndLogProb(n) - _script.MeanLogProb, Floor);
        }

        return score;
    }

    private float EndLogProb(in Node n) => _script!.LogProb(ContextSpan(n, stackalloc byte[8]), ScriptModel.End);

    private static ReadOnlySpan<byte> ContextSpan(in Node n, Span<byte> buffer)
    {
        int count = Math.Min(n.ContextLength, 8);
        ulong c = n.Context;
        for (int i = count - 1; i >= 0; i--)
        {
            buffer[i] = (byte)(c & 0xFF);
            c >>= 8;
        }

        return buffer[..count];
    }

    private int Child(int parent, int cls)
    {
        long key = ((long)parent << 20) | (uint)cls;
        if (_child.TryGetValue(key, out int found))
        {
            return found;
        }

        Node p = _nodes[parent];
        int length = p.Length + 1;
        Node n;
        if (_prep.IsSpace[cls])
        {
            double committed = p.Committed + (p.TokenLength > 0 ? FinalTokenScore(p) : 0.0);
            n = Fresh(parent, cls, length, committed, _prep.TrieChildren is null ? -1 : 0);
        }
        else
        {
            n = p;
            n.Parent = parent;
            n.Cls = cls;
            n.Length = length;
            n.TokenLength = p.TokenLength + 1;
            int sym = _map is null ? -1 : _map[cls];
            if (_script is not null && (p.Protected || _prep.Protected[cls] || sym < 0))
            {
                n.Protected = true;
            }
            else if (_script is not null)
            {
                float lp = _script.LogProb(ContextSpan(p, stackalloc byte[8]), (byte)sym);
                n.LmToken = p.LmToken + Math.Max(lp - _script.MeanLogProb, Floor);
                n.Context = (p.Context << 8) | (byte)sym;
                n.ContextLength = p.ContextLength + 1;
            }

            if (p.Lex >= 0 && _prep.TrieChildren is { } trie)
            {
                n.Lex = trie.TryGetValue(((long)p.Lex << 20) | (uint)cls, out int next) ? next : -1;
            }
        }

        int id = _nodes.Count;
        _nodes.Add(n);
        _child[key] = id;
        return id;
    }

    private double Rank(int node)
    {
        Node n = _nodes[node];
        return n.Committed + (_options.InsertionBonus * n.Length) + TokenScore(n);
    }

    private RecognizedLine Run(ReadOnlySpan<float> logits, int steps, int classes, KernelPath path, RecognizedLine greedy)
    {
        int width = _options.BeamWidth;
        var nodeIds = new List<int> { 0 };
        var pbs = new List<double> { 0.0 };
        var pnbs = new List<double> { double.NegativeInfinity };
        var nextIds = new List<int>();
        var nextPb = new List<double>();
        var nextPnb = new List<double>();
        var slot = new Dictionary<int, int>();
        Span<int> candCls = stackalloc int[MaxCandidates];
        Span<float> candD = stackalloc float[MaxCandidates];

        for (int t = 0; t < steps; t++)
        {
            ReadOnlySpan<float> frame = logits.Slice(t * classes, classes);
            float max = frame[0];
            for (int j = 1; j < frame.Length; j++)
            {
                if (frame[j] > max)
                {
                    max = frame[j];
                }
            }

            double logSum = DetMath.Log(Elementwise.SumExp(path, frame, max));
            double blank = (double)(frame[0] - max) - logSum;

            // The most probable non-blank classes within PruneDelta of the largest logit, best first; ties keep the lower class.
            int count = 0;
            for (int j = 1; j < frame.Length; j++)
            {
                float d = frame[j] - max;
                if (d < PruneDelta || (count == MaxCandidates && !(d > candD[count - 1])))
                {
                    continue;
                }

                int at = count < MaxCandidates ? count : count - 1;
                while (at > 0 && d > candD[at - 1])
                {
                    candD[at] = candD[at - 1];
                    candCls[at] = candCls[at - 1];
                    at--;
                }

                candD[at] = d;
                candCls[at] = j;
                if (count < MaxCandidates)
                {
                    count++;
                }
            }

            nextIds.Clear();
            nextPb.Clear();
            nextPnb.Clear();
            slot.Clear();

            void Add(int node, double pb, double pnb)
            {
                if (slot.TryGetValue(node, out int i))
                {
                    nextPb[i] = DetMath.LogAdd(nextPb[i], pb);
                    nextPnb[i] = DetMath.LogAdd(nextPnb[i], pnb);
                }
                else
                {
                    slot[node] = nextIds.Count;
                    nextIds.Add(node);
                    nextPb.Add(pb);
                    nextPnb.Add(pnb);
                }
            }

            for (int b = 0; b < nodeIds.Count; b++)
            {
                int node = nodeIds[b];
                double pb = pbs[b], pnb = pnbs[b];
                double total = DetMath.LogAdd(pb, pnb);
                int lastCls = node == 0 ? 0 : _nodes[node].Cls;

                // A blank keeps the prefix and ends any run of its last character.
                Add(node, total + blank, double.NegativeInfinity);

                if (lastCls > 0 && !double.IsNegativeInfinity(pnb))
                {
                    // The last character continues through this step (no new character).
                    double lpLast = (double)(frame[lastCls] - max) - logSum;
                    Add(node, double.NegativeInfinity, pnb + lpLast);
                }

                for (int c = 0; c < count; c++)
                {
                    int cls = candCls[c];
                    int child = Child(node, cls);
                    // The same character again starts a new character only after a blank; otherwise it extends the run.
                    double from = cls == lastCls ? pb : total;
                    if (!double.IsNegativeInfinity(from))
                    {
                        Add(child, double.NegativeInfinity, from + ((double)candD[c] - logSum));
                    }
                }
            }

            // Keep the best prefixes: acoustic evidence plus the prefix score; equal scores keep the earlier-created prefix.
            int m = nextIds.Count;
            var order = new int[m];
            var score = new double[m];
            for (int i = 0; i < m; i++)
            {
                order[i] = i;
                score[i] = DetMath.LogAdd(nextPb[i], nextPnb[i]) + Rank(nextIds[i]);
            }

            Array.Sort(order, (x, y) =>
            {
                int cmp = score[y].CompareTo(score[x]);
                return cmp != 0 ? cmp : nextIds[x].CompareTo(nextIds[y]);
            });
            m = Math.Min(m, width);

            nodeIds.Clear();
            pbs.Clear();
            pnbs.Clear();
            for (int i = 0; i < m; i++)
            {
                nodeIds.Add(nextIds[order[i]]);
                pbs.Add(nextPb[order[i]]);
                pnbs.Add(nextPnb[order[i]]);
            }
        }

        // Final choice: best total of acoustic evidence and prefix score (the open word is closed with its end-of-word score).
        int count0 = nodeIds.Count;
        var finalOrder = new int[count0];
        var finalScore = new double[count0];
        var finalAcoustic = new double[count0];
        for (int b = 0; b < count0; b++)
        {
            Node n = _nodes[nodeIds[b]];
            finalOrder[b] = b;
            finalAcoustic[b] = DetMath.LogAdd(pbs[b], pnbs[b]);
            finalScore[b] = finalAcoustic[b] + n.Committed + (_options.InsertionBonus * n.Length) + FinalTokenScore(n);
        }

        Array.Sort(finalOrder, (x, y) =>
        {
            int cmp = finalScore[y].CompareTo(finalScore[x]);
            return cmp != 0 ? cmp : nodeIds[x].CompareTo(nodeIds[y]);
        });

        List<string>? greedyCodes = _options.LanguageModel is null ? null : Prepared.CodeWords(greedy.Text);
        foreach (int b in finalOrder)
        {
            int id = nodeIds[b];
            var parts = new List<string>();
            int emitted = 0;
            while (id > 0)
            {
                Node n = _nodes[id];
                parts.Add(_prep.Charset[n.Cls]);
                emitted++;
                id = n.Parent;
            }

            parts.Reverse();
            string text = string.Concat(parts);
            if (greedyCodes is not null && !KeepsCodeWords(greedyCodes, text))
            {
                continue;
            }

            // Confidence: geometric mean, per emitted character, of the probability the recognizer gives the text (all alignments).
            float confidence = emitted == 0 ? 0f : (float)DetMath.Exp(finalAcoustic[b] / emitted);
            return new RecognizedLine(text, Math.Clamp(confidence, 0f, 1f));
        }

        return greedy;
    }

    // Numbers, identifiers and the like are read as the recognizer sees them: same words in the same order, except that a word
    // the user's lexicon lists may stand in for the one it replaces.
    private bool KeepsCodeWords(List<string> greedyCodes, string text)
    {
        List<string> codes = Prepared.CodeWords(text);
        if (codes.Count != greedyCodes.Count)
        {
            return false;
        }

        for (int i = 0; i < codes.Count; i++)
        {
            if (codes[i] != greedyCodes[i] && _prep.LexiconWords?.Contains(codes[i]) != true)
            {
                return false;
            }
        }

        return true;
    }
}
