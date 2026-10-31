using OneOcr.Kernels;

namespace OneOcr.Documents;

/// <summary>
/// Turns the model's word labels into fields (docs/spec/docextract-v0.md, "Assembly"). Deterministic and independent of the kernel path:
/// every probability comes from the library's own scalar exp, summed in index order.
/// </summary>
internal static class DocAssembly
{
    private static readonly Dictionary<string, FieldValueKind> s_kinds = new(StringComparer.Ordinal)
    {
        ["BT-2"] = FieldValueKind.Date,
        ["BT-9"] = FieldValueKind.Date,
        ["BT-72"] = FieldValueKind.Date,
        ["receipt.date"] = FieldValueKind.Date,
        ["receipt.time"] = FieldValueKind.Time,
        ["BT-5"] = FieldValueKind.Currency,
        ["BT-106"] = FieldValueKind.Amount,
        ["BT-110"] = FieldValueKind.Amount,
        ["BT-112"] = FieldValueKind.Amount,
        ["BT-115"] = FieldValueKind.Amount,
        ["BT-131"] = FieldValueKind.Amount,
        ["BT-146"] = FieldValueKind.Amount,
        ["BT-116"] = FieldValueKind.Amount,
        ["BT-117"] = FieldValueKind.Amount,
        ["receipt.subtotal"] = FieldValueKind.Amount,
        ["receipt.tax"] = FieldValueKind.Amount,
        ["receipt.tip"] = FieldValueKind.Amount,
        ["receipt.total"] = FieldValueKind.Amount,
        ["receipt.unitPrice"] = FieldValueKind.Amount,
        ["receipt.amount"] = FieldValueKind.Amount,
        ["BT-129"] = FieldValueKind.Quantity,
        ["receipt.quantity"] = FieldValueKind.Quantity,
        ["BT-152"] = FieldValueKind.Percent,
        ["BT-119"] = FieldValueKind.Percent,
    };

    private sealed class LabelSpan
    {
        public required int Key { get; init; }
        public required List<int> Words { get; init; }
        public required List<double> P { get; init; }
        public double Confidence { get; set; }
    }

    /// <summary>
    /// Assembles one page.
    /// </summary>
    /// <param name="m">The model, for its schema.</param>
    /// <param name="o">The model's outputs.</param>
    /// <param name="texts">The text of each word the model read.</param>
    /// <param name="boxes">The box of each word.</param>
    /// <param name="pageIndex">For each model word, its index in <see cref="OcrPage.Words"/>.</param>
    /// <param name="sources">For each model word, where it came from.</param>
    /// <param name="forcedDocType">When not null, the document-type name whose fields are read, instead of the one the model predicts.</param>
    /// <param name="style">Number style for amounts.</param>
    /// <param name="order">Date order for numeric dates.</param>
    public static DocumentExtraction Assemble(
        DocExtractModel m, DocOutputs o, IReadOnlyList<string> texts, IReadOnlyList<OcrBox> boxes, IReadOnlyList<int> pageIndex,
        IReadOnlyList<OcrSource> sources, string? forcedDocType, NumberStyle style, DateOrder order)
    {
        int n = o.N;
        DocSchema schema = m.Schema;
        var notes = new List<OcrDiagnostic>();

        // Document type: the largest logit (lowest index on a tie), its probability from the softmax.
        int docIndex = DocForward.ArgMax(o.Doc);
        double docConfidence = SoftmaxOf(o.Doc, docIndex);
        string predicted = schema.DocTypes[docIndex];
        string docType = forcedDocType ?? predicted;
        if (forcedDocType is not null && forcedDocType != predicted)
        {
            notes.Add(Diag.Info(OcrDiagnosticCodes.DocumentTypeDiffers, $"The model reads this page as '{predicted}'; its fields were read with the '{forcedDocType}' schema as asked."));
        }

        // Tags: restricted to the classes the document type allows.
        var allowed = new bool[m.Tags];
        allowed[0] = true;
        for (int c = 0; c < schema.Classes.Count; c++)
        {
            if (schema.Allowed(docType, c))
            {
                allowed[1 + (2 * c)] = true;
                allowed[2 + (2 * c)] = true;
            }
        }

        var labels = new (bool Begin, int Key)?[n];
        var probs = new double[n];
        for (int i = 0; i < n; i++)
        {
            ReadOnlySpan<float> row = o.Tags.AsSpan(i * m.Tags, m.Tags);
            int best = -1;
            for (int t = 0; t < row.Length; t++)
            {
                if (allowed[t] && (best < 0 || row[t] > row[best]))
                {
                    best = t;
                }
            }

            probs[i] = SoftmaxOf(row, best, allowed);
            labels[i] = best == 0 ? null : (((best - 1) % 2) == 0, (best - 1) / 2);
        }

        List<LabelSpan> spans = SpansFromLabels(labels, probs);
        int[] groupPointer = o.GroupPointer;

        var header = new Dictionary<int, ExtractedField>();
        var grouped = new SortedDictionary<(int Family, int Anchor), Dictionary<int, ExtractedField>>();
        foreach (LabelSpan s in spans)
        {
            ExtractedField f = MakeField(schema.Classes[s.Key], s, texts, boxes, pageIndex, style, order);
            if (schema.Groups[s.Key] == FieldGroup.Header)
            {
                if (!header.TryGetValue(s.Key, out ExtractedField? have) || f.Confidence > have.Confidence)
                {
                    header[s.Key] = f;
                }
            }
            else
            {
                int anchor = Anchor(groupPointer, s.Words[0]);
                var key = (schema.Groups[s.Key] == FieldGroup.Tax ? 1 : 0, anchor);
                if (!grouped.TryGetValue(key, out Dictionary<int, ExtractedField>? g))
                {
                    grouped[key] = g = [];
                }

                if (!g.TryGetValue(s.Key, out ExtractedField? have) || f.Confidence > have.Confidence)
                {
                    g[s.Key] = f;
                }
            }
        }

        List<ExtractedField> fields = [.. header.OrderBy(static kv => kv.Key).Select(static kv => kv.Value)];
        List<ExtractedGroup> lineItems = [];
        List<ExtractedGroup> taxRows = [];
        foreach (((int family, int anchor), Dictionary<int, ExtractedField> g) in grouped)
        {
            var group = new ExtractedGroup(pageIndex[anchor], [.. g.Values.OrderBy(static f => f.WordIndices[0])]);
            (family == 0 ? lineItems : taxRows).Add(group);
        }

        // Key and value roles.
        var roleLabels = new (bool Begin, int Key)?[n];
        var roleProbs = new double[n];
        for (int i = 0; i < n; i++)
        {
            ReadOnlySpan<float> row = o.Roles.AsSpan(i * m.Roles, m.Roles);
            int best = DocForward.ArgMax(row);
            roleProbs[i] = SoftmaxOf(row, best);
            roleLabels[i] = best == 0 ? null : (best is 1 or 3, best is 1 or 2 ? 0 : 1);
        }

        List<LabelSpan> roleSpans = SpansFromLabels(roleLabels, roleProbs);
        var keyOfWord = new Dictionary<int, LabelSpan>();
        foreach (LabelSpan k in roleSpans.Where(static s => s.Key == 0))
        {
            foreach (int w in k.Words)
            {
                keyOfWord[w] = k;
            }
        }

        var keyValues = new List<OcrKeyValue>();
        foreach (LabelSpan v in roleSpans.Where(static s => s.Key == 1))
        {
            if (!keyOfWord.TryGetValue(o.KeyValuePointer[v.Words[0]], out LabelSpan? k))
            {
                continue;
            }

            OcrBox box = Union(k.Words.Concat(v.Words), boxes);
            keyValues.Add(new OcrKeyValue(
                [.. k.Words.Select(w => pageIndex[w])], [.. v.Words.Select(w => pageIndex[w])], box, (float)Math.Min(k.Confidence, v.Confidence),
                sources[v.Words[0]]));
        }

        DocumentType type = predicted switch { "invoice" => DocumentType.Invoice, "receipt" => DocumentType.Receipt, _ => DocumentType.Other };
        return new DocumentExtraction(type, (float)docConfidence, fields, lineItems, taxRows, keyValues, notes);
    }

    /// <summary>
    /// Follows an item pointer to the word that points to itself, at most four steps; on a cycle, the smallest index on it.
    /// </summary>
    internal static int Anchor(int[] pointer, int word)
    {
        Span<int> seen = stackalloc int[5];
        int count = 1;
        seen[0] = word;
        for (int step = 0; step < 4; step++)
        {
            int j = pointer[seen[count - 1]];
            if (j == seen[count - 1])
            {
                return j;
            }

            int at = seen[..count].IndexOf(j);
            if (at >= 0)
            {
                return seen[at..count].ToArray().Min();
            }

            seen[count++] = j;
        }

        return seen[count - 1];
    }

    private static List<LabelSpan> SpansFromLabels((bool Begin, int Key)?[] labels, double[] probs)
    {
        var spans = new List<LabelSpan>();
        LabelSpan? cur = null;
        for (int i = 0; i < labels.Length; i++)
        {
            if (labels[i] is not { } lab)
            {
                cur = null;
                continue;
            }

            if (cur is not null && !lab.Begin && cur.Key == lab.Key && cur.Words[^1] == i - 1)
            {
                cur.Words.Add(i);
                cur.P.Add(probs[i]);
            }
            else
            {
                cur = new LabelSpan { Key = lab.Key, Words = [i], P = [probs[i]] };
                spans.Add(cur);
            }
        }

        foreach (LabelSpan s in spans)
        {
            double sum = 0;
            foreach (double p in s.P)
            {
                sum += p;
            }

            s.Confidence = sum / s.P.Count;
        }

        return spans;
    }

    /// <summary>The probability of one entry: exp(l - max) / sum of exp(l' - max) over the allowed entries, in index order.</summary>
    private static double SoftmaxOf(ReadOnlySpan<float> logits, int index, bool[]? allowed = null)
    {
        float max = logits[index];
        float sum = 0f;
        for (int t = 0; t < logits.Length; t++)
        {
            if (allowed is null || allowed[t])
            {
                sum += FastMath.Exp(logits[t] - max);
            }
        }

        return 1f / sum;
    }

    private static ExtractedField MakeField(
        string id, LabelSpan s, IReadOnlyList<string> texts, IReadOnlyList<OcrBox> boxes, IReadOnlyList<int> pageIndex, NumberStyle style, DateOrder order)
    {
        string text = string.Join(' ', s.Words.Select(w => texts[w]));
        OcrBox box = Union(s.Words, boxes);
        var indices = s.Words.Select(w => pageIndex[w]).ToArray();
        FieldValueKind kind = s_kinds.GetValueOrDefault(id, FieldValueKind.Text);
        string value = text;
        decimal? number = null;
        DateOnly? date = null;
        string? currency = null;
        switch (kind)
        {
            case FieldValueKind.Amount or FieldValueKind.Quantity or FieldValueKind.Percent:
                string numeric = kind == FieldValueKind.Percent ? text.Replace("%", string.Empty, StringComparison.Ordinal) : text;
                if (ValueNormalize.TryParseAmount(numeric, style, out ParsedAmount amount))
                {
                    number = amount.Value;
                    value = ValueNormalize.CanonicalAmount(amount.Value);
                    currency = amount.Currencies is { Length: 1 } one ? one[0] : null;
                }

                break;
            case FieldValueKind.Date:
                List<ParsedDate> dates = ValueNormalize.ParseDate(text, order);
                if (dates.Count == 1)
                {
                    value = ValueNormalize.CanonicalDate(dates[0]);
                    if (dates[0].Year is { } y && y is >= 1 and <= 9999 && dates[0].Month is >= 1 and <= 12 && dates[0].Day >= 1
                        && dates[0].Day <= DateTime.DaysInMonth(y, dates[0].Month))
                    {
                        date = new DateOnly(y, dates[0].Month, dates[0].Day);
                    }
                }

                break;
            case FieldValueKind.Currency:
                string[]? codes = ValueNormalize.CurrencyOf(text);
                if (codes is { Length: 1 })
                {
                    value = codes[0];
                    currency = codes[0];
                }

                break;
        }

        return new ExtractedField(id, text, indices, box, (float)s.Confidence, kind, value, number, date, currency);
    }

    private static OcrBox Union(IEnumerable<int> words, IReadOnlyList<OcrBox> boxes)
    {
        OcrBox? acc = null;
        foreach (int w in words)
        {
            acc = acc is null ? boxes[w] : OcrBox.Union(acc.Value, boxes[w]);
        }

        return acc ?? default;
    }
}
