namespace OneOcr.Documents;

/// <summary>
/// Decides between a due date (BT-9) and a delivery date (BT-72) by the label printed next to the value. The tagging model sees both as
/// "a date after a label" and, on an invoice that prints only one of them, can tag a due date as the delivery date. The label is on the
/// page, in the same row to the left of the value, so it settles the question: a value whose label says "due" (or its translation in the
/// labels the model was trained on) is BT-9, one whose label says "delivery" is BT-72, and anything else keeps the model's tag.
/// </summary>
internal static class DateLabel
{
    private const int MaxLabelWords = 4;

    private static readonly string[] s_due = ["due", "fällig", "faellig", "échéance", "echeance", "vencimiento", "scadenza", "vervaldatum", "vervaldag"];

    private static readonly string[] s_delivery = ["deliver", "liefer", "livraison", "entrega", "consegna", "leverdatum", "leverings"];

    /// <summary>The class index a date span should carry: <paramref name="key"/>, or the other of BT-9 and BT-72 when the printed label says so.</summary>
    public static int Retag(DocSchema schema, int key, IReadOnlyList<int> words, IReadOnlyList<string> texts, IReadOnlyList<OcrBox> boxes)
    {
        string id = schema.Classes[key];
        if (id is not ("BT-9" or "BT-72"))
        {
            return key;
        }

        bool? due = LabelSays(words, texts, boxes);
        if (due is null)
        {
            return key;
        }

        string want = due.Value ? "BT-9" : "BT-72";
        if (want == id)
        {
            return key;
        }

        int other = schema.Classes.ToList().IndexOf(want);
        return other >= 0 ? other : key;
    }

    // True: the label says due. False: it says delivery. Null: no label of either kind in the row to the left.
    private static bool? LabelSays(IReadOnlyList<int> words, IReadOnlyList<string> texts, IReadOnlyList<OcrBox> boxes)
    {
        int first = words[0];
        OcrBox v = boxes[first];
        var label = new List<(double Right, string Text)>();
        for (int i = 0; i < texts.Count; i++)
        {
            OcrBox b = boxes[i];
            double overlap = Math.Min(b.Bottom, v.Bottom) - Math.Max(b.Top, v.Top);
            if (i == first || b.Right > v.Left + 1e-6 || overlap < 0.5 * Math.Min(b.Height, v.Height))
            {
                continue;
            }

            label.Add((b.Right, texts[i]));
        }

        label.Sort(static (a, b) => b.Right.CompareTo(a.Right));
        foreach ((_, string text) in label.Take(MaxLabelWords))
        {
            string t = text.Trim().TrimEnd(':', '.').ToLowerInvariant();
            if (s_delivery.Any(p => t.StartsWith(p, StringComparison.Ordinal)))
            {
                return false;
            }

            if (s_due.Any(p => t == p || (p.Length > 3 && t.StartsWith(p, StringComparison.Ordinal))))
            {
                return true;
            }
        }

        return null;
    }
}
