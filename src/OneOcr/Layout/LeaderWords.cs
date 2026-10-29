namespace OneOcr.Layout;

/// <summary>
/// Puts the words read before and after a dot leader together with the leader itself (<c>docs/spec/oneocr-page-v1.md</c>, "Dot leaders").
/// The leader is a word of full stops, as many as the pixels show. A page number that follows the leader closely (as the number of a
/// table of contents does) is part of the same word, the way a text layer has it: <c>.........12</c>.
/// </summary>
internal static class LeaderWords
{
    /// <summary>A word that starts within this many pitches of the last dot is joined to the leader.</summary>
    internal const double GlueGapPitches = 2.5;

    /// <summary>The confidence reported for a leader that stands alone; its dots were counted from the pixels and no network read them. It is above the noise filter's bar for the short lines a bare run of dots makes.</summary>
    internal const float Confidence = 0.995f;

    public static List<OcrWord> Join(List<OcrWord> before, List<OcrWord> after, Leader leader, bool includeChars)
    {
        // Full stops right of the run are dots of the leader that the run did not take in (the last one or two, half hidden by the page
        // number's box); they are not a word.
        after.RemoveAll(w => w.Text.All(c => c == '.'));
        var words = new List<OcrWord>(before.Count + after.Count + 1);
        words.AddRange(before);
        if (leader.Spaced)
        {
            return JoinSpaced(after, leader, includeChars, words);
        }

        string dots = new('.', leader.Count);
        var box = new OcrBox(leader.From, leader.Top, leader.To, leader.Bottom);
        IReadOnlyList<OcrChar> dotChars = [];
        if (includeChars)
        {
            var list = new List<OcrChar>(leader.Count);
            double half = Math.Max(1.0, leader.Pitch / 2);
            foreach (int center in leader.Centers)
            {
                list.Add(new OcrChar(".", new OcrBox(Math.Max(leader.From, center - half), leader.Top, Math.Min(leader.To, center + half), leader.Bottom), 1f));
            }

            dotChars = list;
        }

        int next = 0;
        if (after.Count > 0 && after[0].Box.Left - leader.To <= GlueGapPitches * leader.Pitch)
        {
            OcrWord number = after[0];
            var union = new OcrBox(box.Left, Math.Min(box.Top, number.Box.Top), Math.Max(box.Right, number.Box.Right), Math.Max(box.Bottom, number.Box.Bottom));
            IReadOnlyList<OcrChar> chars = includeChars ? [.. dotChars, .. number.Chars] : [];
            words.Add(new OcrWord(dots + number.Text, union, number.Confidence, number.RawConfidence, OcrSource.Ocr, chars));
            next = 1;
        }
        else
        {
            words.Add(new OcrWord(dots, box, Confidence, 1f, OcrSource.Ocr, dotChars));
        }

        for (int i = next; i < after.Count; i++)
        {
            words.Add(after[i]);
        }

        return words;
    }

    // Stops set one by one are words of their own, and the page number after them is a word of its own, as in a text layer.
    private static List<OcrWord> JoinSpaced(List<OcrWord> after, Leader leader, bool includeChars, List<OcrWord> words)
    {
        double half = Math.Max(1.0, leader.Pitch / 4);
        foreach (int center in leader.Centers)
        {
            var box = new OcrBox(Math.Max(leader.From, center - half), leader.Top, Math.Min(leader.To, center + half), leader.Bottom);
            IReadOnlyList<OcrChar> chars = includeChars ? [new OcrChar(".", box, 1f)] : [];
            words.Add(new OcrWord(".", box, Confidence, 1f, OcrSource.Ocr, chars));
        }

        words.AddRange(after);
        return words;
    }
}
