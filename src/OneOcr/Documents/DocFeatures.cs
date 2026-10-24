using System.Text;

using OneOcr.Unicode;

namespace OneOcr.Documents;

/// <summary>
/// The inputs of the extraction model for the words of one page (docs/spec/docextract-v0.md, "Featurization"). Computed in double precision
/// and stored as the model reads them: character ids, numeric features, position-grid cells and word boxes in units of the median word height.
/// </summary>
internal sealed class DocFeatures
{
    private DocFeatures(int count, int[] chars, float[] feats, int[] bins, float[] geo, double unit)
    {
        Count = count;
        Chars = chars;
        Feats = feats;
        Bins = bins;
        Geo = geo;
        Unit = unit;
    }

    /// <summary>Number of words.</summary>
    public int Count { get; }

    /// <summary>Character ids <c>[Count, MaxChars]</c>, right-padded with 0.</summary>
    public int[] Chars { get; }

    /// <summary>Numeric features <c>[Count, 8]</c>.</summary>
    public float[] Feats { get; }

    /// <summary>Position-grid cells <c>[Count, 4]</c>: x0, y0, x1, y1.</summary>
    public int[] Bins { get; }

    /// <summary>Word boxes in median word heights <c>[Count, 4]</c>.</summary>
    public float[] Geo { get; }

    /// <summary>The median word height in page units (at least 1).</summary>
    public double Unit { get; }

    /// <summary>
    /// Builds the inputs. <paramref name="boxes"/> are <c>[x0, y0, x1, y1]</c> in page units with the origin at the top left. Non-finite
    /// coordinates count as 0 and every coordinate is limited to ±1e9 so that the arithmetic stays finite.
    /// </summary>
    public static DocFeatures Build(DocExtractModel m, IReadOnlyList<string> texts, IReadOnlyList<OcrBox> boxes, double width, double height)
    {
        ArgumentNullException.ThrowIfNull(m);
        ArgumentNullException.ThrowIfNull(texts);
        ArgumentNullException.ThrowIfNull(boxes);
        int n = texts.Count;
        if (boxes.Count != n)
        {
            throw new ArgumentException("Every word needs a box.", nameof(boxes));
        }

        int mc = m.MaxChars;
        var chars = new int[checked(n * mc)];
        for (int i = 0; i < n; i++)
        {
            EncodeWord(m, texts[i], chars.AsSpan(i * mc, mc));
        }

        double w = Math.Max(Clean(width), 1.0);
        double h = Math.Max(Clean(height), 1.0);
        var x0 = new double[n];
        var y0 = new double[n];
        var x1 = new double[n];
        var y1 = new double[n];
        var bh = new double[n];
        for (int i = 0; i < n; i++)
        {
            x0[i] = Clean(boxes[i].Left);
            y0[i] = Clean(boxes[i].Top);
            x1[i] = Clean(boxes[i].Right);
            y1[i] = Clean(boxes[i].Bottom);
            bh[i] = Math.Max(y1[i] - y0[i], 1e-3);
        }

        double unit = 1.0;
        if (n > 0)
        {
            double[] sorted = (double[])bh.Clone();
            Array.Sort(sorted);
            unit = n % 2 == 1 ? sorted[n / 2] : (sorted[(n / 2) - 1] + sorted[n / 2]) / 2.0;
            unit = Math.Max(unit, 1.0);
        }

        var feats = new float[n * 8];
        var bins = new int[n * 4];
        var geo = new float[n * 4];
        int last = m.Bins - 1;
        for (int i = 0; i < n; i++)
        {
            double bw = Math.Max(x1[i] - x0[i], 1e-3);
            double a = bw / bh[i];
            double r = Math.Min(bh[i] / unit, 8.0);
            double[] nb = [x0[i] / w, y0[i] / h, x1[i] / w, y1[i] / h];
            feats[(i * 8) + 0] = (float)nb[0];
            feats[(i * 8) + 1] = (float)nb[1];
            feats[(i * 8) + 2] = (float)nb[2];
            feats[(i * 8) + 3] = (float)nb[3];
            feats[(i * 8) + 4] = (float)(bw / w);
            feats[(i * 8) + 5] = (float)(bh[i] / h);
            feats[(i * 8) + 6] = (float)(r - 1.0);
            feats[(i * 8) + 7] = (float)(a / (1.0 + a));
            for (int q = 0; q < 4; q++)
            {
                bins[(i * 4) + q] = (int)Math.Clamp(Math.Floor(nb[q] * m.Bins), 0, last);
            }

            geo[(i * 4) + 0] = (float)(x0[i] / unit);
            geo[(i * 4) + 1] = (float)(y0[i] / unit);
            geo[(i * 4) + 2] = (float)(x1[i] / unit);
            geo[(i * 4) + 3] = (float)(y1[i] / unit);
        }

        return new DocFeatures(n, chars, feats, bins, geo, unit);
    }

    /// <summary>
    /// Character ids of a word: NFC, each code point mapped through the charset (an unknown one to the unknown id), the first
    /// <c>max_chars</c> kept, the rest of the row 0. Invalid UTF-16 reads as U+FFFD.
    /// </summary>
    internal static void EncodeWord(DocExtractModel m, string text, Span<int> ids)
    {
        ids.Clear();
        // Our own NFC: the BCL's depends on the globalization mode. Unpaired surrogates pass through and read as U+FFFD below.
        string nfc = TextNormalization.Nfc(text);

        int k = 0;
        foreach (Rune rune in nfc.EnumerateRunes())
        {
            if (k == ids.Length)
            {
                break;
            }

            ids[k++] = m.CharIds.TryGetValue(rune.Value, out int id) ? id : m.UnkChar;
        }
    }

    private static double Clean(double v) => double.IsFinite(v) ? Math.Clamp(v, -1e9, 1e9) : 0.0;
}
