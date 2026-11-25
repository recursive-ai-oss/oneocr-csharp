using System.Globalization;
using System.Text;

namespace OneOcr.Pdf;

/// <summary>
/// Marks the visible text shows of a page whose character map cannot be trusted with an empty <c>/ActualText</c> (ISO 32000-1
/// section 14.9.4), so that text extraction and copy skip them and return the words of the recognized layer instead.
/// </summary>
/// <remarks>
/// <para>
/// Each text-showing operator (<c>Tj</c>, <c>TJ</c>, <c>'</c>, <c>"</c>) that paints (render mode other than 3) and starts inside one of
/// the given boxes is wrapped as <c>/Span &lt;&lt;/ActualText ()&gt;&gt; BDC … EMC</c>. Those operators only open and close a marked-content
/// sequence, which paints nothing and changes no graphics state, and every other byte is copied as it was, so the page looks exactly as
/// before. The sequence is exactly one operator (with its operands), so it nests correctly with <c>BT</c>/<c>ET</c>, <c>q</c>/<c>Q</c> and
/// any marked content already on the page.
/// </para>
/// <para>
/// The position of a show is the origin of the current text matrix mapped through the current transformation matrix. Text-matrix
/// advances from earlier shows are not tracked (that needs the fonts' widths), so a show that continues a line takes the line's start;
/// the boxes are whole lines, so the start is inside the box too. The box is the line's own extent with a margin of a tenth of its
/// smaller side, so a neighbouring line's origin is outside it.
/// </para>
/// <para>
/// The input is untrusted: one pass, no recursion, allocation proportional to the content, and any ambiguity (an inline image that
/// cannot be delimited, a graphics state nested deeper than <see cref="InvisibleTextRemover.MaxStateDepth"/>) gives up with null.
/// Form XObjects are not entered.
/// </para>
/// </remarks>
internal static class ActualTextMarker
{
    private const string Open = "/Span <</ActualText ()>> BDC ";
    private const string Close = " EMC";

    /// <summary>The bytes each wrapped show adds.</summary>
    public static int AddedBytesPerShow => Open.Length + Close.Length;

    /// <summary>Wraps the shows that start inside a box; returns the new content or null when nothing was marked or the content could not be handled safely.</summary>
    /// <param name="content">The decoded content stream (several streams joined by a line feed), in default user space.</param>
    /// <param name="boxes">Rectangles (left, bottom, right, top) in user space, already widened by nothing: the margin is added here.</param>
    /// <param name="marked">The number of shows wrapped.</param>
    public static byte[]? Mark(ReadOnlySpan<byte> content, IReadOnlyList<(double Left, double Bottom, double Right, double Top)> boxes, out int marked)
    {
        marked = 0;
        if (boxes.Count == 0)
        {
            return null;
        }

        var hit = new (double L, double B, double R, double T)[boxes.Count];
        for (int i = 0; i < hit.Length; i++)
        {
            (double l, double b, double r, double t) = boxes[i];
            double tol = 0.1 * Math.Min(r - l, t - b);
            hit[i] = (l - tol, b - tol, r + tol, t + tol);
        }

        var cuts = new List<(int Start, int End)>();
        var saved = new Stack<(Matrix Ctm, int Mode, double Leading)>();
        Matrix ctm = Matrix.Identity, tm = Matrix.Identity, tlm = Matrix.Identity;
        int mode = 0;
        double leading = 0;
        int pos = 0, operandStart = -1;
        var operands = new List<(int Start, int End)>(8);
        while (InvisibleTextRemover.Next(content, ref pos, out InvisibleTextRemover.Token t))
        {
            if (t.Kind != InvisibleTextRemover.TokenKind.Operator)
            {
                if (operandStart < 0)
                {
                    operandStart = t.Start;
                }

                if (operands.Count < 8)
                {
                    operands.Add((t.Start, t.End));
                }

                continue;
            }

            ReadOnlySpan<byte> op = content[t.Start..t.End];
            int from = operandStart < 0 ? t.Start : operandStart;
            if (op.SequenceEqual("q"u8))
            {
                if (saved.Count >= InvisibleTextRemover.MaxStateDepth)
                {
                    return null;
                }

                saved.Push((ctm, mode, leading));
            }
            else if (op.SequenceEqual("Q"u8))
            {
                if (saved.Count > 0)
                {
                    (ctm, mode, leading) = saved.Pop();
                }
            }
            else if (op.SequenceEqual("cm"u8))
            {
                if (Numbers(content, operands, 6, out double[] m))
                {
                    ctm = new Matrix(m[0], m[1], m[2], m[3], m[4], m[5]).Then(ctm);
                }
            }
            else if (op.SequenceEqual("BT"u8))
            {
                tm = tlm = Matrix.Identity;
            }
            else if (op.SequenceEqual("Tm"u8))
            {
                if (Numbers(content, operands, 6, out double[] m))
                {
                    tm = tlm = new Matrix(m[0], m[1], m[2], m[3], m[4], m[5]);
                }
            }
            else if (op.SequenceEqual("Td"u8) || op.SequenceEqual("TD"u8))
            {
                if (Numbers(content, operands, 2, out double[] m))
                {
                    if (op[1] == (byte)'D')
                    {
                        leading = -m[1];
                    }

                    tm = tlm = new Matrix(1, 0, 0, 1, m[0], m[1]).Then(tlm);
                }
            }
            else if (op.SequenceEqual("TL"u8))
            {
                if (Numbers(content, operands, 1, out double[] m))
                {
                    leading = m[0];
                }
            }
            else if (op.SequenceEqual("T*"u8))
            {
                tm = tlm = new Matrix(1, 0, 0, 1, 0, -leading).Then(tlm);
            }
            else if (op.SequenceEqual("Tr"u8))
            {
                if (Numbers(content, operands, 1, out double[] m) && m[0] is >= 0 and <= 7)
                {
                    mode = (int)m[0];
                }
            }
            else if (op.SequenceEqual("Tj"u8) || op.SequenceEqual("TJ"u8) || op.SequenceEqual("'"u8) || op.SequenceEqual("\""u8))
            {
                if (op[0] is (byte)'\'' or (byte)'"')
                {
                    tm = tlm = new Matrix(1, 0, 0, 1, 0, -leading).Then(tlm);
                }

                if (mode != 3 && operandStart >= 0 && Starts(tm, ctm, hit))
                {
                    cuts.Add((from, t.End));
                }
            }
            else if (op.SequenceEqual("BI"u8))
            {
                if (!InvisibleTextRemover.SkipInlineImage(content, ref pos))
                {
                    return null;
                }
            }

            operandStart = -1;
            operands.Clear();
        }

        if (cuts.Count == 0)
        {
            return null;
        }

        var result = new MemoryStream(content.Length + (cuts.Count * AddedBytesPerShow));
        int copied = 0;
        foreach ((int start, int end) in cuts)
        {
            result.Write(content[copied..start]);
            result.Write(Encoding.ASCII.GetBytes(Open));
            result.Write(content[start..end]);
            result.Write(Encoding.ASCII.GetBytes(Close));
            copied = end;
        }

        result.Write(content[copied..]);
        marked = cuts.Count;
        return result.ToArray();
    }

    private static bool Starts(Matrix tm, Matrix ctm, (double L, double B, double R, double T)[] boxes)
    {
        double x = (tm.E * ctm.A) + (tm.F * ctm.C) + ctm.E;
        double y = (tm.E * ctm.B) + (tm.F * ctm.D) + ctm.F;
        if (!double.IsFinite(x) || !double.IsFinite(y))
        {
            return false;
        }

        foreach ((double l, double b, double r, double t) in boxes)
        {
            if (x >= l && x <= r && y >= b && y <= t)
            {
                return true;
            }
        }

        return false;
    }

    // The last `count` operands as numbers; false when there are fewer, or one is not a plain number.
    private static bool Numbers(ReadOnlySpan<byte> content, List<(int Start, int End)> operands, int count, out double[] values)
    {
        values = new double[count];
        if (operands.Count != count)
        {
            return false;
        }

        for (int i = 0; i < count; i++)
        {
            ReadOnlySpan<byte> s = content[operands[i].Start..operands[i].End];
            if (!InvisibleTextRemover.IsNumber(s) || !double.TryParse(Encoding.ASCII.GetString(s), NumberStyles.Float, CultureInfo.InvariantCulture, out values[i]) || !double.IsFinite(values[i]))
            {
                return false;
            }
        }

        return true;
    }

    private readonly record struct Matrix(double A, double B, double C, double D, double E, double F)
    {
        public static Matrix Identity => new(1, 0, 0, 1, 0, 0);

        /// <summary>This transform first, then <paramref name="o"/> (row-vector convention, as the PDF operators are written).</summary>
        public Matrix Then(Matrix o) => new(
            (A * o.A) + (B * o.C), (A * o.B) + (B * o.D),
            (C * o.A) + (D * o.C), (C * o.B) + (D * o.D),
            (E * o.A) + (F * o.C) + o.E, (E * o.B) + (F * o.D) + o.F);
    }
}
