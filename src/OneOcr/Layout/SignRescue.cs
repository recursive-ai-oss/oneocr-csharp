namespace OneOcr.Layout;

/// <summary>
/// Finds the sign of an amount that lies just outside a line's crop. A minus beside a number is a short, thin stroke; the detector's
/// box often ends inside it or just before it, and the recognizer does not read ink in the margin of its crop, so the line comes
/// back as the bare number. A line that ends (or starts) with a number is therefore looked at again: if such a stroke lies past
/// the end of the ink, the line is read once more with the stroke inside the crop, and the second reading replaces the first only
/// when it differs from it by that one sign. Nothing else in the line can change, and a sign is never added unless the
/// recognizer reads it from the pixels.
/// </summary>
internal static class SignRescue
{
    /// <summary>Whether lines ending or starting with a number are read again with a minus beside them inside the crop.</summary>
    internal static bool Enabled = true;

    /// <summary>How far past a line's end a sign mark is looked for, in band heights, gap included.</summary>
    internal static double MarkReach = 0.8;

    /// <summary>The widest blank stretch between a line's end and its sign mark, in band heights.</summary>
    internal static double MarkGap = 0.5;

    /// <summary>The narrowest a sign mark may be, in band heights (and never under 3 pixels).</summary>
    internal static double MarkMinWidth = 0.08;

    /// <summary>The widest a sign mark may be, in band heights.</summary>
    internal static double MarkMaxWidth = 0.5;

    /// <summary>The most a minus may be tall, in band heights.</summary>
    internal static double MinusThickness = 0.25;

    /// <summary>A minus' vertical centre lies between these shares of the band height, counted from the top.</summary>
    internal static double MinusMiddleFrom = 0.25, MinusMiddleTo = 0.75;

    /// <summary>A minus is at least this many times longer than it is thick (a dot is not).</summary>
    internal static double MinusLength = 1.5;

    /// <summary>
    /// How far toward the page background the ink threshold moves when the mark is looked for in the scaled crop, as a share of the
    /// distance from the threshold to the crop's median gray: scaling a crop down thins and lightens a stroke of two or three pixels.
    /// </summary>
    internal static double ScaledSlack = 0.3;


    /// <summary>The fewest digits a word needs to count as a number.</summary>
    private const int MinDigits = 2;

    /// <summary>Whether a line's first word is a number that does not already start with a sign or bracket.</summary>
    public static bool WantsLeadingSign(IReadOnlyList<OcrWord> words) =>
        words.Count > 0 && IsNumber(words[0].Text) && !IsMinus(words[0].Text[0]) && words[0].Text[0] != '(';

    /// <summary>Whether a line's last word is a number that does not already end with a sign or bracket.</summary>
    public static bool WantsTrailingSign(IReadOnlyList<OcrWord> words)
    {
        if (words.Count == 0)
        {
            return false;
        }

        string t = words[^1].Text;
        return IsNumber(t) && !IsMinus(t[^1]) && t[^1] != ')';
    }

    private static bool IsNumber(string text)
    {
        int digits = 0;
        foreach (char c in text)
        {
            if (c is >= '0' and <= '9')
            {
                digits++;
            }
        }

        return digits >= MinDigits;
    }

    private static bool IsMinus(char c) => c is '-' or '\u2212' or '\u2013' or '\u2014';

    /// <summary>
    /// Whether <paramref name="after"/> is <paramref name="before"/> with a minus added at the start (when <paramref name="leading"/>)
    /// or at the end (when <paramref name="trailing"/>) or both, spaces ignored.
    /// </summary>
    public static bool DiffersByASign(IReadOnlyList<OcrWord> before, IReadOnlyList<OcrWord> after, bool leading, bool trailing)
    {
        string a = Join(before), b = Join(after);
        return (leading && b.Length == a.Length + 1 && IsMinus(b[0]) && b.AsSpan(1).SequenceEqual(a))
            || (trailing && b.Length == a.Length + 1 && IsMinus(b[^1]) && b.AsSpan(0, a.Length).SequenceEqual(a))
            || (leading && trailing && b.Length == a.Length + 2 && IsMinus(b[0]) && IsMinus(b[^1]) && b.AsSpan(1, a.Length).SequenceEqual(a));
    }

    private static string Join(IReadOnlyList<OcrWord> words)
    {
        var sb = new System.Text.StringBuilder();
        foreach (OcrWord w in words)
        {
            sb.Append(w.Text);
        }

        return sb.ToString();
    }

    /// <summary>
    /// Looks in a scaled crop (<paramref name="pixels"/>, <paramref name="cropWidth"/> by the model's input height) for a minus
    /// beyond the ink's end at column <paramref name="end"/>, going <paramref name="direction"/> (+1 right, -1 left) over the rows
    /// <paramref name="rowFrom"/> to <paramref name="rowTo"/> of the ink band. Returns how many columns past <paramref name="end"/>
    /// the mark's far edge lies, or 0 when there is none.
    /// </summary>
    public static int FindBeyond(ReadOnlySpan<byte> pixels, int cropWidth, int rowFrom, int rowTo, int end, int direction, int threshold)
    {
        int median = Median(pixels);
        int slackThreshold = threshold + (int)(ScaledSlack * (median - threshold));
        if (median <= threshold || slackThreshold >= median)
        {
            return 0;
        }

        int? far = FindMark(pixels, cropWidth, rowFrom, rowTo, end, direction, direction > 0 ? cropWidth : -1, slackThreshold);
        return far is int f ? Math.Abs(f - end) : 0;
    }

    private static int Median(ReadOnlySpan<byte> pixels)
    {
        Span<int> histogram = stackalloc int[256];
        histogram.Clear();
        foreach (byte v in pixels)
        {
            histogram[v]++;
        }

        int seen = 0, median = 0;
        for (; median < 255 && seen + histogram[median] <= pixels.Length / 2; median++)
        {
            seen += histogram[median];
        }

        return median;
    }

    // Past the end of a line (direction +1 or -1 from column `end`, never beyond `limit`), the first runs of inked columns are
    // looked at: a run that is a minus, a thin unbroken stroke at mid height with a blank stretch of at most MarkGap before it, is the
    // mark. A full stop or comma sits low, the dots of a leader are broken and square, the edge of a letter is narrow or tall, so
    // none of them passes. Returns the far column of the mark, or null. Dark ink on a light ground only.
    private static int? FindMark(ReadOnlySpan<byte> page, int width, int rowFrom, int rowTo, int end, int direction, int limit, int threshold)
    {
        int bandHeight = rowTo - rowFrom + 1;
        if (bandHeight < 1)
        {
            return null;
        }

        int window = (int)Math.Ceiling(MarkReach * bandHeight);
        int maxGap = (int)Math.Round(MarkGap * bandHeight);
        int to = direction > 0 ? Math.Min(limit - 1, end + window) : Math.Max(limit + 1, end - window);
        int count = (to - end) * direction;
        if (count < 1)
        {
            return null;
        }

        // Runs of inked columns, as offsets from the line's end (1 is the column beside it), nearest first.
        Span<int> runStart = stackalloc int[3], runEnd = stackalloc int[3];
        int runs = 0;
        bool inRun = false;
        for (int k = 1; k <= count && runs < 3; k++)
        {
            int x = end + (k * direction);
            bool ink = false;
            for (int y = rowFrom; y <= rowTo && !ink; y++)
            {
                ink = page[(y * width) + x] <= threshold;
            }

            if (ink && !inRun)
            {
                runStart[runs] = k;
                inRun = true;
            }
            else if (!ink && inRun)
            {
                runEnd[runs++] = k - 1;
                inRun = false;
            }
        }

        // A run that reaches the end of the window (the crop's edge, for a mark in its margin) is judged by the part that can be seen.
        if (inRun && runs < 3)
        {
            runEnd[runs++] = count;
        }

        for (int r = 0; r < runs; r++)
        {
            bool touching = runStart[r] == 1;
            int columns = runEnd[r] - runStart[r] + 1;
            if (runStart[r] - 1 - (r > 0 ? runEnd[r - 1] : 0) > maxGap || columns > MarkMaxWidth * bandHeight)
            {
                return null;
            }

            if (columns >= Math.Max(3, (int)Math.Ceiling(MarkMinWidth * bandHeight)))
            {
                int top = int.MaxValue, bottom = -1;
                for (int y = rowFrom; y <= rowTo; y++)
                {
                    for (int k = runStart[r]; k <= runEnd[r]; k++)
                    {
                        if (page[(y * width) + end + (k * direction)] <= threshold)
                        {
                            top = Math.Min(top, y - rowFrom);
                            bottom = Math.Max(bottom, y - rowFrom);
                            break;
                        }
                    }
                }

                int tall = bottom - top + 1;
                double middle = (top + bottom + 1) / 2.0 / bandHeight;
                bool thin = tall <= Math.Max(2, MinusThickness * bandHeight) && middle >= MinusMiddleFrom && middle <= MinusMiddleTo;

                // A stroke the box cuts in two (touching the ink's end) may be a part of a minus, so its length is not judged.
                if (thin && columns >= Math.Max(3, (touching ? 0 : 1) * MinusLength * tall))
                {
                    return end + (runEnd[r] * direction);
                }
            }

            // Only the edge of the line's last letter (a narrow run beside it) may be passed over.
            if (!(touching && columns <= 2))
            {
                return null;
            }
        }

        return null;
    }
}
