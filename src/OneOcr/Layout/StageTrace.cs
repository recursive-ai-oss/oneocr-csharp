namespace OneOcr.Layout;

/// <summary>One thing the page reader saw or dropped, in the coordinates of the page that was read (or of the input image once mapped).</summary>
internal sealed record TraceEntry(string Stage, OcrBox Box, string Text, double Confidence, int Reading)
{
    /// <summary>The box after it was carried back to the input image.</summary>
    public OcrBox Mapped { get; set; } = Box;

    public bool IsMapped { get; set; }
}

/// <summary>
/// A diagnostic record of what the page reader did with each line and word: the lines it found, the words it read, and the words
/// a later rule took away. It is off (<see cref="Sink"/> null) in normal use and exists so a benchmark can say which stage lost a
/// word; it never changes a result.
/// </summary>
internal static class StageTrace
{
    private static readonly object Gate = new();
    private static int _reading;

    /// <summary>Where entries go; null turns tracing off.</summary>
    internal static List<TraceEntry>? Sink { get; set; }

    internal static bool On => Sink is not null;

    internal static void Add(string stage, OcrBox box, string text, double confidence)
    {
        lock (Gate)
        {
            Sink?.Add(new TraceEntry(stage, box, text, confidence, _reading));
        }
    }

    /// <summary>Starts a new reading of the page (the first one, or one turned by the angle search).</summary>
    internal static int NextReading()
    {
        lock (Gate)
        {
            return ++_reading;
        }
    }

    /// <summary>Carries the entries that are not mapped yet back through <paramref name="frame"/>.</summary>
    internal static void MapBack(PageFrame frame)
    {
        lock (Gate)
        {
            if (Sink is null)
            {
                return;
            }

            foreach (TraceEntry e in Sink)
            {
                if (!e.IsMapped)
                {
                    OcrPoint a = frame.ToInput(e.Box.Left, e.Box.Top), b = frame.ToInput(e.Box.Right, e.Box.Top);
                    OcrPoint c = frame.ToInput(e.Box.Right, e.Box.Bottom), d = frame.ToInput(e.Box.Left, e.Box.Bottom);
                    e.Mapped = new OcrBox(
                        Math.Min(Math.Min(a.X, b.X), Math.Min(c.X, d.X)), Math.Min(Math.Min(a.Y, b.Y), Math.Min(c.Y, d.Y)),
                        Math.Max(Math.Max(a.X, b.X), Math.Max(c.X, d.X)), Math.Max(Math.Max(a.Y, b.Y), Math.Max(c.Y, d.Y)));
                    e.IsMapped = true;
                }
            }
        }
    }
}
