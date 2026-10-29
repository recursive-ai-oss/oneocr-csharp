using System.Diagnostics;

namespace OneOcr;

/// <summary>
/// Optional per-stage wall-clock totals for the page reader, read by the benchmark tool. Off unless a tool switches it on, and then
/// costs one timestamp per stage; it never changes what is computed.
/// </summary>
internal static class StageClock
{
    private static readonly object Gate = new();
    private static readonly Dictionary<string, double> Totals = new(StringComparer.Ordinal);

    /// <summary>Whether stages are being timed.</summary>
    public static bool Enabled { get; set; }

    /// <summary>The time now, or 0 when timing is off.</summary>
    public static long Start() => Enabled ? Stopwatch.GetTimestamp() : 0;

    /// <summary>Adds the time since <paramref name="start"/> to <paramref name="stage"/>.</summary>
    public static void Stop(string stage, long start)
    {
        if (start == 0)
        {
            return;
        }

        double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        lock (Gate)
        {
            Totals[stage] = Totals.GetValueOrDefault(stage) + ms;
        }
    }

    /// <summary>Returns the totals (milliseconds per stage) and clears them.</summary>
    public static Dictionary<string, double> Take()
    {
        lock (Gate)
        {
            var copy = new Dictionary<string, double>(Totals, StringComparer.Ordinal);
            Totals.Clear();
            return copy;
        }
    }
}
