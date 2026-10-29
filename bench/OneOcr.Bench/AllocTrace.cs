using System.Diagnostics.Tracing;

/// <summary>
/// Samples the runtime's allocation ticks (one per ~100 KB allocated, large objects always) and totals them by type, kind and object size,
/// to show where a run's allocations come from. Also counts collections by generation.
/// </summary>
internal sealed class AllocTrace : EventListener
{
    private readonly Dictionary<string, (long Bytes, int Ticks)> _totals = new(StringComparer.Ordinal);
    private readonly int[] _collections = new int[3];
    private readonly object _gate = new();

    protected override void OnEventSourceCreated(EventSource source)
    {
        if (source.Name == "Microsoft-Windows-DotNETRuntime")
        {
            EnableEvents(source, EventLevel.Verbose, (EventKeywords)1);
        }
    }

    protected override void OnEventWritten(EventWrittenEventArgs e)
    {
        if (e.Payload is null)
        {
            return;
        }

        if (e.EventName?.StartsWith("GCAllocationTick", StringComparison.Ordinal) == true)
        {
            long amount = Convert.ToInt64(e.Payload[e.PayloadNames!.IndexOf("AllocationAmount64")], System.Globalization.CultureInfo.InvariantCulture);
            uint kind = Convert.ToUInt32(e.Payload[e.PayloadNames.IndexOf("AllocationKind")], System.Globalization.CultureInfo.InvariantCulture);
            string type = (string?)e.Payload[e.PayloadNames.IndexOf("TypeName")] ?? "?";
            long size = e.PayloadNames.IndexOf("ObjectSize") is int i and >= 0 ? Convert.ToInt64(e.Payload[i], System.Globalization.CultureInfo.InvariantCulture) : 0;
            string bucket = size >= 85000 ? (size >= (1 << 20) ? ">=1MB" : ">=85KB") : "small";
            string key = $"{type} {(kind == 1 ? "LOH" : kind == 2 ? "POH" : "SOH")} {(Environment.GetEnvironmentVariable("ALLOC_SIZES") is null || size < 85000 ? bucket : (size >> 16 << 16).ToString(System.Globalization.CultureInfo.InvariantCulture))}";
            lock (_gate)
            {
                _totals.TryGetValue(key, out var t);
                _totals[key] = (t.Bytes + amount, t.Ticks + 1);
            }
        }
        else if (e.EventName?.StartsWith("GCStart", StringComparison.Ordinal) == true)
        {
            int gen = Convert.ToInt32(e.Payload[e.PayloadNames!.IndexOf("Depth")], System.Globalization.CultureInfo.InvariantCulture);
            lock (_gate)
            {
                _collections[Math.Clamp(gen, 0, 2)]++;
            }
        }
    }

    public void Print()
    {
        lock (_gate)
        {
            Console.WriteLine($"  collections gen0 {_collections[0]} gen1 {_collections[1]} gen2 {_collections[2]}");
            foreach (var (key, v) in _totals.OrderByDescending(p => p.Value.Bytes).Take(25))
            {
                Console.WriteLine($"  alloc {v.Bytes / 1048576.0,8:F1} MB  {key}");
            }
        }
    }
}
