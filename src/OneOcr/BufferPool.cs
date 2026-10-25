namespace OneOcr;

/// <summary>
/// A small, private pool of scratch arrays that keeps at most a fixed number of bytes for reuse. Page-sized scratch
/// arrays are large objects; with a shared pool, every distinct size class stays alive and lines of different widths pile up
/// buffers that are each used rarely. Here a request takes the smallest free array that is long enough, sizes are rounded to a
/// coarse grid so one array serves many widths, and an array returned over the budget is dropped. Thread-safe.
/// </summary>
/// <typeparam name="T">The element type; arrays are handed out with unspecified contents.</typeparam>
internal sealed class BufferPool<T>
    where T : struct
{
    // Requests up to this many elements round up to a power of two; larger ones to a half-octave class.
    private const int Grid = 1 << 16;

    private readonly object _gate = new();
    private readonly List<T[]> _free = [];
    private readonly long _maxRetainedBytes;
    private readonly int _elementSize;
    private long _retainedBytes;
    private long _outstandingBytes;

    public BufferPool(long maxRetainedBytes, int elementSize)
    {
        _maxRetainedBytes = Math.Max(0, maxRetainedBytes);
        _elementSize = elementSize;
    }

    /// <summary>The most bytes that were handed out and not yet given back at any one time (for tests and measurements).</summary>
    public long PeakOutstandingBytes { get; private set; }

    /// <summary>Bytes of all arrays this pool has created so far (for tests and measurements).</summary>
    public long CreatedBytes { get; private set; }

    /// <summary>An array of at least <paramref name="length"/> elements (more when a larger free one was reused). Contents are unspecified.</summary>
    public T[] Rent(long length)
    {
        if (length <= 0)
        {
            return [];
        }

        int needed = checked((int)length);
        lock (_gate)
        {
            int best = -1;
            for (int i = 0; i < _free.Count; i++)
            {
                if (_free[i].Length >= needed && (best < 0 || _free[i].Length < _free[best].Length))
                {
                    best = i;
                }
            }

            if (best >= 0)
            {
                T[] hit = _free[best];
                _free[best] = _free[^1];
                _free.RemoveAt(_free.Count - 1);
                _retainedBytes -= (long)hit.Length * _elementSize;
                Rented(hit.Length);
                return hit;
            }
        }

        int rounded = Round(needed);
        lock (_gate)
        {
            Rented(rounded);
            CreatedBytes += (long)rounded * _elementSize;
        }

        // The contents are unspecified by contract, so large arrays skip the zeroing; the pages past what a request uses are never touched.
        return rounded >= Grid ? GC.AllocateUninitializedArray<T>(rounded) : new T[rounded];
    }

    /// <summary>Gives an array back; it is kept only while the pool stays within its byte budget.</summary>
    public void Return(T[]? array)
    {
        if (array is null || array.Length == 0)
        {
            return;
        }

        long bytes = (long)array.Length * _elementSize;
        lock (_gate)
        {
            // An array given back twice would be handed to two renters at once and each would overwrite the other's data;
            // the second return is ignored so a bug in a caller cannot turn into silent corruption.
            foreach (T[] free in _free)
            {
                if (ReferenceEquals(free, array))
                {
                    return;
                }
            }

            _outstandingBytes -= bytes;
            if (_retainedBytes + bytes <= _maxRetainedBytes)
            {
                _free.Add(array);
                _retainedBytes += bytes;
            }
        }
    }

    // Caller holds the gate.
    private void Rented(int elements)
    {
        _outstandingBytes += (long)elements * _elementSize;
        PeakOutstandingBytes = Math.Max(PeakOutstandingBytes, _outstandingBytes);
    }

    private static int Round(int length)
    {
        if (length <= Grid)
        {
            return (int)Math.Min(int.MaxValue, System.Numerics.BitOperations.RoundUpToPowerOf2((uint)Math.Max(length, 16)));
        }

        // Two size classes per doubling (at most 50% slack): lines and pages of different sizes ask for different sizes, and with
        // classes this coarse a returned array fits the next request of its class instead of leaving a near miss behind. The slack
        // costs address space, not memory: arrays this large are handed out unzeroed and only the part a request uses is ever written.
        long step = System.Numerics.BitOperations.RoundUpToPowerOf2((uint)length) / 4;
        long rounded = ((long)length + step - 1) / step * step;
        return (int)Math.Min(rounded, Array.MaxLength);
    }
}
