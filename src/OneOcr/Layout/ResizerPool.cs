using OneImage;

namespace OneOcr.Layout;

/// <summary>
/// Hands out reusable single-threaded resizers so that each worker thread scales its lines with warm scratch buffers and no
/// per-line allocation. A resizer is rented for one line and given back; at most one per processor is kept.
/// </summary>
internal sealed class ResizerPool
{
    private readonly Stack<PixelResizer> _free = new();
    private readonly int _keep = Math.Max(4, Environment.ProcessorCount);

    public PixelResizer Rent()
    {
        lock (_free)
        {
            if (_free.TryPop(out PixelResizer? resizer))
            {
                return resizer;
            }
        }

        // Degree 1: the line is already the unit of parallelism.
        return new PixelResizer(maxDegree: 1);
    }

    public void Return(PixelResizer resizer)
    {
        lock (_free)
        {
            if (_free.Count < _keep)
            {
                _free.Push(resizer);
                return;
            }
        }

        resizer.Dispose();
    }
}
