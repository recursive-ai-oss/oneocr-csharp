using System.Buffers;

namespace OneOcr.Layout;

/// <summary>
/// The page's ink counted on a coarse grid (about 1200 cells on the long side at most): the input of skew estimation and of the
/// sideways-text check. A cell holds the number of ink pixels in its block.
/// </summary>
internal sealed class InkGrid : IDisposable
{
    private ushort[]? _rented;

    // `counts` comes from the shared array pool: it can be longer than Width * Height, and Dispose gives it back.
    private InkGrid(int width, int height, int factor, ushort[] counts)
    {
        Width = width;
        Height = height;
        Factor = factor;
        Counts = counts;
        _rented = counts;
    }

    /// <summary>Gives the counts back to the pool; the grid must not be used afterwards.</summary>
    public void Dispose()
    {
        if (_rented is not null)
        {
            ArrayPool<ushort>.Shared.Return(_rented);
            _rented = null;
        }
    }

    private static ushort[] Zeroed(int length)
    {
        ushort[] a = ArrayPool<ushort>.Shared.Rent(length);
        Array.Clear(a, 0, length);
        return a;
    }

    public int Width { get; }

    public int Height { get; }

    /// <summary>Pixels per cell side.</summary>
    public int Factor { get; }

    /// <summary>Row-major ink pixel counts, at most <c>Factor * Factor</c> each; only the first <c>Width * Height</c> entries belong to the grid.</summary>
    public ushort[] Counts { get; }

    private const int LongSide = 1200;

    /// <summary>Counts the ink of a page (ink is the dark class, or the bright class when <paramref name="inkIsBright"/>).</summary>
    public static InkGrid Build(ReadOnlySpan<byte> gray, int width, int height, int threshold, bool inkIsBright)
    {
        int factor = Math.Max(1, (Math.Max(width, height) + LongSide - 1) / LongSide);
        int gw = (width + factor - 1) / factor, gh = (height + factor - 1) / factor;
        ushort[] counts = Zeroed(checked(gw * gh));
        if (threshold < 0)
        {
            return new InkGrid(gw, gh, factor, counts);
        }

        // Each grid cell is counted over its own run of pixels, so no pixel pays for a division or a read-modify-write of its cell.
        for (int y = 0; y < height; y++)
        {
            ReadOnlySpan<byte> row = gray.Slice(y * width, width);
            int rowBase = (y / factor) * gw;
            for (int cell = 0, x0 = 0; x0 < width; cell++, x0 += factor)
            {
                ReadOnlySpan<byte> run = row.Slice(x0, Math.Min(factor, width - x0));
                int n = 0;
                if (inkIsBright)
                {
                    foreach (byte v in run)
                    {
                        n += v > threshold ? 1 : 0;
                    }
                }
                else
                {
                    foreach (byte v in run)
                    {
                        n += v <= threshold ? 1 : 0;
                    }
                }

                counts[rowBase + cell] += (ushort)n;
            }
        }

        return new InkGrid(gw, gh, factor, counts);
    }

    /// <summary>The grid with blocks of <paramref name="by"/> by <paramref name="by"/> cells added up: a coarser picture for a first, cheap search.</summary>
    public InkGrid Shrink(int by)
    {
        int gw = (Width + by - 1) / by, gh = (Height + by - 1) / by;
        ushort[] counts = Zeroed(checked(gw * gh));
        for (int y = 0; y < Height; y++)
        {
            int to = (y / by) * gw;
            ReadOnlySpan<ushort> row = Counts.AsSpan(y * Width, Width);
            for (int cell = 0, x0 = 0; x0 < Width; cell++, x0 += by)
            {
                int sum = 0;
                foreach (ushort c in row.Slice(x0, Math.Min(by, Width - x0)))
                {
                    sum += c;
                }

                counts[to + cell] += (ushort)sum;
            }
        }

        return new InkGrid(gw, gh, Factor * by, counts);
    }

    /// <summary>The grid of the same page turned a quarter turn clockwise: cell (x, y) goes to (height - 1 - y, x).</summary>
    public InkGrid TurnClockwise()
    {
        ushort[] turned = Zeroed(Width * Height);
        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                turned[(x * Height) + (Height - 1 - y)] = Counts[(y * Width) + x];
            }
        }

        return new InkGrid(Height, Width, Factor, turned);
    }
}

/// <summary>The outcome of a skew search.</summary>
/// <param name="Slope">The text's slope times 1024: lines descend to the right by this many units per 1024 across. Zero when the evidence is too weak.</param>
/// <param name="Sharpness">The profile sharpness (sum of squared row sums) at the best slope; 0 for a page without text.</param>
/// <param name="Gain">Sharpness at the best slope divided by sharpness at slope 0.</param>
/// <param name="Contrast">
/// Sharpness at the best slope relative to what the same ink spread evenly over the profile would give (<c>bins * sum p^2 / (sum p)^2</c>):
/// 1 for no structure, larger for ruled lines of text. Unlike <paramref name="Sharpness"/> it can be compared between a page and the
/// same page turned a quarter, whose profiles have different lengths.
/// </param>
internal readonly record struct SkewEstimate(int Slope, long Sharpness, double Gain, double Contrast);

/// <summary>
/// Estimates the tilt of text lines by the projection-profile method (Postl 1986, "Detection of linear oblique structures and
/// skew in digitized documents"; Baird 1987, "The skew angle of printed documents"): the ink is projected onto the axis
/// perpendicular to a candidate line direction, and the candidate whose profile is sharpest (largest sum of squares: text lines
/// pile ink into few bins, a wrong direction smears it) is the text direction. Candidates are slopes k/1024 so that projection
/// is integer arithmetic and the answer cannot differ between machines. A coarse search over the whole range on a shrunken grid is
/// followed by a fine search around its result on the full grid.
/// </summary>
internal static class SkewEstimator
{
    /// <summary>Slopes are in units of 1/Scale.</summary>
    public const int Scale = 1024;

    /// <summary>Largest slope searched: tan(15.3°).</summary>
    public const int MaxSlope = 280;

    private const int CoarseShrink = 4;
    private const int CoarseStep = 2;
    private const int FineReach = 6;

    // Cells denser than this share of their pixels are solid areas (pictures, rules, scan borders), not text strokes.
    private const double SolidShare = 0.6;

    // The outer share of the grid ignored on every side: scanner shadows and page borders.
    private const double EdgeShare = 0.02;

    public static SkewEstimate Estimate(InkGrid grid) => Estimate(grid, transposed: false);

    /// <summary>
    /// As <see cref="Estimate(InkGrid)"/> for the grid turned a quarter clockwise, without making that grid: cell <c>(x, y)</c> is
    /// read as <c>(height − 1 − y, x)</c>.
    /// </summary>
    public static SkewEstimate EstimateTurned(InkGrid grid) => Estimate(grid, transposed: true);

    private static SkewEstimate Estimate(InkGrid grid, bool transposed)
    {
        using var fine = new Projector(grid, transposed);
        if (fine.Count < 32)
        {
            return new SkewEstimate(0, 0, 1, 1);
        }

        using InkGrid shrunk = grid.Shrink(CoarseShrink);
        using var coarse = new Projector(shrunk, transposed);
        int center = 0;
        long best = -1;
        for (int k = -MaxSlope; k <= MaxSlope; k += CoarseStep)
        {
            long s = coarse.Sharpness(k);
            if (s > best || (s == best && Math.Abs(k) < Math.Abs(center)))
            {
                best = s;
                center = k;
            }
        }

        long zero = fine.Sharpness(0);
        long top = zero;
        int bestSlope = 0;
        for (int k = Math.Max(-MaxSlope, center - FineReach); k <= Math.Min(MaxSlope, center + FineReach); k++)
        {
            long s = k == 0 ? zero : fine.Sharpness(k);
            if (s > top || (s == top && Math.Abs(k) < Math.Abs(bestSlope)))
            {
                top = s;
                bestSlope = k;
            }
        }

        return new SkewEstimate(bestSlope, top, zero == 0 ? 1 : (double)top / zero, top * (double)fine.GridHeight / ((double)fine.Total * fine.Total));
    }

    // The ink cells of a grid (without the border and the solid areas) as a flat list, and the profile they project onto. A turned grid
    // is read with swapped axes instead of being copied. Every cell is kept as the two integers its bin is computed from, bin =
    // (Along - slope * Across) >> 10, so one slope is a single branch-free pass over the list; the sums are integers, so the
    // order of the cells does not matter.
    private sealed class Projector : IDisposable
    {
        private int[]? _along, _across;
        private ushort[]? _weight;
        private readonly long[] _profile;
        private readonly int _offset;

        public Projector(InkGrid grid, bool transposed)
        {
            int gw = grid.Width, gh = grid.Height;
            int solid = Math.Max(1, (int)(SolidShare * grid.Factor * grid.Factor));
            // Margins are 2% of the (possibly turned) grid's own width and height, which are this grid's height and width.
            int turnedWidth = transposed ? gh : gw, turnedHeight = transposed ? gw : gh;
            int mx = (int)(turnedWidth * EdgeShare), my = (int)(turnedHeight * EdgeShare);
            (mx, my) = transposed ? (my, mx) : (mx, my);
            int capacity = Math.Max(0, gh - (2 * my)) * Math.Max(0, gw - (2 * mx));
            _along = ArrayPool<int>.Shared.Rent(capacity);
            _across = ArrayPool<int>.Shared.Rent(capacity);
            _weight = ArrayPool<ushort>.Shared.Rent(capacity);
            long total = 0;
            int count = 0;
            for (int y = my; y < gh - my; y++)
            {
                ReadOnlySpan<ushort> row = grid.Counts.AsSpan(y * gw, gw);
                for (int x = mx; x < gw - mx; x++)
                {
                    int c = row[x];
                    if (c > 0 && c <= solid)
                    {
                        // Turned grid: cell (x, y) is at (gh - 1 - y, x).
                        (_along[count], _across[count]) = transposed ? (Scale * x, gh - 1 - y) : (Scale * y, x);
                        _weight[count] = (ushort)c;
                        total += c;
                        count++;
                    }
                }
            }

            Count = count;
            Total = total;
            GridHeight = turnedHeight;
            // Bin = floor(y - slope * x), shifted to be non-negative.
            _offset = (((MaxSlope * turnedWidth) + Scale - 1) / Scale) + 1;
            _profile = new long[turnedHeight + (2 * _offset) + 2];
        }

        public int Count { get; }

        public long Total { get; }

        public int GridHeight { get; }

        public void Dispose()
        {
            if (_along is not null)
            {
                ArrayPool<int>.Shared.Return(_along);
                ArrayPool<int>.Shared.Return(_across!);
                ArrayPool<ushort>.Shared.Return(_weight!);
                _along = null;
            }
        }

        public long Sharpness(int slope)
        {
            Array.Clear(_profile);
            ReadOnlySpan<int> along = _along.AsSpan(0, Count), across = _across.AsSpan(0, Count);
            ReadOnlySpan<ushort> weight = _weight.AsSpan(0, Count);
            long[] profile = _profile;
            int offset = _offset;
            for (int i = 0; i < along.Length; i++)
            {
                profile[((along[i] - (slope * across[i])) >> 10) + offset] += weight[i];
            }

            long sum = 0;
            foreach (long p in profile)
            {
                sum += p * p;
            }

            return sum;
        }
    }
}
