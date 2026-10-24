namespace OneOcr.Layout;

/// <summary>
/// The page's ink counted on a coarse grid (about 2400 cells on the long side at most): the input of skew estimation and of the
/// sideways-text check. A cell holds the number of ink pixels in its block.
/// </summary>
internal sealed class InkGrid
{
    public InkGrid(int width, int height, int factor, ushort[] counts)
    {
        Width = width;
        Height = height;
        Factor = factor;
        Counts = counts;
    }

    public int Width { get; }

    public int Height { get; }

    /// <summary>Pixels per cell side.</summary>
    public int Factor { get; }

    /// <summary>Row-major ink pixel counts, at most <c>Factor * Factor</c> each.</summary>
    public ushort[] Counts { get; }

    private const int LongSide = 2400;

    /// <summary>Counts the ink of a page (ink is the dark class, or the bright class when <paramref name="inkIsBright"/>).</summary>
    public static InkGrid Build(ReadOnlySpan<byte> gray, int width, int height, int threshold, bool inkIsBright)
    {
        int factor = Math.Max(1, (Math.Max(width, height) + LongSide - 1) / LongSide);
        int gw = (width + factor - 1) / factor, gh = (height + factor - 1) / factor;
        var counts = new ushort[checked(gw * gh)];
        if (threshold < 0)
        {
            return new InkGrid(gw, gh, factor, counts);
        }

        for (int y = 0; y < height; y++)
        {
            ReadOnlySpan<byte> row = gray.Slice(y * width, width);
            int rowBase = (y / factor) * gw;
            if (inkIsBright)
            {
                for (int x = 0; x < row.Length; x++)
                {
                    if (row[x] > threshold)
                    {
                        counts[rowBase + (x / factor)]++;
                    }
                }
            }
            else
            {
                for (int x = 0; x < row.Length; x++)
                {
                    if (row[x] <= threshold)
                    {
                        counts[rowBase + (x / factor)]++;
                    }
                }
            }
        }

        return new InkGrid(gw, gh, factor, counts);
    }

    /// <summary>The grid with blocks of <paramref name="by"/> by <paramref name="by"/> cells added up: a coarser picture for a first, cheap search.</summary>
    public InkGrid Shrink(int by)
    {
        int gw = (Width + by - 1) / by, gh = (Height + by - 1) / by;
        var counts = new ushort[checked(gw * gh)];
        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                counts[((y / by) * gw) + (x / by)] += Counts[(y * Width) + x];
            }
        }

        return new InkGrid(gw, gh, Factor * by, counts);
    }

    /// <summary>The grid of the same page turned a quarter turn clockwise: cell (x, y) goes to (height - 1 - y, x).</summary>
    public InkGrid TurnClockwise()
    {
        var turned = new ushort[Counts.Length];
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

    public static SkewEstimate Estimate(InkGrid grid)
    {
        var fine = new Projector(grid);
        if (fine.Count < 32)
        {
            return new SkewEstimate(0, 0, 1, 1);
        }

        var coarse = new Projector(grid.Shrink(CoarseShrink));
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

    // The ink cells of a grid (without the border and the solid areas) and the profile they project onto.
    private sealed class Projector
    {
        private readonly int[] _x, _y, _w;
        private readonly long[] _profile;
        private readonly int _offset;

        public Projector(InkGrid grid)
        {
            int gw = grid.Width, gh = grid.Height;
            int solid = Math.Max(1, (int)(SolidShare * grid.Factor * grid.Factor));
            int mx = (int)(gw * EdgeShare), my = (int)(gh * EdgeShare);
            var xs = new List<int>();
            var ys = new List<int>();
            var ws = new List<int>();
            for (int y = my; y < gh - my; y++)
            {
                for (int x = mx; x < gw - mx; x++)
                {
                    int c = grid.Counts[(y * gw) + x];
                    if (c > 0 && c <= solid)
                    {
                        xs.Add(x);
                        ys.Add(y);
                        ws.Add(c);
                        Total += c;
                    }
                }
            }

            _x = [.. xs];
            _y = [.. ys];
            _w = [.. ws];
            GridHeight = gh;
            // Bin = floor(y - slope * x), shifted to be non-negative.
            _offset = (((MaxSlope * gw) + Scale - 1) / Scale) + 1;
            _profile = new long[gh + (2 * _offset) + 2];
        }

        public int Count => _x.Length;

        public long Total { get; }

        public int GridHeight { get; }

        public long Sharpness(int slope)
        {
            Array.Clear(_profile);
            for (int i = 0; i < _x.Length; i++)
            {
                int bin = ((Scale * _y[i]) - (slope * _x[i])) >> 10;
                _profile[bin + _offset] += _w[i];
            }

            long sum = 0;
            foreach (long p in _profile)
            {
                sum += p * p;
            }

            return sum;
        }
    }
}
