using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OneOcr.Kernels;

/// <summary>
/// Dot products and matrix products over fp32 rows, with the library's fixed accumulation order (docs/spec/numerics.md): element k
/// of a dot product is fused-multiply-added into partial sum (k mod 16), whatever the vector width, and the 16 partial sums are
/// then added in a fixed tree. A tail shorter than 16 continues the same lanes. A bias, when given, is added to the finished sum.
/// Scalar uses <see cref="MathF.FusedMultiplyAdd"/>, which is exact on every platform; 128-bit vectors keep the partials in four
/// registers, AVX2 in two, AVX-512 in one.
///
/// Speed comes from register blocking (Goto and van de Geijn, "Anatomy of high-performance matrix multiplication", 2008): a tile
/// of several A rows against several B rows shares every load between the tile's outputs, and each output keeps its own
/// accumulators, so blocking never changes the order within an output. Tiles whose rows run past the matrix repeat the last valid
/// row and discard those outputs. The reduction of the 16 partials is done for four outputs at a time with the same additions.
/// This is the only file group in the library that handles raw pointers; the wrappers validate every length first.
/// </summary>
internal static unsafe partial class Gemm
{
    /// <summary>Dot product of two rows in the fixed order.</summary>
    public static float Dot(KernelPath path, ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("The two rows must have the same length.");
        }

        if (a.Length == 0)
        {
            return 0f;
        }

        Dispatch.Require(path);
        fixed (float* pa = a, pb = b)
        {
            return DotRow(path, pa, pb, a.Length);
        }
    }

    /// <summary>
    /// C[i, j] = dot(A row i, B row j) (+ bias[j]) for i in [0, m) and j in [0, n), where A is m rows of k elements at stride
    /// <paramref name="lda"/> and B is n rows of k elements at stride <paramref name="ldb"/>. C[i, j] is stored at
    /// <c>i * ldcRow + j * ldcCol</c>.
    /// </summary>
    public static void NT(
        KernelPath path,
        ReadOnlySpan<float> a, int lda,
        ReadOnlySpan<float> b, int ldb,
        int m, int n, int k,
        ReadOnlySpan<float> bias,
        Span<float> c, long ldcRow, long ldcCol)
    {
        if (m < 0 || n < 0 || k < 0 || lda < k || ldb < k || ldcRow < 0 || ldcCol < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(m), "Matrix dimensions and strides must be non-negative and strides at least k.");
        }

        if (m == 0 || n == 0)
        {
            return;
        }

        if (RowsNeeded(m, lda, k) > a.Length || RowsNeeded(n, ldb, k) > b.Length)
        {
            throw new ArgumentException("An input matrix is smaller than its declared shape.");
        }

        if (!bias.IsEmpty && bias.Length < n)
        {
            throw new ArgumentException("The bias has fewer than n elements.");
        }

        // Largest index written: (m-1)*ldcRow + (n-1)*ldcCol.
        if (((m - 1) * ldcRow) + ((n - 1) * ldcCol) >= c.Length)
        {
            throw new ArgumentException("The output is smaller than its declared shape.");
        }

        Dispatch.Require(path);
        if (k == 0)
        {
            // Empty dot products are +0; keep the bias rule.
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    c[(int)((i * ldcRow) + (j * ldcCol))] = bias.IsEmpty ? 0f : 0f + bias[j];
                }
            }

            return;
        }

        fixed (float* pa = a, pb = b, pbias = bias, pc = c)
        {
            float* biasPtr = bias.IsEmpty ? null : pbias;
            NTCore(path, pa, lda, pb, ldb, m, n, k, biasPtr, pc, ldcRow, ldcCol);
        }
    }

    private static long RowsNeeded(int rows, int stride, int k) => (((long)rows - 1) * stride) + k;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]

    private static void NTCore(
        KernelPath path, float* a, int lda, float* b, int ldb, int m, int n, int k, float* bias, float* c, long ldcRow, long ldcCol)
    {
        // Tiles need whole blocks of 16 along k; other lengths and thin matrices use the row kernel.
        int mr = path switch { KernelPath.Avx512 => 4, KernelPath.Avx2 => 3, _ => 0 };
        if (mr == 0 || (k & 15) != 0 || m < mr)
        {
            for (int i = 0; i < m; i++)
            {
                RowKernel(path, a + ((long)i * lda), b, ldb, n, k, bias, c + (i * ldcRow), ldcCol);
            }

            return;
        }

        int nr = path == KernelPath.Avx512 ? 4 : 2;
        float** ap = stackalloc float*[4];
        float** bp = stackalloc float*[4];
        float* tile = stackalloc float[4 * 4];
        // Rows of A are taken in blocks that fit the second-level cache (about 256 KB); inside a block the B tile (nr rows) is the outer
        // loop, so it is read from memory once per block and stays in the first-level cache while every row tile of the block runs
        // against it. With A as the outer loop, B (a whole weight matrix in the recurrent layers) would be streamed once per row tile.
        // Each output is still one tile accumulation over the same k, so the order of every dot product is unchanged.
        int blockRows = (int)Math.Clamp(262144L / ((long)lda * sizeof(float)) / mr * mr, mr, 1 << 20);
        for (int iBlock = 0; iBlock < m; iBlock += blockRows)
        {
            int iEnd = Math.Min(m, iBlock + blockRows);
            for (int j0 = 0; j0 < n; j0 += nr)
            {
                int cols = Math.Min(nr, n - j0);
                for (int q = 0; q < nr; q++)
                {
                    bp[q] = b + ((long)(j0 + Math.Min(q, cols - 1)) * ldb);
                }

                for (int i0 = iBlock; i0 < iEnd; i0 += mr)
                {
                    int rows = Math.Min(mr, iEnd - i0);
                    for (int r = 0; r < mr; r++)
                    {
                        ap[r] = a + ((long)(i0 + Math.Min(r, rows - 1)) * lda);
                    }

                    if (path == KernelPath.Avx512)
                    {
                        Tile4x4Avx512(ap, bp, 1, k, 0, tile);
                    }
                    else
                    {
                        Tile3x2Avx2(ap, bp, 1, k, 0, tile);
                    }

                    StoreTile(tile, rows, cols, bias == null ? null : bias + j0, c + (i0 * ldcRow) + (j0 * ldcCol), ldcRow, ldcCol, path);
                }
            }
        }
    }

    // tile[q * 4 + r] holds the output of A row r against B row q; bias points at the bias of B row 0 of the tile.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreTile(float* tile, int rows, int cols, float* bias, float* c, long ldcRow, long ldcCol, KernelPath path)
    {
        if (path == KernelPath.Avx512 && rows == 4 && cols == 4 && ldcCol == 1)
        {
            // Transpose so that each A row's four outputs are adjacent, then four 128-bit stores.
            Vector128<float> t0 = Sse.LoadVector128(tile), t1 = Sse.LoadVector128(tile + 4);
            Vector128<float> t2 = Sse.LoadVector128(tile + 8), t3 = Sse.LoadVector128(tile + 12);
            Vector128<float> a0 = Sse.UnpackLow(t0, t1), a1 = Sse.UnpackLow(t2, t3);
            Vector128<float> a2 = Sse.UnpackHigh(t0, t1), a3 = Sse.UnpackHigh(t2, t3);
            Vector128<float> r0 = Sse.MoveLowToHigh(a0, a1), r1 = Sse.MoveHighToLow(a1, a0);
            Vector128<float> r2 = Sse.MoveLowToHigh(a2, a3), r3 = Sse.MoveHighToLow(a3, a2);
            if (bias != null)
            {
                Vector128<float> bv = Sse.LoadVector128(bias);
                r0 = Sse.Add(r0, bv);
                r1 = Sse.Add(r1, bv);
                r2 = Sse.Add(r2, bv);
                r3 = Sse.Add(r3, bv);
            }

            Sse.Store(c, r0);
            Sse.Store(c + ldcRow, r1);
            Sse.Store(c + (2 * ldcRow), r2);
            Sse.Store(c + (3 * ldcRow), r3);
            return;
        }

        for (int q = 0; q < cols; q++)
        {
            float add = bias == null ? 0f : bias[q];
            for (int r = 0; r < rows; r++)
            {
                float v = tile[(q * 4) + r];
                c[(r * ldcRow) + (q * ldcCol)] = bias == null ? v : v + add;
            }
        }
    }

    // One A row against n B rows, any k.
    private static void RowKernel(KernelPath path, float* ar, float* b, int ldb, int n, int k, float* bias, float* crow, long ldcCol)
    {
        int j = 0;
        if (path == KernelPath.Avx512)
        {
            // Eight independent accumulators hide the latency of the fused multiply-add (a single row has no other reuse to hide it).
            for (; j + 8 <= n; j += 8)
            {
                Row8Avx512(ar, b + ((long)j * ldb), ldb, k, bias == null ? null : bias + j, crow + (j * ldcCol), ldcCol);
            }

            for (; j + 4 <= n; j += 4)
            {
                Row4Avx512(ar, b + ((long)j * ldb), ldb, k, bias == null ? null : bias + j, crow + (j * ldcCol), ldcCol);
            }
        }
        else if (path == KernelPath.Avx2)
        {
            for (; j + 4 <= n; j += 4)
            {
                Row4Avx2(ar, b + ((long)j * ldb), ldb, k, bias == null ? null : bias + j, crow + (j * ldcCol), ldcCol);
            }
        }

        for (; j < n; j++)
        {
            float v = DotRow(path, ar, b + ((long)j * ldb), k);
            crow[j * ldcCol] = bias == null ? v : v + bias[j];
        }
    }

    private static float DotRow(KernelPath path, float* a, float* b, int n) => path switch
    {
        KernelPath.Avx512 => DotAvx512(a, b, n),
        KernelPath.Avx2 => DotAvx2(a, b, n),
        KernelPath.Vector128 => DotVector128(a, b, n),
        _ => DotScalar(a, b, n),
    };

    // The 16-partial tree, in place.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Reduce16(float* p)
    {
        for (int i = 0; i < 8; i++)
        {
            p[i] += p[i + 8];
        }

        for (int i = 0; i < 4; i++)
        {
            p[i] += p[i + 4];
        }

        p[0] += p[2];
        p[1] += p[3];
        return p[0] + p[1];
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]

    internal static float DotScalar(float* a, float* b, int n)
    {
        Debug.Assert(n > 0);
        float* p = stackalloc float[16];
        for (int j = 0; j < 16; j++)
        {
            p[j] = 0f;
        }

        int k = 0;
        for (; k + 16 <= n; k += 16)
        {
            for (int j = 0; j < 16; j++)
            {
                p[j] = MathF.FusedMultiplyAdd(a[k + j], b[k + j], p[j]);
            }
        }

        for (int j = 0; k < n; k++, j++)
        {
            p[j] = MathF.FusedMultiplyAdd(a[k], b[k], p[j]);
        }

        return Reduce16(p);
    }

    // Continues a dot product's lanes through the tail (fewer than 16 elements) and reduces the 16 partials in p.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float FinishWithTail(float* p, float* aTail, float* bTail, int tail)
    {
        for (int j = 0; j < tail; j++)
        {
            p[j] = MathF.FusedMultiplyAdd(aTail[j], bTail[j], p[j]);
        }

        return Reduce16(p);
    }

    // Final additions of 16 partials already folded to four lanes x = [p0+p8+..., ...]: y = x + (x2, x3), result y0 + y1.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Reduce4Lanes(Vector128<float> x)
    {
        Vector128<float> y = Sse.Add(x, Sse.MoveHighToLow(x, x));
        return y.GetElement(0) + y.GetElement(1);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Fold(Vector256<float> a0, Vector256<float> a1)
    {
        Vector256<float> s = Avx.Add(a0, a1);
        return Sse.Add(s.GetLower(), s.GetUpper());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Fold(Vector512<float> a)
    {
        Vector256<float> s = Avx.Add(a.GetLower(), a.GetUpper());
        return Sse.Add(s.GetLower(), s.GetUpper());
    }

    // The final two steps for four outputs at once: lane i of the result is y_i[0] + y_i[1] with y_i = x_i + (x_i[2], x_i[3]).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Reduce4Outputs(Vector128<float> x0, Vector128<float> x1, Vector128<float> x2, Vector128<float> x3)
    {
        Vector128<float> y0 = Sse.Add(x0, Sse.MoveHighToLow(x0, x0));
        Vector128<float> y1 = Sse.Add(x1, Sse.MoveHighToLow(x1, x1));
        Vector128<float> y2 = Sse.Add(x2, Sse.MoveHighToLow(x2, x2));
        Vector128<float> y3 = Sse.Add(x3, Sse.MoveHighToLow(x3, x3));
        Vector128<float> u = Sse.UnpackLow(y0, y1);
        Vector128<float> v = Sse.UnpackLow(y2, y3);
        return Sse.Add(Sse.MoveLowToHigh(u, v), Sse.MoveHighToLow(v, u));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]

    internal static float DotVector128(float* a, float* b, int n)
    {
        Debug.Assert(n > 0);
        Vector128<float> c0 = Vector128<float>.Zero, c1 = c0, c2 = c0, c3 = c0;
        int k = 0;
        for (; k + 16 <= n; k += 16)
        {
            c0 = Vector128.FusedMultiplyAdd(Vector128.Load(a + k), Vector128.Load(b + k), c0);
            c1 = Vector128.FusedMultiplyAdd(Vector128.Load(a + k + 4), Vector128.Load(b + k + 4), c1);
            c2 = Vector128.FusedMultiplyAdd(Vector128.Load(a + k + 8), Vector128.Load(b + k + 8), c2);
            c3 = Vector128.FusedMultiplyAdd(Vector128.Load(a + k + 12), Vector128.Load(b + k + 12), c3);
        }

        float* p = stackalloc float[16];
        Vector128.Store(c0, p);
        Vector128.Store(c1, p + 4);
        Vector128.Store(c2, p + 8);
        Vector128.Store(c3, p + 12);
        return FinishWithTail(p, a + k, b + k, n - k);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]

    internal static float DotAvx2(float* a, float* b, int n)
    {
        Debug.Assert(n > 0);
        Vector256<float> acc0 = Vector256<float>.Zero;
        Vector256<float> acc1 = Vector256<float>.Zero;
        int k = 0;
        for (; k + 16 <= n; k += 16)
        {
            acc0 = Fma.MultiplyAdd(Avx.LoadVector256(a + k), Avx.LoadVector256(b + k), acc0);
            acc1 = Fma.MultiplyAdd(Avx.LoadVector256(a + k + 8), Avx.LoadVector256(b + k + 8), acc1);
        }

        if (k == n)
        {
            return Reduce4Lanes(Fold(acc0, acc1));
        }

        float* p = stackalloc float[16];
        Avx.Store(p, acc0);
        Avx.Store(p + 8, acc1);
        return FinishWithTail(p, a + k, b + k, n - k);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]

    internal static float DotAvx512(float* a, float* b, int n)
    {
        Debug.Assert(n > 0);
        Vector512<float> acc = Vector512<float>.Zero;
        int k = 0;
        for (; k + 16 <= n; k += 16)
        {
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(a + k), Avx512F.LoadVector512(b + k), acc);
        }

        if (k == n)
        {
            return Reduce4Lanes(Fold(acc));
        }

        float* p = stackalloc float[16];
        Avx512F.Store(p, acc);
        return FinishWithTail(p, a + k, b + k, n - k);
    }

    // ---- Row kernels: one A row against four consecutive B rows ----

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]

    private static void Row4Avx2(float* ar, float* b0, int ldb, int k, float* bias, float* c, long ldcCol)
    {
        float* b1 = b0 + ldb;
        float* b2 = b1 + ldb;
        float* b3 = b2 + ldb;
        Vector256<float> c00 = Vector256<float>.Zero, c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00, c30 = c00, c31 = c00;
        int kMain = k & ~15;
        for (int x = 0; x < kMain; x += 16)
        {
            Vector256<float> a0 = Avx.LoadVector256(ar + x);
            Vector256<float> a1 = Avx.LoadVector256(ar + x + 8);
            c00 = Fma.MultiplyAdd(a0, Avx.LoadVector256(b0 + x), c00);
            c01 = Fma.MultiplyAdd(a1, Avx.LoadVector256(b0 + x + 8), c01);
            c10 = Fma.MultiplyAdd(a0, Avx.LoadVector256(b1 + x), c10);
            c11 = Fma.MultiplyAdd(a1, Avx.LoadVector256(b1 + x + 8), c11);
            c20 = Fma.MultiplyAdd(a0, Avx.LoadVector256(b2 + x), c20);
            c21 = Fma.MultiplyAdd(a1, Avx.LoadVector256(b2 + x + 8), c21);
            c30 = Fma.MultiplyAdd(a0, Avx.LoadVector256(b3 + x), c30);
            c31 = Fma.MultiplyAdd(a1, Avx.LoadVector256(b3 + x + 8), c31);
        }

        if (kMain == k)
        {
            Vector128<float> r = Reduce4Outputs(Fold(c00, c01), Fold(c10, c11), Fold(c20, c21), Fold(c30, c31));
            Store4(c, ldcCol, bias, r);
            return;
        }

        float* p = stackalloc float[16];
        float* at = ar + kMain;
        int tail = k - kMain;
        Avx.Store(p, c00); Avx.Store(p + 8, c01);
        Put(c, 0, ldcCol, bias, FinishWithTail(p, at, b0 + kMain, tail));
        Avx.Store(p, c10); Avx.Store(p + 8, c11);
        Put(c, 1, ldcCol, bias, FinishWithTail(p, at, b1 + kMain, tail));
        Avx.Store(p, c20); Avx.Store(p + 8, c21);
        Put(c, 2, ldcCol, bias, FinishWithTail(p, at, b2 + kMain, tail));
        Avx.Store(p, c30); Avx.Store(p + 8, c31);
        Put(c, 3, ldcCol, bias, FinishWithTail(p, at, b3 + kMain, tail));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Row8Avx512(float* ar, float* b0, int ldb, int k, float* bias, float* c, long ldcCol)
    {
        float* b1 = b0 + ldb, b2 = b1 + ldb, b3 = b2 + ldb, b4 = b3 + ldb, b5 = b4 + ldb, b6 = b5 + ldb, b7 = b6 + ldb;
        Vector512<float> z = Vector512<float>.Zero;
        Vector512<float> c0 = z, c1 = z, c2 = z, c3 = z, c4 = z, c5 = z, c6 = z, c7 = z;
        int kMain = k & ~15;
        for (int x = 0; x < kMain; x += 16)
        {
            Vector512<float> a = Avx512F.LoadVector512(ar + x);
            c0 = Avx512F.FusedMultiplyAdd(a, Avx512F.LoadVector512(b0 + x), c0);
            c1 = Avx512F.FusedMultiplyAdd(a, Avx512F.LoadVector512(b1 + x), c1);
            c2 = Avx512F.FusedMultiplyAdd(a, Avx512F.LoadVector512(b2 + x), c2);
            c3 = Avx512F.FusedMultiplyAdd(a, Avx512F.LoadVector512(b3 + x), c3);
            c4 = Avx512F.FusedMultiplyAdd(a, Avx512F.LoadVector512(b4 + x), c4);
            c5 = Avx512F.FusedMultiplyAdd(a, Avx512F.LoadVector512(b5 + x), c5);
            c6 = Avx512F.FusedMultiplyAdd(a, Avx512F.LoadVector512(b6 + x), c6);
            c7 = Avx512F.FusedMultiplyAdd(a, Avx512F.LoadVector512(b7 + x), c7);
        }

        if (kMain == k)
        {
            Store4(c, ldcCol, bias, Reduce4Outputs(Fold(c0), Fold(c1), Fold(c2), Fold(c3)));
            Store4(c + (4 * ldcCol), ldcCol, bias == null ? null : bias + 4, Reduce4Outputs(Fold(c4), Fold(c5), Fold(c6), Fold(c7)));
            return;
        }

        float* p = stackalloc float[16];
        float* at = ar + kMain;
        int tail = k - kMain;
        Avx512F.Store(p, c0);
        Put(c, 0, ldcCol, bias, FinishWithTail(p, at, b0 + kMain, tail));
        Avx512F.Store(p, c1);
        Put(c, 1, ldcCol, bias, FinishWithTail(p, at, b1 + kMain, tail));
        Avx512F.Store(p, c2);
        Put(c, 2, ldcCol, bias, FinishWithTail(p, at, b2 + kMain, tail));
        Avx512F.Store(p, c3);
        Put(c, 3, ldcCol, bias, FinishWithTail(p, at, b3 + kMain, tail));
        Avx512F.Store(p, c4);
        Put(c, 4, ldcCol, bias, FinishWithTail(p, at, b4 + kMain, tail));
        Avx512F.Store(p, c5);
        Put(c, 5, ldcCol, bias, FinishWithTail(p, at, b5 + kMain, tail));
        Avx512F.Store(p, c6);
        Put(c, 6, ldcCol, bias, FinishWithTail(p, at, b6 + kMain, tail));
        Avx512F.Store(p, c7);
        Put(c, 7, ldcCol, bias, FinishWithTail(p, at, b7 + kMain, tail));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Row4Avx512(float* ar, float* b0, int ldb, int k, float* bias, float* c, long ldcCol)
    {
        float* b1 = b0 + ldb;
        float* b2 = b1 + ldb;
        float* b3 = b2 + ldb;
        Vector512<float> c0 = Vector512<float>.Zero, c1 = c0, c2 = c0, c3 = c0;
        int kMain = k & ~15;
        for (int x = 0; x < kMain; x += 16)
        {
            Vector512<float> a = Avx512F.LoadVector512(ar + x);
            c0 = Avx512F.FusedMultiplyAdd(a, Avx512F.LoadVector512(b0 + x), c0);
            c1 = Avx512F.FusedMultiplyAdd(a, Avx512F.LoadVector512(b1 + x), c1);
            c2 = Avx512F.FusedMultiplyAdd(a, Avx512F.LoadVector512(b2 + x), c2);
            c3 = Avx512F.FusedMultiplyAdd(a, Avx512F.LoadVector512(b3 + x), c3);
        }

        if (kMain == k)
        {
            Store4(c, ldcCol, bias, Reduce4Outputs(Fold(c0), Fold(c1), Fold(c2), Fold(c3)));
            return;
        }

        float* p = stackalloc float[16];
        float* at = ar + kMain;
        int tail = k - kMain;
        Avx512F.Store(p, c0);
        Put(c, 0, ldcCol, bias, FinishWithTail(p, at, b0 + kMain, tail));
        Avx512F.Store(p, c1);
        Put(c, 1, ldcCol, bias, FinishWithTail(p, at, b1 + kMain, tail));
        Avx512F.Store(p, c2);
        Put(c, 2, ldcCol, bias, FinishWithTail(p, at, b2 + kMain, tail));
        Avx512F.Store(p, c3);
        Put(c, 3, ldcCol, bias, FinishWithTail(p, at, b3 + kMain, tail));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Put(float* c, int index, long ldcCol, float* bias, float v) =>
        c[index * ldcCol] = bias == null ? v : v + bias[index];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store4(float* c, long ldcCol, float* bias, Vector128<float> v)
    {
        if (bias != null)
        {
            v = Sse.Add(v, Sse.LoadVector128(bias));
        }

        if (ldcCol == 1)
        {
            Sse.Store(c, v);
        }
        else
        {
            c[0] = v.GetElement(0);
            c[ldcCol] = v.GetElement(1);
            c[2 * ldcCol] = v.GetElement(2);
            c[3 * ldcCol] = v.GetElement(3);
        }
    }

    // ---- Tile kernels: results go to tile[q * 4 + r] (B row q against A row r) ----

    // 4 A rows x 4 B rows, one 512-bit accumulator per output: 16 accumulators, 8 loads per 16 fused multiply-adds.
    // The k range is nSeg segments of kSeg elements (a multiple of 16): segment s of A row r starts at ap[r] + s * aSegStride, and
    // of B row q at bp[q] + s * kSeg. A plain matrix product is one segment. Walking the segments in order is the same as
    // walking one long row, so lane k mod 16 and the order of additions are those of the flat row.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile4x4Avx512(float** ap, float** bp, int nSeg, int kSeg, long aSegStride, float* tile)
    {
        float* a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
        float* b0 = bp[0], b1 = bp[1], b2 = bp[2], b3 = bp[3];
        Vector512<float> z = Vector512<float>.Zero;
        Vector512<float> c00 = z, c01 = z, c02 = z, c03 = z, c10 = z, c11 = z, c12 = z, c13 = z;
        Vector512<float> c20 = z, c21 = z, c22 = z, c23 = z, c30 = z, c31 = z, c32 = z, c33 = z;
        for (int seg = 0; seg < nSeg; seg++, a0 += aSegStride, a1 += aSegStride, a2 += aSegStride, a3 += aSegStride, b0 += kSeg, b1 += kSeg, b2 += kSeg, b3 += kSeg)
        for (int x = 0; x < kSeg; x += 16)
        {
            Vector512<float> va0 = Avx512F.LoadVector512(a0 + x);
            Vector512<float> va1 = Avx512F.LoadVector512(a1 + x);
            Vector512<float> va2 = Avx512F.LoadVector512(a2 + x);
            Vector512<float> va3 = Avx512F.LoadVector512(a3 + x);
            Vector512<float> vb = Avx512F.LoadVector512(b0 + x);
            c00 = Avx512F.FusedMultiplyAdd(va0, vb, c00);
            c10 = Avx512F.FusedMultiplyAdd(va1, vb, c10);
            c20 = Avx512F.FusedMultiplyAdd(va2, vb, c20);
            c30 = Avx512F.FusedMultiplyAdd(va3, vb, c30);
            vb = Avx512F.LoadVector512(b1 + x);
            c01 = Avx512F.FusedMultiplyAdd(va0, vb, c01);
            c11 = Avx512F.FusedMultiplyAdd(va1, vb, c11);
            c21 = Avx512F.FusedMultiplyAdd(va2, vb, c21);
            c31 = Avx512F.FusedMultiplyAdd(va3, vb, c31);
            vb = Avx512F.LoadVector512(b2 + x);
            c02 = Avx512F.FusedMultiplyAdd(va0, vb, c02);
            c12 = Avx512F.FusedMultiplyAdd(va1, vb, c12);
            c22 = Avx512F.FusedMultiplyAdd(va2, vb, c22);
            c32 = Avx512F.FusedMultiplyAdd(va3, vb, c32);
            vb = Avx512F.LoadVector512(b3 + x);
            c03 = Avx512F.FusedMultiplyAdd(va0, vb, c03);
            c13 = Avx512F.FusedMultiplyAdd(va1, vb, c13);
            c23 = Avx512F.FusedMultiplyAdd(va2, vb, c23);
            c33 = Avx512F.FusedMultiplyAdd(va3, vb, c33);
        }

        // cRQ: row R against B row Q. Reduce the four rows of each B row together.
        Sse.Store(tile, Reduce4Outputs(Fold(c00), Fold(c10), Fold(c20), Fold(c30)));
        Sse.Store(tile + 4, Reduce4Outputs(Fold(c01), Fold(c11), Fold(c21), Fold(c31)));
        Sse.Store(tile + 8, Reduce4Outputs(Fold(c02), Fold(c12), Fold(c22), Fold(c32)));
        Sse.Store(tile + 12, Reduce4Outputs(Fold(c03), Fold(c13), Fold(c23), Fold(c33)));
    }

    // 3 A rows x 2 B rows, two 256-bit accumulators per output: 12 accumulators; the A rows stay in registers and the B vectors
    // are used as memory operands.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile3x2Avx2(float** ap, float** bp, int nSeg, int kSeg, long aSegStride, float* tile)
    {
        float* a0 = ap[0], a1 = ap[1], a2 = ap[2];
        float* b0 = bp[0], b1 = bp[1];
        Vector256<float> z = Vector256<float>.Zero;
        Vector256<float> c00l = z, c00h = z, c10l = z, c10h = z, c20l = z, c20h = z;
        Vector256<float> c01l = z, c01h = z, c11l = z, c11h = z, c21l = z, c21h = z;
        for (int seg = 0; seg < nSeg; seg++, a0 += aSegStride, a1 += aSegStride, a2 += aSegStride, b0 += kSeg, b1 += kSeg)
        for (int x = 0; x < kSeg; x += 16)
        {
            Vector256<float> va0 = Avx.LoadVector256(a0 + x);
            Vector256<float> va1 = Avx.LoadVector256(a1 + x);
            Vector256<float> va2 = Avx.LoadVector256(a2 + x);
            c00l = Fma.MultiplyAdd(va0, Avx.LoadVector256(b0 + x), c00l);
            c10l = Fma.MultiplyAdd(va1, Avx.LoadVector256(b0 + x), c10l);
            c20l = Fma.MultiplyAdd(va2, Avx.LoadVector256(b0 + x), c20l);
            c01l = Fma.MultiplyAdd(va0, Avx.LoadVector256(b1 + x), c01l);
            c11l = Fma.MultiplyAdd(va1, Avx.LoadVector256(b1 + x), c11l);
            c21l = Fma.MultiplyAdd(va2, Avx.LoadVector256(b1 + x), c21l);
            va0 = Avx.LoadVector256(a0 + x + 8);
            va1 = Avx.LoadVector256(a1 + x + 8);
            va2 = Avx.LoadVector256(a2 + x + 8);
            c00h = Fma.MultiplyAdd(va0, Avx.LoadVector256(b0 + x + 8), c00h);
            c10h = Fma.MultiplyAdd(va1, Avx.LoadVector256(b0 + x + 8), c10h);
            c20h = Fma.MultiplyAdd(va2, Avx.LoadVector256(b0 + x + 8), c20h);
            c01h = Fma.MultiplyAdd(va0, Avx.LoadVector256(b1 + x + 8), c01h);
            c11h = Fma.MultiplyAdd(va1, Avx.LoadVector256(b1 + x + 8), c11h);
            c21h = Fma.MultiplyAdd(va2, Avx.LoadVector256(b1 + x + 8), c21h);
        }

        Vector128<float> zero = Vector128<float>.Zero;
        Sse.Store(tile, Reduce4Outputs(Fold(c00l, c00h), Fold(c10l, c10h), Fold(c20l, c20h), zero));
        Sse.Store(tile + 4, Reduce4Outputs(Fold(c01l, c01h), Fold(c11l, c11h), Fold(c21l, c21h), zero));
    }
}
