using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OneOcr.Kernels;

/// <summary>
/// Dot products and matrix products over fp32 rows, with the library's fixed accumulation order: element k of a dot product
/// is fused-multiply-added into partial sum (k mod 16), whatever the vector width, and the 16 partial sums are then added in a
/// fixed tree: p[i] += p[i+8] for i &lt; 8, p[i] += p[i+4] for i &lt; 4, p[i] += p[i+2] for i &lt; 2, result p[0] + p[1]. A tail shorter
/// than 16 continues the same lanes (element k still goes to lane k mod 16). A bias, when given, is added to the finished sum.
/// Scalar uses <see cref="MathF.FusedMultiplyAdd"/>, which is exact on every platform; AVX2 uses two 8-lane accumulators.
/// This is the only file group in the library that handles raw pointers; the public-facing wrappers validate every length first.
/// </summary>
internal static unsafe class Gemm
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
            return path == KernelPath.Avx2 ? DotAvx2(pa, pb, a.Length) : DotScalar(pa, pb, a.Length);
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
        if ((m - 1) * ldcRow + (n - 1) * ldcCol >= c.Length)
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
                    c[(int)(i * ldcRow + j * ldcCol)] = bias.IsEmpty ? 0f : 0f + bias[j];
                }
            }

            return;
        }

        fixed (float* pa = a, pb = b, pbias = bias, pc = c)
        {
            float* biasPtr = bias.IsEmpty ? null : pbias;
            if (path == KernelPath.Avx2)
            {
                NTAvx2(pa, lda, pb, ldb, m, n, k, biasPtr, pc, ldcRow, ldcCol);
            }
            else
            {
                NTScalar(pa, lda, pb, ldb, m, n, k, biasPtr, pc, ldcRow, ldcCol);
            }
        }
    }

    private static long RowsNeeded(int rows, int stride, int k) => ((long)rows - 1) * stride + k;

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

    // Finishes a dot product from its two accumulators; the tail continues in the same lanes through memory.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Finish(Vector256<float> acc0, Vector256<float> acc1, float* aTail, float* bTail, int tail)
    {
        if (tail == 0)
        {
            Vector256<float> s = Avx.Add(acc0, acc1);
            Vector128<float> t = Sse.Add(s.GetLower(), s.GetUpper());
            Vector128<float> u = Sse.Add(t, Sse.MoveHighToLow(t, t));
            return u.GetElement(0) + u.GetElement(1);
        }

        float* p = stackalloc float[16];
        Avx.Store(p, acc0);
        Avx.Store(p + 8, acc1);
        for (int j = 0; j < tail; j++)
        {
            p[j] = MathF.FusedMultiplyAdd(aTail[j], bTail[j], p[j]);
        }

        return Reduce16(p);
    }

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

        return Finish(acc0, acc1, a + k, b + k, n - k);
    }

    private static void NTScalar(
        float* a, int lda, float* b, int ldb, int m, int n, int k, float* bias, float* c, long ldcRow, long ldcCol)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float v = DotScalar(a + (long)i * lda, b + (long)j * ldb, k);
                c[i * ldcRow + j * ldcCol] = bias == null ? v : v + bias[j];
            }
        }
    }

    // Four B rows against one A row at a time: the A loads are shared, each output keeps its own pair of accumulators.
    private static void NTAvx2(
        float* a, int lda, float* b, int ldb, int m, int n, int k, float* bias, float* c, long ldcRow, long ldcCol)
    {
        int kMain = k & ~15;
        int tail = k - kMain;
        for (int i = 0; i < m; i++)
        {
            float* ar = a + (long)i * lda;
            int j = 0;
            for (; j + 4 <= n; j += 4)
            {
                float* b0 = b + (long)j * ldb;
                float* b1 = b0 + ldb;
                float* b2 = b1 + ldb;
                float* b3 = b2 + ldb;
                Vector256<float> c00 = Vector256<float>.Zero, c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00, c30 = c00, c31 = c00;
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

                float* at = ar + kMain;
                Store(c, i, j, 0, Finish(c00, c01, at, b0 + kMain, tail), bias, ldcRow, ldcCol);
                Store(c, i, j, 1, Finish(c10, c11, at, b1 + kMain, tail), bias, ldcRow, ldcCol);
                Store(c, i, j, 2, Finish(c20, c21, at, b2 + kMain, tail), bias, ldcRow, ldcCol);
                Store(c, i, j, 3, Finish(c30, c31, at, b3 + kMain, tail), bias, ldcRow, ldcCol);
            }

            for (; j < n; j++)
            {
                Store(c, i, j, 0, DotAvx2(ar, b + (long)j * ldb, k), bias, ldcRow, ldcCol);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(float* c, int i, int j, int dj, float v, float* bias, long ldcRow, long ldcCol)
    {
        int col = j + dj;
        c[i * ldcRow + col * ldcCol] = bias == null ? v : v + bias[col];
    }
}
