using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OneOcr.Kernels;

/// <summary>LSTM cell update and softmax normalizer, scalar and AVX2, bit-identical.</summary>
internal static unsafe class Elementwise
{
    /// <summary>
    /// One LSTM step after the gate pre-activations are known. <paramref name="z"/> holds 4H values in PyTorch gate order
    /// (input, forget, cell, output). With i, f, o = sigmoid and g = tanh of their slices:
    /// <c>t = i * g</c> (rounded), <c>c' = fma(f, c, t)</c>, <c>h = o * tanh(c')</c>. Updates <paramref name="c"/> in place and writes <paramref name="h"/>.
    /// </summary>
    public static void LstmCell(KernelPath path, ReadOnlySpan<float> z, Span<float> c, Span<float> h)
    {
        int hidden = c.Length;
        if (h.Length != hidden || z.Length != 4 * hidden)
        {
            throw new ArgumentException("z must hold 4H values and c and h H values each.");
        }

        Dispatch.Require(path);
        if (hidden == 0)
        {
            return;
        }

        fixed (float* pz = z, pc = c, ph = h)
        {
            int j = 0;
            if (path == KernelPath.Avx2)
            {
                for (; j + 8 <= hidden; j += 8)
                {
                    Vector256<float> i = FastMathAvx2.Sigmoid(Avx.LoadVector256(pz + j));
                    Vector256<float> f = FastMathAvx2.Sigmoid(Avx.LoadVector256(pz + hidden + j));
                    Vector256<float> g = FastMathAvx2.Tanh(Avx.LoadVector256(pz + 2 * hidden + j));
                    Vector256<float> o = FastMathAvx2.Sigmoid(Avx.LoadVector256(pz + 3 * hidden + j));
                    Vector256<float> cNew = Fma.MultiplyAdd(f, Avx.LoadVector256(pc + j), Avx.Multiply(i, g));
                    Avx.Store(pc + j, cNew);
                    Avx.Store(ph + j, Avx.Multiply(o, FastMathAvx2.Tanh(cNew)));
                }
            }

            for (; j < hidden; j++)
            {
                float i = FastMath.Sigmoid(pz[j]);
                float f = FastMath.Sigmoid(pz[hidden + j]);
                float g = FastMath.Tanh(pz[2 * hidden + j]);
                float o = FastMath.Sigmoid(pz[3 * hidden + j]);
                float cNew = MathF.FusedMultiplyAdd(f, pc[j], i * g);
                pc[j] = cNew;
                ph[j] = o * FastMath.Tanh(cNew);
            }
        }
    }

    /// <summary>
    /// Sum over j of exp(x[j] - shift), accumulated like a dot product: element j adds into partial (j mod 16), and the 16 partials
    /// are combined in the same fixed tree. Used for the softmax denominator.
    /// </summary>
    public static float SumExp(KernelPath path, ReadOnlySpan<float> x, float shift)
    {
        Dispatch.Require(path);
        if (x.IsEmpty)
        {
            return 0f;
        }

        float* p = stackalloc float[16];
        for (int q = 0; q < 16; q++)
        {
            p[q] = 0f;
        }

        fixed (float* px = x)
        {
            int n = x.Length;
            int k = 0;
            if (path == KernelPath.Avx2)
            {
                Vector256<float> acc0 = Vector256<float>.Zero;
                Vector256<float> acc1 = Vector256<float>.Zero;
                Vector256<float> s = Vector256.Create(shift);
                for (; k + 16 <= n; k += 16)
                {
                    acc0 = Avx.Add(acc0, FastMathAvx2.Exp(Avx.Subtract(Avx.LoadVector256(px + k), s)));
                    acc1 = Avx.Add(acc1, FastMathAvx2.Exp(Avx.Subtract(Avx.LoadVector256(px + k + 8), s)));
                }

                Avx.Store(p, acc0);
                Avx.Store(p + 8, acc1);
            }
            else
            {
                for (; k + 16 <= n; k += 16)
                {
                    for (int q = 0; q < 16; q++)
                    {
                        p[q] += FastMath.Exp(px[k + q] - shift);
                    }
                }
            }

            for (int q = 0; k < n; k++, q++)
            {
                p[q] += FastMath.Exp(px[k] - shift);
            }
        }

        for (int q = 0; q < 8; q++)
        {
            p[q] += p[q + 8];
        }

        for (int q = 0; q < 4; q++)
        {
            p[q] += p[q + 4];
        }

        p[0] += p[2];
        p[1] += p[3];
        return p[0] + p[1];
    }

    /// <summary>Applies exp (0), sigmoid (1) or tanh (anything else) elementwise on the chosen path; lets tests compare the paths directly.</summary>
    public static void Apply(KernelPath path, int function, ReadOnlySpan<float> x, Span<float> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Lengths differ.");
        }

        Dispatch.Require(path);
        int k = 0;
        if (path == KernelPath.Avx2)
        {
            fixed (float* px = x, py = y)
            {
                for (; k + 8 <= x.Length; k += 8)
                {
                    Vector256<float> v = Avx.LoadVector256(px + k);
                    Avx.Store(py + k, function switch { 0 => FastMathAvx2.Exp(v), 1 => FastMathAvx2.Sigmoid(v), _ => FastMathAvx2.Tanh(v) });
                }
            }
        }

        for (; k < x.Length; k++)
        {
            y[k] = function switch { 0 => FastMath.Exp(x[k]), 1 => FastMath.Sigmoid(x[k]), _ => FastMath.Tanh(x[k]) };
        }
    }
}
