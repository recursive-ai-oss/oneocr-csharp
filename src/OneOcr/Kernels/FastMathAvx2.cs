using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OneOcr.Kernels;

/// <summary>Eight-lane versions of <see cref="FastMath"/>, the same operations in the same order.</summary>
internal static class FastMathAvx2
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Exp(Vector256<float> x)
    {
        x = Avx.Min(x, Vector256.Create(FastMath.Limit));
        x = Avx.Max(x, Vector256.Create(-FastMath.Limit));
        Vector256<float> n = Avx.RoundToNearestInteger(Avx.Multiply(x, Vector256.Create(FastMath.Log2E)));
        Vector256<float> r = Fma.MultiplyAddNegated(n, Vector256.Create(FastMath.Ln2Hi), x);
        r = Fma.MultiplyAddNegated(n, Vector256.Create(FastMath.Ln2Lo), r);
        Vector256<float> p = Vector256.Create(FastMath.C7);
        p = Fma.MultiplyAdd(p, r, Vector256.Create(FastMath.C6));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(FastMath.C5));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(FastMath.C4));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(FastMath.C3));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(FastMath.C2));
        Vector256<float> y = Avx.Add(Fma.MultiplyAdd(p, Avx.Multiply(r, r), r), Vector256.Create(1f));
        Vector256<int> bits = Avx2.ShiftLeftLogical(Avx2.Add(Avx.ConvertToVector256Int32WithTruncation(n), Vector256.Create(127)), 23);
        return Avx.Multiply(y, bits.AsSingle());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Sigmoid(Vector256<float> x)
    {
        Vector256<float> negated = Avx.Xor(x, Vector256.Create(-0f));
        return Avx.Divide(Vector256.Create(1f), Avx.Add(Vector256.Create(1f), Exp(negated)));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Tanh(Vector256<float> x)
    {
        Vector256<float> e = Exp(Avx.Add(x, x));
        return Avx.Subtract(Vector256.Create(1f), Avx.Divide(Vector256.Create(2f), Avx.Add(e, Vector256.Create(1f))));
    }
}
