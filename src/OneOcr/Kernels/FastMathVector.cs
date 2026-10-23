using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace OneOcr.Kernels;

// Vector versions of FastMath: the same operations in the same order on 4, 8 and 16 lanes. The three classes are the same code
// with the vector type changed. Comparisons and selects are used instead of Min/Max so that a NaN input becomes the bound on
// every path, as in the scalar code; Round is round-half-to-even like MathF.Round.

/// <summary>4-lane versions of <see cref="FastMath"/>.</summary>
internal static class FastMathV128
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Exp(Vector128<float> x)
    {
        Vector128<float> hi = Vector128.Create(FastMath.Limit);
        Vector128<float> lo = Vector128.Create(-FastMath.Limit);
        x = Vector128.ConditionalSelect(Vector128.LessThan(x, hi), x, hi);
        x = Vector128.ConditionalSelect(Vector128.GreaterThan(x, lo), x, lo);
        Vector128<float> n = Vector128.Round(Vector128.Multiply(x, Vector128.Create(FastMath.Log2E)), MidpointRounding.ToEven);
        Vector128<float> negN = Vector128.Negate(n);
        Vector128<float> r = Vector128.FusedMultiplyAdd(negN, Vector128.Create(FastMath.Ln2Hi), x);
        r = Vector128.FusedMultiplyAdd(negN, Vector128.Create(FastMath.Ln2Lo), r);
        Vector128<float> p = Vector128.Create(FastMath.C7);
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(FastMath.C6));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(FastMath.C5));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(FastMath.C4));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(FastMath.C3));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(FastMath.C2));
        Vector128<float> y = Vector128.Add(Vector128.FusedMultiplyAdd(p, Vector128.Multiply(r, r), r), Vector128.Create(1f));
        Vector128<int> bits = Vector128.ShiftLeft(Vector128.Add(Vector128.ConvertToInt32(n), Vector128.Create(127)), 23);
        return Vector128.Multiply(y, bits.AsSingle());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Sigmoid(Vector128<float> x) =>
        Vector128.Divide(Vector128.Create(1f), Vector128.Add(Vector128.Create(1f), Exp(Vector128.Negate(x))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Tanh(Vector128<float> x)
    {
        Vector128<float> e = Exp(Vector128.Add(x, x));
        return Vector128.Subtract(Vector128.Create(1f), Vector128.Divide(Vector128.Create(2f), Vector128.Add(e, Vector128.Create(1f))));
    }
}

/// <summary>8-lane versions of <see cref="FastMath"/>.</summary>
internal static class FastMathV256
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Exp(Vector256<float> x)
    {
        Vector256<float> hi = Vector256.Create(FastMath.Limit);
        Vector256<float> lo = Vector256.Create(-FastMath.Limit);
        x = Vector256.ConditionalSelect(Vector256.LessThan(x, hi), x, hi);
        x = Vector256.ConditionalSelect(Vector256.GreaterThan(x, lo), x, lo);
        Vector256<float> n = Vector256.Round(Vector256.Multiply(x, Vector256.Create(FastMath.Log2E)), MidpointRounding.ToEven);
        Vector256<float> negN = Vector256.Negate(n);
        Vector256<float> r = Vector256.FusedMultiplyAdd(negN, Vector256.Create(FastMath.Ln2Hi), x);
        r = Vector256.FusedMultiplyAdd(negN, Vector256.Create(FastMath.Ln2Lo), r);
        Vector256<float> p = Vector256.Create(FastMath.C7);
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(FastMath.C6));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(FastMath.C5));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(FastMath.C4));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(FastMath.C3));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(FastMath.C2));
        Vector256<float> y = Vector256.Add(Vector256.FusedMultiplyAdd(p, Vector256.Multiply(r, r), r), Vector256.Create(1f));
        Vector256<int> bits = Vector256.ShiftLeft(Vector256.Add(Vector256.ConvertToInt32(n), Vector256.Create(127)), 23);
        return Vector256.Multiply(y, bits.AsSingle());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Sigmoid(Vector256<float> x) =>
        Vector256.Divide(Vector256.Create(1f), Vector256.Add(Vector256.Create(1f), Exp(Vector256.Negate(x))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Tanh(Vector256<float> x)
    {
        Vector256<float> e = Exp(Vector256.Add(x, x));
        return Vector256.Subtract(Vector256.Create(1f), Vector256.Divide(Vector256.Create(2f), Vector256.Add(e, Vector256.Create(1f))));
    }
}

/// <summary>16-lane versions of <see cref="FastMath"/>.</summary>
internal static class FastMathV512
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Exp(Vector512<float> x)
    {
        Vector512<float> hi = Vector512.Create(FastMath.Limit);
        Vector512<float> lo = Vector512.Create(-FastMath.Limit);
        x = Vector512.ConditionalSelect(Vector512.LessThan(x, hi), x, hi);
        x = Vector512.ConditionalSelect(Vector512.GreaterThan(x, lo), x, lo);
        Vector512<float> n = Vector512.Round(Vector512.Multiply(x, Vector512.Create(FastMath.Log2E)), MidpointRounding.ToEven);
        Vector512<float> negN = Vector512.Negate(n);
        Vector512<float> r = Vector512.FusedMultiplyAdd(negN, Vector512.Create(FastMath.Ln2Hi), x);
        r = Vector512.FusedMultiplyAdd(negN, Vector512.Create(FastMath.Ln2Lo), r);
        Vector512<float> p = Vector512.Create(FastMath.C7);
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(FastMath.C6));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(FastMath.C5));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(FastMath.C4));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(FastMath.C3));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(FastMath.C2));
        Vector512<float> y = Vector512.Add(Vector512.FusedMultiplyAdd(p, Vector512.Multiply(r, r), r), Vector512.Create(1f));
        Vector512<int> bits = Vector512.ShiftLeft(Vector512.Add(Vector512.ConvertToInt32(n), Vector512.Create(127)), 23);
        return Vector512.Multiply(y, bits.AsSingle());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Sigmoid(Vector512<float> x) =>
        Vector512.Divide(Vector512.Create(1f), Vector512.Add(Vector512.Create(1f), Exp(Vector512.Negate(x))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Tanh(Vector512<float> x)
    {
        Vector512<float> e = Exp(Vector512.Add(x, x));
        return Vector512.Subtract(Vector512.Create(1f), Vector512.Divide(Vector512.Create(2f), Vector512.Add(e, Vector512.Create(1f))));
    }
}
