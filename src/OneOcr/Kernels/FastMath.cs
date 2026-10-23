using System.Runtime.CompilerServices;

namespace OneOcr.Kernels;

/// <summary>
/// Exponential, sigmoid and tanh built from IEEE operations only (add, multiply, divide, fused multiply-add, round to
/// nearest even), in one fixed order, so every path and every platform produces the same bits. The runtime's own math
/// library is deliberately not used. The vector versions in <see cref="FastMathAvx2"/> repeat this sequence operation for operation.
///
/// exp(x): clamp x to [-80, 80]; n = round(x * log2(e)); r = x - n * ln2 (two fused steps, ln2 split in high and low parts,
/// Cody-Waite); exp(r) ~ 1 + r + r^2 * P(r) with P a degree-5 Horner polynomial from the Taylor series (|r| &lt;= ln2/2, relative
/// error below 1e-8); the result is scaled by 2^n built directly in the exponent bits. Worst-case relative error about 2 ulp.
/// sigmoid(x) = 1 / (1 + exp(-x)). tanh(x) = 1 - 2 / (exp(2x) + 1), absolute error about 1.2e-7.
/// </summary>
internal static class FastMath
{
    public const float Limit = 80f;
    public const float Log2E = 1.44269504088896341f;
    public const float Ln2Hi = 0.693145751953125f;
    public const float Ln2Lo = 1.42860677e-6f;
    public const float C2 = 0.5f;
    public const float C3 = 1f / 6f;
    public const float C4 = 1f / 24f;
    public const float C5 = 1f / 120f;
    public const float C6 = 1f / 720f;
    public const float C7 = 1f / 5040f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Exp(float x)
    {
        // The comparisons mirror Avx.Min / Avx.Max: a NaN input becomes the bound.
        x = x < Limit ? x : Limit;
        x = x > -Limit ? x : -Limit;
        float n = MathF.Round(x * Log2E);
        float r = MathF.FusedMultiplyAdd(-n, Ln2Hi, x);
        r = MathF.FusedMultiplyAdd(-n, Ln2Lo, r);
        float p = C7;
        p = MathF.FusedMultiplyAdd(p, r, C6);
        p = MathF.FusedMultiplyAdd(p, r, C5);
        p = MathF.FusedMultiplyAdd(p, r, C4);
        p = MathF.FusedMultiplyAdd(p, r, C3);
        p = MathF.FusedMultiplyAdd(p, r, C2);
        float y = MathF.FusedMultiplyAdd(p, r * r, r) + 1f;
        return y * BitConverter.Int32BitsToSingle(((int)n + 127) << 23);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float x) => 1f / (1f + Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Tanh(float x) => 1f - 2f / (Exp(x + x) + 1f);
}
