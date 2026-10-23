using System.Runtime.CompilerServices;

namespace OneOcr.Decoding;

/// <summary>
/// Exponential and natural logarithm in double precision from additions, subtractions, multiplications and divisions only, in a
/// fixed order, so the beam search gives the same bits on every platform (the runtime's own <c>Math.Exp</c>/<c>Math.Log</c> are
/// not guaranteed to agree across operating systems and CPUs). The operation order is specified in <c>docs/spec/decoding.md</c>.
/// Relative error is below 1e-15 on the ranges the decoder uses.
/// </summary>
internal static class DetMath
{
    private const double Log2E = 1.4426950408889634;
    private const double Ln2Hi = 6.93147180369123816490e-01; // 32 significant bits, so k * Ln2Hi is exact for |k| < 2^20
    private const double Ln2Lo = 1.90821492927058770002e-10;
    private const double Ln2 = 0.6931471805599453;

    /// <summary>exp(x) for x in [-700, 700]; smaller inputs give 0 and larger ones are clamped. Not a number gives 0.</summary>
    public static double Exp(double x)
    {
        if (!(x > -700.0))
        {
            return 0.0;
        }

        if (x > 700.0)
        {
            x = 700.0;
        }

        double k = Math.Floor((x * Log2E) + 0.5);
        double r = (x - (k * Ln2Hi)) - (k * Ln2Lo);
        // Taylor series of exp on |r| <= 0.35, degree 13, Horner's rule.
        double p = 1.0 / 6227020800.0;
        p = (p * r) + (1.0 / 479001600.0);
        p = (p * r) + (1.0 / 39916800.0);
        p = (p * r) + (1.0 / 3628800.0);
        p = (p * r) + (1.0 / 362880.0);
        p = (p * r) + (1.0 / 40320.0);
        p = (p * r) + (1.0 / 5040.0);
        p = (p * r) + (1.0 / 720.0);
        p = (p * r) + (1.0 / 120.0);
        p = (p * r) + (1.0 / 24.0);
        p = (p * r) + (1.0 / 6.0);
        p = (p * r) + 0.5;
        p = (p * r) + 1.0;
        p = (p * r) + 1.0;
        return p * BitConverter.Int64BitsToDouble((long)((int)k + 1023) << 52);
    }

    /// <summary>ln(x) for positive finite x; 0 gives negative infinity, negative numbers and NaN give NaN.</summary>
    public static double Log(double x)
    {
        if (x == 0.0)
        {
            return double.NegativeInfinity;
        }

        if (!(x > 0.0) || double.IsPositiveInfinity(x))
        {
            return x;
        }

        long bits = BitConverter.DoubleToInt64Bits(x);
        int e = (int)((bits >> 52) & 0x7FF);
        if (e == 0)
        {
            // Subnormal: scale by 2^54 first.
            bits = BitConverter.DoubleToInt64Bits(x * 18014398509481984.0);
            e = (int)((bits >> 52) & 0x7FF) - 54;
        }

        e -= 1023;
        double m = BitConverter.Int64BitsToDouble((bits & 0x000FFFFFFFFFFFFFL) | 0x3FF0000000000000L);
        if (m > 1.4142135623730951)
        {
            m *= 0.5;
            e++;
        }

        // ln m = 2 atanh(s), s = (m - 1) / (m + 1), |s| <= 0.1716: 2 s (1 + s^2/3 + s^4/5 + ... + s^18/19).
        double s = (m - 1.0) / (m + 1.0);
        double z = s * s;
        double q = 1.0 / 19.0;
        q = (q * z) + (1.0 / 17.0);
        q = (q * z) + (1.0 / 15.0);
        q = (q * z) + (1.0 / 13.0);
        q = (q * z) + (1.0 / 11.0);
        q = (q * z) + (1.0 / 9.0);
        q = (q * z) + (1.0 / 7.0);
        q = (q * z) + (1.0 / 5.0);
        q = (q * z) + (1.0 / 3.0);
        q = (q * z) + 1.0;
        return (e * Ln2) + (2.0 * s * q);
    }

    /// <summary>ln(exp(a) + exp(b)); negative infinity stands for probability 0.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double LogAdd(double a, double b)
    {
        if (a < b)
        {
            (a, b) = (b, a);
        }

        if (double.IsNegativeInfinity(b))
        {
            return a;
        }

        double d = b - a;
        return d < -40.0 ? a : a + Log(1.0 + Exp(d));
    }
}
