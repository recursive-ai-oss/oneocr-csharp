using OneOcr.Kernels;

namespace OneOcr.Tests;

public class KernelTests
{
    private static bool Avx2 => Dispatch.Avx2Available;

    private static float[] Random(Random rng, int n, double scale = 1)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++)
        {
            v[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        }

        return v;
    }

    private static void AssertBitEqual(ReadOnlySpan<float> a, ReadOnlySpan<float> b, string what)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
            {
                Assert.Fail($"{what}: element {i} differs: {a[i]:R} vs {b[i]:R}");
            }
        }
    }

    [Fact]
    public void DotMatchesTheSpecifiedOrderAndAvx2MatchesScalarBitForBit()
    {
        var rng = new Random(1);
        for (int n = 1; n <= 300; n++)
        {
            float[] a = Random(rng, n), b = Random(rng, n);
            float scalar = Gemm.Dot(KernelPath.Scalar, a, b);

            // The order rule written out independently: lane k mod 16, then the fixed tree.
            var p = new float[16];
            for (int k = 0; k < n; k++)
            {
                p[k % 16] = MathF.FusedMultiplyAdd(a[k], b[k], p[k % 16]);
            }

            float[] l8 = new float[8], l4 = new float[4];
            for (int i = 0; i < 8; i++) { l8[i] = p[i] + p[i + 8]; }
            for (int i = 0; i < 4; i++) { l4[i] = l8[i] + l8[i + 4]; }
            float expected = (l4[0] + l4[2]) + (l4[1] + l4[3]);
            Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(scalar));

            if (Avx2)
            {
                float v = Gemm.Dot(KernelPath.Avx2, a, b);
                Assert.Equal(BitConverter.SingleToInt32Bits(scalar), BitConverter.SingleToInt32Bits(v));
            }

            double exact = 0;
            for (int k = 0; k < n; k++)
            {
                exact += (double)a[k] * b[k];
            }

            Assert.True(Math.Abs(scalar - exact) <= 1e-5 * Math.Sqrt(n) + 1e-6 * Math.Abs(exact), $"n={n}");
        }
    }

    [Fact]
    public void GemmPathsAgreeOnOddShapesStridesAndBias()
    {
        if (!Avx2)
        {
            return;
        }

        var rng = new Random(2);
        foreach (int k in new[] { 1, 5, 15, 16, 17, 31, 32, 33, 100, 576 })
        {
            foreach (int m in new[] { 1, 2, 7 })
            {
                foreach (int n in new[] { 1, 3, 4, 5, 8, 13 })
                {
                    int lda = k + 3, ldb = k + 1;
                    float[] a = Random(rng, m * lda), b = Random(rng, n * ldb), bias = Random(rng, n);
                    foreach (bool useBias in new[] { false, true })
                    {
                        var c1 = new float[m * n * 2 + 5];
                        var c2 = new float[c1.Length];
                        // Transposed, strided output.
                        Gemm.NT(KernelPath.Scalar, a, lda, b, ldb, m, n, k, useBias ? bias : default, c1, 1, m + 1);
                        Gemm.NT(KernelPath.Avx2, a, lda, b, ldb, m, n, k, useBias ? bias : default, c2, 1, m + 1);
                        AssertBitEqual(c1, c2, $"gemm m={m} n={n} k={k} bias={useBias}");
                    }
                }
            }
        }
    }

    [Fact]
    public void GemmRejectsSpansSmallerThanTheirShape()
    {
        var a = new float[10];
        var c = new float[4];
        Assert.Throws<ArgumentException>(() => Gemm.NT(KernelPath.Scalar, a, 5, a, 5, 3, 1, 5, default, c, 1, 1));
        Assert.Throws<ArgumentException>(() => Gemm.NT(KernelPath.Scalar, a, 5, a, 5, 2, 2, 5, default, new float[3], 2, 1));
    }

    [Fact]
    public void TranscendentalsAreBitIdenticalAcrossPathsAndAccurate()
    {
        var rng = new Random(3);
        var xs = new List<float> { 0f, -0f, 1f, -1f, 80f, -80f, 81f, -1000f, 1000f, 1e-20f, -1e-20f, 0.34657f, float.NaN, float.PositiveInfinity, float.NegativeInfinity };
        for (int i = 0; i < 200_000; i++)
        {
            xs.Add((float)((rng.NextDouble() * 2 - 1) * (i % 3 == 0 ? 100 : i % 3 == 1 ? 10 : 0.5)));
        }

        // Random bit patterns too: every exponent, both signs.
        for (int i = 0; i < 100_000; i++)
        {
            xs.Add(BitConverter.Int32BitsToSingle(rng.Next()));
        }

        float[] x = [.. xs];
        for (int function = 0; function < 3; function++)
        {
            var s = new float[x.Length];
            Elementwise.Apply(KernelPath.Scalar, function, x, s);
            if (Avx2)
            {
                var v = new float[x.Length];
                Elementwise.Apply(KernelPath.Avx2, function, x, v);
                AssertBitEqual(s, v, $"function {function}");
            }

            double maxRel = 0, maxAbs = 0;
            for (int i = 0; i < x.Length; i++)
            {
                if (!float.IsFinite(x[i]))
                {
                    continue;
                }

                double xd = Math.Clamp(x[i], -80.0, 80.0);
                double reference = function switch { 0 => Math.Exp(xd), 1 => 1 / (1 + Math.Exp(-xd)), _ => Math.Tanh(xd) };
                double err = Math.Abs(s[i] - reference);
                maxAbs = Math.Max(maxAbs, err);
                if (reference > 1e-30)
                {
                    maxRel = Math.Max(maxRel, err / reference);
                }
            }

            // exp: relative error within a few ulp (2^-23 each); sigmoid: relative; tanh: absolute.
            if (function == 2)
            {
                Assert.True(maxAbs < 2e-7, $"tanh abs error {maxAbs}");
            }
            else
            {
                Assert.True(maxRel < 4e-7, $"function {function} rel error {maxRel}");
            }
        }
    }

    [Fact]
    public void LstmCellAndSumExpAgreeAcrossPaths()
    {
        if (!Avx2)
        {
            return;
        }

        var rng = new Random(4);
        foreach (int hidden in new[] { 1, 7, 8, 9, 24, 33, 256 })
        {
            float[] z = Random(rng, 4 * hidden, 6);
            float[] c0 = Random(rng, hidden, 2);
            float[] c1 = (float[])c0.Clone();
            var h0 = new float[hidden];
            var h1 = new float[hidden];
            Elementwise.LstmCell(KernelPath.Scalar, z, c0, h0);
            Elementwise.LstmCell(KernelPath.Avx2, z, c1, h1);
            AssertBitEqual(c0, c1, "cell state");
            AssertBitEqual(h0, h1, "hidden state");
        }

        foreach (int n in new[] { 1, 2, 15, 16, 17, 100, 700, 701 })
        {
            float[] x = Random(rng, n, 20);
            float max = x.Max();
            float a = Elementwise.SumExp(KernelPath.Scalar, x, max);
            float b = Elementwise.SumExp(KernelPath.Avx2, x, max);
            Assert.Equal(BitConverter.SingleToInt32Bits(a), BitConverter.SingleToInt32Bits(b));
        }
    }
}
