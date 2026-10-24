using OneOcr.Kernels;

namespace OneOcr.Tests;

public class KernelTests
{
    // Every vector path this machine has, to be compared with scalar.
    private static KernelPath[] Vectors => Dispatch.Available().Where(p => p != KernelPath.Scalar).ToArray();

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

            foreach (KernelPath path in Vectors)
            {
                float v = Gemm.Dot(path, a, b);
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
        var rng = new Random(2);
        foreach (int k in new[] { 1, 5, 15, 16, 17, 31, 32, 33, 48, 100, 192, 576 })
        {
            foreach (int m in new[] { 1, 2, 3, 4, 5, 7, 8, 9, 33 })
            {
                foreach (int n in new[] { 1, 2, 3, 4, 5, 8, 13, 16, 31 })
                {
                    int lda = k + 3, ldb = k + 1;
                    float[] a = Random(rng, m * lda), b = Random(rng, n * ldb), bias = Random(rng, n);
                    foreach (bool useBias in new[] { false, true })
                    {
                        foreach (bool transposed in new[] { false, true })
                        {
                            // Transposed, strided output, or plain row-major.
                            long ldcRow = transposed ? 1 : n, ldcCol = transposed ? m + 1 : 1;
                            var c1 = new float[(m * n * 2) + 5];
                            Gemm.NT(KernelPath.Scalar, a, lda, b, ldb, m, n, k, useBias ? bias : default, c1, ldcRow, ldcCol);
                            foreach (KernelPath path in Vectors)
                            {
                                var c2 = new float[c1.Length];
                                Gemm.NT(path, a, lda, b, ldb, m, n, k, useBias ? bias : default, c2, ldcRow, ldcCol);
                                AssertBitEqual(c1, c2, $"{path} gemm m={m} n={n} k={k} bias={useBias} transposed={transposed}");
                            }
                        }
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
            foreach (KernelPath path in Vectors)
            {
                var v = new float[x.Length];
                Elementwise.Apply(path, function, x, v);
                AssertBitEqual(s, v, $"{path} function {function}");
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
        var rng = new Random(4);
        foreach (int hidden in new[] { 1, 7, 8, 9, 16, 17, 24, 33, 192, 256 })
        {
            float[] z = Random(rng, 4 * hidden, 6);
            float[] c0 = Random(rng, hidden, 2);
            var h0 = new float[hidden];
            float[] cs = (float[])c0.Clone();
            Elementwise.LstmCell(KernelPath.Scalar, z, cs, h0);
            foreach (KernelPath path in Vectors)
            {
                float[] c1 = (float[])c0.Clone();
                var h1 = new float[hidden];
                Elementwise.LstmCell(path, z, c1, h1);
                AssertBitEqual(cs, c1, $"{path} cell state");
                AssertBitEqual(h0, h1, $"{path} hidden state");
            }
        }

        foreach (int n in new[] { 1, 2, 15, 16, 17, 31, 32, 100, 622, 701 })
        {
            float[] x = Random(rng, n, 20);
            float max = x.Max();
            float a = Elementwise.SumExp(KernelPath.Scalar, x, max);
            foreach (KernelPath path in Vectors)
            {
                float b = Elementwise.SumExp(path, x, max);
                Assert.Equal(BitConverter.SingleToInt32Bits(a), BitConverter.SingleToInt32Bits(b));
            }
        }
    }

    [Fact]
    public void DirectConvolutionMatchesDotsOverCopiedWindows()
    {
        var rng = new Random(7);
        foreach (int c in new[] { 16, 32, 48 })
        {
            foreach (var (kh, kw) in new[] { (3, 3), (1, 3), (3, 1), (2, 5) })
            {
                foreach (int oc in new[] { 1, 3, 4, 5, 16, 33 })
                {
                    foreach (int n in new[] { 1, 2, 3, 4, 5, 7, 96 })
                    {
                        int wp = 120, hp = kh + 3, y = 2, x0 = 5;
                        float[] input = Random(rng, hp * wp * c);
                        int k = kh * kw * c;
                        float[] weights = Random(rng, oc * k, 0.3);
                        float[] bias = Random(rng, oc);
                        // Reference: copy each window into a row, one dot product per output.
                        var windows = new float[n * k];
                        for (int i = 0; i < n; i++)
                        {
                            for (int ky = 0; ky < kh; ky++)
                            {
                                Array.Copy(input, (((y + ky) * wp) + x0 + i) * c, windows, (i * k) + (ky * kw * c), kw * c);
                            }
                        }

                        var expected = new float[n * oc];
                        Gemm.NT(KernelPath.Scalar, windows, k, weights, k, n, oc, k, bias, expected, oc, 1);
                        foreach (KernelPath path in Vectors.Where(p => p is KernelPath.Avx2 or KernelPath.Avx512))
                        {
                            var got = new float[n * oc];
                            Gemm.ConvDirect(path, input, wp, c, kh, kw, y, x0, n, weights, oc, bias, got);
                            AssertBitEqual(expected, got, $"{path} direct conv c={c} k={kh}x{kw} oc={oc} n={n}");
                        }
                    }
                }
            }
        }
    }

    [Fact]
    public void DirectConvolutionMayLeaveOutKernelRowsOfZeros()
    {
        // Rows of the zero border add nothing to a finite sum, so a result computed without them is the same to the last bit.
        var rng = new Random(9);
        foreach (int c in new[] { 16, 32 })
        {
            foreach (int oc in new[] { 1, 5, 16, 33 })
            {
                foreach (int n in new[] { 1, 3, 4, 7, 40 })
                {
                    const int kh = 3, kw = 3, wp = 64;
                    int k = kh * kw * c;
                    float[] weights = Random(rng, oc * k, 0.3);
                    float[] bias = Random(rng, oc);
                    foreach (var (zeroTop, zeroBottom) in new[] { (true, false), (false, true), (true, true) })
                    {
                        float[] input = Random(rng, 6 * wp * c);
                        // Output row 2 reads padded rows 2..4; zero the first and/or the last of them.
                        if (zeroTop)
                        {
                            Array.Clear(input, 2 * wp * c, wp * c);
                        }

                        if (zeroBottom)
                        {
                            Array.Clear(input, 4 * wp * c, wp * c);
                        }

                        int start = zeroTop ? 1 : 0, rows = kh - start - (zeroBottom ? 1 : 0);
                        foreach (KernelPath path in Vectors.Where(p => p is KernelPath.Avx2 or KernelPath.Avx512))
                        {
                            var full = new float[n * oc];
                            var skipped = new float[n * oc];
                            Gemm.ConvDirect(path, input, wp, c, kh, kw, 2, 3, n, weights, oc, bias, full);
                            Gemm.ConvDirect(path, input, wp, c, kh, kw, 2, 3, n, weights, oc, bias, skipped, start, rows);
                            AssertBitEqual(full, skipped, $"{path} c={c} oc={oc} n={n} rows {start}+{rows}");
                        }
                    }
                }
            }

            // No kernel row at all leaves the bias.
            if (Vectors.Any(p => p is KernelPath.Avx2 or KernelPath.Avx512))
            {
                KernelPath path = Vectors.First(p => p is KernelPath.Avx2 or KernelPath.Avx512);
                float[] weights = Random(rng, 4 * 3 * 3 * c, 0.3);
                float[] bias = Random(rng, 4);
                var none = new float[4];
                Gemm.ConvDirect(path, new float[6 * 64 * c], 64, c, 3, 3, 2, 3, 1, weights, 4, bias, none, 3, 0);
                AssertBitEqual(bias, none, "no rows");
                Assert.Throws<ArgumentOutOfRangeException>(() => Gemm.ConvDirect(path, new float[6 * 64 * c], 64, c, 3, 3, 2, 3, 1, weights, 4, bias, none, 2, 2));
            }
        }
    }

    [Fact]
    public void SingleChannelConvolutionMatchesDotsOverCopiedWindows()
    {
        if (!Dispatch.Avx2Available)
        {
            return;
        }

        var rng = new Random(8);
        foreach (var (kh, kw) in new[] { (3, 3), (1, 5), (4, 4), (1, 1), (2, 7) })
        {
            foreach (int oc in new[] { 8, 16, 40 })
            {
                foreach (int n in new[] { 1, 3, 96 })
                {
                    int wp = 130, hp = kh + 2, y = 1, x0 = 3, taps = kh * kw;
                    float[] input = Random(rng, hp * wp);
                    float[] weights = Random(rng, oc * taps, 0.5);
                    float[] bias = Random(rng, oc);
                    var weightsT = new float[taps * oc];
                    var padded = new float[oc * 16];
                    for (int o = 0; o < oc; o++)
                    {
                        for (int t = 0; t < taps; t++)
                        {
                            weightsT[(t * oc) + o] = weights[(o * taps) + t];
                            padded[(o * 16) + t] = weights[(o * taps) + t];
                        }
                    }

                    var windows = new float[n * 16];
                    for (int i = 0; i < n; i++)
                    {
                        for (int ky = 0; ky < kh; ky++)
                        {
                            Array.Copy(input, ((y + ky) * wp) + x0 + i, windows, (i * 16) + (ky * kw), kw);
                        }
                    }

                    var expected = new float[n * oc];
                    Gemm.NT(KernelPath.Scalar, windows, 16, padded, 16, n, oc, 16, bias, expected, oc, 1);
                    var got = new float[n * oc];
                    Gemm.ConvSingleChannel(input, wp, kh, kw, y, x0, n, weightsT, oc, bias, got);
                    AssertBitEqual(expected, got, $"single channel {kh}x{kw} oc={oc} n={n}");
                }
            }
        }
    }

    [Fact]
    public void ConvolutionKernelsRejectBuffersThatAreTooSmall()
    {
        if (Dispatch.Best is not (KernelPath.Avx2 or KernelPath.Avx512))
        {
            return;
        }

        KernelPath path = Dispatch.Best;
        var input = new float[3 * 20 * 16];
        var weights = new float[4 * 3 * 3 * 16];
        var bias = new float[4];
        var output = new float[10 * 4];
        // A window that runs past the input.
        Assert.Throws<ArgumentException>(() => Gemm.ConvDirect(path, input, 20, 16, 3, 3, 1, 5, 10, weights, 4, bias, output));
        // Output too small.
        Assert.Throws<ArgumentException>(() => Gemm.ConvDirect(path, input, 20, 16, 3, 3, 0, 0, 10, weights, 4, bias, new float[39]));
        // Row length not a multiple of 16.
        Assert.Throws<ArgumentOutOfRangeException>(() => Gemm.ConvDirect(path, input, 20, 5, 3, 3, 0, 0, 4, weights, 4, bias, output));
        Assert.Throws<ArgumentException>(() => Gemm.ConvSingleChannel(new float[10], 20, 3, 3, 0, 0, 4, new float[72], 8, new float[8], new float[32]));
    }
}
