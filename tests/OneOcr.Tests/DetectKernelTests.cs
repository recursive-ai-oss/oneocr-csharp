using OneOcr.Kernels;

namespace OneOcr.Tests;

/// <summary>The detector's kernels: every vector path equals the scalar path bit for bit, and scalar equals the order rule written out.</summary>
public class DetectKernelTests
{
    private static KernelPath[] Vectors => Dispatch.Available().Where(p => p != KernelPath.Scalar).ToArray();

    private static float[] Random(Random rng, int n, double scale = 1)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++)
        {
            // Some exact zeros and negative zeros, to exercise the sign-of-zero cases of the tree.
            double r = rng.NextDouble();
            v[i] = r < 0.03 ? 0f : r < 0.05 ? -0f : (float)(((rng.NextDouble() * 2) - 1) * scale);
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

    // The order rule of numerics.md for a sum of up to nine products: partial k = fma(w[k], x[k], +0), the rest +0, the fixed tree, the bias.
    private static float RuleSum(float[] w, float[] x, int taps, float bias)
    {
        var p = new float[16];
        for (int k = 0; k < taps; k++)
        {
            p[k] = MathF.FusedMultiplyAdd(w[k], x[k], 0f);
        }

        for (int i = 0; i < 8; i++) { p[i] += p[i + 8]; }
        for (int i = 0; i < 4; i++) { p[i] += p[i + 4]; }
        p[0] += p[2];
        p[1] += p[3];
        return (p[0] + p[1]) + bias;
    }

    [Fact]
    public void DepthwiseFollowsTheOrderRuleOnEveryPath()
    {
        var rng = new Random(11);
        foreach (int c in new[] { 16, 32, 48, 80, 224 })
        {
            foreach ((int h, int w) in new[] { (2, 2), (4, 6), (6, 4), (8, 10), (14, 2), (2, 18) })
            {
                foreach (int stride in new[] { 1, 2 })
                {
                    foreach (bool relu in new[] { false, true })
                    {
                        float[] input = Random(rng, h * w * c), weights = Random(rng, 9 * c, 0.7), bias = Random(rng, c, 0.3);
                        int oh = h / stride, ow = w / stride;
                        int y0 = rng.Next(0, oh), y1 = rng.Next(y0, oh + 1);
                        var expected = new float[(y1 - y0) * ow * c];
                        for (int oy = y0; oy < y1; oy++)
                        {
                            for (int ox = 0; ox < ow; ox++)
                            {
                                for (int k = 0; k < c; k++)
                                {
                                    var wv = new float[9];
                                    var xv = new float[9];
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            int iy = (oy * stride) - 1 + ky, ix = (ox * stride) - 1 + kx;
                                            wv[(ky * 3) + kx] = weights[(((ky * 3) + kx) * c) + k];
                                            // Padding is a zero input, written out here instead of skipped.
                                            xv[(ky * 3) + kx] = iy >= 0 && iy < h && ix >= 0 && ix < w ? input[((((iy * w) + ix)) * c) + k] : 0f;
                                        }
                                    }

                                    float v = RuleSum(wv, xv, 9, bias[k]);
                                    expected[(((((oy - y0) * ow) + ox)) * c) + k] = relu ? (v > 0 ? v : 0f) : v;
                                }
                            }
                        }

                        foreach (KernelPath path in Dispatch.Available())
                        {
                            var got = new float[expected.Length];
                            DetectKernels.Depthwise(path, input, h, w, c, stride, weights, bias, relu, y0, y1, got);
                            AssertBitEqual(expected, got, $"depthwise {path} c={c} {h}x{w} s={stride} relu={relu}");
                        }
                    }
                }
            }
        }
    }

    [Fact]
    public void StemFollowsTheOrderRuleOnEveryPath()
    {
        var rng = new Random(12);
        foreach (int oc in new[] { 16, 32 })
        {
            foreach ((int h, int w) in new[] { (2, 2), (4, 8), (10, 6), (32, 40) })
            {
                float[] input = Random(rng, h * w), weights = Random(rng, 9 * oc, 0.7), bias = Random(rng, oc, 0.3);
                int oh = h / 2, ow = w / 2;
                var expected = new float[oh * ow * oc];
                for (int oy = 0; oy < oh; oy++)
                {
                    for (int ox = 0; ox < ow; ox++)
                    {
                        var xv = new float[9];
                        for (int ky = 0; ky < 3; ky++)
                        {
                            for (int kx = 0; kx < 3; kx++)
                            {
                                int iy = (oy * 2) - 1 + ky, ix = (ox * 2) - 1 + kx;
                                xv[(ky * 3) + kx] = iy >= 0 && iy < h && ix >= 0 && ix < w ? input[(iy * w) + ix] : 0f;
                            }
                        }

                        for (int k = 0; k < oc; k++)
                        {
                            var wv = new float[9];
                            for (int t = 0; t < 9; t++)
                            {
                                wv[t] = weights[(t * oc) + k];
                            }

                            float v = RuleSum(wv, xv, 9, bias[k]);
                            expected[((((oy * ow) + ox)) * oc) + k] = v > 0 ? v : 0f;
                        }
                    }
                }

                foreach (KernelPath path in Dispatch.Available())
                {
                    var got = new float[expected.Length];
                    DetectKernels.Stem(path, input, h, w, weights, bias, oc, 0, oh, got);
                    AssertBitEqual(expected, got, $"stem {path} oc={oc} {h}x{w}");
                }
            }
        }
    }

    [Fact]
    public void UpsampledAdditionReluAndMeansAgreeOnEveryPath()
    {
        var rng = new Random(13);
        foreach (int c in new[] { 16, 48, 80 })
        {
            foreach ((int h, int w) in new[] { (2, 2), (6, 4), (10, 14) })
            {
                float[] dst0 = Random(rng, h * w * c), src = Random(rng, (h / 2) * (w / 2) * c);
                var expected = (float[])dst0.Clone();
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        for (int k = 0; k < c; k++)
                        {
                            expected[(((y * w) + x) * c) + k] += src[((((y / 2) * (w / 2)) + (x / 2)) * c) + k];
                        }
                    }
                }

                foreach (KernelPath path in Dispatch.Available())
                {
                    float[] dst = (float[])dst0.Clone();
                    int split = h / 2;
                    DetectKernels.AddUpsampled(path, dst, h, w, c, src, 0, split);
                    DetectKernels.AddUpsampled(path, dst, h, w, c, src, split, h);
                    AssertBitEqual(expected, dst, $"add-upsampled {path} c={c} {h}x{w}");
                }

                var mean = new float[c];
                for (int k = 0; k < c; k++)
                {
                    float sum = 0f;
                    for (int p = 0; p < h * w; p++)
                    {
                        sum += dst0[(p * c) + k];
                    }

                    mean[k] = sum / (h * w);
                }

                foreach (KernelPath path in Dispatch.Available())
                {
                    var got = new float[c];
                    DetectKernels.ChannelMeans(path, dst0, h * w, c, got);
                    AssertBitEqual(mean, got, $"channel means {path} c={c} {h}x{w}");
                }
            }
        }

        float[] values = Random(rng, 160);
        values[3] = float.NaN;
        values[4] = -0f;
        float[] relu = values.Select(v => v > 0 ? v : 0f).ToArray();
        foreach (KernelPath path in Dispatch.Available())
        {
            float[] got = (float[])values.Clone();
            DetectKernels.Relu(path, got);
            AssertBitEqual(relu, got, $"relu {path}");
        }
    }

    [Fact]
    public void KernelsRejectShapesThatDoNotFit()
    {
        float[] x = new float[16 * 16], w = new float[9 * 16], b = new float[16], o = new float[16 * 16];
        Assert.Throws<ArgumentOutOfRangeException>(() => DetectKernels.Depthwise(KernelPath.Scalar, x, 3, 4, 16, 2, w, b, true, 0, 1, o));
        Assert.Throws<ArgumentOutOfRangeException>(() => DetectKernels.Depthwise(KernelPath.Scalar, x, 4, 4, 15, 1, w, b, true, 0, 1, o));
        Assert.Throws<ArgumentException>(() => DetectKernels.Depthwise(KernelPath.Scalar, x, 4, 4, 16, 1, w, b, true, 0, 4, new float[10]));
        Assert.Throws<ArgumentException>(() => DetectKernels.Depthwise(KernelPath.Scalar, new float[10], 4, 4, 16, 1, w, b, true, 0, 4, o));
        Assert.Throws<ArgumentOutOfRangeException>(() => DetectKernels.Stem(KernelPath.Scalar, x, 4, 4, w, b, 16, 0, 3, o));
        Assert.Throws<ArgumentException>(() => DetectKernels.AddUpsampled(KernelPath.Scalar, o, 4, 4, 16, new float[3], 0, 4));

        // A band of input rows must hold every row the output rows read: output row 2 of a 4-row image reads rows 1..3.
        float[] band = new float[2 * 4 * 16];
        Assert.Throws<ArgumentException>(() => DetectKernels.Depthwise(KernelPath.Scalar, band, 4, 4, 16, 1, w, b, true, 2, 3, o, 1, 2));
        DetectKernels.Depthwise(KernelPath.Scalar, new float[3 * 4 * 16], 4, 4, 16, 1, w, b, true, 2, 3, o, 1, 3);
        Assert.Throws<ArgumentOutOfRangeException>(() => DetectKernels.AddUpsampled(KernelPath.Scalar, o, 4, 4, 16, new float[2 * 2 * 16], 1, 3, 2));
    }
}
