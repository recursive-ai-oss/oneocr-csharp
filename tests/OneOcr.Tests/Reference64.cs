using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Tests;

/// <summary>
/// A straightforward double-precision implementation of the <c>crnn-v0</c> forward pass (exact math functions, plain loops),
/// used as the reference the float32 kernels are measured against.
/// </summary>
internal static class Reference64
{
    public static Dictionary<string, (int[] Shape, double[] Data)> Forward(
        ModelContainer c, TestModels.Shape s, ReadOnlySpan<byte> gray, int width, out int steps)
    {
        var layers = new Dictionary<string, (int[], double[])>();
        int h = s.Height, w = width;
        double[] x = new double[gray.Length];
        for (int i = 0; i < x.Length; i++)
        {
            // Same single-precision arithmetic as the model's normalization table; this stage is not under test.
            float f = gray[i] / 255f;
            x[i] = (f - c.Input.Mean) / c.Input.Std;
        }

        layers["input"] = ([1, h, w], x);
        int ch = 1;
        for (int i = 0; i < s.Conv.Length; i++)
        {
            double[] weight = D(c.ReadTensor($"conv.{i}.weight"));
            double[] bias = D(c.ReadTensor($"conv.{i}.bias"));
            int oc = s.Conv[i].Out;
            var y = new double[oc * h * w];
            for (int o = 0; o < oc; o++)
            {
                for (int yy = 0; yy < h; yy++)
                {
                    for (int xx = 0; xx < w; xx++)
                    {
                        double sum = 0;
                        for (int ic = 0; ic < ch; ic++)
                        {
                            for (int ky = 0; ky < 3; ky++)
                            {
                                for (int kx = 0; kx < 3; kx++)
                                {
                                    int sy = yy + ky - 1, sx = xx + kx - 1;
                                    if (sy < 0 || sy >= h || sx < 0 || sx >= w)
                                    {
                                        continue;
                                    }

                                    sum += weight[(((((o * ch) + ic) * 3) + ky) * 3) + kx] * x[(((ic * h) + sy) * w) + sx];
                                }
                            }
                        }

                        y[(((o * h) + yy) * w) + xx] = Math.Max(sum + bias[o], 0);
                    }
                }
            }

            int ph = s.Conv[i].PoolH, pw = s.Conv[i].PoolW;
            int nh = h / ph, nw = w / pw;
            var p = new double[oc * nh * nw];
            for (int o = 0; o < oc; o++)
            {
                for (int yy = 0; yy < nh; yy++)
                {
                    for (int xx = 0; xx < nw; xx++)
                    {
                        double m = 0;
                        for (int dy = 0; dy < ph; dy++)
                        {
                            for (int dx = 0; dx < pw; dx++)
                            {
                                m = Math.Max(m, y[(((o * h) + (yy * ph) + dy) * w) + (xx * pw) + dx]);
                            }
                        }

                        p[(((o * nh) + yy) * nw) + xx] = m;
                    }
                }
            }

            x = p;
            h = nh;
            w = nw;
            ch = oc;
            layers[$"conv.{i}"] = ([ch, h, w], x);
        }

        steps = w;
        var seq = new double[w * ch];
        for (int cc = 0; cc < ch; cc++)
        {
            for (int t = 0; t < w; t++)
            {
                seq[(t * ch) + cc] = x[(cc * w) + t];
            }
        }

        int hid = s.Hidden;
        int inSize = ch;
        double[] cur = seq;
        for (int l = 0; l < s.LstmLayers; l++)
        {
            var next = new double[w * 2 * hid];
            for (int dir = 0; dir < 2; dir++)
            {
                string sfx = dir == 0 ? string.Empty : "_reverse";
                double[] wih = D(c.ReadTensor($"lstm.weight_ih_l{l}{sfx}"));
                double[] whh = D(c.ReadTensor($"lstm.weight_hh_l{l}{sfx}"));
                double[] bih = D(c.ReadTensor($"lstm.bias_ih_l{l}{sfx}"));
                double[] bhh = D(c.ReadTensor($"lstm.bias_hh_l{l}{sfx}"));
                var hs = new double[hid];
                var cs = new double[hid];
                for (int step = 0; step < w; step++)
                {
                    int t = dir == 0 ? step : w - 1 - step;
                    var z = new double[4 * hid];
                    for (int g = 0; g < 4 * hid; g++)
                    {
                        double sum = bih[g] + bhh[g];
                        for (int k = 0; k < inSize; k++)
                        {
                            sum += wih[(g * inSize) + k] * cur[(t * inSize) + k];
                        }

                        for (int k = 0; k < hid; k++)
                        {
                            sum += whh[(g * hid) + k] * hs[k];
                        }

                        z[g] = sum;
                    }

                    for (int j = 0; j < hid; j++)
                    {
                        double ig = Sig(z[j]), fg = Sig(z[hid + j]), gg = Math.Tanh(z[(2 * hid) + j]), og = Sig(z[(3 * hid) + j]);
                        cs[j] = (fg * cs[j]) + (ig * gg);
                        hs[j] = og * Math.Tanh(cs[j]);
                        next[(t * 2 * hid) + (dir * hid) + j] = hs[j];
                    }
                }
            }

            cur = next;
            inSize = 2 * hid;
            layers[$"lstm.l{l}"] = ([w, 2 * hid], cur);
        }

        double[] fw = D(c.ReadTensor("fc.weight"));
        double[] fb = D(c.ReadTensor("fc.bias"));
        var logits = new double[w * s.Classes];
        for (int t = 0; t < w; t++)
        {
            for (int k = 0; k < s.Classes; k++)
            {
                double sum = fb[k];
                for (int j = 0; j < 2 * hid; j++)
                {
                    sum += fw[(k * 2 * hid) + j] * cur[(t * 2 * hid) + j];
                }

                logits[(t * s.Classes) + k] = sum;
            }
        }

        layers["fc"] = ([w, s.Classes], logits);
        return layers;
    }

    private static double Sig(double v) => 1 / (1 + Math.Exp(-v));

    private static double[] D(float[] v)
    {
        var r = new double[v.Length];
        for (int i = 0; i < r.Length; i++)
        {
            r[i] = v[i];
        }

        return r;
    }
}
