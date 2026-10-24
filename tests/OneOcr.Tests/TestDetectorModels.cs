using System.Text;
using OneOcr.Model;

namespace OneOcr.Tests;

/// <summary>Builds random <c>dbnet-lite-v0</c> containers (the layer sizes of the shipped architecture or smaller ones).</summary>
internal static class TestDetectorModels
{
    public sealed record Shape(int Stem, int[] Widths, int[] Depths, int Fpn, int Text, int Layout, int LayoutBlocks, string[] Classes)
    {
        public int ClassCount => Classes.Length;
    }

    public static readonly string[] AllClasses = ["text", "title", "list", "table", "figure", "header_footer", "form_field", "checkbox"];

    /// <summary>The configuration the architecture draft gives: 309,077 parameters.</summary>
    public static Shape Default { get; } = new(16, [32, 72, 144, 224], [2, 3, 4, 3], 56, 32, 112, 3, AllClasses);

    /// <summary>Fewer, narrower blocks, with widths that are not multiples of 16, for quick tests.</summary>
    public static Shape Small { get; } = new(16, [16, 24, 40, 56], [1, 2, 1, 1], 20, 12, 28, 2, AllClasses);

    public static string ConfigJson(Shape s, string? trainedJson = null)
    {
        string trained = trainedJson is null ? string.Empty : $",\"layout_trained\":{trainedJson}";
        return $"{{\"arch\":\"dbnet-lite\",\"stem\":{s.Stem},\"widths\":[{string.Join(',', s.Widths)}],\"depths\":[{string.Join(',', s.Depths)}]," +
            $"\"fpn\":{s.Fpn},\"text\":{s.Text},\"layout\":{s.Layout},\"layout_blocks\":{s.LayoutBlocks},\"classes\":{s.ClassCount}," +
            $"\"layout_classes\":[{string.Join(',', s.Classes.Select(c => $"\"{c}\""))}]{trained}}}";
    }

    public static List<TestTensor> RandomTensors(Shape s, int seed)
    {
        var rng = new Random(seed);
        var list = new List<TestTensor>();
        void Add(string name, double scale, double offset, params int[] shape)
        {
            int n = 1;
            foreach (int d in shape)
            {
                n *= d;
            }

            var v = new float[n];
            for (int i = 0; i < n; i++)
            {
                v[i] = (float)(offset + (((rng.NextDouble() * 2) - 1) * scale));
            }

            list.Add(new TestTensor { Name = name, Shape = shape, Values = v });
        }

        // He-style scale so activations neither vanish nor blow up; positive-ish biases keep many ReLUs open.
        void Dw(string path, int cin, int cout)
        {
            Add($"{path}.dw.weight", Math.Sqrt(6.0 / 9), 0, cin, 1, 3, 3);
            Add($"{path}.dw.bias", 0.2, 0.05, cin);
            Add($"{path}.pw.weight", Math.Sqrt(6.0 / cin), 0, cout, cin, 1, 1);
            Add($"{path}.pw.bias", 0.2, 0.05, cout);
        }

        void Plain(string path, int cin, int cout)
        {
            Add($"{path}.weight", Math.Sqrt(3.0 / cin), 0, cout, cin, 1, 1);
            Add($"{path}.bias", 0.2, 0, cout);
        }

        Add("stem.weight", Math.Sqrt(6.0 / 9), 0, s.Stem, 1, 3, 3);
        Add("stem.bias", 0.2, 0.05, s.Stem);
        int cin = s.Stem;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < s.Depths[i]; j++)
            {
                Dw($"stage{i}.{j}", cin, s.Widths[i]);
                cin = s.Widths[i];
            }
        }

        for (int i = 0; i < 4; i++)
        {
            Plain($"lat{i}", s.Widths[i], s.Fpn);
        }

        Dw("smooth8", s.Fpn, s.Fpn);
        Dw("smooth4", s.Fpn, s.Fpn);
        cin = s.Fpn;
        for (int j = 0; j < s.LayoutBlocks; j++)
        {
            Dw($"layout.{j}", cin, s.Layout);
            cin = s.Layout;
        }

        Plain("layout_out", s.Layout, s.ClassCount);
        Dw("text4", s.Fpn, s.Text);
        Plain("stem_proj", s.Stem, s.Text);
        Dw("text2", s.Text, s.Text);
        Plain("prob_out", s.Text, 1);
        Add("orient.weight", 1.0 / Math.Sqrt(s.Widths[3]), 0, 4, s.Widths[3]);
        Add("orient.bias", 0.1, 0, 4);
        return list;
    }

    public static string[] Charset(Shape s) => ["<blank>", .. s.Classes];

    public static byte[] Random(Shape s, int seed, string? trainedJson = null, IReadOnlyList<TestTensor>? tensors = null) =>
        TestContainerWriter.Write(
            "dbnet-lite-v0", 0, Charset(s), ConfigJson(s, trainedJson), tensors ?? RandomTensors(s, seed),
            calibrationJson: "{\"raw\":\"word-mean-max-run\",\"scripts\":{}}");

    /// <summary>A gray image with smooth structure and noise, <c>[height][width]</c>.</summary>
    public static byte[] Image(int height, int width, int seed)
    {
        var rng = new Random(seed);
        var px = new byte[height * width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double v = 150 + (90 * Math.Sin((x * 0.31) + (y * 0.07))) + (rng.NextDouble() * 50 - 25);
                px[(y * width) + x] = (byte)Math.Clamp((int)v, 0, 255);
            }
        }

        return px;
    }

    /// <summary>The model's input normalization applied to a gray image: <c>(p / 255 - mean) / std</c> in float32.</summary>
    public static float[] Normalize(byte[] gray, float mean = 0.5f, float std = 0.5f) =>
        gray.Select(p => ((p / 255f) - mean) / std).ToArray();
}

/// <summary>
/// The detector graph in double precision on plain loops (exact math, channels-first), written from the architecture table and
/// independent of the kernels: the reference our float32 results are measured against.
/// </summary>
internal sealed class DetectorReference64
{
    private readonly ModelContainer _c;
    private readonly TestDetectorModels.Shape _s;

    public DetectorReference64(ModelContainer container, TestDetectorModels.Shape shape)
    {
        _c = container;
        _s = shape;
    }

    private double[] T(string name) => _c.ReadTensor(name).Select(v => (double)v).ToArray();

    private static double[] Conv(double[] x, int cin, int h, int w, double[] wt, double[] b, int cout, int k, int stride, int groups, bool relu, out int oh, out int ow)
    {
        int pad = k / 2;
        oh = (h + (2 * pad) - k) / stride + 1;
        ow = (w + (2 * pad) - k) / stride + 1;
        int cinPerGroup = cin / groups, coutPerGroup = cout / groups;
        var y = new double[cout * oh * ow];
        for (int o = 0; o < cout; o++)
        {
            int g = o / coutPerGroup;
            for (int yy = 0; yy < oh; yy++)
            {
                for (int xx = 0; xx < ow; xx++)
                {
                    double sum = 0;
                    for (int ic = 0; ic < cinPerGroup; ic++)
                    {
                        for (int ky = 0; ky < k; ky++)
                        {
                            for (int kx = 0; kx < k; kx++)
                            {
                                int sy = (yy * stride) - pad + ky, sx = (xx * stride) - pad + kx;
                                if (sy < 0 || sy >= h || sx < 0 || sx >= w)
                                {
                                    continue;
                                }

                                sum += wt[((((o * cinPerGroup) + ic) * k + ky) * k) + kx] * x[((((g * cinPerGroup) + ic) * h + sy) * w) + sx];
                            }
                        }
                    }

                    sum += b[o];
                    y[(((o * oh) + yy) * ow) + xx] = relu ? Math.Max(sum, 0) : sum;
                }
            }
        }

        return y;
    }

    private double[] Block(string path, double[] x, int cin, int cout, int h, int w, int stride, out int oh, out int ow)
    {
        double[] d = Conv(x, cin, h, w, T($"{path}.dw.weight"), T($"{path}.dw.bias"), cin, 3, stride, cin, true, out oh, out ow);
        return Conv(d, cin, oh, ow, T($"{path}.pw.weight"), T($"{path}.pw.bias"), cout, 1, 1, 1, true, out _, out _);
    }

    private static double[] Up2(double[] x, int c, int h, int w)
    {
        var y = new double[c * h * 2 * w * 2];
        for (int k = 0; k < c; k++)
        {
            for (int yy = 0; yy < h * 2; yy++)
            {
                for (int xx = 0; xx < w * 2; xx++)
                {
                    y[(((k * h * 2) + yy) * w * 2) + xx] = x[(((k * h) + (yy / 2)) * w) + (xx / 2)];
                }
            }
        }

        return y;
    }

    private static double[] Add(double[] a, double[] b) => a.Zip(b, (u, v) => u + v).ToArray();

    /// <summary>Channels-first layers by the names the fixtures use, from a normalized <c>[h][w]</c> input.</summary>
    public Dictionary<string, (int[] Shape, double[] Data)> Forward(float[] input, int h, int w)
    {
        var o = new Dictionary<string, (int[], double[])>();
        double[] x = input.Select(v => (double)v).ToArray();
        double[] stem = Conv(x, 1, h, w, T("stem.weight"), T("stem.bias"), _s.Stem, 3, 2, 1, true, out int sh, out int sw);
        o["stem"] = ([_s.Stem, sh, sw], stem);
        double[] cur = stem;
        int cin = _s.Stem, ch = sh, cw = sw;
        var feats = new double[4][];
        var dims = new (int H, int W)[4];
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < _s.Depths[i]; j++)
            {
                cur = Block($"stage{i}.{j}", cur, cin, _s.Widths[i], ch, cw, j == 0 ? 2 : 1, out ch, out cw);
                cin = _s.Widths[i];
            }

            feats[i] = cur;
            dims[i] = (ch, cw);
            o[$"stage{i}"] = ([cin, ch, cw], cur);
        }

        var lat = new double[4][];
        for (int i = 0; i < 4; i++)
        {
            lat[i] = Conv(feats[i], _s.Widths[i], dims[i].H, dims[i].W, T($"lat{i}.weight"), T($"lat{i}.bias"), _s.Fpn, 1, 1, 1, false, out _, out _);
        }

        double[] p16 = Add(lat[2], Up2(lat[3], _s.Fpn, dims[3].H, dims[3].W));
        double[] p8 = Block("smooth8", Add(lat[1], Up2(p16, _s.Fpn, dims[2].H, dims[2].W)), _s.Fpn, _s.Fpn, dims[1].H, dims[1].W, 1, out _, out _);
        o["p8"] = ([_s.Fpn, dims[1].H, dims[1].W], p8);
        double[] p4 = Block("smooth4", Add(lat[0], Up2(p8, _s.Fpn, dims[1].H, dims[1].W)), _s.Fpn, _s.Fpn, dims[0].H, dims[0].W, 1, out _, out _);
        o["p4"] = ([_s.Fpn, dims[0].H, dims[0].W], p4);

        double[] f = p8;
        int fin = _s.Fpn;
        for (int j = 0; j < _s.LayoutBlocks; j++)
        {
            f = Block($"layout.{j}", f, fin, _s.Layout, dims[1].H, dims[1].W, 1, out _, out _);
            fin = _s.Layout;
        }

        o["layout_feat"] = ([_s.Layout, dims[1].H, dims[1].W], f);
        o["layout"] = ([_s.ClassCount, dims[1].H, dims[1].W],
            Conv(f, _s.Layout, dims[1].H, dims[1].W, T("layout_out.weight"), T("layout_out.bias"), _s.ClassCount, 1, 1, 1, false, out _, out _));

        double[] t4 = Block("text4", p4, _s.Fpn, _s.Text, dims[0].H, dims[0].W, 1, out _, out _);
        o["text4"] = ([_s.Text, dims[0].H, dims[0].W], t4);
        double[] proj = Conv(stem, _s.Stem, sh, sw, T("stem_proj.weight"), T("stem_proj.bias"), _s.Text, 1, 1, 1, false, out _, out _);
        double[] t2 = Block("text2", Add(proj, Up2(t4, _s.Text, dims[0].H, dims[0].W)), _s.Text, _s.Text, sh, sw, 1, out _, out _);
        o["text2"] = ([_s.Text, sh, sw], t2);
        o["prob"] = ([1, sh, sw], Conv(t2, _s.Text, sh, sw, T("prob_out.weight"), T("prob_out.bias"), 1, 1, 1, 1, false, out _, out _));

        int c3 = _s.Widths[3], px = dims[3].H * dims[3].W;
        double[] ow = T("orient.weight"), ob = T("orient.bias");
        var logits = new double[4];
        for (int k = 0; k < 4; k++)
        {
            double sum = ob[k];
            for (int c = 0; c < c3; c++)
            {
                double mean = 0;
                for (int p = 0; p < px; p++)
                {
                    mean += feats[3][(c * px) + p];
                }

                sum += ow[(k * c3) + c] * (mean / px);
            }

            logits[k] = sum;
        }

        o["orient"] = ([4], logits);
        return o;
    }
}
