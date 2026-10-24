using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace OneOcr.Tests;

/// <summary>Builds random <c>docextract-v0</c> containers of a small shape, for loader, determinism and assembly tests.</summary>
internal static class TestDocModels
{
    public sealed record Shape(
        int D, int Layers, int Heads, int Ffn, int CharDim, int CharConv, int PairHidden, int RelDim, int MaxWords, string[] Classes, int Chars = 41);

    public static Shape Small { get; } = new(
        32, 2, 2, 48, 8, 16, 8, 8, 16, ["BT-1", "BT-2", "BT-112", "BT-129", "BT-153", "BT-116", "receipt.merchant", "receipt.total", "receipt.itemName", "receipt.amount"]);

    /// <summary>A window of 12 words, so that pages of 13 or more words need several windows.</summary>
    public static Shape SmallWindow { get; } = Small with { MaxWords = 12 };

    public static string WordInputJson(Shape s) => $"{{\"kind\":\"words\",\"max_words\":{s.MaxWords},\"max_chars\":8,\"bins\":16,\"features\":8}}";

    public static string ConfigJson(Shape s)
    {
        int k = s.Classes.Length;
        string list = string.Join(",", s.Classes.Select(c => $"\"{c}\""));
        return $"{{\"d\":{s.D},\"layers\":{s.Layers},\"heads\":{s.Heads},\"ffn\":{s.Ffn},\"char_dim\":{s.CharDim},\"char_conv\":{s.CharConv}," +
            $"\"pair_hidden\":{s.PairHidden},\"rel_dim\":{s.RelDim},\"max_words\":{s.MaxWords},\"classes\":{k},\"tags\":{1 + (2 * k)},\"roles\":5,\"doc_types\":3," +
            $"\"classes_list\":[{list}],\"roles_list\":[\"O\",\"B-KEY\",\"I-KEY\",\"B-VAL\",\"I-VAL\"],\"doc_types_list\":[\"invoice\",\"receipt\",\"other\"],\"unk_char\":{s.Chars}}}";
    }

    public static List<TestTensor> RandomTensors(Shape s, int seed, Func<string, int[], bool>? drop = null)
    {
        var rng = new Random(seed);
        var list = new List<TestTensor>();
        void Add(string name, double scale, params int[] shape)
        {
            if (drop?.Invoke(name, shape) == true)
            {
                return;
            }

            int n = shape.Aggregate(1, (a, b) => a * b);
            var v = new float[n];
            for (int i = 0; i < n; i++)
            {
                v[i] = (float)(((rng.NextDouble() * 2) - 1) * scale);
            }

            list.Add(new TestTensor { Name = name, Shape = shape, Values = v });
        }

        void Linear(string name, int outDim, int inDim, double scale = 1.0)
        {
            Add(name + ".weight", scale / Math.Sqrt(inDim) * 2, outDim, inDim);
            Add(name + ".bias", 0.1, outDim);
        }

        void Norm(string name, int d)
        {
            Add(name + ".weight", 0.3, d);
            float[] gain = list[^1].Values;
            for (int i = 0; i < gain.Length; i++)
            {
                gain[i] += 1f;
            }

            Add(name + ".bias", 0.1, d);
        }

        int d = s.D;
        Add("char_emb.weight", 0.5, s.Chars + 1, s.CharDim);
        Add("char_conv.weight", 0.5, s.CharConv, s.CharDim, 3);
        Add("char_conv.bias", 0.1, s.CharConv);
        Linear("char_proj", d, s.CharConv);
        foreach (string p in new[] { "pos_x0", "pos_y0", "pos_x1", "pos_y1" })
        {
            Add(p + ".weight", 0.3, 16, d);
        }

        Linear("feat_proj", d, 8);
        Norm("emb_ln", d);
        for (int l = 0; l < s.Layers; l++)
        {
            string p = $"blocks.{l}.";
            Norm(p + "ln1", d);
            Linear(p + "qkv", 3 * d, d);
            Linear(p + "proj", d, d);
            Norm(p + "ln2", d);
            Linear(p + "fc1", s.Ffn, d);
            Linear(p + "fc2", d, s.Ffn);
        }

        Linear("pair1", s.PairHidden, 13, 2.0);
        Linear("pair2", (s.Layers * s.Heads) + 2, s.PairHidden, 2.0);
        Norm("final_ln", d);
        int tags = 1 + (2 * s.Classes.Length);
        Linear("tag_head", tags, d, 4.0);
        Linear("role_head", 5, d, 4.0);
        Linear("doc_head", 3, d, 4.0);
        Linear("rel_q", 2 * s.RelDim, d, 2.0);
        Linear("rel_k", 2 * s.RelDim, d, 2.0);
        return list;
    }

    public static string[] Charset(Shape s) => TestModels.Charset(s.Chars);

    public static byte[] RandomContainer(Shape s, int seed, Func<string, int[], bool>? drop = null, Func<string, string>? editConfig = null) =>
        TestContainerWriter.Write(
            "docextract-v0", 0, Charset(s), editConfig is null ? ConfigJson(s) : editConfig(ConfigJson(s)), RandomTensors(s, seed, drop),
            calibrationJson: "{\"raw\":\"word-mean-max-run\",\"scripts\":{}}", wordInputJson: WordInputJson(s));

    /// <summary>A page of <paramref name="words"/> words laid out in rows, with texts that cover known and unknown characters.</summary>
    public static (string[] Texts, OcrBox[] Boxes, double Width, double Height) Page(int words, int seed)
    {
        var rng = new Random(seed);
        var texts = new string[words];
        var boxes = new OcrBox[words];
        double x = 10, y = 10;
        for (int i = 0; i < words; i++)
        {
            int len = i % 5 == 0 ? 11 : rng.Next(1, 9);
            var sb = new StringBuilder();
            for (int k = 0; k < len; k++)
            {
                sb.Append(rng.Next(12) == 0 ? '\u4e2d' : (char)(0x101 + rng.Next(40)));
            }

            texts[i] = sb.ToString();
            double w = 7 * len;
            if (x + w > 400)
            {
                x = 10;
                y += 16;
            }

            boxes[i] = new OcrBox(x, y, x + w, y + 12);
            x += w + 6;
        }

        return (texts, boxes, 420, y + 40);
    }
}
