using System;
using System.Collections.Generic;
using System.Linq;
using System.Buffers.Binary;
using System.Text;

namespace OneOcr.Tests;

/// <summary>Writes small character language model files for tests and fuzz seeds: maximum-likelihood n-grams, no discounting.</summary>
internal static class TestLm
{
    /// <summary>A one-script model of <paramref name="words"/> over <paramref name="alphabet"/> (one string per symbol).</summary>
    public static byte[] Build(string scriptName, string[] alphabet, string[] words, int order = 3, float mean = -1.5f) =>
        Assemble([(scriptName, alphabet, words)], order, mean);

    public static byte[] Assemble((string Name, string[] Alphabet, string[] Words)[] models, int order = 3, float mean = -1.5f)
    {
        var o = new List<byte>();
        o.AddRange("OCLM"u8.ToArray());
        o.AddRange(BitConverter.GetBytes((ushort)1));
        o.AddRange(BitConverter.GetBytes((ushort)models.Length));
        foreach (var (name, alphabet, words) in models)
        {
            Script(o, name, alphabet, words, order, mean);
        }

        return [.. o];
    }

    private static void Script(List<byte> o, string name, string[] alphabet, string[] words, int order, float mean)
    {
        byte[] nameBytes = Encoding.ASCII.GetBytes(name);
        o.Add((byte)nameBytes.Length);
        o.AddRange(nameBytes);
        o.Add((byte)order);
        o.Add((byte)alphabet.Length);
        foreach (string a in alphabet)
        {
            byte[] b = Encoding.UTF8.GetBytes(a);
            o.Add((byte)b.Length);
            o.AddRange(b);
        }

        var f = new byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(f, mean);
        o.AddRange(f);

        var counts = new SortedDictionary<string, int>[order + 1];
        for (int n = 1; n <= order; n++)
        {
            counts[n] = new SortedDictionary<string, int>(StringComparer.Ordinal);
        }

        foreach (string word in words)
        {
            var seq = new StringBuilder();
            seq.Append('\0');
            foreach (char c in word)
            {
                int at = Array.IndexOf(alphabet, c.ToString());
                seq.Append((char)(at + 2));
            }

            seq.Append('\u0001');
            string s = seq.ToString();
            for (int n = 1; n <= order; n++)
            {
                for (int i = 0; i + n <= s.Length; i++)
                {
                    string g = s.Substring(i, n);
                    if (n == 1 && g == "\0")
                    {
                        continue;
                    }

                    counts[n][g] = counts[n].GetValueOrDefault(g) + 1;
                }
            }
        }

        // Every unigram exists (add-one), plus the start symbol, which only carries a back-off weight.
        for (int sym = 1; sym < alphabet.Length + 2; sym++)
        {
            string g = ((char)sym).ToString();
            counts[1][g] = counts[1].GetValueOrDefault(g) + 1;
        }

        counts[1]["\0"] = 1;
        double unigramTotal = counts[1].Where(kv => kv.Key != "\0").Sum(kv => kv.Value);
        for (int n = 1; n <= order; n++)
        {
            var grams = counts[n].Keys.OrderBy(k => k, StringComparer.Ordinal).ToList();
            o.AddRange(BitConverter.GetBytes((uint)grams.Count));
            foreach (string g in grams)
            {
                double p;
                if (n == 1)
                {
                    p = g == "\0" ? 1e-6 : counts[1][g] / unigramTotal;
                }
                else
                {
                    double total = counts[n].Where(kv => kv.Key.StartsWith(g[..^1], StringComparison.Ordinal)).Sum(kv => kv.Value);
                    p = counts[n][g] / total;
                }

                foreach (char c in g)
                {
                    o.Add((byte)c);
                }

                o.Add((byte)Math.Clamp((int)Math.Round(-Math.Log(p) / 0.1), 0, 254));
                if (n < order)
                {
                    o.Add(0);
                }
            }
        }
    }
}
