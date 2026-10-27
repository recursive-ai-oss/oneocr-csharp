using System.Text;

namespace OneOcr.Tests;

/// <summary>Writes word frequency files in the layout of <c>docs/spec/word-lm-v1.md</c>, for tests and fuzz seeds.</summary>
internal static class TestWordFrequencies
{
    internal const int Total = 1_000_000;

    // Builds a file as docs/spec/word-lm-v1.md lays it out.
    internal static byte[] Build(IEnumerable<(string Word, int Count)> words, int total = Total)
    {
        var sorted = words.Select(w => (Bytes: Encoding.UTF8.GetBytes(w.Word), w.Count)).OrderBy(w => w.Bytes, Comparer<byte[]>.Create((a, b) => a.AsSpan().SequenceCompareTo(b))).ToList();
        var blocks = new List<int>();
        var data = new List<byte>();
        byte[] previous = [];
        for (int i = 0; i < sorted.Count; i++)
        {
            if (i % 16 == 0)
            {
                blocks.Add(data.Count);
                previous = [];
            }

            byte[] w = sorted[i].Bytes;
            int shared = 0;
            while (shared < Math.Min(previous.Length, w.Length) && previous[shared] == w[shared])
            {
                shared++;
            }

            data.Add((byte)shared);
            data.Add((byte)(w.Length - shared));
            data.AddRange(w.AsSpan(shared).ToArray());
            data.Add(Quarter(sorted[i].Count));
            previous = w;
        }

        var o = new List<byte>();
        o.AddRange("OCWF"u8.ToArray());
        o.AddRange(BitConverter.GetBytes((ushort)1));
        o.AddRange(BitConverter.GetBytes((ushort)0));
        o.AddRange(BitConverter.GetBytes((uint)total));
        o.AddRange(BitConverter.GetBytes((uint)sorted.Count));
        o.Add(Quarter(total));
        o.Add(2);
        o.AddRange(BitConverter.GetBytes((uint)blocks.Count));
        foreach (int b in blocks)
        {
            o.AddRange(BitConverter.GetBytes((uint)b));
        }

        o.AddRange(data);
        return [.. o];
    }

    internal static byte Quarter(int count) => (byte)Math.Clamp((int)Math.Round(4 * Math.Log2(count)), 1, 255);
}
