using System.Buffers.Binary;
using System.Security.Cryptography;

namespace OneOcr.Training;

/// <summary>
/// The state needed to continue a run exactly: the step, the weights and the AdamW moments, tied to the run by a fingerprint of
/// the starting model, the labels and the settings. A checkpoint is read back like any other input: lengths are checked against
/// the model before anything is allocated.
/// </summary>
internal static class Checkpoint
{
    private static ReadOnlySpan<byte> Magic => "ONEOCRC\0"u8;
    private const uint Version = 1;
    private const int HeaderSize = 8 + 4 + 32 + 8 + 8 + 8 + 4;

    public sealed record State(long Step, double FirstLoss, double LastLoss, float[][] Weights, float[][] M, float[][] V);

    public static void Save(string path, byte[] fingerprint, State state)
    {
        long floats = 0;
        foreach (float[] w in state.Weights)
        {
            floats += w.Length;
        }

        var bytes = new byte[checked(HeaderSize + (3 * floats * 4) + (state.Weights.Length * 4L))];
        Span<byte> s = bytes;
        Magic.CopyTo(s);
        BinaryPrimitives.WriteUInt32LittleEndian(s[8..], Version);
        fingerprint.CopyTo(s[12..]);
        BinaryPrimitives.WriteInt64LittleEndian(s[44..], state.Step);
        BinaryPrimitives.WriteDoubleLittleEndian(s[52..], state.FirstLoss);
        BinaryPrimitives.WriteDoubleLittleEndian(s[60..], state.LastLoss);
        BinaryPrimitives.WriteInt32LittleEndian(s[68..], state.Weights.Length);
        int pos = HeaderSize;
        for (int i = 0; i < state.Weights.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(s[pos..], state.Weights[i].Length);
            pos += 4;
            foreach (float[] set in new[] { state.Weights[i], state.M[i], state.V[i] })
            {
                foreach (float f in set)
                {
                    BinaryPrimitives.WriteSingleLittleEndian(s[pos..], f);
                    pos += 4;
                }
            }
        }

        WorkStore.WriteAtomic(path, bytes);
    }

    /// <summary>Reads a checkpoint for tensors of the given lengths; null when there is none or it belongs to another run.</summary>
    /// <exception cref="OcrFormatException">The file is damaged or does not fit the model.</exception>
    public static State? TryLoad(string path, byte[] fingerprint, int[] tensorLengths)
    {
        if (!File.Exists(path))
        {
            return null;
        }

        long expected = HeaderSize + (tensorLengths.Length * 4L);
        foreach (int n in tensorLengths)
        {
            expected += 3L * n * 4;
        }

        long actual = new FileInfo(path).Length;
        if (actual != expected)
        {
            throw new OcrFormatException($"The checkpoint '{path}' has {actual} bytes; this model needs {expected}. Delete it to start over.");
        }

        byte[] bytes = File.ReadAllBytes(path);
        ReadOnlySpan<byte> s = bytes;
        if (!s[..8].SequenceEqual(Magic) || BinaryPrimitives.ReadUInt32LittleEndian(s[8..]) != Version)
        {
            throw new OcrFormatException($"'{path}' is not a checkpoint of this version.");
        }

        if (!s.Slice(12, 32).SequenceEqual(fingerprint))
        {
            throw new InvalidOperationException(
                $"The checkpoint '{path}' belongs to a different run (other model, labels or settings). Set Restart (--restart), or use another work directory, to start over.");
        }

        long step = BinaryPrimitives.ReadInt64LittleEndian(s[44..]);
        double first = BinaryPrimitives.ReadDoubleLittleEndian(s[52..]);
        double last = BinaryPrimitives.ReadDoubleLittleEndian(s[60..]);
        if (step < 0 || BinaryPrimitives.ReadInt32LittleEndian(s[68..]) != tensorLengths.Length)
        {
            throw new OcrFormatException($"The checkpoint '{path}' does not fit the model.");
        }

        var weights = new float[tensorLengths.Length][];
        var m = new float[tensorLengths.Length][];
        var v = new float[tensorLengths.Length][];
        int pos = HeaderSize;
        for (int i = 0; i < tensorLengths.Length; i++)
        {
            if (BinaryPrimitives.ReadInt32LittleEndian(s[pos..]) != tensorLengths[i])
            {
                throw new OcrFormatException($"The checkpoint '{path}' does not fit the model.");
            }

            pos += 4;
            float[][] targets = [new float[tensorLengths[i]], new float[tensorLengths[i]], new float[tensorLengths[i]]];
            foreach (float[] t in targets)
            {
                for (int j = 0; j < t.Length; j++)
                {
                    t[j] = BinaryPrimitives.ReadSingleLittleEndian(s[pos..]);
                    pos += 4;
                }
            }

            (weights[i], m[i], v[i]) = (targets[0], targets[1], targets[2]);
        }

        return new State(step, first, last, weights, m, v);
    }

    /// <summary>SHA-256 over the parts, each length-prefixed so that parts cannot run into one another.</summary>
    public static byte[] Fingerprint(IEnumerable<string> parts)
    {
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> len = stackalloc byte[4];
        foreach (string p in parts)
        {
            byte[] b = System.Text.Encoding.UTF8.GetBytes(p);
            BinaryPrimitives.WriteInt32LittleEndian(len, b.Length);
            sha.AppendData(len);
            sha.AppendData(b);
        }

        return sha.GetHashAndReset();
    }
}

/// <summary>A small fixed generator (SplitMix64) so batch order and splits are the same on every runtime.</summary>
internal sealed class SplitMix
{
    private ulong _state;

    public SplitMix(ulong seed) => _state = seed;

    public ulong Next()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    /// <summary>A value in [0, n) by rejection, so it is unbiased.</summary>
    public int Below(int n)
    {
        ulong limit = ulong.MaxValue - (ulong.MaxValue % (ulong)n);
        ulong x;
        do
        {
            x = Next();
        }
        while (x >= limit);
        return (int)(x % (ulong)n);
    }

    /// <summary>A Fisher–Yates permutation of 0..n-1.</summary>
    public static int[] Permutation(ulong seed, int n)
    {
        var rng = new SplitMix(seed);
        int[] p = new int[n];
        for (int i = 0; i < n; i++)
        {
            p[i] = i;
        }

        for (int i = n - 1; i > 0; i--)
        {
            int j = rng.Below(i + 1);
            (p[i], p[j]) = (p[j], p[i]);
        }

        return p;
    }
}
