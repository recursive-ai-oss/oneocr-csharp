using System.Globalization;
using System.Security.Cryptography;
using OneOcr.Kernels;

// OneOcr.Bench mathhash [--n 4000000]
//   SHA-256 of the bits of FastMath.Exp / Sigmoid / Tanh over a fixed pseudo-random input sweep, of SumExp over fixed frames on every
//   path this machine runs, and of a float mean. Run on two platforms and compare the lines: they must match exactly.
internal static class MathHash
{
    public static int Run(Dictionary<string, string> o)
    {
        int n = o.TryGetValue("n", out string? ns) ? int.Parse(ns, CultureInfo.InvariantCulture) : 4_000_000;
        uint state = 12345;
        float Next(float lo, float hi)
        {
            state = (state * 1664525u) + 1013904223u;
            return lo + (((state >> 8) / 16777216f) * (hi - lo));
        }

        using var he = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        using var hs = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        using var ht = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> b = stackalloc byte[4];
        for (int i = 0; i < n; i++)
        {
            float x = Next(-95f, 95f);
            System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(b, BitConverter.SingleToInt32Bits(FastMath.Exp(x)));
            he.AppendData(b);
            System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(b, BitConverter.SingleToInt32Bits(FastMath.Sigmoid(x)));
            hs.AppendData(b);
            System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(b, BitConverter.SingleToInt32Bits(FastMath.Tanh(x * 0.2f)));
            ht.AppendData(b);
        }

        Console.WriteLine("exp " + Convert.ToHexString(he.GetHashAndReset())[..16]);
        Console.WriteLine("sigmoid " + Convert.ToHexString(hs.GetHashAndReset())[..16]);
        Console.WriteLine("tanh " + Convert.ToHexString(ht.GetHashAndReset())[..16]);

        // Frames shaped like recognizer output: one large winner, the rest spread over a wide range.
        const int classes = 622;
        var frame = new float[classes];
        foreach (KernelPath path in Dispatch.Available())
        {
            state = 777;
            using var hf = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            for (int f = 0; f < 4000; f++)
            {
                float max = float.MinValue;
                for (int j = 0; j < classes; j++)
                {
                    frame[j] = Next(-12f, 4f) - (f % 3 == 0 ? 14f * (j % 5 == 0 ? 0f : 1f) : 0f);
                    max = Math.Max(max, frame[j]);
                }

                float s = Elementwise.SumExp(path, frame, max);
                float p = 1f / s;
                System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(b, BitConverter.SingleToInt32Bits(p));
                hf.AppendData(b);
            }

            Console.WriteLine("sumexp-" + path.ToString().ToLowerInvariant() + " " + Convert.ToHexString(hf.GetHashAndReset())[..16]);
        }

        SrgbTables();
        return 0;
    }

    // The sRGB curve OneImage uses to resample in linear light (IEC 61966-2-1), with the platform's MathF.Pow, and the same curve with
    // the power taken in double and rounded once. Lines tagged `pow` come from the platform's math library; `powd` from the double route.
    private static void SrgbTables()
    {
        static float Decode(float v, bool dbl)
        {
            float a = MathF.Abs(v);
            float x = (a + 0.055f) / 1.055f;
            return a <= 0.04045f ? a / 12.92f : dbl ? (float)Math.Pow(x, 2.4f) : MathF.Pow(x, 2.4f);
        }

        static float Encode(float v, bool dbl)
        {
            float a = MathF.Abs(v);
            return a <= 0.0031308f ? a * 12.92f : (1.055f * (dbl ? (float)Math.Pow(a, 1f / 2.4f) : MathF.Pow(a, 1f / 2.4f))) - 0.055f;
        }

        foreach (bool dbl in new[] { false, true })
        {
            using var hd = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            using var he = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            byte[] b = new byte[4];
            for (int i = 0; i <= 255; i++)
            {
                System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(b, BitConverter.SingleToInt32Bits(Decode((float)i / 255, dbl)));
                hd.AppendData(b);
            }

            for (int i = 0; i <= 65535; i++)
            {
                int code = Math.Clamp((int)Math.Round(Encode((float)i / 65535, dbl) * 255, MidpointRounding.AwayFromZero), 0, 255);
                he.AppendData([(byte)code]);
            }

            string tag = dbl ? "powd" : "pow";
            Console.WriteLine(tag + "-decode8 " + Convert.ToHexString(hd.GetHashAndReset())[..16]);
            Console.WriteLine(tag + "-encode8 " + Convert.ToHexString(he.GetHashAndReset())[..16]);
        }

        int decodeDiff = 0, encodeDiff = 0;
        for (int i = 0; i <= 255; i++)
        {
            decodeDiff += Decode((float)i / 255, false) == Decode((float)i / 255, true) ? 0 : 1;
        }

        for (int i = 0; i <= 65535; i++)
        {
            int c0 = Math.Clamp((int)Math.Round(Encode((float)i / 65535, false) * 255, MidpointRounding.AwayFromZero), 0, 255);
            int c1 = Math.Clamp((int)Math.Round(Encode((float)i / 65535, true) * 255, MidpointRounding.AwayFromZero), 0, 255);
            encodeDiff += c0 == c1 ? 0 : 1;
        }

        Console.WriteLine($"pow-vs-powd decode8 entries differing: {decodeDiff} of 256; encode8 codes differing: {encodeDiff} of 65536");
    }
}
