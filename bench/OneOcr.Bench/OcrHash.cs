using System.Buffers.Binary;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using OneImage;
using OneOcr;

// OneOcr.Bench ocrhash --pages DIR [--model F --detector F --wordlm F --wordfreq F] [--degree N] [--out FILE]
//   One line per page: SHA-256 over the plain text, the page JSON and, for every word, the exact bits of its box and confidences.
//   A change that must leave the output untouched leaves every line untouched. Set DOTNET_EnableAVX512F=0, DOTNET_EnableAVX2=0 or
//   DOTNET_EnableHWIntrinsic=0 in the environment to run the other kernel paths.
internal static class OcrHash
{
    public static int Run(Dictionary<string, string> o)
    {
        int degree = o.TryGetValue("degree", out string? d) ? int.Parse(d, CultureInfo.InvariantCulture) : 1;
        // --path scalar|vector128|avx2|avx512 forces a kernel path on this machine (the environment switches cannot select 128-bit vectors).
        OneOcr.Kernels.KernelPath? forced = o.TryGetValue("path", out string? pathName) ? Enum.Parse<OneOcr.Kernels.KernelPath>(pathName, ignoreCase: true) : null;
        OcrEngine engine;
        if (forced is { } fp)
        {
            engine = OcrEngine.Create(OneOcr.Model.ModelContainer.Load(o["model"]), null, fp);
            if (o.TryGetValue("detector", out string? det))
            {
                engine = engine.WithDetector(OneOcr.Model.ModelContainer.Load(det), fp);
            }
        }
        else
        {
            engine = o.TryGetValue("model", out string? model) ? OcrEngine.Load(model) : OcrEngine.LoadBundled();
            if (o.TryGetValue("detector", out string? detector))
            {
                engine = engine.WithDetector(detector);
            }
        }

        if (o.TryGetValue("wordlm", out string? wordLm))
        {
            engine = engine.WithWordModel(OneOcr.Decoding.CharLanguageModel.Load(wordLm));
        }

        if (o.TryGetValue("wordfreq", out string? wordFreq))
        {
            engine = engine.WithWordFrequencies(OneOcr.Decoding.WordFrequencyModel.Load(wordFreq));
        }

        Span<byte> bits = stackalloc byte[8];
        var lines = new StringBuilder();
        using var all = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        string[] files = [.. Directory.GetFiles(o["pages"]).Where(f => f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase)).Order(StringComparer.Ordinal)];
        foreach (string file in files)
        {
            OcrPage page = engine.Ocr(Image.Load(file), new OcrOptions { MaxDegreeOfParallelism = degree });
            using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            h.AppendData(Encoding.UTF8.GetBytes(page.ToText()));
            h.AppendData(Encoding.UTF8.GetBytes(OcrPageJson.ToJson([page])));
            foreach (OcrWord word in page.Words)
            {
                foreach (double v in new[] { word.Box.Left, word.Box.Top, word.Box.Right, word.Box.Bottom })
                {
                    BinaryPrimitives.WriteInt64LittleEndian(bits, BitConverter.DoubleToInt64Bits(v));
                    h.AppendData(bits);
                }

                BinaryPrimitives.WriteInt32LittleEndian(bits, BitConverter.SingleToInt32Bits(word.Confidence));
                h.AppendData(bits[..4]);
                BinaryPrimitives.WriteInt32LittleEndian(bits, BitConverter.SingleToInt32Bits(word.RawConfidence));
                h.AppendData(bits[..4]);
            }

            byte[] digest = h.GetHashAndReset();
            all.AppendData(digest);
            lines.Append(Path.GetFileNameWithoutExtension(file)).Append(' ').Append(Convert.ToHexString(digest)[..16]).Append('\n');
        }

        lines.Append("ALL ").Append(files.Length).Append(' ').Append(Convert.ToHexString(all.GetHashAndReset())[..16]).Append('\n');
        if (o.TryGetValue("out", out string? outFile))
        {
            File.WriteAllText(outFile, lines.ToString());
        }

        Console.Write(lines.ToString().Split('\n')[^2]);
        Console.WriteLine();
        return 0;
    }
}
