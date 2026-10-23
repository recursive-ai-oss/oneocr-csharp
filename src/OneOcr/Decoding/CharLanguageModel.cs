using System.Buffers.Binary;
using System.Text;

namespace OneOcr.Decoding;

/// <summary>
/// A character n-gram model of words, one model per script, used to rank candidate readings during beam search. A word is a
/// run of non-space characters; the model scores its characters one at a time from a start-of-word symbol up to an end-of-word
/// symbol, with no context across spaces. Probabilities are interpolated Kneser-Ney estimates stored in back-off form and
/// quantized to 0.1 nats (format and scoring in <c>docs/spec/char-lm-v1.md</c>). Instances are immutable and thread-safe.
/// </summary>
public sealed class CharLanguageModel
{
    private const int HeaderSize = 8;

    private static ReadOnlySpan<byte> Magic => "OCLM"u8;

    internal CharLanguageModel(IReadOnlyList<ScriptModel> scripts)
    {
        Scripts = scripts;
    }

    /// <summary>The script model names, for example <c>latin</c>, <c>greek</c>, <c>cyrillic</c>.</summary>
    public IEnumerable<string> ScriptNames => Scripts.Select(s => s.Name);

    internal IReadOnlyList<ScriptModel> Scripts { get; }

    /// <summary>Reads a language model file.</summary>
    /// <param name="path">A file in the format of <c>docs/spec/char-lm-v1.md</c>.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The model.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="path"/> is null.</exception>
    /// <exception cref="OcrModelException">The file is malformed.</exception>
    /// <exception cref="OcrLimitException">The file or a table in it exceeds <paramref name="limits"/>.</exception>
    public static CharLanguageModel Load(string path, OcrLimits? limits = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        using FileStream stream = File.OpenRead(path);
        return Load(stream, limits);
    }

    /// <summary>Reads a language model from a stream, taking at most <see cref="OcrLimits.MaxLanguageModelFileBytes"/> bytes.</summary>
    /// <param name="stream">A stream positioned at the start of a language model file.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The model.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null.</exception>
    /// <exception cref="OcrModelException">The data is malformed.</exception>
    /// <exception cref="OcrLimitException">The data or a table in it exceeds <paramref name="limits"/>.</exception>
    public static CharLanguageModel Load(Stream stream, OcrLimits? limits = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        limits ??= OcrLimits.Default;
        using var buffer = new MemoryStream();
        var chunk = new byte[81920];
        long total = 0;
        int read;
        while ((read = stream.Read(chunk, 0, chunk.Length)) > 0)
        {
            total += read;
            if (total > limits.MaxLanguageModelFileBytes)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxLanguageModelFileBytes), limits.MaxLanguageModelFileBytes, total);
            }

            buffer.Write(chunk, 0, read);
        }

        return Parse(buffer.GetBuffer().AsSpan(0, (int)buffer.Length), limits);
    }

    /// <summary>Parses language model bytes.</summary>
    /// <param name="data">The whole file.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The model.</returns>
    /// <exception cref="OcrModelException">The data is malformed.</exception>
    /// <exception cref="OcrLimitException">The data or a table in it exceeds <paramref name="limits"/>.</exception>
    public static CharLanguageModel Parse(ReadOnlySpan<byte> data, OcrLimits? limits = null)
    {
        limits ??= OcrLimits.Default;
        if (data.Length > limits.MaxLanguageModelFileBytes)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxLanguageModelFileBytes), limits.MaxLanguageModelFileBytes, data.Length);
        }

        if (data.Length < HeaderSize || !data[..4].SequenceEqual(Magic))
        {
            throw new OcrModelException("Not a character language model file (bad magic).");
        }

        int version = BinaryPrimitives.ReadUInt16LittleEndian(data[4..]);
        if (version != 1)
        {
            throw new OcrModelException($"Unsupported character language model version {version}.");
        }

        int scriptCount = BinaryPrimitives.ReadUInt16LittleEndian(data[6..]);
        if (scriptCount < 1)
        {
            throw new OcrModelException("The file holds no script models.");
        }

        if (scriptCount > limits.MaxLanguageModelScripts)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxLanguageModelScripts), limits.MaxLanguageModelScripts, scriptCount);
        }

        int pos = HeaderSize;
        var scripts = new List<ScriptModel>(scriptCount);
        long ngramBudget = limits.MaxLanguageModelNgrams;
        for (int i = 0; i < scriptCount; i++)
        {
            scripts.Add(ParseScript(data, ref pos, limits, ref ngramBudget));
        }

        if (pos != data.Length)
        {
            throw new OcrModelException("Unexpected bytes after the last script model.");
        }

        return new CharLanguageModel(scripts);
    }

    private static ScriptModel ParseScript(ReadOnlySpan<byte> data, ref int pos, OcrLimits limits, ref long ngramBudget)
    {
        int nameLength = Need(data, ref pos, 1)[0];
        if (nameLength == 0)
        {
            throw new OcrModelException("A script model has an empty name.");
        }

        ReadOnlySpan<byte> nameBytes = Need(data, ref pos, nameLength);
        foreach (byte b in nameBytes)
        {
            if (b < 0x21 || b > 0x7E)
            {
                throw new OcrModelException("A script name must be printable ASCII.");
            }
        }

        string name = Encoding.ASCII.GetString(nameBytes);
        ReadOnlySpan<byte> head = Need(data, ref pos, 2);
        int order = head[0];
        int alphabetCount = head[1];
        if (order < 1 || order > limits.MaxLanguageModelOrder)
        {
            throw order < 1
                ? new OcrModelException("The model order must be at least 1.")
                : new OcrLimitException(nameof(OcrLimits.MaxLanguageModelOrder), limits.MaxLanguageModelOrder, order);
        }

        if (alphabetCount < 1 || alphabetCount > ScriptModel.MaxAlphabet)
        {
            throw new OcrModelException($"The alphabet size must be 1 to {ScriptModel.MaxAlphabet}, not {alphabetCount}.");
        }

        var alphabet = new string[alphabetCount];
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < alphabetCount; i++)
        {
            int length = Need(data, ref pos, 1)[0];
            if (length < 1 || length > 8)
            {
                throw new OcrModelException("An alphabet entry must be 1 to 8 bytes.");
            }

            try
            {
                alphabet[i] = new UTF8Encoding(false, true).GetString(Need(data, ref pos, length));
            }
            catch (ArgumentException ex)
            {
                throw new OcrModelException("An alphabet entry is not valid UTF-8.", ex);
            }

            if (!seen.Add(alphabet[i]))
            {
                throw new OcrModelException("The alphabet repeats an entry.");
            }
        }

        float mean = BinaryPrimitives.ReadSingleLittleEndian(Need(data, ref pos, 4));
        if (!float.IsFinite(mean) || mean > 0f || mean < -50f)
        {
            throw new OcrModelException("The mean log probability is out of range.");
        }

        int symbols = alphabetCount + 2;
        var keys = new byte[order][];
        var probs = new byte[order][];
        var bows = new sbyte[order][];
        for (int n = 1; n <= order; n++)
        {
            long count = BinaryPrimitives.ReadUInt32LittleEndian(Need(data, ref pos, 4));
            if (count > ngramBudget)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxLanguageModelNgrams), limits.MaxLanguageModelNgrams, limits.MaxLanguageModelNgrams - ngramBudget + count);
            }

            int entry = n + 1 + (n < order ? 1 : 0);
            // The bytes must be there before anything is allocated for them.
            if (count * entry > data.Length - pos)
            {
                throw new OcrModelException("The file ends inside an n-gram table.");
            }

            ngramBudget -= count;
            int c = (int)count;
            var k = new byte[c * n];
            var p = new byte[c];
            var w = new sbyte[n < order ? c : 0];
            ReadOnlySpan<byte> previous = default;
            for (int i = 0; i < c; i++)
            {
                ReadOnlySpan<byte> row = Need(data, ref pos, entry);
                ReadOnlySpan<byte> key = row[..n];
                foreach (byte s in key)
                {
                    // Symbol 0 is the start of a word, 1 its end, alphabet entries follow. The start can only lead a key.
                    if (s >= symbols)
                    {
                        throw new OcrModelException("An n-gram holds a symbol outside the alphabet.");
                    }
                }

                for (int j = 1; j < n; j++)
                {
                    if (key[j] == 0)
                    {
                        throw new OcrModelException("The start symbol may only begin an n-gram.");
                    }
                }

                for (int j = 0; j < n - 1; j++)
                {
                    if (key[j] == 1)
                    {
                        throw new OcrModelException("The end symbol may only finish an n-gram.");
                    }
                }

                if (i > 0 && previous.SequenceCompareTo(key) >= 0)
                {
                    throw new OcrModelException("The n-grams are not in strictly increasing order.");
                }

                key.CopyTo(k.AsSpan(i * n, n));
                p[i] = row[n];
                if (n < order)
                {
                    w[i] = (sbyte)row[n + 1];
                }

                previous = k.AsSpan(i * n, n);
            }

            keys[n - 1] = k;
            probs[n - 1] = p;
            bows[n - 1] = w;
        }

        return new ScriptModel(name, order, alphabet, mean, keys, probs, bows);
    }

    private static ReadOnlySpan<byte> Need(ReadOnlySpan<byte> data, ref int pos, int count)
    {
        if (count < 0 || count > data.Length - pos)
        {
            throw new OcrModelException("The file ends early.");
        }

        ReadOnlySpan<byte> slice = data.Slice(pos, count);
        pos += count;
        return slice;
    }
}

/// <summary>The n-gram tables of one script and the scoring over them.</summary>
internal sealed class ScriptModel
{
    /// <summary>Most alphabet entries; symbol ids are one byte, with 0 and 1 reserved for the word start and end.</summary>
    public const int MaxAlphabet = 250;

    public const byte Start = 0;
    public const byte End = 1;
    public const byte FirstLetter = 2;

    /// <summary>Log probability of a symbol with no unigram entry (never in files the builder writes).</summary>
    private const float UnknownLogProb = -14f;
    private const float Step = 0.1f;

    private readonly byte[][] _keys;
    private readonly byte[][] _probs;
    private readonly sbyte[][] _bows;

    public ScriptModel(string name, int order, string[] alphabet, float mean, byte[][] keys, byte[][] probs, sbyte[][] bows)
    {
        Name = name;
        Order = order;
        Alphabet = alphabet;
        MeanLogProb = mean;
        _keys = keys;
        _probs = probs;
        _bows = bows;
    }

    public string Name { get; }

    public int Order { get; }

    /// <summary>The characters of the script's model; symbol id = index + <see cref="FirstLetter"/>.</summary>
    public string[] Alphabet { get; }

    /// <summary>Average log probability per symbol on the training words, in nats (negative); used to centre scores.</summary>
    public float MeanLogProb { get; }

    /// <summary>
    /// ln P(<paramref name="symbol"/> | <paramref name="context"/>) with back-off. The context holds the symbols of the word so
    /// far, led by <see cref="Start"/>; only its last <c>Order - 1</c> symbols count.
    /// </summary>
    public float LogProb(ReadOnlySpan<byte> context, byte symbol)
    {
        float back = 0f;
        int longest = Math.Min(context.Length, Order - 1);
        Span<byte> key = stackalloc byte[Order];
        for (int length = longest; length >= 0; length--)
        {
            ReadOnlySpan<byte> ctx = context[(context.Length - length)..];
            Span<byte> probe = key[..(length + 1)];
            ctx.CopyTo(probe);
            probe[length] = symbol;
            int at = Find(length + 1, probe);
            if (at >= 0)
            {
                return back - (_probs[length][at] * Step);
            }

            if (length > 0)
            {
                int bowAt = Find(length, key[..length]);
                if (bowAt >= 0)
                {
                    back += _bows[length - 1][bowAt] * Step;
                }
            }
        }

        return back + UnknownLogProb;
    }

    private int Find(int n, ReadOnlySpan<byte> query)
    {
        byte[] keys = _keys[n - 1];
        int low = 0, high = keys.Length / n - 1;
        while (low <= high)
        {
            int mid = low + ((high - low) >> 1);
            int cmp = keys.AsSpan(mid * n, n).SequenceCompareTo(query);
            if (cmp == 0)
            {
                return mid;
            }

            if (cmp < 0)
            {
                low = mid + 1;
            }
            else
            {
                high = mid - 1;
            }
        }

        return -1;
    }
}
