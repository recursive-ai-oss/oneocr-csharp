using System.Buffers.Binary;
using System.Text;

namespace OneOcr.Decoding;

/// <summary>
/// How often words occur in running text: a sorted table of lowercase Latin-script words with a count each (format in
/// <c>docs/spec/word-lm-v1.md</c>). The recognizer uses it to take back a space the network put inside a word: a joined word
/// that is far more frequent than its two pieces together ("condition" "ally") was probably never two words. The table knows
/// nothing about numbers, codes or names, and those are never judged with it. Instances are immutable and thread-safe.
/// </summary>
public sealed class WordFrequencyModel
{
    private const int HeaderSize = 22;
    private const int BlockSize = 16;
    private const int MaxWordBytes = 40;

    private static ReadOnlySpan<byte> Magic => "OCWF"u8;

    private readonly byte[] _data;
    private readonly int[] _blocks;

    private WordFrequencyModel(byte[] data, int[] blocks, int words, int totalQuarterBits, int missingQuarterBits)
    {
        _data = data;
        _blocks = blocks;
        WordCount = words;
        TotalQuarterBits = totalQuarterBits;
        MissingQuarterBits = missingQuarterBits;
    }

    /// <summary>The number of words in the table.</summary>
    public int WordCount { get; }

    /// <summary>Four times the base-2 logarithm of the number of words the table was counted from.</summary>
    internal int TotalQuarterBits { get; }

    /// <summary>The value, in the same unit, taken for a word the table does not hold.</summary>
    internal int MissingQuarterBits { get; }

    /// <summary>Reads a word frequency file.</summary>
    /// <param name="path">A file in the format of <c>docs/spec/word-lm-v1.md</c>.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The model.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="path"/> is null.</exception>
    /// <exception cref="OcrModelException">The file is malformed.</exception>
    /// <exception cref="OcrLimitException">The file exceeds <paramref name="limits"/>.</exception>
    public static WordFrequencyModel Load(string path, OcrLimits? limits = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        using FileStream stream = File.OpenRead(path);
        return Load(stream, limits);
    }

    /// <summary>Reads a word frequency file from a stream, taking at most <see cref="OcrLimits.MaxLanguageModelFileBytes"/> bytes.</summary>
    /// <param name="stream">A stream positioned at the start of the file.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The model.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null.</exception>
    /// <exception cref="OcrModelException">The data is malformed.</exception>
    /// <exception cref="OcrLimitException">The data exceeds <paramref name="limits"/>.</exception>
    public static WordFrequencyModel Load(Stream stream, OcrLimits? limits = null)
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

    /// <summary>Parses word frequency bytes; every entry is checked (order, lengths, bounds) before the model is returned.</summary>
    /// <param name="data">The whole file.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The model.</returns>
    /// <exception cref="OcrModelException">The data is malformed.</exception>
    /// <exception cref="OcrLimitException">The data exceeds <paramref name="limits"/>.</exception>
    public static WordFrequencyModel Parse(ReadOnlySpan<byte> data, OcrLimits? limits = null)
    {
        limits ??= OcrLimits.Default;
        if (data.Length > limits.MaxLanguageModelFileBytes)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxLanguageModelFileBytes), limits.MaxLanguageModelFileBytes, data.Length);
        }

        if (data.Length < HeaderSize || !data[..4].SequenceEqual(Magic))
        {
            throw new OcrModelException("Not a word frequency file (bad magic).");
        }

        int version = BinaryPrimitives.ReadUInt16LittleEndian(data[4..]);
        if (version != 1)
        {
            throw new OcrModelException($"Unsupported word frequency file version {version}.");
        }

        if (BinaryPrimitives.ReadUInt16LittleEndian(data[6..]) != 0)
        {
            throw new OcrModelException("A word frequency file has flags this reader does not know.");
        }

        uint words = BinaryPrimitives.ReadUInt32LittleEndian(data[12..]);
        int totalBits = data[16], missingBits = data[17];
        uint blockCount = BinaryPrimitives.ReadUInt32LittleEndian(data[18..]);
        if (words > limits.MaxWordModelWords)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxWordModelWords), limits.MaxWordModelWords, words);
        }

        if (blockCount != (words + BlockSize - 1) / BlockSize)
        {
            throw new OcrModelException("The block count does not match the word count.");
        }

        long indexEnd = HeaderSize + (4L * blockCount);
        if (indexEnd > data.Length)
        {
            throw new OcrModelException("The file ends inside the block index.");
        }

        ReadOnlySpan<byte> body = data[(int)indexEnd..];
        var blocks = new int[blockCount];
        for (int i = 0; i < blocks.Length; i++)
        {
            uint offset = BinaryPrimitives.ReadUInt32LittleEndian(data[(HeaderSize + (4 * i))..]);
            if (offset >= body.Length && !(offset == 0 && body.Length == 0))
            {
                throw new OcrModelException("A block starts outside the word data.");
            }

            blocks[i] = (int)offset;
        }

        // Walk every entry once: prefixes and lengths stay inside the word, words rise strictly, each block starts where the last
        // one ended. Lookups then need no checks beyond the slice bounds.
        Span<byte> previous = stackalloc byte[MaxWordBytes];
        Span<byte> current = stackalloc byte[MaxWordBytes];
        int previousLength = 0, pos = 0;
        for (uint w = 0; w < words; w++)
        {
            bool first = w % BlockSize == 0;
            if (first)
            {
                if (blocks[w / BlockSize] != pos)
                {
                    throw new OcrModelException("A block does not start where the previous block ended.");
                }

                previousLength = 0;
            }

            if (pos + 2 > body.Length)
            {
                throw new OcrModelException("The word data ends inside an entry.");
            }

            int shared = body[pos], suffix = body[pos + 1];
            pos += 2;
            if (shared > previousLength || shared + suffix > MaxWordBytes || shared + suffix == 0 || (first && shared != 0))
            {
                throw new OcrModelException("A word entry has an impossible length.");
            }

            if (pos + suffix + 1 > body.Length)
            {
                throw new OcrModelException("The word data ends inside an entry.");
            }

            previous[..shared].CopyTo(current);
            body.Slice(pos, suffix).CopyTo(current[shared..]);
            int length = shared + suffix;
            pos += suffix;
            if (body[pos] == 0)
            {
                throw new OcrModelException("A word has a zero count.");
            }

            pos++;
            if (!first && current[..length].SequenceCompareTo(previous[..previousLength]) <= 0)
            {
                throw new OcrModelException("The words are not in ascending order.");
            }

            current[..length].CopyTo(previous);
            previousLength = length;
        }

        if (pos != body.Length)
        {
            throw new OcrModelException("Unexpected bytes after the last word.");
        }

        return new WordFrequencyModel(body.ToArray(), blocks, (int)words, totalBits, missingBits);
    }

    /// <summary>
    /// The count of a word, as four times the base-2 logarithm of how often it was seen.
    /// </summary>
    /// <param name="word">The word, lowercase (the table's own folding), UTF-8.</param>
    /// <returns>The value, or <see cref="MissingQuarterBits"/> when the table does not hold the word.</returns>
    internal int Quarter(ReadOnlySpan<byte> word) => TryFind(word, out int q) ? q : MissingQuarterBits;

    /// <summary>Looks a word up: the last block whose first word is not above it, then a scan of that block.</summary>
    internal bool TryFind(ReadOnlySpan<byte> word, out int quarterBits)
    {
        quarterBits = 0;
        if (word.Length is 0 or > MaxWordBytes || _blocks.Length == 0)
        {
            return false;
        }

        int lo = 0, hi = _blocks.Length - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) / 2;
            if (FirstWord(mid).SequenceCompareTo(word) <= 0)
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1;
            }
        }

        Span<byte> current = stackalloc byte[MaxWordBytes];
        int pos = _blocks[lo];
        int entries = Math.Min(BlockSize, WordCount - (lo * BlockSize));
        for (int i = 0; i < entries; i++)
        {
            int shared = _data[pos], suffix = _data[pos + 1];
            _data.AsSpan(pos + 2, suffix).CopyTo(current[shared..]);
            int length = shared + suffix;
            int order = current[..length].SequenceCompareTo(word);
            if (order == 0)
            {
                quarterBits = _data[pos + 2 + suffix];
                return true;
            }

            if (order > 0)
            {
                return false;
            }

            pos += 2 + suffix + 1;
        }

        return false;
    }

    private ReadOnlySpan<byte> FirstWord(int block)
    {
        int pos = _blocks[block];
        return _data.AsSpan(pos + 2, _data[pos + 1]);
    }

    /// <summary>Like <see cref="Quarter(ReadOnlySpan{byte})"/> for a lowercase word given as text.</summary>
    internal int Quarter(string word)
    {
        Span<byte> bytes = stackalloc byte[MaxWordBytes * 4];
        if (word.Length > MaxWordBytes || Encoding.UTF8.GetByteCount(word) > MaxWordBytes)
        {
            return MissingQuarterBits;
        }

        int n = Encoding.UTF8.GetBytes(word, bytes);
        return Quarter(bytes[..n]);
    }

    /// <summary>Whether the table holds the lowercase word.</summary>
    internal bool Contains(string word)
    {
        Span<byte> bytes = stackalloc byte[MaxWordBytes * 4];
        if (word.Length > MaxWordBytes || Encoding.UTF8.GetByteCount(word) > MaxWordBytes)
        {
            return false;
        }

        int n = Encoding.UTF8.GetBytes(word, bytes);
        return TryFind(bytes[..n], out _);
    }
}
