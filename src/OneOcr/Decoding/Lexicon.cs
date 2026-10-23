using System.Text;

namespace OneOcr.Decoding;

/// <summary>
/// A list of words the user expects to see (product names, surnames, codes). During beam search a reading that spells a
/// listed word, or is on its way to spelling one, gets a bounded bonus (<see cref="DecodeOptions.LexiconBoost"/>), so a
/// listed word wins close calls. It never forces anything: a reading with enough acoustic evidence against it still loses.
/// Matching is exact (case and accents count) on whole space-delimited words. Immutable and thread-safe.
/// </summary>
public sealed class Lexicon
{
    private Lexicon(string[] words)
    {
        Words = words;
    }

    /// <summary>The number of distinct words.</summary>
    public int Count => Words.Length;

    internal string[] Words { get; }

    /// <summary>Builds a lexicon from words. Duplicates collapse. Words are matched code point by code point against the model's charset, which is NFC, so give NFC text.</summary>
    /// <param name="words">The words; none may be empty or contain whitespace.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The lexicon.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="words"/> is null.</exception>
    /// <exception cref="ArgumentException">A word is empty or contains whitespace.</exception>
    /// <exception cref="OcrLimitException">There are more words than <see cref="OcrLimits.MaxLexiconWords"/> or a word is longer than <see cref="OcrLimits.MaxLexiconWordLength"/>.</exception>
    public static Lexicon FromWords(IEnumerable<string> words, OcrLimits? limits = null)
    {
        ArgumentNullException.ThrowIfNull(words);
        limits ??= OcrLimits.Default;
        var set = new SortedSet<string>(StringComparer.Ordinal);
        foreach (string raw in words)
        {
            if (string.IsNullOrEmpty(raw) || raw.Any(char.IsWhiteSpace))
            {
                throw new ArgumentException("A lexicon word must be non-empty and contain no whitespace.", nameof(words));
            }

            if (raw.Length > limits.MaxLexiconWordLength)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxLexiconWordLength), limits.MaxLexiconWordLength, raw.Length);
            }

            set.Add(raw);
            if (set.Count > limits.MaxLexiconWords)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxLexiconWords), limits.MaxLexiconWords, set.Count);
            }
        }

        return new Lexicon([.. set]);
    }

    /// <summary>Reads a lexicon from a UTF-8 text file with one word per line; blank lines are skipped.</summary>
    /// <param name="path">The file, at most <see cref="OcrLimits.MaxLexiconFileBytes"/> bytes.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The lexicon.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="path"/> is null.</exception>
    /// <exception cref="OcrModelException">A line is not valid UTF-8 or contains whitespace inside a word.</exception>
    /// <exception cref="OcrLimitException">The file or the list exceeds <paramref name="limits"/>.</exception>
    public static Lexicon Load(string path, OcrLimits? limits = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        limits ??= OcrLimits.Default;
        long size = new FileInfo(path).Length;
        if (size > limits.MaxLexiconFileBytes)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxLexiconFileBytes), limits.MaxLexiconFileBytes, size);
        }

        return Parse(File.ReadAllBytes(path), limits);
    }

    /// <summary>Parses lexicon file bytes (UTF-8, one word per line).</summary>
    internal static Lexicon Parse(ReadOnlySpan<byte> data, OcrLimits limits)
    {
        if (data.Length > limits.MaxLexiconFileBytes)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxLexiconFileBytes), limits.MaxLexiconFileBytes, data.Length);
        }

        string text;
        try
        {
            text = new UTF8Encoding(false, true).GetString(data);
        }
        catch (ArgumentException ex)
        {
            throw new OcrModelException("The lexicon file is not valid UTF-8.", ex);
        }

        var words = new List<string>();
        foreach (string line in text.Split('\n'))
        {
            string word = line.Trim();
            if (word.Length == 0)
            {
                continue;
            }

            if (word.Any(char.IsWhiteSpace))
            {
                throw new OcrModelException("A lexicon line holds more than one word.");
            }

            words.Add(word);
            if (words.Count > limits.MaxLexiconWords)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxLexiconWords), limits.MaxLexiconWords, words.Count);
            }
        }

        return FromWords(words, limits);
    }
}
