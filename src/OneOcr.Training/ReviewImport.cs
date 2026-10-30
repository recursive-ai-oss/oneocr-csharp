using System.Text;
using System.Text.Json;

namespace OneOcr.Training;

/// <summary>One correction read from a review file: the corrected text of a queued line.</summary>
/// <param name="Id">The queue item's id.</param>
/// <param name="Text">The line as it should be read. Empty is allowed and marks a line with no text (it is dropped from training).</param>
public readonly record struct ReviewCorrection(string Id, string Text);

/// <summary>What a review file held after reading, and what became of each correction.</summary>
/// <param name="Corrections">Corrections that were read.</param>
/// <param name="Unreviewed">Items without a <c>text</c> member (or with <c>null</c>), left in the queue.</param>
internal sealed record ReviewFile(IReadOnlyList<ReviewCorrection> Corrections, int Unreviewed);

/// <summary>
/// Reader for review files (<c>{"format":"oneocr.review","version":1,"items":[{"id":"…","text":"…"}]}</c>). A review file is
/// input from outside: it is read with <see cref="Utf8JsonReader"/> under the depth, size, item-count and text-length limits of
/// <see cref="OcrLimits"/>, items are never executed or looked up on disk, and only <c>id</c> and <c>text</c> are used (other
/// members of an exported item are skipped).
/// </summary>
internal static class ReviewImport
{
    public const string Format = "oneocr.review";
    private const int MaxIdLength = 128;

    /// <summary>Parses a review file.</summary>
    /// <exception cref="OcrFormatException">The data is not a review file of this version, or an id or text is malformed.</exception>
    /// <exception cref="OcrLimitException">The data is larger than the limits allow, or has too many items or too long a text.</exception>
    public static ReviewFile Parse(ReadOnlySpan<byte> data, OcrLimits limits)
    {
        ArgumentNullException.ThrowIfNull(limits);
        if (data.Length > limits.MaxReviewImportBytes)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxReviewImportBytes), limits.MaxReviewImportBytes, data.Length);
        }

        // Strings are decoded only after the whole input is known to be UTF-8, so no later step meets invalid bytes.
        if (!System.Text.Unicode.Utf8.IsValid(data))
        {
            throw new OcrFormatException("The review file is not valid UTF-8.");
        }

        // Escapes of lone surrogates would make a later string read throw InvalidOperationException.
        JsonEscapes.RejectLoneSurrogates(data, "The review file");

        var corrections = new List<ReviewCorrection>();
        int unreviewed = 0;
        try
        {
            var reader = new Utf8JsonReader(data, new JsonReaderOptions { MaxDepth = Math.Max(2, limits.MaxJsonDepth) });
            Expect(ref reader, JsonTokenType.StartObject, "the document");
            bool sawFormat = false, sawItems = false, sawVersion = false;
            while (Next(ref reader) && reader.TokenType == JsonTokenType.PropertyName)
            {
                if (reader.ValueTextEquals("format"u8))
                {
                    Next(ref reader);
                    if (ReadString(ref reader, 64, "format", nullable: false) != Format)
                    {
                        throw new OcrFormatException($"'format' must be \"{Format}\".");
                    }

                    sawFormat = true;
                }
                else if (reader.ValueTextEquals("version"u8))
                {
                    Next(ref reader);
                    if (reader.TokenType != JsonTokenType.Number || !reader.TryGetInt32(out int v) || v != 1)
                    {
                        throw new OcrFormatException("'version' must be 1.");
                    }

                    sawVersion = true;
                }
                else if (reader.ValueTextEquals("items"u8))
                {
                    if (sawItems)
                    {
                        throw new OcrFormatException("'items' appears twice.");
                    }

                    Next(ref reader);
                    ReadItems(ref reader, limits, corrections, ref unreviewed);
                    sawItems = true;
                }
                else
                {
                    Next(ref reader);
                    reader.Skip();
                }
            }

            if (reader.TokenType != JsonTokenType.EndObject)
            {
                throw new OcrFormatException("The document is not a JSON object.");
            }

            if (!sawFormat || !sawVersion || !sawItems)
            {
                throw new OcrFormatException("A review file needs 'format', 'version' and 'items'.");
            }

            if (reader.Read())
            {
                throw new OcrFormatException("Unexpected content after the document.");
            }
        }
        catch (JsonException ex)
        {
            throw new OcrFormatException("The review file is not valid JSON: " + ex.Message, ex);
        }

        return new ReviewFile(corrections, unreviewed);
    }

    private static void ReadItems(ref Utf8JsonReader reader, OcrLimits limits, List<ReviewCorrection> corrections, ref int unreviewed)
    {
        if (reader.TokenType != JsonTokenType.StartArray)
        {
            throw new OcrFormatException("'items' must be an array.");
        }

        int count = 0;
        while (Next(ref reader) && reader.TokenType != JsonTokenType.EndArray)
        {
            if (++count > limits.MaxReviewItems)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxReviewItems), limits.MaxReviewItems, count);
            }

            if (reader.TokenType != JsonTokenType.StartObject)
            {
                throw new OcrFormatException($"Item {count - 1} is not an object.");
            }

            string? id = null, text = null;
            while (Next(ref reader) && reader.TokenType == JsonTokenType.PropertyName)
            {
                if (reader.ValueTextEquals("id"u8))
                {
                    Next(ref reader);
                    id = ReadString(ref reader, MaxIdLength, "id", nullable: false);
                }
                else if (reader.ValueTextEquals("text"u8))
                {
                    Next(ref reader);
                    text = ReadString(ref reader, limits.MaxReviewTextLength, "text", nullable: true);
                }
                else
                {
                    Next(ref reader);
                    reader.Skip();
                }
            }

            if (id is null || id.Length == 0)
            {
                throw new OcrFormatException($"Item {count - 1} has no 'id'.");
            }

            if (text is null)
            {
                unreviewed++;
            }
            else
            {
                corrections.Add(new ReviewCorrection(id, text));
            }
        }

        if (reader.TokenType != JsonTokenType.EndArray)
        {
            throw new OcrFormatException("'items' is not closed.");
        }
    }

    private static string? ReadString(ref Utf8JsonReader reader, int maxLength, string name, bool nullable)
    {
        if (reader.TokenType == JsonTokenType.Null && nullable)
        {
            return null;
        }

        if (reader.TokenType != JsonTokenType.String)
        {
            throw new OcrFormatException($"'{name}' must be a string.");
        }

        // The UTF-8 length bounds the decoded length from below, so a huge string is refused before it is decoded.
        int bytes = reader.HasValueSequence ? checked((int)reader.ValueSequence.Length) : reader.ValueSpan.Length;
        if (bytes / 6 > maxLength)
        {
            throw TooLong(name, maxLength, bytes / 6);
        }

        string value;
        try
        {
            value = reader.GetString() ?? "";
        }
        catch (InvalidOperationException ex)
        {
            throw new OcrFormatException($"'{name}' is not valid text.", ex);
        }

        if (value.Length > maxLength)
        {
            throw TooLong(name, maxLength, value.Length);
        }

        foreach (Rune r in value.EnumerateRunes())
        {
            // Replacement characters stand for bytes the writer could not encode; control characters never belong in a line.
            if (r.Value == 0xFFFD || Rune.IsControl(r))
            {
                throw new OcrFormatException($"'{name}' holds a control or replacement character.");
            }
        }

        return value;
    }

    // A text over the limit is a limit error; an id is a name, so one that long is simply not an id we issued.
    private static OcrException TooLong(string name, int max, long actual) =>
        name == "id" ? new OcrFormatException("An 'id' is longer than " + max + " characters.") : new OcrLimitException(nameof(OcrLimits.MaxReviewTextLength), max, actual);

    private static bool Next(ref Utf8JsonReader reader) =>
        reader.Read() ? true : throw new OcrFormatException("The review file ends early.");

    private static void Expect(ref Utf8JsonReader reader, JsonTokenType type, string what)
    {
        if (!Next(ref reader) || reader.TokenType != type)
        {
            throw new OcrFormatException($"Expected {type} for {what}.");
        }
    }
}
