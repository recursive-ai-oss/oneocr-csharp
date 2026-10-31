using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace OneOcr.Documents;

/// <summary>
/// The values a custom extraction model read, shaped into the caller's own type, with where each came from on the page.
/// </summary>
/// <typeparam name="T">The caller's type.</typeparam>
public sealed class ExtractionResult<T>
{
    private readonly OcrPage _page;
    private readonly HashSet<string> _propertyNames;

    internal ExtractionResult(
        T value, IReadOnlyDictionary<string, ExtractedField> fields, DocumentExtraction extraction, IReadOnlyList<OcrDiagnostic> diagnostics, OcrPage page,
        HashSet<string> propertyNames)
    {
        _propertyNames = propertyNames;
        Value = value;
        Fields = fields;
        Extraction = extraction;
        Diagnostics = diagnostics;
        _page = page;
    }

    /// <summary>The value. A property whose field the model did not find, or whose text could not be read as the property's type, keeps its default (null for reference and nullable types).</summary>
    public T Value { get; }

    /// <summary>The fields that were found and used, by the property's JSON name: text as read, the words' indices into <see cref="OcrPage.Words"/>, the box around them, and the confidence (the mean probability of the words' tags, not a calibrated probability of being right).</summary>
    public IReadOnlyDictionary<string, ExtractedField> Fields { get; }

    /// <summary>The underlying extraction, with the generic key–value pairs.</summary>
    public DocumentExtraction Extraction { get; }

    /// <summary>Notes: properties without a class in the model, text that did not parse as the property's type, and the extraction's own notes.</summary>
    public IReadOnlyList<OcrDiagnostic> Diagnostics { get; }

    /// <summary>The boxes of the words a field was read from, one per word, in reading order.</summary>
    /// <param name="name">The property's JSON name.</param>
    /// <returns>The boxes; empty when the property exists but its field was not found on the page.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="name"/> is null.</exception>
    /// <exception cref="KeyNotFoundException"><paramref name="name"/> is not the JSON name of a property of <typeparamref name="T"/> (a misspelt name, or the C# name of a renamed property).</exception>
    public IReadOnlyList<OcrBox> WordBoxes(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        if (!_propertyNames.Contains(name))
        {
            throw new KeyNotFoundException($"'{name}' is not the JSON name of a property of {typeof(T).Name}. Names are: {string.Join(", ", _propertyNames.Order(StringComparer.Ordinal))}.");
        }

        if (!Fields.TryGetValue(name, out ExtractedField? f))
        {
            return [];
        }

        IReadOnlyList<OcrWord> words = _page.Words;
        return [.. f.WordIndices.Select(i => words[i].Box)];
    }
}

public sealed partial class DocumentExtractor
{
    /// <summary>
    /// Reads a page with a custom model (one trained for a schema with <c>OcrTrainer</c>'s extraction counterpart, <c>ExtractionTrainer</c>) and returns
    /// the fields as <typeparamref name="T"/>. A property is matched to the model's class of the same JSON name, so a type with source-generated
    /// type information and a schema made from it line up without further mapping. Text is turned into the property's type by the same rules
    /// that check LLM answers against a page: amounts and quantities for numeric properties (the page's own number style), dates for
    /// <see cref="DateOnly"/>, <see cref="DateTime"/> and <see cref="DateTimeOffset"/> (a date needs a year, and a day-and-month pair that reads two ways
    /// stays unread), <c>true</c>/<c>false</c>/yes/no for <see cref="bool"/>, the words as read for strings. Nothing uses reflection.
    /// Nested objects and arrays are not read: a property of such a type is left at its default and noted.
    /// </summary>
    /// <typeparam name="T">The caller's type; a class or struct with an object contract.</typeparam>
    /// <param name="page">The page; its words, in reading order, are what the model reads.</param>
    /// <param name="typeInfo">Type information from a <see cref="System.Text.Json.Serialization.JsonSerializerContext"/>.</param>
    /// <param name="options">Options for this call; null for the defaults. <see cref="DocumentExtractionOptions.Strict"/> turns an unmapped property into an error.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The value, and per field its words, box and confidence.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="page"/> or <paramref name="typeInfo"/> is null.</exception>
    /// <exception cref="ArgumentException">The type has no object contract; none of its properties has a class of the same JSON name in the model (or, with <see cref="DocumentExtractionOptions.Strict"/>, one has not); or the values cannot be put into it (for example a required member whose field was not found). The message names the properties and the model's classes.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="options"/> holds a value out of range.</exception>
    /// <exception cref="OcrLimitException">The page has more words than <see cref="OcrLimits.MaxWordsPerPage"/>.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public ExtractionResult<T> Extract<T>(OcrPage page, JsonTypeInfo<T> typeInfo, DocumentExtractionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(page);
        ArgumentNullException.ThrowIfNull(typeInfo);
        if (typeInfo.Kind != JsonTypeInfoKind.Object)
        {
            throw new ArgumentException("The type needs an object contract (a class or struct with properties).", nameof(typeInfo));
        }

        var unmapped = new List<string>();
        var propertyNames = new HashSet<string>(StringComparer.Ordinal);
        foreach (JsonPropertyInfo p in typeInfo.Properties)
        {
            propertyNames.Add(p.Name);
            if (!_model.Schema.Classes.Contains(p.Name))
            {
                unmapped.Add(p.Name);
            }
        }

        // A type none of whose properties the model can fill would return defaults that look like "nothing found"; that is a mismatch, not a result.
        if (propertyNames.Count > 0 && unmapped.Count == propertyNames.Count || (options?.Strict == true && unmapped.Count > 0))
        {
            throw new ArgumentException(
                $"{(unmapped.Count == propertyNames.Count ? "No property" : "Not every property")} of {typeof(T).Name} has a class of the same JSON name in the model. Unmapped: {string.Join(", ", unmapped)}. The model's classes: {string.Join(", ", _model.Schema.Classes)}. "
                + "Name the properties after the classes, for example with [JsonPropertyName(\"BT-112\")].",
                nameof(typeInfo));
        }

        DocumentExtraction ex = Run(page, null, options, cancellationToken);
        var notes = new List<OcrDiagnostic>(ex.Diagnostics);
        var byName = new Dictionary<string, ExtractedField>(StringComparer.Ordinal);
        foreach (ExtractedField f in ex.Fields)
        {
            byName[f.Id] = f;
        }

        var texts = page.Words.Where(static w => w.Text.Length > 0).Select(static w => w.Text).ToList();
        NumberStyle style = ValueNormalize.DetectStyle(texts);
        if (style == NumberStyle.Auto)
        {
            style = NumberStyle.PointDecimal;
        }

        DateOrder order = ValueNormalize.DetectOrder(texts);
        var used = new Dictionary<string, ExtractedField>(StringComparer.Ordinal);
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms))
        {
            w.WriteStartObject();
            foreach (JsonPropertyInfo p in typeInfo.Properties)
            {
                if (!_model.Schema.Classes.Contains(p.Name))
                {
                    notes.Add(Diag.Warn(OcrDiagnosticCodes.PropertyUnmapped, $"The model has no class '{p.Name}'; the property keeps its default."));
                    continue;
                }

                if (!byName.TryGetValue(p.Name, out ExtractedField? field))
                {
                    continue;
                }

                if (WriteValue(w, p, field, style, order, out string? problem))
                {
                    used[p.Name] = field;
                }
                else
                {
                    notes.Add(Diag.Warn(OcrDiagnosticCodes.ValueNotConverted, $"'{p.Name}': {problem}"));
                }
            }

            w.WriteEndObject();
        }

        T? value;
        try
        {
            value = JsonSerializer.Deserialize(ms.ToArray(), typeInfo);
        }
        catch (JsonException e)
        {
            throw new ArgumentException("The values cannot be put into the type: " + e.Message, nameof(typeInfo), e);
        }

        return new ExtractionResult<T>(value ?? throw new ArgumentException("The type reads an object as null.", nameof(typeInfo)), used, ex, notes, page, propertyNames);
    }

    /// <summary>Asynchronous <see cref="Extract{T}(OcrPage, JsonTypeInfo{T}, DocumentExtractionOptions?, CancellationToken)"/>: runs the same local work on the thread pool.</summary>
    /// <typeparam name="T">The caller's type.</typeparam>
    /// <param name="page">The page.</param>
    /// <param name="typeInfo">Type information from a <see cref="System.Text.Json.Serialization.JsonSerializerContext"/>.</param>
    /// <param name="options">Options for this call; null for the defaults. <see cref="DocumentExtractionOptions.Strict"/> turns an unmapped property into an error.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The value, and per field its words, box and confidence.</returns>
    public Task<ExtractionResult<T>> ExtractAsync<T>(OcrPage page, JsonTypeInfo<T> typeInfo, DocumentExtractionOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Extract(page, typeInfo, options, cancellationToken), cancellationToken);

    // Writes "name": value for one property when the field's text reads as the property's type; otherwise says why not.
    private static bool WriteValue(Utf8JsonWriter w, JsonPropertyInfo p, ExtractedField field, NumberStyle style, DateOrder order, out string? problem)
    {
        problem = null;
        Type t = Nullable.GetUnderlyingType(p.PropertyType) ?? p.PropertyType;
        string text = field.Text;
        if (t == typeof(string))
        {
            w.WriteString(p.Name, text);
            return true;
        }

        if (t == typeof(bool))
        {
            string v = text.Trim().Trim('.', ':', ';').ToLowerInvariant();
            if (v is "true" or "yes" or "y" or "ja" or "oui" or "si" or "sí" or "x" or "1" or "\u2611" or "\u2713" or "\u2714")
            {
                w.WriteBoolean(p.Name, true);
                return true;
            }

            if (v is "false" or "no" or "n" or "nein" or "non" or "0" or "\u2610")
            {
                w.WriteBoolean(p.Name, false);
                return true;
            }

            problem = $"'{text}' is not a yes/no value.";
            return false;
        }

        if (t == typeof(decimal) || t == typeof(double) || t == typeof(float) || IsIntegral(t))
        {
            if (!ValueNormalize.TryParseAmount(text, style, out ParsedAmount amount))
            {
                problem = $"'{text}' does not read as a number.";
                return false;
            }

            decimal d = amount.Value;
            if (IsIntegral(t) && (d != decimal.Truncate(d) || !FitsIntegral(t, d)))
            {
                problem = $"{d.ToString(CultureInfo.InvariantCulture)} is not a whole number in the range of {t.Name}.";
                return false;
            }

            if ((t == typeof(double) && Math.Abs(d) > 1e15m) || (t == typeof(float) && Math.Abs(d) > 1e7m))
            {
                problem = $"{d.ToString(CultureInfo.InvariantCulture)} is outside the exact range of {t.Name}.";
                return false;
            }

            w.WriteNumber(p.Name, d);
            return true;
        }

        if (t == typeof(DateOnly) || t == typeof(DateTime) || t == typeof(DateTimeOffset))
        {
            List<ParsedDate> dates = ValueNormalize.ParseDate(text, order);
            if (dates.Count != 1 || dates[0].Year is not { } y || y is < 1 or > 9999 || dates[0].Month is < 1 or > 12 || dates[0].Day < 1
                || dates[0].Day > DateTime.DaysInMonth(y, dates[0].Month))
            {
                problem = $"'{text}' does not read as one complete date.";
                return false;
            }

            w.WriteString(p.Name, new DateOnly(y, dates[0].Month, dates[0].Day).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture));
            return true;
        }

        // Other scalars the type's own converter reads from a string.
        if (t.IsEnum || t == typeof(Guid) || t == typeof(TimeSpan) || t == typeof(char) || t == typeof(Uri))
        {
            w.WriteString(p.Name, text);
            return true;
        }

        problem = "its type is not a scalar this reader fills (strings, numbers, booleans, dates, enums and GUIDs are); it keeps its default.";
        return false;
    }

    private static bool IsIntegral(Type t) =>
        t == typeof(int) || t == typeof(long) || t == typeof(short) || t == typeof(byte) || t == typeof(sbyte) || t == typeof(uint) || t == typeof(ulong)
        || t == typeof(ushort);

    private static bool FitsIntegral(Type t, decimal d) =>
        t == typeof(int) ? d is >= int.MinValue and <= int.MaxValue
        : t == typeof(long) ? d is >= long.MinValue and <= long.MaxValue
        : t == typeof(short) ? d is >= short.MinValue and <= short.MaxValue
        : t == typeof(byte) ? d is >= byte.MinValue and <= byte.MaxValue
        : t == typeof(sbyte) ? d is >= sbyte.MinValue and <= sbyte.MaxValue
        : t == typeof(uint) ? d is >= uint.MinValue and <= uint.MaxValue
        : t == typeof(ushort) ? d is >= ushort.MinValue and <= ushort.MaxValue
        : d is >= ulong.MinValue and <= ulong.MaxValue;
}
