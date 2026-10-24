using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;
using OneOcr.Llm;

namespace OneOcr.Training;

/// <summary>One scalar field of an <see cref="ExtractionSchema"/>.</summary>
/// <param name="Name">The JSON property name; it becomes the model's class name and the key in the labelled JSON.</param>
/// <param name="Type">The JSON type: <c>string</c>, <c>number</c>, <c>integer</c> or <c>boolean</c>.</param>
/// <param name="Description">The schema's description of the field, shown to the labelling model; may be null.</param>
/// <param name="Format">The schema's <c>format</c> (for example <c>date</c>), or null.</param>
public sealed record ExtractionField(string Name, string Type, string? Description, string? Format);

/// <summary>
/// The fields of a custom document type: the top-level scalar properties of a JSON Schema (or of a type's source-generated schema).
/// Each field becomes a class of the fine-tuned model that tags the words holding its value. Nested objects and arrays are not read by
/// the model and are left out with a note; a line-items table is a separate job.
/// </summary>
public sealed class ExtractionSchema
{
    private ExtractionSchema(IReadOnlyList<ExtractionField> fields, IReadOnlyList<string> notes)
    {
        Fields = fields;
        Notes = notes;
    }

    /// <summary>The fields in schema order; the order fixes the model's class order.</summary>
    public IReadOnlyList<ExtractionField> Fields { get; }

    /// <summary>What was left out and why (nested properties, properties without a scalar type).</summary>
    public IReadOnlyList<string> Notes { get; }

    /// <summary>Reads the scalar properties of a JSON Schema object.</summary>
    /// <param name="jsonSchema">A JSON Schema whose root is <c>"type": "object"</c> with <c>properties</c>.</param>
    /// <param name="limits">Bounds for the schema text; null for the defaults.</param>
    /// <returns>The schema.</returns>
    /// <exception cref="ArgumentException">The text is not a usable object schema, or no property is a scalar, or a name is not allowed for a class.</exception>
    /// <exception cref="OcrLimitException">The text is over the limits' schema size or depth.</exception>
    public static ExtractionSchema FromJsonSchema(string jsonSchema, OcrLimits? limits = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(jsonSchema);
        limits ??= OcrLimits.Default;
        JsonObject root = SchemaTools.Parse(jsonSchema, limits);
        if (root["properties"] is not JsonObject props)
        {
            throw new ArgumentException("The schema has no 'properties'.", nameof(jsonSchema));
        }

        var fields = new List<ExtractionField>();
        var notes = new List<string>();
        foreach ((string name, JsonNode? node) in props)
        {
            if (node is not JsonObject p)
            {
                notes.Add($"'{name}': not a schema object; left out.");
                continue;
            }

            string? type = null;
            switch (p["type"])
            {
                case JsonValue v when v.TryGetValue(out string? s):
                    type = s;
                    break;
                case JsonArray a:
                    foreach (JsonNode? t in a)
                    {
                        if (t is JsonValue tv && tv.TryGetValue(out string? ts) && ts != "null")
                        {
                            type = type is null ? ts : "mixed";
                        }
                    }

                    break;
            }

            if (type is not ("string" or "number" or "integer" or "boolean"))
            {
                notes.Add($"'{name}': {(type is "object" or "array" ? "nested values are not read by the extraction model" : "no single scalar type")}; left out.");
                continue;
            }

            string? description = p["description"] is JsonValue dv && dv.TryGetValue(out string? d) ? d : null;
            string? format = p["format"] is JsonValue fv && fv.TryGetValue(out string? f) ? f : null;
            fields.Add(new ExtractionField(name, type, description, format));
        }

        if (fields.Count == 0)
        {
            throw new ArgumentException("The schema has no scalar properties to extract.", nameof(jsonSchema));
        }

        DocContainerWriter.ValidateClasses([.. fields.Select(static f => f.Name)], limits);
        return new ExtractionSchema(fields, notes);
    }

    /// <summary>Reads the scalar properties of a type from its source-generated type information.</summary>
    /// <typeparam name="T">The type.</typeparam>
    /// <param name="typeInfo">Type information from a <see cref="System.Text.Json.Serialization.JsonSerializerContext"/>.</param>
    /// <param name="limits">Bounds; null for the defaults.</param>
    /// <returns>The schema.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="typeInfo"/> is null.</exception>
    /// <exception cref="ArgumentException">See <see cref="FromJsonSchema"/>.</exception>
    public static ExtractionSchema From<T>(JsonTypeInfo<T> typeInfo, OcrLimits? limits = null)
    {
        ArgumentNullException.ThrowIfNull(typeInfo);
        return FromJsonSchema(SchemaTools.Export(typeInfo), limits);
    }

    /// <summary>
    /// The JSON Schema sent to the labelling model: the same fields, every one nullable, so that a value the page does not show comes back
    /// as <c>null</c> rather than a guess.
    /// </summary>
    /// <returns>The schema text.</returns>
    public string ToJsonSchema()
    {
        var props = new JsonObject();
        foreach (ExtractionField f in Fields)
        {
            var p = new JsonObject { ["type"] = new JsonArray(f.Type, "null") };
            if (f.Format is not null)
            {
                p["format"] = f.Format;
            }

            if (f.Description is not null)
            {
                p["description"] = f.Description;
            }

            props[f.Name] = p;
        }

        return new JsonObject { ["type"] = "object", ["properties"] = props }.ToJsonString();
    }
}
