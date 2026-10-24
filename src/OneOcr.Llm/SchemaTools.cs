using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Schema;
using System.Text.Json.Serialization.Metadata;

namespace OneOcr.Llm;

/// <summary>JSON Schema handling: export from source-generated type information, validation of caller schemas, and the per-provider dialect changes.</summary>
internal static class SchemaTools
{
    private static readonly string[] GeminiDropped = ["additionalProperties", "$schema", "$id", "$comment", "pattern", "default", "const", "examples", "unevaluatedProperties"];
    private static readonly string[] SchemaMapKeywords = ["properties", "$defs", "definitions", "patternProperties"];
    private static readonly string[] SchemaKeywords = ["items", "additionalProperties", "not", "if", "then", "else", "contains", "propertyNames", "unevaluatedProperties"];
    private static readonly string[] SchemaListKeywords = ["anyOf", "oneOf", "allOf", "prefixItems"];

    /// <summary>The schema of a type, from its source-generated type information (no reflection). Reference types are non-nullable unless annotated.</summary>
    public static string Export(JsonTypeInfo typeInfo)
    {
        ArgumentNullException.ThrowIfNull(typeInfo);
        JsonNode node = JsonSchemaExporter.GetJsonSchemaAsNode(typeInfo, new JsonSchemaExporterOptions { TreatNullObliviousAsNonNullable = true });
        return node.ToJsonString();
    }

    /// <summary>Validates a caller's schema text: size and depth under the limits, valid JSON, an object schema with <c>"type": "object"</c> at the root.</summary>
    public static JsonObject Parse(string schema, OcrLimits limits)
    {
        ArgumentNullException.ThrowIfNull(schema);
        int size = Encoding.UTF8.GetByteCount(schema);
        if (size > limits.MaxLlmSchemaBytes)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxLlmSchemaBytes), limits.MaxLlmSchemaBytes, size);
        }

        byte[] bytes = Encoding.UTF8.GetBytes(schema);
        JsonNode? node;
        try
        {
            ProviderJson.CheckDepth(bytes, limits.MaxLlmJsonDepth);
            node = JsonNode.Parse(bytes, null, new JsonDocumentOptions { MaxDepth = limits.MaxLlmJsonDepth });
        }
        catch (JsonException e)
        {
            throw new ArgumentException("The schema is not valid JSON: " + e.Message, nameof(schema), e);
        }

        if (node is not JsonObject root || root["type"] is not JsonValue type || !type.TryGetValue(out string? t) || t != "object")
        {
            throw new ArgumentException("The schema must be a JSON object with \"type\": \"object\" at the root.", nameof(schema));
        }

        return root;
    }

    /// <summary>
    /// The form OpenAI's strict structured output accepts: every object lists all its properties as required and forbids
    /// extra ones. Optional values must therefore be nullable types in the caller's schema.
    /// </summary>
    public static JsonObject ForStrict(JsonObject schema)
    {
        var copy = (JsonObject)schema.DeepClone();
        Walk(copy, o =>
        {
            if (o["properties"] is JsonObject props)
            {
                var required = new JsonArray();
                foreach (var kv in props)
                {
                    required.Add((JsonNode?)JsonValue.Create(kv.Key));
                }

                o["required"] = required;
                o["additionalProperties"] = false;
            }
            else if (o["type"] is JsonValue v && v.TryGetValue(out string? type) && type == "object")
            {
                o["properties"] = new JsonObject();
                o["required"] = new JsonArray();
                o["additionalProperties"] = false;
            }
        });
        return copy;
    }

    /// <summary>
    /// Gemini's <c>responseSchema</c> dialect, an OpenAPI subset: no <c>additionalProperties</c>, <c>pattern</c>, <c>default</c>
    /// or <c>$ref</c>; nullable types are written <c>"nullable": true</c>; only the formats Gemini lists are kept.
    /// A schema that needs <c>$ref</c> (a recursive type) cannot be expressed and throws <see cref="ArgumentException"/>.
    /// </summary>
    public static JsonObject ForGemini(JsonObject schema)
    {
        var copy = (JsonObject)schema.DeepClone();
        copy.Remove("$defs");
        copy.Remove("definitions");
        Walk(copy, o =>
        {
            if (o.ContainsKey("$ref"))
            {
                throw new ArgumentException("Gemini's response schema has no $ref, so recursive or shared types cannot be sent.", nameof(schema));
            }

            foreach (string key in GeminiDropped)
            {
                o.Remove(key);
            }

            if (o["format"] is JsonValue f && f.TryGetValue(out string? format) && format is not ("date-time" or "enum" or "int32" or "int64" or "float" or "double"))
            {
                o.Remove("format");
            }

            if (o["type"] is JsonArray types)
            {
                var names = new List<string>();
                bool nullable = false;
                foreach (JsonNode? n in types)
                {
                    string? name = n is JsonValue v && v.TryGetValue(out string? s) ? s : null;
                    if (name == "null")
                    {
                        nullable = true;
                    }
                    else if (name is not null)
                    {
                        names.Add(name);
                    }
                }

                o.Remove("type");
                if (names.Count == 1)
                {
                    o["type"] = names[0];
                }
                else if (names.Count > 1)
                {
                    var any = new JsonArray();
                    foreach (string name in names)
                    {
                        any.Add((JsonNode)new JsonObject { ["type"] = name });
                    }

                    o["anyOf"] = any;
                }

                if (nullable)
                {
                    o["nullable"] = true;
                }
            }
        });
        return copy;
    }

    /// <summary>Visits every schema object in the tree, parents first, following only schema-bearing keywords. Iterative: depth of the input never becomes call depth.</summary>
    private static void Walk(JsonObject root, Action<JsonObject> visit)
    {
        var stack = new Stack<JsonObject>();
        stack.Push(root);
        while (stack.Count > 0)
        {
            JsonObject o = stack.Pop();
            visit(o);
            foreach (string key in SchemaMapKeywords)
            {
                if (o[key] is JsonObject map)
                {
                    foreach (var kv in map)
                    {
                        if (kv.Value is JsonObject child)
                        {
                            stack.Push(child);
                        }
                    }
                }
            }

            foreach (string key in SchemaKeywords)
            {
                if (o[key] is JsonObject child)
                {
                    stack.Push(child);
                }
                else if (key == "items" && o[key] is JsonArray tuple)
                {
                    PushAll(stack, tuple);
                }
            }

            foreach (string key in SchemaListKeywords)
            {
                if (o[key] is JsonArray list)
                {
                    PushAll(stack, list);
                }
            }
        }
    }

    private static void PushAll(Stack<JsonObject> stack, JsonArray list)
    {
        foreach (JsonNode? n in list)
        {
            if (n is JsonObject child)
            {
                stack.Push(child);
            }
        }
    }
}
