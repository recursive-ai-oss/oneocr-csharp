using System.Text.Json;
using System.Text.Json.Nodes;

namespace OneOcr.Llm;

internal abstract class SNode
{
    public bool Nullable { get; init; }

    public string? Description { get; init; }
}

internal sealed class SObject : SNode
{
    public List<(string Name, SNode Node)> Props { get; } = [];
}

internal sealed class SArray(SNode items) : SNode
{
    public SNode Items { get; } = items;

    public JsonNode? MinItems { get; init; }

    public JsonNode? MaxItems { get; init; }
}

internal sealed class SLeaf(JsonObject schema) : SNode
{
    public JsonObject Schema { get; } = schema;
}

/// <summary>
/// The caller's schema read into a small tree, so the same walk can wrap every scalar value in
/// <c>{value, words, derived}</c> for the model, read the answer back, and write the plain JSON the caller asked for.
/// Recursion follows the schema tree, never the answer; the tree's size is capped.
/// </summary>
internal static class GroundedSchema
{
    private const int MaxRefDepth = 8;
    private const int MaxNodes = 20000;

    public static SNode Build(JsonObject root)
    {
        int budget = MaxNodes;
        return BuildNode(root, root, 0, ref budget, false);
    }

    private static SNode BuildNode(JsonObject o, JsonObject root, int refDepth, ref int budget, bool forceNullable)
    {
        if (--budget < 0)
        {
            throw new ArgumentException("The schema is too large to ground (shared definitions expand too far).");
        }

        string? description = o["description"] is JsonValue dv && dv.TryGetValue(out string? d) ? d : null;
        if (o["$ref"] is JsonValue rv && rv.TryGetValue(out string? reference))
        {
            if (refDepth >= MaxRefDepth)
            {
                throw new ArgumentException("The schema nests $ref more than " + MaxRefDepth + " levels deep, which grounding does not follow.");
            }

            JsonObject target = Resolve(root, reference) ?? throw new ArgumentException($"The schema reference '{reference}' cannot be resolved.");
            SNode inner = BuildNode(target, root, refDepth + 1, ref budget, forceNullable);
            return description is null || inner.Description is not null ? inner : With(inner, description);
        }

        foreach (string unsupported in new[] { "oneOf", "allOf", "prefixItems" })
        {
            if (o.ContainsKey(unsupported))
            {
                throw new ArgumentException($"'{unsupported}' is not supported when grounding values.");
            }
        }

        if (o["anyOf"] is JsonArray any)
        {
            JsonObject? only = null;
            bool nullBranch = false;
            foreach (JsonNode? b in any)
            {
                if (b is JsonObject bo && bo["type"] is JsonValue tv && tv.TryGetValue(out string? tn) && tn == "null")
                {
                    nullBranch = true;
                }
                else if (b is JsonObject bo2 && only is null)
                {
                    only = bo2;
                }
                else
                {
                    throw new ArgumentException("anyOf is supported only as one type or null when grounding values.");
                }
            }

            if (only is null)
            {
                throw new ArgumentException("anyOf has no usable branch.");
            }

            SNode inner = BuildNode(only, root, refDepth, ref budget, forceNullable || nullBranch);
            return description is null || inner.Description is not null ? inner : With(inner, description);
        }

        (string? type, bool nullable) = TypeOf(o);
        nullable |= forceNullable;
        switch (type)
        {
            case "object":
                {
                    if (o["properties"] is not JsonObject props)
                    {
                        throw new ArgumentException("An object without properties cannot be grounded.");
                    }

                    var node = new SObject { Nullable = nullable, Description = description };
                    foreach (var kv in props)
                    {
                        node.Props.Add((kv.Key, BuildNode(kv.Value as JsonObject ?? throw new ArgumentException("A property schema must be an object."), root, refDepth, ref budget, false)));
                    }

                    return node;
                }

            case "array":
                {
                    if (o["items"] is not JsonObject items)
                    {
                        throw new ArgumentException("An array needs an 'items' schema to be grounded.");
                    }

                    return new SArray(BuildNode(items, root, refDepth, ref budget, false))
                    {
                        Nullable = nullable, Description = description, MinItems = o["minItems"]?.DeepClone(), MaxItems = o["maxItems"]?.DeepClone(),
                    };
                }

            default:
                return new SLeaf((JsonObject)o.DeepClone()) { Nullable = nullable, Description = description };
        }
    }

    private static SNode With(SNode n, string description) => n switch
    {
        SObject so => CopyObject(so, description),
        SArray sa => new SArray(sa.Items) { Nullable = sa.Nullable, Description = description, MinItems = sa.MinItems, MaxItems = sa.MaxItems },
        SLeaf sl => new SLeaf(sl.Schema) { Nullable = sl.Nullable, Description = description },
        _ => n,
    };

    private static SObject CopyObject(SObject so, string description)
    {
        var copy = new SObject { Nullable = so.Nullable, Description = description };
        copy.Props.AddRange(so.Props);
        return copy;
    }

    private static JsonObject? Resolve(JsonObject root, string reference)
    {
        const string A = "#/$defs/", B = "#/definitions/";
        string? key = reference.StartsWith(A, StringComparison.Ordinal) ? reference[A.Length..] : reference.StartsWith(B, StringComparison.Ordinal) ? reference[B.Length..] : null;
        if (key is null)
        {
            return null;
        }

        string map = reference.StartsWith(A, StringComparison.Ordinal) ? "$defs" : "definitions";
        return root[map] is JsonObject defs && defs[key] is JsonObject def ? def : null;
    }

    private static (string? Type, bool Nullable) TypeOf(JsonObject o)
    {
        switch (o["type"])
        {
            case JsonValue v when v.TryGetValue(out string? s):
                return (s == "null" ? null : s, s == "null");
            case JsonArray arr:
                {
                    string? main = null;
                    bool nullable = false;
                    foreach (JsonNode? n in arr)
                    {
                        if (n is JsonValue nv && nv.TryGetValue(out string? t))
                        {
                            if (t == "null")
                            {
                                nullable = true;
                            }
                            else
                            {
                                main ??= t;
                            }
                        }
                    }

                    return (main, nullable);
                }

            default:
                return (null, o["enum"] is JsonArray e && e.Any(static x => x is null));
        }
    }

    // ---- the schema the model sees ----

    public static JsonObject Emit(SNode n)
    {
        switch (n)
        {
            case SObject so:
                {
                    var props = new JsonObject();
                    var required = new JsonArray();
                    foreach ((string name, SNode child) in so.Props)
                    {
                        props[name] = Emit(child);
                        required.Add((JsonNode?)JsonValue.Create(name));
                    }

                    var o = new JsonObject { ["type"] = Type("object", so.Nullable), ["properties"] = props, ["required"] = required, ["additionalProperties"] = false };
                    Describe(o, so.Description);
                    return o;
                }

            case SArray sa:
                {
                    var o = new JsonObject { ["type"] = Type("array", sa.Nullable), ["items"] = Emit(sa.Items) };
                    if (sa.MinItems is not null)
                    {
                        o["minItems"] = sa.MinItems.DeepClone();
                    }

                    if (sa.MaxItems is not null)
                    {
                        o["maxItems"] = sa.MaxItems.DeepClone();
                    }

                    Describe(o, sa.Description);
                    return o;
                }

            default:
                {
                    var sl = (SLeaf)n;
                    JsonObject inner = (JsonObject)sl.Schema.DeepClone();
                    inner.Remove("$defs");
                    inner.Remove("default");
                    MakeNullable(inner);
                    var words = new JsonObject
                    {
                        ["type"] = "array",
                        ["items"] = new JsonObject { ["type"] = "string" },
                        ["description"] = "Ids of the listed words the value was read from; every word needed, in order. Empty only when derived.",
                    };
                    var derived = new JsonObject
                    {
                        ["type"] = "boolean",
                        ["description"] = "True only when the value is computed or inferred and not printed on the page.",
                    };
                    var o = new JsonObject
                    {
                        ["type"] = "object",
                        ["properties"] = new JsonObject { ["value"] = inner, ["words"] = words, ["derived"] = derived },
                        ["required"] = new JsonArray("value", "words", "derived"),
                        ["additionalProperties"] = false,
                    };
                    Describe(o, sl.Description);
                    return o;
                }
        }
    }

    private static void Describe(JsonObject o, string? description)
    {
        if (description is not null)
        {
            o["description"] = description;
        }
    }

    private static JsonNode Type(string t, bool nullable) => nullable ? new JsonArray(t, "null") : JsonValue.Create(t)!;

    private static void MakeNullable(JsonObject o)
    {
        switch (o["type"])
        {
            case JsonValue v when v.TryGetValue(out string? s):
                if (s != "null")
                {
                    o["type"] = new JsonArray(s, "null");
                }

                break;
            case JsonArray arr when !arr.Any(static x => x is JsonValue xv && xv.TryGetValue(out string? t) && t == "null"):
                arr.Add((JsonNode?)JsonValue.Create("null"));
                break;
        }

        if (o["enum"] is JsonArray e && !e.Any(static x => x is null))
        {
            e.Add(null);
        }
    }

    // ---- reading the answer ----

    public static void Collect(JsonElement el, SNode n, string path, List<LeafAnswer> leaves)
    {
        switch (n)
        {
            case SObject so when el.ValueKind == JsonValueKind.Object:
                foreach ((string name, SNode child) in so.Props)
                {
                    if (el.TryGetProperty(name, out JsonElement c))
                    {
                        Collect(c, child, path + "/" + Escape(name), leaves);
                    }
                }

                break;
            case SArray sa when el.ValueKind == JsonValueKind.Array:
                int i = 0;
                foreach (JsonElement item in el.EnumerateArray())
                {
                    Collect(item, sa.Items, path + "/" + i.ToString(System.Globalization.CultureInfo.InvariantCulture), leaves);
                    i++;
                }

                break;
            case SLeaf:
                {
                    JsonElement value = el;
                    var ids = new List<string>();
                    bool derived = false;
                    if (el.ValueKind == JsonValueKind.Object && el.TryGetProperty("value", out JsonElement inner))
                    {
                        value = inner;
                        if (el.TryGetProperty("words", out JsonElement w) && w.ValueKind == JsonValueKind.Array)
                        {
                            foreach (JsonElement id in w.EnumerateArray())
                            {
                                if (id.ValueKind == JsonValueKind.String && id.GetString() is { } s)
                                {
                                    ids.Add(s);
                                }
                            }
                        }

                        derived = el.TryGetProperty("derived", out JsonElement dv) && dv.ValueKind == JsonValueKind.True;
                    }

                    string? text = value.ValueKind switch
                    {
                        JsonValueKind.String => value.GetString(),
                        JsonValueKind.Number => value.GetRawText(),
                        JsonValueKind.True => "true",
                        JsonValueKind.False => "false",
                        JsonValueKind.Object or JsonValueKind.Array => value.GetRawText(),
                        _ => null,
                    };
                    if (text is not null)
                    {
                        leaves.Add(new LeafAnswer(path, text, value.ValueKind, ids, derived));
                    }

                    break;
                }
        }
    }

    // ---- the plain JSON ----

    public static void Render(JsonElement el, SNode n, string path, Utf8JsonWriter w, IReadOnlyDictionary<string, LeafAnswer> final, ISet<string> nulled)
    {
        switch (n)
        {
            case SObject so:
                w.WriteStartObject();
                foreach ((string name, SNode child) in so.Props)
                {
                    w.WritePropertyName(name);
                    JsonElement c = default;
                    bool has = el.ValueKind == JsonValueKind.Object && el.TryGetProperty(name, out c);
                    Render(has ? c : default, child, path + "/" + Escape(name), w, final, nulled);
                }

                w.WriteEndObject();
                break;
            case SArray sa:
                w.WriteStartArray();
                if (el.ValueKind == JsonValueKind.Array)
                {
                    int i = 0;
                    foreach (JsonElement item in el.EnumerateArray())
                    {
                        Render(item, sa.Items, path + "/" + i.ToString(System.Globalization.CultureInfo.InvariantCulture), w, final, nulled);
                        i++;
                    }
                }

                w.WriteEndArray();
                break;
            default:
                if (!final.TryGetValue(path, out LeafAnswer? leaf) || nulled.Contains(path))
                {
                    w.WriteNullValue();
                    break;
                }

                switch (leaf.Kind)
                {
                    case JsonValueKind.String:
                        w.WriteStringValue(leaf.Value);
                        break;
                    case JsonValueKind.Number:
                    case JsonValueKind.Object:
                    case JsonValueKind.Array:
                        w.WriteRawValue(leaf.Value);
                        break;
                    case JsonValueKind.True:
                        w.WriteBooleanValue(true);
                        break;
                    case JsonValueKind.False:
                        w.WriteBooleanValue(false);
                        break;
                    default:
                        w.WriteNullValue();
                        break;
                }

                break;
        }
    }

    private static string Escape(string s) => s.Replace("~", "~0", StringComparison.Ordinal).Replace("/", "~1", StringComparison.Ordinal);
}
