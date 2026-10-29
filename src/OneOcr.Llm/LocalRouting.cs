using System.Text.Json;

namespace OneOcr.Llm;

/// <summary>
/// Which schema values a local extractor answers confidently under <see cref="LlmRouting.LocalFirst"/>, and the schema left for the model.
/// A scalar is local when the extractor lists it at or above the threshold with a value. An array is local only as a whole: the extractor
/// has at least one element and every value of every element is local; otherwise the whole array goes to the model, because the model
/// and the local reader may disagree about how many elements there are.
/// </summary>
internal sealed class LocalSplit
{
    private readonly float _threshold;
    private readonly Dictionary<string, GroundedField> _fields = new(StringComparer.Ordinal);
    private readonly Dictionary<string, LeafAnswer> _leaves = new(StringComparer.Ordinal);

    public LocalSplit(SNode tree, LocalFieldResult local, JsonElement localRoot, float threshold)
    {
        _threshold = threshold;
        foreach (GroundedField f in local.Fields)
        {
            _fields.TryAdd(f.Path, f);
        }

        var all = new List<LeafAnswer>();
        GroundedSchema.Collect(localRoot, tree, "", all);
        foreach (LeafAnswer l in all)
        {
            _leaves[l.Path] = l;
        }

        var target = new Taken();
        Classify(tree, "", localRoot, target);
        Paths = target.Paths;
        Leaves = target.Leaves;
        Fields = target.Fields;
        LocalArrays = target.Arrays;
        Reduced = Reduce(tree, "", true);
        AllLocal = Reduced is not SObject { Props.Count: > 0 };
        LocalRoot = localRoot;
    }

    private sealed class Taken
    {
        public HashSet<string> Paths { get; } = new(StringComparer.Ordinal);

        public List<LeafAnswer> Leaves { get; } = [];

        public List<GroundedField> Fields { get; } = [];

        public HashSet<string> Arrays { get; } = new(StringComparer.Ordinal);
    }

    public HashSet<string> Paths { get; }

    public List<LeafAnswer> Leaves { get; }

    public List<GroundedField> Fields { get; }

    public HashSet<string> LocalArrays { get; }

    public SNode? Reduced { get; }

    public bool AllLocal { get; }

    public JsonElement LocalRoot { get; }

    // Visits every node (no short-circuit) and says whether all leaves below it are local.
    private bool Classify(SNode n, string path, JsonElement loc, Taken into)
    {
        switch (n)
        {
            case SObject so:
                bool all = true;
                foreach ((string name, SNode child) in so.Props)
                {
                    JsonElement c = default;
                    if (loc.ValueKind == JsonValueKind.Object)
                    {
                        loc.TryGetProperty(name, out c);
                    }

                    all &= Classify(child, path + "/" + GroundedSchema.Escape(name), c, into);
                }

                return all;
            case SArray sa:
                {
                    var tmp = new Taken();
                    bool ok = loc.ValueKind == JsonValueKind.Array && loc.GetArrayLength() > 0;
                    if (ok)
                    {
                        int i = 0;
                        foreach (JsonElement item in loc.EnumerateArray())
                        {
                            ok &= Classify(sa.Items, path + "/" + i.ToString(System.Globalization.CultureInfo.InvariantCulture), item, tmp);
                            i++;
                        }
                    }

                    if (ok)
                    {
                        into.Paths.UnionWith(tmp.Paths);
                        into.Leaves.AddRange(tmp.Leaves);
                        into.Fields.AddRange(tmp.Fields);
                        into.Arrays.UnionWith(tmp.Arrays);
                        into.Arrays.Add(path);
                    }

                    return ok;
                }

            default:
                if (_fields.TryGetValue(path, out GroundedField? f) && f.Confidence >= _threshold
                    && f.State is GroundingState.Grounded or GroundingState.GroundedNormalized
                    && _leaves.TryGetValue(path, out LeafAnswer? leaf))
                {
                    into.Paths.Add(path);
                    into.Leaves.Add(leaf);
                    into.Fields.Add(f.Source == OcrSource.Llm ? f with { Source = OcrSource.Ocr } : f);
                    return true;
                }

                return false;
        }
    }

    private SNode? Reduce(SNode n, string path, bool root)
    {
        switch (n)
        {
            case SObject so:
                var r = new SObject { Nullable = so.Nullable, Description = so.Description };
                foreach ((string name, SNode child) in so.Props)
                {
                    SNode? c = Reduce(child, path + "/" + GroundedSchema.Escape(name), false);
                    if (c is not null)
                    {
                        r.Props.Add((name, c));
                    }
                }

                return r.Props.Count > 0 || root ? r : null;
            case SArray:
                return LocalArrays.Contains(path) ? null : n;
            default:
                return Paths.Contains(path) ? null : n;
        }
    }

    /// <summary>
    /// The answer's shape for rendering the full schema: objects follow the schema, a local array is the extractor's, a model array the model's.
    /// Scalars are placeholders; their values come from the final-value map.
    /// </summary>
    public void WriteShape(SNode n, string path, JsonElement llm, JsonElement loc, Utf8JsonWriter w)
    {
        switch (n)
        {
            case SObject so:
                w.WriteStartObject();
                foreach ((string name, SNode child) in so.Props)
                {
                    JsonElement l = default, c = default;
                    if (llm.ValueKind == JsonValueKind.Object)
                    {
                        llm.TryGetProperty(name, out l);
                    }

                    if (loc.ValueKind == JsonValueKind.Object)
                    {
                        loc.TryGetProperty(name, out c);
                    }

                    w.WritePropertyName(name);
                    WriteShape(child, path + "/" + GroundedSchema.Escape(name), l, c, w);
                }

                w.WriteEndObject();
                break;
            case SArray:
                JsonElement src = LocalArrays.Contains(path) ? loc : llm;
                if (src.ValueKind == JsonValueKind.Array)
                {
                    src.WriteTo(w);
                }
                else
                {
                    w.WriteStartArray();
                    w.WriteEndArray();
                }

                break;
            default:
                w.WriteNullValue();
                break;
        }
    }
}

/// <summary>The role check: a model value whose cited words the local model reads in another role.</summary>
internal static class RoleCheck
{
    /// <summary>
    /// Returns the field marked <see cref="GroundingState.RoleConflict"/>, or the field unchanged. A conflict is one of:
    /// a cited word is part of a confident local field of another path; or the local model confidently reads this very path from other
    /// words and its value differs. Confident means at or above <paramref name="threshold"/>. Values without cited words (derived,
    /// ungrounded) are not checked, and neither are values of a property the local model has no class for (<see cref="LocalFieldResult.MappedPaths"/>):
    /// the local model says nothing about what such a property means, so its words belonging to some other class is no contradiction.
    /// </summary>
    public static GroundedField Apply(GroundedField g, LocalFieldResult local, float threshold)
    {
        if (g.State is not (GroundingState.Grounded or GroundingState.GroundedNormalized) || g.Words.Count == 0)
        {
            return g;
        }

        if (local.MappedPaths is { } mapped && !mapped.Contains(StripIndices(g.Path)))
        {
            return g;
        }

        var cited = new HashSet<(int, int)>();
        foreach (LlmWordRef w in g.Words)
        {
            cited.Add((w.PageIndex, w.WordIndex));
        }

        foreach (GroundedField lf in Candidates(local))
        {
            if (lf.Confidence < threshold || lf.State is not (GroundingState.Grounded or GroundingState.GroundedNormalized) || lf.Words.Count == 0)
            {
                continue;
            }

            bool overlap = lf.Words.Any(w => cited.Contains((w.PageIndex, w.WordIndex)));
            if (lf.Path != g.Path && overlap)
            {
                return Conflict(g, $"the local model reads the cited words as '{lf.Path}' ({lf.Value})");
            }

            if (lf.Path == g.Path && !overlap
                && !string.Equals(lf.NormalizedValue ?? lf.Value, g.NormalizedValue ?? g.Value, StringComparison.OrdinalIgnoreCase))
            {
                return Conflict(g, $"the local model reads '{lf.Value}' as this field");
            }
        }

        return g;
    }

    /// <summary>A schema path without its array indices: <c>/lines/3/qty</c> becomes <c>/lines/qty</c>.</summary>
    public static string StripIndices(string path)
    {
        var sb = new System.Text.StringBuilder(path.Length);
        foreach (string seg in path.Split('/', StringSplitOptions.RemoveEmptyEntries))
        {
            if (!seg.All(char.IsAsciiDigit))
            {
                sb.Append('/').Append(seg);
            }
        }

        return sb.ToString();
    }

    /// <summary>The local value for a path, for the re-ask's hint; null when the local model has none.</summary>
    public static string? LocalValue(LocalFieldResult local, string path) => local.Fields.FirstOrDefault(f => f.Path == path)?.Value;

    private static IEnumerable<GroundedField> Candidates(LocalFieldResult local) => local.OtherFields is { } o ? local.Fields.Concat(o) : local.Fields;

    private static GroundedField Conflict(GroundedField g, string why) =>
        g with { State = GroundingState.RoleConflict, Confidence = 0f, Note = (g.Note is null ? "" : g.Note + "; ") + why };
}
