using System.Buffers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;
using OneImage;

namespace OneOcr.Llm;

public sealed partial class LlmClient
{
    private const int MaxCorrectionLines = 64;

    /// <summary>
    /// Reads one or more pages and returns the values as <typeparamref name="T"/> with their grounding. The schema is derived from <paramref name="typeInfo"/>
    /// with the BCL's <c>JsonSchemaExporter</c> (no reflection); <see cref="LlmReadOptions.JsonSchema"/> is ignored and <see cref="LlmReadOptions.Outputs"/> gets
    /// <see cref="LlmOutputs.Json"/> added. Everything else in <paramref name="options"/> works as for <see cref="ReadAsync(IReadOnlyList{LlmPageInput}, LlmReadOptions, CancellationToken)"/>.
    /// </summary>
    /// <typeparam name="T">The answer type.</typeparam>
    /// <param name="pages">The pages, between one and <see cref="OcrLimits.MaxPagesPerCall"/>.</param>
    /// <param name="typeInfo">Type information from a <c>JsonSerializerContext</c>.</param>
    /// <param name="options">What to read and how.</param>
    /// <param name="cancellationToken">Cancels the call.</param>
    /// <returns>The value (null only when the answer is JSON null) and the read.</returns>
    /// <exception cref="OcrProviderException">As for the schema route; <see cref="LlmFailure.Malformed"/> when the values do not fit <typeparamref name="T"/>.</exception>
    public async Task<LlmReadResult<T>> ReadAsync<T>(
        IReadOnlyList<LlmPageInput> pages, JsonTypeInfo<T> typeInfo, LlmReadOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(typeInfo);
        ArgumentNullException.ThrowIfNull(options);
        var typed = new LlmReadOptions
        {
            Outputs = options.Outputs | LlmOutputs.Json,
            JsonSchema = SchemaTools.Export(typeInfo),
            SchemaName = options.SchemaName,
            Instruction = options.Instruction,
            Routing = options.Routing,
            ConfidenceThreshold = options.ConfidenceThreshold,
            LocalExtractor = options.LocalExtractor,
            RoleCheck = options.RoleCheck,
            UngroundedPolicy = options.UngroundedPolicy,
            Reask = options.Reask,
            NumberStyle = options.NumberStyle,
            DateOrder = options.DateOrder,
            KeepLlmOnlyWords = options.KeepLlmOnlyWords,
        };
        LlmReadResult read = await ReadAsync(pages, typed, cancellationToken).ConfigureAwait(false);
        try
        {
            return new LlmReadResult<T>(read.GetValue(typeInfo), read);
        }
        catch (JsonException e)
        {
            throw new OcrProviderException(_provider.Name, LlmFailure.Malformed, _provider.Scrub("the values do not fit the requested type: " + e.Message));
        }
    }

    /// <summary>Reads one page; see <see cref="ReadAsync(IReadOnlyList{LlmPageInput}, LlmReadOptions, CancellationToken)"/>.</summary>
    /// <param name="image">The page image.</param>
    /// <param name="page">The page model read from it.</param>
    /// <param name="options">What to produce and how.</param>
    /// <param name="cancellationToken">Cancels the read, including the re-ask.</param>
    /// <returns>The requested outputs.</returns>
    /// <exception cref="OcrProviderException">Offline, unreachable, an unusable answer, or ungrounded values under <see cref="UngroundedPolicy.Throw"/>.</exception>
    /// <exception cref="OcrBudgetException">A cost bound would be passed by the first call. (When only the re-ask would pass it, the re-ask is skipped and a diagnostic says so.)</exception>
    /// <exception cref="OcrLimitException">An <see cref="OcrLimits"/> bound was exceeded.</exception>
    /// <exception cref="ArgumentException">The options are inconsistent or the schema cannot be grounded.</exception>
    public Task<LlmReadResult> ReadAsync(Image image, OcrPage page, LlmReadOptions options, CancellationToken cancellationToken = default) =>
        ReadAsync([new LlmPageInput(image, page)], options, cancellationToken);

    /// <summary>
    /// Produces every requested output from one model call: <see cref="LlmOutputs.Text"/> (the model's transcription merged word by
    /// word with the local reading), <see cref="LlmOutputs.Markdown"/> (the model's Markdown with disagreeing spans flagged),
    /// <see cref="LlmOutputs.Json"/> (values for a schema, each grounded against the page) and <see cref="LlmOutputs.PageModel"/>
    /// (the merged page model). Ungrounded values are re-asked once in a second call. With <see cref="LlmRouting.LocalOnly"/>, or
    /// when <see cref="LlmRouting.LocalFirst"/> finds the local result confident enough, no call is made and nothing is sent.
    /// </summary>
    /// <param name="pages">The pages, one or more; each page's model comes from the local reader.</param>
    /// <param name="options">What to produce and how.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>The requested outputs, the grounding of every value, what was spent and what was left local.</returns>
    /// <exception cref="OcrProviderException">See the single-page overload.</exception>
    /// <exception cref="OcrBudgetException">See the single-page overload.</exception>
    /// <exception cref="OcrLimitException">See the single-page overload.</exception>
    /// <exception cref="ArgumentException">See the single-page overload.</exception>
    public async Task<LlmReadResult> ReadAsync(IReadOnlyList<LlmPageInput> pages, LlmReadOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(pages);
        ArgumentNullException.ThrowIfNull(options);
        if (pages.Count == 0)
        {
            throw new ArgumentException("At least one page is required.", nameof(pages));
        }

        LlmOutputs outputs = options.Outputs;
        if (outputs == LlmOutputs.None || (outputs & ~(LlmOutputs.Text | LlmOutputs.Markdown | LlmOutputs.Json | LlmOutputs.PageModel)) != 0)
        {
            throw new ArgumentException("Outputs must be a non-empty combination of Text, Markdown, Json and PageModel.", nameof(options));
        }

        if (!(options.ConfidenceThreshold is >= 0f and <= 1f))
        {
            throw new ArgumentOutOfRangeException(nameof(options), "ConfidenceThreshold must be between 0 and 1.");
        }

        bool wantText = (outputs & (LlmOutputs.Text | LlmOutputs.PageModel)) != 0;
        bool wantMarkdown = (outputs & LlmOutputs.Markdown) != 0;
        bool wantJson = (outputs & LlmOutputs.Json) != 0;
        SNode? tree = null;
        string? userSchema = null;
        if (wantJson)
        {
            if (string.IsNullOrWhiteSpace(options.JsonSchema))
            {
                throw new ArgumentException("Outputs has Json, so JsonSchema is required.", nameof(options));
            }

            JsonObject root = SchemaTools.Parse(options.JsonSchema, _options.Limits);
            userSchema = root.ToJsonString();
            tree = GroundedSchema.Build(root);
        }

        if (options.Routing == LlmRouting.LocalOnly && wantJson && options.LocalExtractor is null)
        {
            throw new ArgumentException("LocalOnly with Json needs a LocalExtractor.", nameof(options));
        }

        var diagnostics = new List<string>();
        LocalFieldResult? local = null;
        if (wantJson && options.LocalExtractor is { } extractor && (options.Routing != LlmRouting.LlmOnly || options.RoleCheck))
        {
            local = extractor.TryExtract([.. pages.Select(static p => p.Page)], userSchema!);
            if (options.Routing == LlmRouting.LocalOnly && local is null)
            {
                throw new ArgumentException("The local extractor has nothing for this schema, and LocalOnly never calls the model.", nameof(options));
            }
        }

        // LocalFirst: an extractor that lists fields is routed per field; one that lists none keeps the whole-document rule.
        using JsonDocument? localDoc = TryParseLocal(local, diagnostics);
        LocalSplit? split = null;
        bool wholeLocal = false;
        if (options.Routing == LlmRouting.LocalFirst && local is not null)
        {
            if (local.Fields.Count == 0)
            {
                wholeLocal = local.MinConfidence >= options.ConfidenceThreshold;
            }
            else if (localDoc is not null)
            {
                split = new LocalSplit(tree!, local, localDoc.RootElement, options.ConfidenceThreshold);
                wholeLocal = split.AllLocal;
            }
        }

        bool jsonFromModel = wantJson && options.Routing != LlmRouting.LocalOnly && !wholeLocal;
        SNode? askedTree = split is { Paths.Count: > 0 } ? split.Reduced : tree;
        LocalSplit? used = split is { Paths.Count: > 0 } ? split : null;
        if (jsonFromModel && used is not null)
        {
            diagnostics.Add($"{used.Paths.Count} value(s) were kept from the local extractor; the model was asked for the rest.");
        }
        bool mdFromModel = wantMarkdown && options.Routing != LlmRouting.LocalOnly;
        if (wantMarkdown && !mdFromModel)
        {
            diagnostics.Add("Markdown needs the model; no local Markdown is available, so it was not produced.");
        }

        // Which pages the model sees.
        var sent = new List<int>();
        bool textFromModel = false;
        if (options.Routing == LlmRouting.LlmOnly)
        {
            sent.AddRange(Enumerable.Range(0, pages.Count));
            textFromModel = wantText;
        }
        else if (options.Routing == LlmRouting.LocalFirst)
        {
            if (jsonFromModel || mdFromModel)
            {
                sent.AddRange(Enumerable.Range(0, pages.Count));
                textFromModel = wantText;
            }
            else if (wantText)
            {
                for (int i = 0; i < pages.Count; i++)
                {
                    if (pages[i].Page.Words.Any(w => w.Confidence < options.ConfidenceThreshold))
                    {
                        sent.Add(i);
                    }
                }

                textFromModel = sent.Count > 0;
                if (sent.Count < pages.Count)
                {
                    diagnostics.Add($"{pages.Count - sent.Count} of {pages.Count} page(s) had only confident words and stayed local.");
                }
            }
        }

        if (sent.Count == 0)
        {
            return LocalResult(pages, options, local, wantText, diagnostics);
        }

        var sentInputs = sent.Select(i => pages[i]).ToList();
        bool pagesAsked = textFromModel, mdAsked = mdFromModel, dataAsked = jsonFromModel;
        JsonObject wrapper = Wrapper(pagesAsked, mdAsked, dataAsked ? askedTree : null);
        string instruction = Instruction(options.Instruction, pagesAsked, mdAsked, dataAsked);

        LlmExtraction first = await ExtractCoreAsync(sentInputs, instruction, wrapper.ToJsonString(), options.SchemaName, null, cancellationToken).ConfigureAwait(false);
        int calls = 1;
        using JsonDocument doc = ParseAnswer(first.Json);
        JsonElement answer = doc.RootElement;
        decimal? cost = first.CostUsd;
        long inTok = first.Usage.InputTokens, outTok = first.Usage.OutputTokens;
        bool reported = first.Usage.Reported;

        // ---- text and markdown ----
        var disagreements = new List<TextDisagreement>();
        var markdownFlags = new List<TextDisagreement>();
        string[] pageTexts = [.. pages.Select(static p => p.Page.Text)];
        var mergedEntries = new Dictionary<int, List<MergeEntry>>();
        string? llmText = null, markdown = null, text = null;
        if (pagesAsked)
        {
            string[] raw = StringArray(answer, "pages", sent.Count, diagnostics);
            llmText = string.Join("\n\n", raw);
            for (int k = 0; k < sent.Count; k++)
            {
                int pi = sent[k];
                List<Tok> toks = TextMerge.Tokenize(raw[k], false);
                List<MergeEntry>? entries = TextMerge.MergeText(pi, pages[pi].Page.Words, toks, options, disagreements);
                if (entries is null)
                {
                    diagnostics.Add($"Page {pi}: too many words to align with the model's text; the local text was kept.");
                }
                else
                {
                    mergedEntries[pi] = entries;
                }
            }

            var sb = new StringBuilder();
            for (int pi = 0; pi < pages.Count; pi++)
            {
                if (pi > 0)
                {
                    sb.Append("\n\n");
                }

                pageTexts[pi] = mergedEntries.TryGetValue(pi, out List<MergeEntry>? me) ? TextMerge.Assemble(me, disagreements, sb.Length) : pageTexts[pi];
                sb.Append(pageTexts[pi]);
            }

            text = sb.ToString();
        }
        else if ((outputs & LlmOutputs.Text) != 0)
        {
            text = string.Join("\n\n", pageTexts);
        }

        if (mdAsked)
        {
            string[] raw = StringArray(answer, "markdown", sent.Count, diagnostics);
            var sb = new StringBuilder();
            for (int k = 0; k < sent.Count; k++)
            {
                if (k > 0)
                {
                    sb.Append("\n\n");
                }

                List<Tok> toks = TextMerge.Tokenize(raw[k], true);
                if (!TextMerge.FlagMarkdown(sent[k], pages[sent[k]].Page.Words, toks, options, sb.Length, markdownFlags))
                {
                    diagnostics.Add($"Page {sent[k]}: too many words to align the Markdown with the local reading; it is not flagged.");
                }

                sb.Append(raw[k]);
            }

            markdown = sb.ToString();
        }

        OcrPage[] resultPages = [.. pages.Select(static p => p.Page)];
        if ((outputs & LlmOutputs.PageModel) != 0)
        {
            foreach ((int pi, List<MergeEntry> entries) in mergedEntries)
            {
                int n = disagreements.Count(d => d.PageIndex == pi);
                resultPages[pi] = TextMerge.BuildPage(pages[pi].Page, entries, $"Words merged with the model's reading: {n} disagreement(s).");
            }
        }

        // ---- values ----
        GroundedField Check(GroundedField g) =>
            options.RoleCheck && local is not null && options.Routing != LlmRouting.LocalOnly ? RoleCheck.Apply(g, local, options.ConfidenceThreshold) : g;

        string? json = null;
        List<GroundedField> fields = [];
        if (jsonFromModel)
        {
            var ctx = new GroundingContext(first.Words, sentInputs, options.NumberStyle, options.DateOrder);
            JsonElement data = answer.ValueKind == JsonValueKind.Object && answer.TryGetProperty("data", out JsonElement d) ? d : default;
            var leaves = new List<LeafAnswer>();
            GroundedSchema.Collect(data, askedTree!, "", leaves);
            var final = new Dictionary<string, LeafAnswer>(StringComparer.Ordinal);
            var grounded = new Dictionary<string, GroundedField>(StringComparer.Ordinal);
            var order = new List<string>();
            foreach (LeafAnswer leaf in leaves)
            {
                final[leaf.Path] = leaf;
                grounded[leaf.Path] = Check(Grounder.Ground(ctx, leaf, false));
                order.Add(leaf.Path);
            }

            var withdrawn = new HashSet<string>(StringComparer.Ordinal);
            List<string> ungrounded = [.. order.Where(p => grounded[p].State is GroundingState.Ungrounded or GroundingState.RoleConflict)];
            if (ungrounded.Count > 0 && options.Reask)
            {
                try
                {
                    var lines = new List<string>();
                    foreach (string path in ungrounded.Take(MaxCorrectionLines))
                    {
                        lines.Add(CorrectionLine(ctx, final[path], grounded[path], local is null ? null : RoleCheck.LocalValue(local, path)));
                    }

                    if (ungrounded.Count > MaxCorrectionLines)
                    {
                        diagnostics.Add($"Only the first {MaxCorrectionLines} of {ungrounded.Count} ungrounded values were re-asked.");
                    }

                    JsonObject dataOnly = Wrapper(false, false, askedTree);
                    LlmExtraction second = await ExtractCoreAsync(
                        sentInputs, Instruction(options.Instruction, false, false, true), dataOnly.ToJsonString(), options.SchemaName, lines, cancellationToken).ConfigureAwait(false);
                    calls++;
                    cost = cost is { } c1 && second.CostUsd is { } c2 ? c1 + c2 : null;
                    inTok += second.Usage.InputTokens;
                    outTok += second.Usage.OutputTokens;
                    reported &= second.Usage.Reported;
                    using JsonDocument doc2 = ParseAnswer(second.Json);
                    JsonElement data2 = doc2.RootElement.ValueKind == JsonValueKind.Object && doc2.RootElement.TryGetProperty("data", out JsonElement d2) ? d2 : default;
                    var again = new List<LeafAnswer>();
                    GroundedSchema.Collect(data2, askedTree!, "", again);
                    var byPath = again.ToDictionary(static l => l.Path, StringComparer.Ordinal);
                    foreach (string path in ungrounded.Take(MaxCorrectionLines))
                    {
                        if (byPath.TryGetValue(path, out LeafAnswer? replacement))
                        {
                            final[path] = replacement;
                            grounded[path] = Check(Grounder.Ground(ctx, replacement, true));
                        }
                        else
                        {
                            withdrawn.Add(path);
                        }
                    }
                }
                catch (Exception e) when (e is OcrBudgetException or OcrProviderException)
                {
                    diagnostics.Add("The re-ask was skipped: " + _provider.Scrub(e.Message));
                }
            }

            foreach (string path in withdrawn)
            {
                diagnostics.Add($"{path}: the model withdrew the value when asked again; it is null.");
            }

            List<string> stillUngrounded = [.. order.Where(p => !withdrawn.Contains(p) && grounded[p].State is GroundingState.Ungrounded or GroundingState.RoleConflict)];
            if (stillUngrounded.Count > 0 && options.UngroundedPolicy == UngroundedPolicy.Throw)
            {
                throw new OcrProviderException(
                    _provider.Name, LlmFailure.Ungrounded, _provider.Scrub($"{stillUngrounded.Count} value(s) are not on the page or conflict with the local reading: {string.Join(", ", stillUngrounded.Take(8))}"));
            }

            var nulled = new HashSet<string>(withdrawn, StringComparer.Ordinal);
            if (options.UngroundedPolicy == UngroundedPolicy.Null)
            {
                nulled.UnionWith(stillUngrounded);
            }

            var buffer = new ArrayBufferWriter<byte>(1024);
            if (used is null)
            {
                using var w = new Utf8JsonWriter(buffer);
                GroundedSchema.Render(data, tree!, "", w, final, nulled);
            }
            else
            {
                // Local values and model values in one answer; each array is wholly one side's.
                foreach (LeafAnswer l in used.Leaves)
                {
                    final[l.Path] = l;
                }

                var shape = new ArrayBufferWriter<byte>(1024);
                using (var sw = new Utf8JsonWriter(shape))
                {
                    used.WriteShape(tree!, "", data, used.LocalRoot, sw);
                }

                using JsonDocument shapeDoc = JsonDocument.Parse(shape.WrittenMemory);
                using var w = new Utf8JsonWriter(buffer);
                GroundedSchema.Render(shapeDoc.RootElement, tree!, "", w, final, nulled);
            }

            json = Encoding.UTF8.GetString(buffer.WrittenSpan);
            fields = [.. (used?.Fields ?? []).Concat(order.Where(p => !withdrawn.Contains(p)).Select(p => grounded[p]))];
        }
        else if (wantJson && local is not null)
        {
            json = local.Json;
            fields = [.. local.Fields];
            diagnostics.Add("Values came from the local extractor; the model was not asked for them.");
        }

        return new LlmReadResult
        {
            Outputs = outputs,
            Text = (outputs & LlmOutputs.Text) != 0 ? text : null,
            LlmText = llmText,
            Markdown = markdown,
            MarkdownFlags = markdownFlags,
            Json = json,
            Fields = fields,
            Pages = resultPages,
            Disagreements = disagreements,
            Calls = calls,
            Usage = new LlmUsage(inTok, outTok, reported),
            CostUsd = cost,
            Provider = _provider.Name,
            Model = _provider.Model,
            Diagnostics = diagnostics,
        };
    }

    private LlmReadResult LocalResult(IReadOnlyList<LlmPageInput> pages, LlmReadOptions options, LocalFieldResult? local, bool wantText, List<string> diagnostics)
    {
        LlmOutputs outputs = options.Outputs;
        bool json = (outputs & LlmOutputs.Json) != 0 && local is not null;
        if (json)
        {
            diagnostics.Add("Values came from the local extractor; the model was not called.");
        }

        return new LlmReadResult
        {
            Outputs = outputs,
            Text = (outputs & LlmOutputs.Text) != 0 ? string.Join("\n\n", pages.Select(static p => p.Page.Text)) : null,
            Json = json ? local!.Json : null,
            Fields = json ? [.. local!.Fields] : [],
            Pages = [.. pages.Select(static p => p.Page)],
            Calls = 0,
            Usage = new LlmUsage(0, 0, true),
            CostUsd = 0m,
            Provider = _provider.Name,
            Model = _provider.Model,
            Diagnostics = diagnostics,
        };
    }

    private JsonDocument ParseAnswer(string json)
    {
        try
        {
            return JsonDocument.Parse(json, new JsonDocumentOptions { MaxDepth = _options.Limits.MaxLlmJsonDepth + 8 });
        }
        catch (JsonException e)
        {
            throw new OcrProviderException(_provider.Name, LlmFailure.Malformed, _provider.Scrub("the answer is not valid JSON: " + e.Message));
        }
    }

    private static string[] StringArray(JsonElement answer, string name, int count, List<string> diagnostics)
    {
        var result = new string[count];
        Array.Fill(result, "");
        if (answer.ValueKind != JsonValueKind.Object || !answer.TryGetProperty(name, out JsonElement arr) || arr.ValueKind != JsonValueKind.Array)
        {
            diagnostics.Add($"The answer has no '{name}' array; those pages are empty.");
            return result;
        }

        int i = 0;
        foreach (JsonElement e in arr.EnumerateArray())
        {
            if (i < count && e.ValueKind == JsonValueKind.String)
            {
                result[i] = e.GetString() ?? "";
            }

            i++;
        }

        if (i != count)
        {
            diagnostics.Add($"The answer's '{name}' has {i} item(s) for {count} page(s).");
        }

        return result;
    }

    private static JsonObject Wrapper(bool pages, bool markdown, SNode? data)
    {
        var props = new JsonObject();
        var required = new JsonArray();
        JsonObject PerPage(string what) => new()
        {
            ["type"] = "array",
            ["items"] = new JsonObject { ["type"] = "string" },
            ["description"] = what,
        };

        if (pages)
        {
            props["pages"] = PerPage("One string per page image, in the order given: the exact text of the page.");
            required.Add((JsonNode?)JsonValue.Create("pages"));
        }

        if (markdown)
        {
            props["markdown"] = PerPage("One string per page image, in the order given: the page as Markdown.");
            required.Add((JsonNode?)JsonValue.Create("markdown"));
        }

        if (data is not null)
        {
            props["data"] = GroundedSchema.Emit(data);
            required.Add((JsonNode?)JsonValue.Create("data"));
        }

        return new JsonObject { ["type"] = "object", ["properties"] = props, ["required"] = required, ["additionalProperties"] = false };
    }

    private static string Instruction(string? caller, bool pages, bool markdown, bool data)
    {
        var sb = new StringBuilder();
        sb.Append(string.IsNullOrWhiteSpace(caller) ? "Read the document." : caller.Trim()).Append("\n\nAnswer with one JSON object with these members:\n");
        if (pages)
        {
            sb.Append("- \"pages\": an array with one string per page image, in the order given. Each string is the exact text of that page in reading order, ")
                .Append("copied character for character as printed: lines separated by a newline, words by single spaces, nothing corrected, nothing omitted, ")
                .Append("nothing added, no commentary.\n");
        }

        if (markdown)
        {
            sb.Append("- \"markdown\": an array with one string per page image, in the order given. Each string is the page as Markdown with the same words as printed: ")
                .Append("'#' headings, '-' lists, tables as rows with a space on both sides of every '|'. No commentary.\n");
        }

        if (data)
        {
            sb.Append("- \"data\": the requested values. For every value give \"value\" (null when it is not printed on the page), \"words\" (the ids from the word list of ")
                .Append("every word the value was read from) and \"derived\" (true only if the value is computed or inferred rather than printed, then cite the words used). ")
                .Append("Copy strings as printed. Number-typed values are plain digits with '.' as the decimal mark. A string with format date is written yyyy-MM-dd.\n");
        }

        return sb.ToString();
    }

    private static JsonDocument? TryParseLocal(LocalFieldResult? local, List<string> diagnostics)
    {
        if (local is null)
        {
            return null;
        }

        try
        {
            return JsonDocument.Parse(local.Json, new JsonDocumentOptions { MaxDepth = 64 });
        }
        catch (JsonException)
        {
            diagnostics.Add("The local extractor's JSON could not be read; its values are not used per field.");
            return null;
        }
    }

    private static string CorrectionLine(GroundingContext ctx, LeafAnswer leaf, GroundedField field, string? localValue)
    {
        var buffer = new ArrayBufferWriter<byte>(256);
        using (var w = new Utf8JsonWriter(buffer, new JsonWriterOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }))
        {
            w.WriteStartArray();
            w.WriteStringValue(leaf.Path);
            w.WriteStringValue(leaf.Value);
            w.WriteStartArray();
            foreach (string id in leaf.WordIds.Take(16))
            {
                if (ctx.TryIndex(id, out int i))
                {
                    w.WriteStringValue(ctx.Words[i].Text);
                }
            }

            w.WriteEndArray();
            string? near = field.State == GroundingState.RoleConflict && localValue is not null
                ? localValue
                : Grounder.Nearest(ctx, new ValueForm(leaf.Value, leaf.Kind, ctx));
            if (near is null)
            {
                w.WriteNullValue();
            }
            else
            {
                w.WriteStringValue(near);
            }

            w.WriteEndArray();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }
}
