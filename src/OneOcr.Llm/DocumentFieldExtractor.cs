using System.Buffers;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using OneOcr.Documents;

namespace OneOcr.Llm;

/// <summary>
/// Plugs a <see cref="DocumentExtractor"/> (the shipped invoice and receipt model, or a custom model from <c>ExtractionTrainer</c>) into
/// <see cref="LlmReadOptions.LocalExtractor"/>. No glue is needed: the schema's properties are matched to the model's classes by name.
/// </summary>
/// <remarks>
/// A property matches a class when, in this order: its name equals a class name (ignoring case and punctuation, so a custom model's own
/// names work); its description names the class (<c>BT-112</c>) as a whole token; or its name is a common word for an EN 16931 business term or
/// a receipt field (<c>total</c>, <c>invoiceNumber</c>, <c>vendor</c>, <c>iban</c> …; where an invoice and a receipt class share a word the one the model found is
/// used). Nested objects are followed. An array of objects reads the model's line items, or its tax rows when the array's name
/// contains <c>tax</c> or <c>vat</c>; element <c>i</c> is the model's <c>i</c>-th group. A property with no class, or whose class the model did not find on the page
/// (or, for a number-typed property, found but could not read as a number), is missing: <see cref="LlmRouting.LocalFirst"/> sends it to the model. The
/// confidence of a field is the extraction model's mean tag probability, which is not a calibrated probability of being right; a threshold that suits it may
/// differ from the one for OCR words. Several pages are read one by one; for a header value the most confident page wins, and items come from the page with the most.
/// </remarks>
/// <param name="extractor">The extraction model.</param>
public sealed class DocumentFieldExtractor(DocumentExtractor extractor) : ILocalFieldExtractor
{
    private const int MaxElements = 4096;

    private static readonly Dictionary<string, string[]> Aliases = BuildAliases();

    private readonly DocumentExtractor _extractor = extractor ?? throw new ArgumentNullException(nameof(extractor));

    /// <inheritdoc/>
    /// <exception cref="ArgumentException">The schema cannot be read.</exception>
    public LocalFieldResult? TryExtract(IReadOnlyList<OcrPage> pages, string jsonSchema)
    {
        ArgumentNullException.ThrowIfNull(pages);
        ArgumentNullException.ThrowIfNull(jsonSchema);
        SNode tree = GroundedSchema.Build(SchemaTools.Parse(jsonSchema, _extractor.Limits));
        if (tree is not SObject root || pages.Count == 0)
        {
            return null;
        }

        var extractions = new DocumentExtraction[pages.Count];
        for (int i = 0; i < pages.Count; i++)
        {
            extractions[i] = _extractor.Extract(pages[i]);
        }

        var run = new Run(pages, extractions, _extractor.Classes);
        var buffer = new ArrayBufferWriter<byte>(1024);
        using (var w = new Utf8JsonWriter(buffer))
        {
            run.WriteObject(root, "", w, null);
        }

        float min = run.Fields.Count == 0 ? 0f : run.Fields.Min(static f => f.Confidence);
        var others = run.Others();
        var mapped = new HashSet<string>(StringComparer.Ordinal);
        run.CollectMapped(root, "", mapped);
        return new LocalFieldResult(System.Text.Encoding.UTF8.GetString(buffer.WrittenSpan), run.Fields, min, others, mapped);
    }

    private sealed class Run
    {
        private readonly IReadOnlyList<OcrPage> _pages;
        private readonly DocumentExtraction[] _ex;
        private readonly HashSet<string> _classes;
        private readonly Dictionary<string, string> _byFolded = new(StringComparer.Ordinal);
        private readonly HashSet<(int, int)> _mappedWords = [];
        private readonly List<(int Page, ExtractedField F)> _all = [];

        public Run(IReadOnlyList<OcrPage> pages, DocumentExtraction[] ex, IReadOnlyList<string> classes)
        {
            _pages = pages;
            _ex = ex;
            _classes = new HashSet<string>(classes, StringComparer.Ordinal);
            foreach (string c in classes)
            {
                _byFolded.TryAdd(Fold(c), c);
            }

            for (int p = 0; p < ex.Length; p++)
            {
                foreach (ExtractedField f in ex[p].Fields)
                {
                    _all.Add((p, f));
                }

                foreach (ExtractedGroup g in ex[p].LineItems.Concat(ex[p].TaxRows))
                {
                    foreach (ExtractedField f in g.Fields)
                    {
                        _all.Add((p, f));
                    }
                }
            }
        }

        public List<GroundedField> Fields { get; } = [];

        // The schema properties some class of the model answers to (found on this page or not), with array indices left out.
        public void CollectMapped(SNode n, string path, HashSet<string> into, string leafName = "")
        {
            switch (n)
            {
                case SObject so:
                    foreach ((string name, SNode child) in so.Props)
                    {
                        CollectMapped(child, path + "/" + GroundedSchema.Escape(name), into, name);
                    }

                    break;
                case SArray sa:
                    CollectMapped(sa.Items, path, into, leafName);
                    break;
                case SLeaf leaf:
                    if (Candidates(leafName, leaf.Description).Any())
                    {
                        into.Add(path);
                    }

                    break;
            }
        }

        // Fields of the model that no schema property took, for the role check.
        public IReadOnlyList<GroundedField> Others() =>
        [
            .. _all.Where(x => !x.F.WordIndices.Any(i => _mappedWords.Contains((x.Page, i)))).Select(x => Make(x.Page, x.F, "class:" + x.F.Id)),
        ];

        public void WriteObject(SObject o, string path, Utf8JsonWriter w, ExtractedGroup? group, int groupPage = 0)
        {
            w.WriteStartObject();
            foreach ((string name, SNode child) in o.Props)
            {
                w.WritePropertyName(name);
                string childPath = path + "/" + GroundedSchema.Escape(name);
                switch (child)
                {
                    case SObject so:
                        WriteObject(so, childPath, w, group, groupPage);
                        break;
                    case SArray sa:
                        WriteArray(name, sa, childPath, w);
                        break;
                    case SLeaf leaf:
                        WriteLeaf(name, leaf, childPath, w, group, groupPage);
                        break;
                }
            }

            w.WriteEndObject();
        }

        private void WriteArray(string name, SArray sa, string path, Utf8JsonWriter w)
        {
            w.WriteStartArray();
            if (sa.Items is SObject item)
            {
                string folded = Fold(name);
                bool tax = folded.Contains("tax", StringComparison.Ordinal) || folded.Contains("vat", StringComparison.Ordinal);
                int best = -1, bestCount = 0;
                for (int p = 0; p < _ex.Length; p++)
                {
                    int n = (tax ? _ex[p].TaxRows : _ex[p].LineItems).Count;
                    if (n > bestCount)
                    {
                        best = p;
                        bestCount = n;
                    }
                }

                if (best >= 0)
                {
                    IReadOnlyList<ExtractedGroup> groups = tax ? _ex[best].TaxRows : _ex[best].LineItems;
                    for (int i = 0; i < groups.Count && i < MaxElements; i++)
                    {
                        WriteObject(item, path + "/" + i.ToString(CultureInfo.InvariantCulture), w, groups[i], best);
                    }
                }
            }

            w.WriteEndArray();
        }

        private void WriteLeaf(string name, SLeaf leaf, string path, Utf8JsonWriter w, ExtractedGroup? group, int groupPage)
        {
            ExtractedField? field = null;
            int page = groupPage;
            foreach (string id in Candidates(name, leaf.Description))
            {
                if (group is not null)
                {
                    field = group.Get(id);
                }
                else
                {
                    for (int p = 0; p < _ex.Length; p++)
                    {
                        ExtractedField? f = _ex[p].Get(id);
                        if (f is not null && (field is null || f.Confidence > field.Confidence))
                        {
                            field = f;
                            page = p;
                        }
                    }
                }

                if (field is not null)
                {
                    break;
                }
            }

            if (field is null)
            {
                w.WriteNullValue();
                return;
            }

            if (IsNumber(leaf))
            {
                if (field.Number is not { } number)
                {
                    w.WriteNullValue();
                    return;
                }

                w.WriteNumberValue(number);
            }
            else
            {
                w.WriteStringValue(field.Value);
            }

            foreach (int i in field.WordIndices)
            {
                _mappedWords.Add((page, i));
            }

            Fields.Add(Make(page, field, path));
        }

        private GroundedField Make(int page, ExtractedField f, string path)
        {
            OcrPage p = _pages[page];
            var words = new List<LlmWordRef>(f.WordIndices.Count);
            foreach (int i in f.WordIndices)
            {
                if (i >= 0 && i < p.Words.Count)
                {
                    OcrWord word = p.Words[i];
                    words.Add(new LlmWordRef("p" + page.ToString(CultureInfo.InvariantCulture) + "w" + i.ToString(CultureInfo.InvariantCulture), page, i, word.Text, word.Box));
                }
            }

            bool normalized = !string.Equals(f.Value, f.Text, StringComparison.Ordinal);
            return new GroundedField(
                path, f.Value, normalized ? GroundingState.GroundedNormalized : GroundingState.Grounded, f.Confidence, normalized ? f.Value : null,
                words, [new GroundedBox(page, f.Box)], Source: OcrSource.Ocr);
        }

        private IEnumerable<string> Candidates(string name, string? description)
        {
            string folded = Fold(name);
            if (_byFolded.TryGetValue(folded, out string? exact))
            {
                yield return exact;
            }

            if (description is not null)
            {
                foreach (string c in _classes)
                {
                    if (NamesToken(description, c))
                    {
                        yield return c;
                    }
                }
            }

            if (Aliases.TryGetValue(folded, out string[]? ids))
            {
                foreach (string id in ids)
                {
                    if (_classes.Contains(id))
                    {
                        yield return id;
                    }
                }
            }
        }
    }

    // Whole-token match, so "BT-1" is not found inside "BT-112".
    private static bool NamesToken(string text, string id)
    {
        int at = 0;
        while ((at = text.IndexOf(id, at, StringComparison.OrdinalIgnoreCase)) >= 0)
        {
            bool before = at == 0 || !char.IsLetterOrDigit(text[at - 1]);
            int end = at + id.Length;
            bool after = end >= text.Length || !(char.IsLetterOrDigit(text[end]) || text[end] == '.');
            if (before && after)
            {
                return true;
            }

            at++;
        }

        return false;
    }

    private static bool IsNumber(SLeaf leaf)
    {
        JsonNode? t = leaf.Schema["type"];
        static bool Num(JsonNode? n) => n is JsonValue v && v.TryGetValue(out string? s) && s is "number" or "integer";
        return t is JsonArray a ? a.Any(Num) : Num(t);
    }

    private static string Fold(string s)
    {
        var sb = new System.Text.StringBuilder(s.Length);
        foreach (char c in s)
        {
            if (char.IsLetterOrDigit(c))
            {
                sb.Append(char.ToLowerInvariant(c));
            }
        }

        return sb.ToString();
    }

    private static Dictionary<string, string[]> BuildAliases()
    {
        var d = new Dictionary<string, string[]>(StringComparer.Ordinal);
        void Add(string[] ids, params string[] names)
        {
            foreach (string n in names)
            {
                d[n] = ids;
            }
        }

        Add(["BT-1"], "number", "invoicenumber", "invoiceno", "invoiceid", "invoicenr");
        Add(["BT-2", "receipt.date"], "date", "issuedate", "invoicedate", "receiptdate");
        Add(["BT-5"], "currency");
        Add(["BT-9"], "duedate");
        Add(["BT-13"], "purchaseorder", "ponumber", "po");
        Add(["BT-20"], "paymentterms");
        Add(["BT-27"], "vendor", "vendorname", "seller", "sellername", "supplier", "suppliername");
        Add(["BT-31"], "sellervatid", "vendorvatid", "sellertaxid", "vatid");
        Add(["BT-35"], "selleraddress", "vendoraddress");
        Add(["BT-37"], "sellercity");
        Add(["BT-38"], "sellerpostcode", "sellerzip");
        Add(["BT-40"], "sellercountry");
        Add(["BT-44"], "buyer", "buyername", "customer", "customername", "billto");
        Add(["BT-48"], "buyervatid", "customervatid");
        Add(["BT-50"], "buyeraddress", "customeraddress");
        Add(["BT-52"], "buyercity");
        Add(["BT-53"], "buyerpostcode", "buyerzip");
        Add(["BT-55"], "buyercountry");
        Add(["BT-72"], "deliverydate");
        Add(["BT-84"], "iban");
        Add(["BT-86"], "bic", "swift");
        Add(["BT-106", "receipt.subtotal"], "subtotal", "net", "netamount", "totalnet");
        Add(["BT-110", "receipt.tax"], "tax", "taxtotal", "vat", "vattotal", "totaltax");
        Add(["BT-112", "receipt.total"], "total", "grandtotal", "totalamount", "totalgross");
        Add(["BT-115"], "amountdue", "balancedue");
        Add(["receipt.merchant"], "merchant", "store", "shop", "storename");
        Add(["receipt.time"], "time");
        Add(["receipt.tip"], "tip");
        Add(["receipt.paymentMethod"], "paymentmethod", "payment");
        Add(["BT-126"], "id", "itemid", "sku");
        Add(["BT-153"], "description", "itemdescription");
        Add(["BT-129", "receipt.quantity"], "quantity", "qty");
        Add(["BT-130"], "unit");
        Add(["BT-146", "receipt.unitPrice"], "unitprice", "price");
        Add(["BT-131", "receipt.amount"], "amount", "lineamount", "linetotal");
        Add(["BT-152", "BT-119"], "taxrate", "vatrate", "rate");
        Add(["receipt.itemName"], "name", "itemname");
        Add(["BT-116"], "taxableamount");
        Add(["BT-117"], "taxamount");
        return d;
    }
}
