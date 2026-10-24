using System.Text.Json;
using OneOcr.Model;

namespace OneOcr.Documents;

/// <summary>Where a field class sits in a document: once per page, or once per line item, or once per tax row.</summary>
internal enum FieldGroup
{
    /// <summary>One value per page.</summary>
    Header = 0,

    /// <summary>One value per line item; the item pointer ties the fields of an item together.</summary>
    Line = 1,

    /// <summary>One value per tax-breakdown row.</summary>
    Tax = 2,
}

/// <summary>
/// The field classes of a model and the document types they belong to. The built-in classes (EN 16931 business terms for invoices,
/// <c>receipt.*</c> for receipts) have fixed groups; a class the schema does not know is a once-per-page field allowed in every document type.
/// </summary>
internal sealed class DocSchema
{
    private static readonly HashSet<string> s_invoiceLine = ["BT-126", "BT-129", "BT-130", "BT-131", "BT-146", "BT-152", "BT-153"];
    private static readonly HashSet<string> s_invoiceTax = ["BT-116", "BT-117", "BT-119"];
    private static readonly HashSet<string> s_receiptLine = ["receipt.itemName", "receipt.quantity", "receipt.unitPrice", "receipt.amount"];

    public DocSchema(IReadOnlyList<string> classes, IReadOnlyList<string> roles, IReadOnlyList<string> docTypes)
    {
        Classes = classes;
        Roles = roles;
        DocTypes = docTypes;
        Groups = new FieldGroup[classes.Count];
        for (int c = 0; c < classes.Count; c++)
        {
            string name = classes[c];
            Groups[c] = s_invoiceTax.Contains(name) ? FieldGroup.Tax : s_invoiceLine.Contains(name) || s_receiptLine.Contains(name) ? FieldGroup.Line : FieldGroup.Header;
        }
    }

    public IReadOnlyList<string> Classes { get; }

    public IReadOnlyList<string> Roles { get; }

    public IReadOnlyList<string> DocTypes { get; }

    public FieldGroup[] Groups { get; }

    /// <summary>Whether a class belongs to a document type's schema. A class with neither the invoice (BT-) nor the receipt prefix belongs to every document type.</summary>
    public bool Allowed(string docType, int classIndex)
    {
        string name = Classes[classIndex];
        bool invoice = name.StartsWith("BT-", StringComparison.Ordinal);
        bool receipt = name.StartsWith("receipt.", StringComparison.Ordinal);
        return docType switch
        {
            "invoice" => !receipt,
            "receipt" => !invoice,
            _ => !invoice && !receipt,
        };
    }
}

/// <summary>
/// The validated, widened weights of a <c>docextract-v0</c> document-extraction model (docs/spec/docextract-v0.md): configuration and
/// every tensor shape are checked against <see cref="OcrLimits"/> and against each other before anything sized by them is allocated.
/// </summary>
internal sealed class DocExtractModel
{
    public const int PairFeaturesPadded = Kernels.DocKernels.PairFeaturesPadded;

    /// <summary>One transformer block's weights.</summary>
    internal sealed class Block
    {
        public required float[] Ln1Weight { get; init; }
        public required float[] Ln1Bias { get; init; }
        public required float[] QkvWeight { get; init; }
        public required float[] QkvBias { get; init; }
        public required float[] ProjWeight { get; init; }
        public required float[] ProjBias { get; init; }
        public required float[] Ln2Weight { get; init; }
        public required float[] Ln2Bias { get; init; }
        public required float[] Fc1Weight { get; init; }
        public required float[] Fc1Bias { get; init; }
        public required float[] Fc2Weight { get; init; }
        public required float[] Fc2Bias { get; init; }
    }

    private DocExtractModel()
    {
    }

    public required int D { get; init; }
    public required int Layers { get; init; }
    public required int Heads { get; init; }
    public required int Ffn { get; init; }
    public required int CharDim { get; init; }
    public required int CharConv { get; init; }
    public required int PairHidden { get; init; }
    public required int RelDim { get; init; }
    public required int MaxWords { get; init; }
    public required int MaxChars { get; init; }
    public required int Bins { get; init; }
    public required int Tags { get; init; }
    public required int Roles { get; init; }
    public required int DocTypes { get; init; }
    public required int UnkChar { get; init; }
    public required DocSchema Schema { get; init; }

    /// <summary>Code point to character id; the id 0 (the blank) is never a text character.</summary>
    public required IReadOnlyDictionary<int, int> CharIds { get; init; }

    public required float[] CharEmb { get; init; }
    public required float[] CharConvWeight { get; init; }
    public required float[] CharConvBias { get; init; }
    public required float[] CharProjWeight { get; init; }
    public required float[] CharProjBias { get; init; }
    public required float[] PosX0 { get; init; }
    public required float[] PosY0 { get; init; }
    public required float[] PosX1 { get; init; }
    public required float[] PosY1 { get; init; }

    /// <summary>Feature projection weight <c>[d, 16]</c>: the eight inputs, zero columns after.</summary>
    public required float[] FeatWeight { get; init; }

    public required float[] FeatBias { get; init; }
    public required float[] EmbLnWeight { get; init; }
    public required float[] EmbLnBias { get; init; }
    public required Block[] Blocks { get; init; }

    /// <summary>First pair layer <c>[pair_hidden, 16]</c>: the 13 features, zero columns after.</summary>
    public required float[] Pair1Weight { get; init; }

    public required float[] Pair1Bias { get; init; }
    public required float[] Pair2Weight { get; init; }
    public required float[] Pair2Bias { get; init; }
    public required float[] FinalLnWeight { get; init; }
    public required float[] FinalLnBias { get; init; }
    public required float[] TagWeight { get; init; }
    public required float[] TagBias { get; init; }
    public required float[] RoleWeight { get; init; }
    public required float[] RoleBias { get; init; }
    public required float[] DocWeight { get; init; }
    public required float[] DocBias { get; init; }
    public required float[] RelQWeight { get; init; }
    public required float[] RelQBias { get; init; }
    public required float[] RelKWeight { get; init; }
    public required float[] RelKBias { get; init; }

    /// <summary>Channels of the pair bias: heads of every block plus the two pointer relations.</summary>
    public int PairChannels => (Layers * Heads) + 2;

    public static DocExtractModel FromContainer(ModelContainer container, OcrLimits limits)
    {
        ArgumentNullException.ThrowIfNull(container);
        ArgumentNullException.ThrowIfNull(limits);
        if (container.Architecture != ModelContainer.DocExtractArchitecture || container.WordInput is not { } input)
        {
            throw new OcrModelException($"Unknown architecture '{container.Architecture}'; this reader runs 'docextract-v0'.");
        }

        DocExtractConfigDto c;
        try
        {
            c = container.ConfigElement.Deserialize(ManifestJsonContext.Default.DocExtractConfigDto)
                ?? throw new OcrModelException("The model configuration is null.");
        }
        catch (JsonException ex)
        {
            throw new OcrModelException("The model configuration is not valid: " + ex.Message, ex);
        }

        if (c.ClassesList is null || c.RolesList is null || c.DocTypesList is null)
        {
            throw new OcrModelException("The model configuration lacks its class, role or document-type lists.");
        }

        CheckRange(c.D, 1, limits.MaxLayerWidth, "d");
        CheckRange(c.Ffn, 1, limits.MaxLayerWidth, "ffn");
        CheckRange(c.CharDim, 1, limits.MaxLayerWidth, "char_dim");
        CheckRange(c.CharConv, 1, limits.MaxLayerWidth, "char_conv");
        CheckRange(c.PairHidden, 1, limits.MaxLayerWidth, "pair_hidden");
        CheckRange(c.RelDim, 1, limits.MaxLayerWidth, "rel_dim");
        CheckRange(c.Layers, 1, limits.MaxTransformerLayers, "layers");
        CheckRange(c.Heads, 1, c.D, "heads");
        if (c.D % c.Heads != 0)
        {
            throw new OcrModelException("config.d must be a multiple of config.heads.");
        }

        if (c.MaxWords != input.MaxWords)
        {
            throw new OcrModelException("config.max_words differs from input.max_words.");
        }

        if (input.Features != 8 || input.Bins < 1 || input.MaxChars < 1)
        {
            throw new OcrModelException("docextract-v0 reads 8 numeric features per word.");
        }

        CheckRange(c.Classes, 1, limits.MaxExtractionClasses, "classes");
        CheckRange(c.Roles, 1, limits.MaxExtractionClasses, "roles");
        CheckRange(c.DocTypes, 1, limits.MaxExtractionClasses, "doc_types");
        if (c.ClassesList.Count != c.Classes || c.Tags != 1 + (2 * c.Classes) || c.RolesList.Count != c.Roles || c.DocTypesList.Count != c.DocTypes)
        {
            throw new OcrModelException("The class, tag, role and document-type counts of the configuration disagree with its lists.");
        }

        if (c.Roles != 5 || c.RolesList[0] != "O" || c.RolesList[1] != "B-KEY" || c.RolesList[2] != "I-KEY" || c.RolesList[3] != "B-VAL" || c.RolesList[4] != "I-VAL")
        {
            throw new OcrModelException("config.roles_list must be O, B-KEY, I-KEY, B-VAL, I-VAL.");
        }

        foreach (string name in c.ClassesList.Concat(c.DocTypesList))
        {
            if (string.IsNullOrEmpty(name) || name.Length > limits.MaxCharsetEntryLength * 8)
            {
                throw new OcrModelException("A class or document-type name is empty or too long.");
            }
        }

        if (new HashSet<string>(c.ClassesList, StringComparer.Ordinal).Count != c.Classes)
        {
            throw new OcrModelException("config.classes_list has duplicate names.");
        }

        int chars = container.Charset.Count + 1;
        if (c.UnkChar != container.Charset.Count)
        {
            throw new OcrModelException("config.unk_char must be the charset length (the id after the last character).");
        }

        int d = c.D;
        int pairOut = (c.Layers * c.Heads) + 2;
        if ((long)pairOut * input.MaxWords * input.MaxWords > limits.MaxActivationElements)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxActivationElements), limits.MaxActivationElements, (long)pairOut * input.MaxWords * input.MaxWords);
        }

        float[] T(string name, params int[] shape)
        {
            TensorInfo info = container.FindTensor(name) ?? throw new OcrModelException($"The model has no tensor named '{name}'.");
            if (!info.Shape.SequenceEqual(shape))
            {
                throw new OcrModelException($"Tensor '{name}' has shape [{string.Join(", ", info.Shape)}]; expected [{string.Join(", ", shape)}].");
            }

            return container.ReadTensor(name);
        }

        float[] charConv = T("char_conv.weight", c.CharConv, c.CharDim, 3);
        // Taps are ordered (k, channel) so the three neighbouring characters of a window are one run of memory.
        var reordered = new float[charConv.Length];
        for (int o = 0; o < c.CharConv; o++)
        {
            for (int ch = 0; ch < c.CharDim; ch++)
            {
                for (int k = 0; k < 3; k++)
                {
                    reordered[(o * 3 * c.CharDim) + (k * c.CharDim) + ch] = charConv[(((o * c.CharDim) + ch) * 3) + k];
                }
            }
        }

        var blocks = new Block[c.Layers];
        for (int l = 0; l < blocks.Length; l++)
        {
            string p = $"blocks.{l}.";
            blocks[l] = new Block
            {
                Ln1Weight = T(p + "ln1.weight", d),
                Ln1Bias = T(p + "ln1.bias", d),
                QkvWeight = T(p + "qkv.weight", 3 * d, d),
                QkvBias = T(p + "qkv.bias", 3 * d),
                ProjWeight = T(p + "proj.weight", d, d),
                ProjBias = T(p + "proj.bias", d),
                Ln2Weight = T(p + "ln2.weight", d),
                Ln2Bias = T(p + "ln2.bias", d),
                Fc1Weight = T(p + "fc1.weight", c.Ffn, d),
                Fc1Bias = T(p + "fc1.bias", c.Ffn),
                Fc2Weight = T(p + "fc2.weight", d, c.Ffn),
                Fc2Bias = T(p + "fc2.bias", d),
            };
        }

        var charIds = new Dictionary<int, int>();
        for (int i = 1; i < container.Charset.Count; i++)
        {
            string entry = container.Charset[i];
            if (entry.Length > 0 && entry.EnumerateRunes().Count() == 1)
            {
                charIds.TryAdd(entry.EnumerateRunes().First().Value, i);
            }
        }

        return new DocExtractModel
        {
            D = d,
            Layers = c.Layers,
            Heads = c.Heads,
            Ffn = c.Ffn,
            CharDim = c.CharDim,
            CharConv = c.CharConv,
            PairHidden = c.PairHidden,
            RelDim = c.RelDim,
            MaxWords = input.MaxWords,
            MaxChars = input.MaxChars,
            Bins = input.Bins,
            Tags = c.Tags,
            Roles = c.Roles,
            DocTypes = c.DocTypes,
            UnkChar = c.UnkChar,
            Schema = new DocSchema(c.ClassesList, c.RolesList, c.DocTypesList),
            CharIds = charIds,
            CharEmb = T("char_emb.weight", chars, c.CharDim),
            CharConvWeight = reordered,
            CharConvBias = T("char_conv.bias", c.CharConv),
            CharProjWeight = T("char_proj.weight", d, c.CharConv),
            CharProjBias = T("char_proj.bias", d),
            PosX0 = T("pos_x0.weight", input.Bins, d),
            PosY0 = T("pos_y0.weight", input.Bins, d),
            PosX1 = T("pos_x1.weight", input.Bins, d),
            PosY1 = T("pos_y1.weight", input.Bins, d),
            FeatWeight = PadColumns(T("feat_proj.weight", d, 8), d, 8, 16),
            FeatBias = T("feat_proj.bias", d),
            EmbLnWeight = T("emb_ln.weight", d),
            EmbLnBias = T("emb_ln.bias", d),
            Blocks = blocks,
            Pair1Weight = PadColumns(T("pair1.weight", c.PairHidden, Kernels.DocKernels.PairFeatures), c.PairHidden, Kernels.DocKernels.PairFeatures, 16),
            Pair1Bias = T("pair1.bias", c.PairHidden),
            Pair2Weight = T("pair2.weight", pairOut, c.PairHidden),
            Pair2Bias = T("pair2.bias", pairOut),
            FinalLnWeight = T("final_ln.weight", d),
            FinalLnBias = T("final_ln.bias", d),
            TagWeight = T("tag_head.weight", c.Tags, d),
            TagBias = T("tag_head.bias", c.Tags),
            RoleWeight = T("role_head.weight", c.Roles, d),
            RoleBias = T("role_head.bias", c.Roles),
            DocWeight = T("doc_head.weight", c.DocTypes, d),
            DocBias = T("doc_head.bias", c.DocTypes),
            RelQWeight = T("rel_q.weight", 2 * c.RelDim, d),
            RelQBias = T("rel_q.bias", 2 * c.RelDim),
            RelKWeight = T("rel_k.weight", 2 * c.RelDim, d),
            RelKBias = T("rel_k.bias", 2 * c.RelDim),
        };

        static void CheckRange(int v, int lo, int hi, string name)
        {
            if (v < lo || v > hi)
            {
                throw new OcrModelException($"config.{name} = {v} is outside {lo}..{hi}.");
            }
        }

        static float[] PadColumns(float[] w, int rows, int cols, int padded)
        {
            var r = new float[rows * padded];
            for (int i = 0; i < rows; i++)
            {
                Array.Copy(w, i * cols, r, i * padded, cols);
            }

            return r;
        }
    }
}
