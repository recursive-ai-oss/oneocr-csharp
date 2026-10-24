using System.Buffers.Binary;
using System.Text.Json;
using System.Text.Json.Nodes;
using OneOcr.Documents;
using OneOcr.Model;

namespace OneOcr.Training;

/// <summary>
/// Writes <c>docextract-v0</c> containers for a custom schema (docs/spec/docextract-v0.md, "Fine-tuning a custom schema"). The container
/// format is unchanged: a custom model is the shipped model with a different <c>config.classes_list</c> and a tag head of the matching
/// size, so every reader that reads the shipped file reads a custom one. <see cref="WithNewHead"/> makes the starting point (every tensor
/// but the tag head copied, the head freshly initialised); <see cref="WithTrained"/> writes the weights a training run ended with, mapped
/// back from the runtime's layouts (character-convolution taps in (tap, channel) order, feature columns padded to 16) to the container's.
/// </summary>
internal static class DocContainerWriter
{
    /// <summary>Checks the class names of a custom schema: the container, the pointer paths and the built-in classes all constrain them.</summary>
    public static void ValidateClasses(IReadOnlyList<string> classes, OcrLimits limits)
    {
        ArgumentNullException.ThrowIfNull(classes);
        if (classes.Count == 0 || classes.Count > limits.MaxExtractionClasses)
        {
            throw new ArgumentException($"A schema needs between 1 and {limits.MaxExtractionClasses} fields.");
        }

        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (string c in classes)
        {
            if (string.IsNullOrWhiteSpace(c) || c.Length > 128 || c.Contains('/', StringComparison.Ordinal) || c.Contains('~', StringComparison.Ordinal))
            {
                throw new ArgumentException($"The field name '{c}' is empty, longer than 128 characters, or contains '/' or '~'.");
            }

            // These prefixes select the invoice and receipt rules of the shipped classes (a custom class is allowed in every document type).
            if (c.StartsWith("BT-", StringComparison.Ordinal) || c.StartsWith("receipt.", StringComparison.Ordinal))
            {
                throw new ArgumentException($"The field name '{c}' starts with a prefix reserved for the built-in invoice and receipt classes.");
            }

            if (!seen.Add(c))
            {
                throw new ArgumentException($"The field name '{c}' appears twice.");
            }
        }
    }

    /// <summary>
    /// The container of <paramref name="baseModel"/> with its tag head replaced by one for <paramref name="classes"/> (tags: O, then B and I per class):
    /// weights uniform in ±1/√d from the seed, biases 0, rounded to fp16 like every stored weight.
    /// </summary>
    public static byte[] WithNewHead(ModelContainer baseModel, IReadOnlyList<string> classes, int seed, OcrLimits limits)
    {
        ArgumentNullException.ThrowIfNull(baseModel);
        ValidateClasses(classes, limits);
        DocExtractModel m = DocExtractModel.FromContainer(baseModel, limits);
        int tags = 1 + (2 * classes.Count);
        int d = m.D;
        var weight = new float[tags * d];
        var rng = new SplitMix((ulong)(uint)seed ^ 0x7461675F68656164UL);
        double bound = 1.0 / Math.Sqrt(d);
        for (int i = 0; i < weight.Length; i++)
        {
            double u = (rng.Next() >> 11) * (1.0 / 9007199254740992.0);
            weight[i] = (float)(((2 * u) - 1) * bound);
        }

        var config = (JsonObject)JsonNode.Parse(baseModel.ConfigElement.GetRawText())!;
        config["classes"] = classes.Count;
        config["tags"] = tags;
        config["classes_list"] = new JsonArray([.. classes.Select(static c => (JsonNode?)JsonValue.Create(c))]);

        return Write(baseModel, config, name => name switch
        {
            "tag_head.weight" => (weight, [tags, d]),
            "tag_head.bias" => (new float[tags], [tags]),
            _ => (baseModel.ReadTensor(name), null),
        }, "new tag head for " + classes.Count.ToString(System.Globalization.CultureInfo.InvariantCulture) + " fields");
    }

    /// <summary>The container of <paramref name="custom"/> with every weight replaced by the one <paramref name="trained"/> holds.</summary>
    public static byte[] WithTrained(ModelContainer custom, DocExtractModel trained, string provenanceNote)
    {
        ArgumentNullException.ThrowIfNull(custom);
        ArgumentNullException.ThrowIfNull(trained);
        var byName = new Dictionary<string, float[]>(StringComparer.Ordinal);
        foreach ((string name, float[] value) in DocParams.List(trained))
        {
            byName[name] = value;
        }

        return Write(custom, (JsonObject)JsonNode.Parse(custom.ConfigElement.GetRawText())!, name =>
        {
            if (!byName.TryGetValue(name, out float[]? v))
            {
                return (custom.ReadTensor(name), null);
            }

            return name switch
            {
                "char_conv.weight" => (UnReorderConv(v, trained.CharConv, trained.CharDim), null),
                "feat_proj.weight" => (Unpad(v, trained.D, 16, 8), null),
                "pair1.weight" => (Unpad(v, trained.PairHidden, DocExtractModel.PairFeaturesPadded, Kernels.DocKernels.PairFeatures), null),
                _ => (v, null),
            };
        }, provenanceNote);
    }

    private static float[] UnReorderConv(float[] w, int outCh, int inCh)
    {
        var r = new float[w.Length];
        for (int o = 0; o < outCh; o++)
        {
            for (int ch = 0; ch < inCh; ch++)
            {
                for (int k = 0; k < 3; k++)
                {
                    r[(((o * inCh) + ch) * 3) + k] = w[(o * 3 * inCh) + (k * inCh) + ch];
                }
            }
        }

        return r;
    }

    private static float[] Unpad(float[] w, int rows, int padded, int cols)
    {
        var r = new float[rows * cols];
        for (int i = 0; i < rows; i++)
        {
            Array.Copy(w, i * padded, r, i * cols, cols);
        }

        return r;
    }

    // value(name) gives the tensor's values and, when the shape differs from the base file's, the new shape.
    private static byte[] Write(ModelContainer c, JsonObject config, Func<string, (float[] Values, int[]? Shape)> value, string note)
    {
        int n = c.Tensors.Count;
        var values = new float[n][];
        var shapes = new int[n][];
        for (int i = 0; i < n; i++)
        {
            (float[] v, int[]? s) = value(c.Tensors[i].Name);
            values[i] = v;
            shapes[i] = s ?? [.. c.Tensors[i].Shape];
        }

        var offsets = new long[n];
        long cursor = 0;
        for (int i = 0; i < n; i++)
        {
            offsets[i] = (cursor + 63) / 64 * 64;
            cursor = offsets[i] + (values[i].Length * 2L);
        }

        byte[] manifest = Manifest(c, config, values, shapes, offsets, note);
        long dataStart = (16 + (long)manifest.Length + 63) / 64 * 64;
        var file = new byte[checked(dataStart + cursor)];
        "ONEOCRM"u8.CopyTo(file);
        BinaryPrimitives.WriteUInt32LittleEndian(file.AsSpan(8), c.FormatVersion);
        BinaryPrimitives.WriteUInt32LittleEndian(file.AsSpan(12), (uint)manifest.Length);
        manifest.CopyTo(file, 16);
        for (int i = 0; i < n; i++)
        {
            Span<byte> dst = file.AsSpan((int)(dataStart + offsets[i]));
            for (int j = 0; j < values[i].Length; j++)
            {
                BinaryPrimitives.WriteHalfLittleEndian(dst[(j * 2)..], (Half)values[i][j]);
            }
        }

        return file;
    }

    private static byte[] Manifest(ModelContainer c, JsonObject config, float[][] values, int[][] shapes, long[] offsets, string note)
    {
        WordModelInput input = c.WordInput ?? throw new OcrModelException("Not a docextract model.");
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms))
        {
            w.WriteStartObject();
            w.WriteString("architecture", c.Architecture);
            w.WriteString("name", c.Name);
            w.WriteString("version", c.Version);
            w.WriteString("license", c.License);
            w.WriteStartObject("input");
            w.WriteString("kind", "words");
            w.WriteNumber("max_words", input.MaxWords);
            w.WriteNumber("max_chars", input.MaxChars);
            w.WriteNumber("bins", input.Bins);
            w.WriteNumber("features", input.Features);
            w.WriteEndObject();
            w.WriteStartArray("charset");
            foreach (string s in c.Charset)
            {
                w.WriteStringValue(s);
            }

            w.WriteEndArray();
            w.WritePropertyName("config");
            config.WriteTo(w);
            w.WriteStartArray("tensors");
            for (int i = 0; i < values.Length; i++)
            {
                w.WriteStartObject();
                w.WriteString("name", c.Tensors[i].Name);
                w.WriteString("dtype", "f16");
                w.WriteStartArray("shape");
                foreach (int d in shapes[i])
                {
                    w.WriteNumberValue(d);
                }

                w.WriteEndArray();
                w.WriteNumber("offset", offsets[i]);
                w.WriteNumber("length", values[i].Length * 2L);
                w.WriteEndObject();
            }

            w.WriteEndArray();
            if (c.CalibrationElement is { } cal)
            {
                w.WritePropertyName("calibration");
                cal.WriteTo(w);
            }

            w.WriteStartObject("provenance");
            w.WriteString("fine_tuned", note);
            w.WriteEndObject();
            w.WriteEndObject();
        }

        return ms.ToArray();
    }
}
