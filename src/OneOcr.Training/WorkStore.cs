using System.Text.Json;
using System.Text.Json.Serialization;

namespace OneOcr.Training;

// On-disk shapes of a work directory. Every file here is read back through the same limits as any other untrusted input.

internal sealed class DocLabelsDto
{
    public string Format { get; set; } = WorkStore.LabelsFormat;
    public int Version { get; set; } = 1;
    public string Id { get; set; } = "";
    public string Document { get; set; } = "";

    /// <summary>Input height of the model whose crops are stored.</summary>
    public int Height { get; set; }
    public decimal CostUsd { get; set; }
    public int Calls { get; set; }
    public List<LineLabelDto> Lines { get; set; } = [];
}

internal sealed class LineLabelDto
{
    public int Index { get; set; }
    public int X { get; set; }
    public int Y { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
    public string Local { get; set; } = "";
    public float Confidence { get; set; }
    public string? Llm { get; set; }

    [JsonConverter(typeof(JsonStringEnumConverter<LineAgreement>))]
    public LineAgreement State { get; set; }
    public double Cer { get; set; }
    public int CropWidth { get; set; }

    /// <summary>The line scaled to the model's input height, 8-bit gray, row-major (base64 in the file).</summary>
    public byte[] Crop { get; set; } = [];
}

internal sealed class ReviewedDto
{
    public string Format { get; set; } = WorkStore.ReviewedFormat;
    public int Version { get; set; } = 1;
    public List<ReviewedItemDto> Items { get; set; } = [];
}

internal sealed class ReviewedItemDto
{
    public string Id { get; set; } = "";
    public string Text { get; set; } = "";
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, WriteIndented = false, UseStringEnumConverter = false)]
[JsonSerializable(typeof(DocLabelsDto))]
[JsonSerializable(typeof(ReviewedDto))]
internal sealed partial class WorkJsonContext : JsonSerializerContext
{
}

/// <summary>The files of one run: labelled documents, the reviewer's corrections, the checkpoint and the report.</summary>
internal sealed class WorkStore
{
    public const string LabelsFormat = "oneocr.labels";
    public const string ReviewedFormat = "oneocr.reviewed";

    private readonly OcrLimits _limits;

    public WorkStore(string directory, OcrLimits limits)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        Directory = directory;
        _limits = limits;
        System.IO.Directory.CreateDirectory(Path.Combine(directory, "labels"));
    }

    public string Directory { get; }

    public string CheckpointPath => Path.Combine(Directory, "checkpoint.bin");

    public string ReportPath => Path.Combine(Directory, "report.json");

    public string ReviewedPath => Path.Combine(Directory, "reviewed.json");

    public static string LineId(string docId, int index) => docId + "-" + index.ToString(System.Globalization.CultureInfo.InvariantCulture);

    private string LabelsPath(string docId) => Path.Combine(Directory, "labels", docId + ".json");

    public bool HasDocument(string docId) => File.Exists(LabelsPath(docId));

    public void Save(DocLabelsDto doc)
    {
        WriteAtomic(LabelsPath(doc.Id), JsonSerializer.SerializeToUtf8Bytes(doc, WorkJsonContext.Default.DocLabelsDto));
    }

    /// <summary>Every labelled document, ordered by id.</summary>
    public List<DocLabelsDto> LoadAll()
    {
        var docs = new List<DocLabelsDto>();
        string dir = Path.Combine(Directory, "labels");
        foreach (string file in System.IO.Directory.EnumerateFiles(dir, "*.json").Order(StringComparer.Ordinal))
        {
            DocLabelsDto doc = Read(file, WorkJsonContext.Default.DocLabelsDto);
            if (doc.Format != LabelsFormat || doc.Version != 1 || doc.Id.Length == 0 || doc.Id != Path.GetFileNameWithoutExtension(file))
            {
                throw new OcrFormatException($"'{file}' is not a label file of this version.");
            }

            if (doc.Lines.Count > _limits.MaxLinesPerPage)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxLinesPerPage), _limits.MaxLinesPerPage, doc.Lines.Count);
            }

            foreach (LineLabelDto l in doc.Lines)
            {
                if (l.CropWidth < 1 || l.CropWidth > _limits.MaxLineWidth || doc.Height < 1 || doc.Height > _limits.MaxInputHeight
                    || l.Crop.Length != (long)l.CropWidth * doc.Height)
                {
                    throw new OcrFormatException($"'{file}' holds a line whose crop does not match its size.");
                }
            }

            docs.Add(doc);
        }

        return docs;
    }

    public Dictionary<string, string> LoadReviewed()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        if (!File.Exists(ReviewedPath))
        {
            return map;
        }

        ReviewedDto dto = Read(ReviewedPath, WorkJsonContext.Default.ReviewedDto);
        if (dto.Format != ReviewedFormat || dto.Version != 1)
        {
            throw new OcrFormatException($"'{ReviewedPath}' is not a reviewed-lines file of this version.");
        }

        foreach (ReviewedItemDto item in dto.Items)
        {
            map[item.Id] = item.Text;
        }

        return map;
    }

    public void SaveReviewed(Dictionary<string, string> reviewed)
    {
        var dto = new ReviewedDto { Items = [.. reviewed.OrderBy(static p => p.Key, StringComparer.Ordinal).Select(static p => new ReviewedItemDto { Id = p.Key, Text = p.Value })] };
        WriteAtomic(ReviewedPath, JsonSerializer.SerializeToUtf8Bytes(dto, WorkJsonContext.Default.ReviewedDto));
    }

    public static void WriteAtomic(string path, byte[] bytes)
    {
        string temp = path + ".tmp";
        File.WriteAllBytes(temp, bytes);
        File.Move(temp, path, overwrite: true);
    }

    private T Read<T>(string file, System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> info) where T : class
    {
        var fi = new FileInfo(file);
        if (fi.Length > _limits.MaxModelFileBytes)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxModelFileBytes), _limits.MaxModelFileBytes, fi.Length);
        }

        byte[] bytes = File.ReadAllBytes(file);
        try
        {
            var reader = new Utf8JsonReader(bytes, new JsonReaderOptions { MaxDepth = _limits.MaxJsonDepth });
            return JsonSerializer.Deserialize(ref reader, info) ?? throw new OcrFormatException($"'{file}' is empty.");
        }
        catch (JsonException ex)
        {
            throw new OcrFormatException($"'{file}' is not valid: {ex.Message}", ex);
        }
    }
}
