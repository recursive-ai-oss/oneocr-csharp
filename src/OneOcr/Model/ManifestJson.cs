using System.Text.Json;
using System.Text.Json.Serialization;

namespace OneOcr.Model;

// Wire shapes of the container manifest (docs/spec/model-container-v1.md). Members the spec does not name are skipped,
// including the free-form "provenance" record.

internal sealed class ManifestDto
{
    public required string Architecture { get; init; }
    public required string Name { get; init; }
    public required string Version { get; init; }
    public required string License { get; init; }
    public required InputDto Input { get; init; }
    public required List<string> Charset { get; init; }
    public required JsonElement Config { get; init; }
    public required List<TensorDto> Tensors { get; init; }

    // Read only from version 1 files; held raw so that a version 0 file never trips over it.
    public JsonElement? Calibration { get; init; }

    // Optional, version 1 only: the curve for the confidence a page reports, when it differs from the one the pipeline decides with.
    [JsonPropertyName("report_calibration")]
    public JsonElement? ReportCalibration { get; init; }
}

// The image members are required for every architecture that reads pixels and absent for the word-based extraction model, whose
// input block names its featurization limits instead (kind, max_words, max_chars, bins, features). The container checks which
// set applies to the architecture.
internal sealed class InputDto
{
    public int? Height { get; init; }
    public int? Channels { get; init; }
    public float? Mean { get; init; }
    public float? Std { get; init; }
    public bool? Invert { get; init; }

    public string? Kind { get; init; }

    [JsonPropertyName("max_words")]
    public int? MaxWords { get; init; }

    [JsonPropertyName("max_chars")]
    public int? MaxChars { get; init; }

    public int? Bins { get; init; }
    public int? Features { get; init; }
}

internal sealed class TensorDto
{
    public required string Name { get; init; }
    public required string Dtype { get; init; }
    public required List<long> Shape { get; init; }
    public required long Offset { get; init; }
    public required long Length { get; init; }
}

internal sealed class CrnnConfigDto
{
    public required List<ConvDto> Conv { get; init; }
    public required LstmDto Lstm { get; init; }
    public required int Classes { get; init; }
}

internal sealed class ConvDto
{
    public required int Out { get; init; }
    public required List<int> Kernel { get; init; }
    public required List<int> Padding { get; init; }
    public List<int>? Pool { get; init; }
}

internal sealed class LstmDto
{
    public required int Hidden { get; init; }
    public required int Layers { get; init; }
}

internal sealed class DbNetConfigDto
{
    public required string Arch { get; init; }
    public required int Stem { get; init; }
    public required List<int> Widths { get; init; }
    public required List<int> Depths { get; init; }
    public required int Fpn { get; init; }
    public required int Text { get; init; }
    public required int Layout { get; init; }

    [JsonPropertyName("layout_blocks")]
    public required int LayoutBlocks { get; init; }

    public required int Classes { get; init; }

    /// <summary>Blocks of the stride-16 layout context branch (<c>dbnet-lite-v1</c>); absent or 0 means none.</summary>
    [JsonPropertyName("layout16_blocks")]
    public int? Layout16Blocks { get; init; }

    /// <summary>Width of the stride-16 branch; required when <c>layout16_blocks</c> is above 0.</summary>
    public int? Layout16 { get; init; }

    /// <summary>Whether the orientation head reads the channel means of stages 2 and 3 (<c>dbnet-lite-v1</c>); absent means false.</summary>
    public bool? Orient16 { get; init; }

    [JsonPropertyName("layout_classes")]
    public required List<string> LayoutClasses { get; init; }

    /// <summary>Names of the layout classes the model was trained on; absent means every class except form_field and checkbox.</summary>
    [JsonPropertyName("layout_trained")]
    public List<string>? LayoutTrained { get; init; }

    /// <summary>Post-processing thresholds and expansions; absent members keep the defaults.</summary>
    public PostprocessDto? Postprocess { get; init; }
}

internal sealed class PostprocessDto
{
    [JsonPropertyName("text_threshold")]
    public double? TextThreshold { get; init; }

    [JsonPropertyName("text_expand")]
    public double? TextExpand { get; init; }

    [JsonPropertyName("layout_threshold")]
    public double? LayoutThreshold { get; init; }

    [JsonPropertyName("layout_expand")]
    public double? LayoutExpand { get; init; }

    /// <summary>Page noise above which the page-conditional text threshold may apply; the rule needs all three <c>rule_*</c> members.</summary>
    [JsonPropertyName("rule_sigma_min")]
    public double? RuleSigmaMin { get; init; }

    /// <summary>Speck density per megapixel below which the page-conditional text threshold applies.</summary>
    [JsonPropertyName("rule_junk_max")]
    public double? RuleJunkMax { get; init; }

    /// <summary>Text threshold used on pages the rule selects.</summary>
    [JsonPropertyName("rule_text_threshold")]
    public double? RuleTextThreshold { get; init; }
}

internal sealed class DocExtractConfigDto
{
    public required int D { get; init; }
    public required int Layers { get; init; }
    public required int Heads { get; init; }
    public required int Ffn { get; init; }

    [JsonPropertyName("char_dim")]
    public required int CharDim { get; init; }

    [JsonPropertyName("char_conv")]
    public required int CharConv { get; init; }

    [JsonPropertyName("pair_hidden")]
    public required int PairHidden { get; init; }

    [JsonPropertyName("rel_dim")]
    public required int RelDim { get; init; }

    [JsonPropertyName("max_words")]
    public required int MaxWords { get; init; }

    public required int Classes { get; init; }
    public required int Tags { get; init; }
    public required int Roles { get; init; }

    [JsonPropertyName("doc_types")]
    public required int DocTypes { get; init; }

    [JsonPropertyName("classes_list")]
    public required List<string> ClassesList { get; init; }

    [JsonPropertyName("roles_list")]
    public required List<string> RolesList { get; init; }

    [JsonPropertyName("doc_types_list")]
    public required List<string> DocTypesList { get; init; }

    [JsonPropertyName("unk_char")]
    public required int UnkChar { get; init; }
}

[JsonSerializable(typeof(DocExtractConfigDto))]
[JsonSerializable(typeof(DbNetConfigDto))]
[JsonSerializable(typeof(PostprocessDto))]
[JsonSerializable(typeof(ManifestDto))]
[JsonSerializable(typeof(CrnnConfigDto))]
[JsonSerializable(typeof(CalibrationDto))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, PropertyNameCaseInsensitive = false)]
internal sealed partial class ManifestJsonContext : JsonSerializerContext
{
}
