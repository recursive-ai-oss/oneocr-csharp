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
}

internal sealed class InputDto
{
    public required int Height { get; init; }
    public required int Channels { get; init; }
    public required float Mean { get; init; }
    public required float Std { get; init; }
    public required bool Invert { get; init; }
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
}

[JsonSerializable(typeof(DbNetConfigDto))]
[JsonSerializable(typeof(PostprocessDto))]
[JsonSerializable(typeof(ManifestDto))]
[JsonSerializable(typeof(CrnnConfigDto))]
[JsonSerializable(typeof(CalibrationDto))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, PropertyNameCaseInsensitive = false)]
internal sealed partial class ManifestJsonContext : JsonSerializerContext
{
}
