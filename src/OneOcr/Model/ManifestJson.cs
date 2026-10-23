using System.Text.Json;
using System.Text.Json.Serialization;

namespace OneOcr.Model;

// Wire shapes of the container manifest (docs/spec/model-container-v0.md). Members the spec does not name are skipped,
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

[JsonSerializable(typeof(ManifestDto))]
[JsonSerializable(typeof(CrnnConfigDto))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, PropertyNameCaseInsensitive = false)]
internal sealed partial class ManifestJsonContext : JsonSerializerContext
{
}
