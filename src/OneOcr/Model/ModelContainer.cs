using System.Buffers.Binary;
using System.Text.Json;
using System.Text.Unicode;

namespace OneOcr.Model;

/// <summary>How a line crop is turned into the model's input tensor.</summary>
/// <param name="Height">Pixel height the crop must be scaled to.</param>
/// <param name="Channels">Channel count; 1 (grayscale) in container version 0.</param>
/// <param name="Mean">Subtracted after scaling to 0..1.</param>
/// <param name="Std">Divided by after subtracting the mean.</param>
/// <param name="Invert">Whether intensities are flipped (<c>1 - p/255</c>) before normalizing.</param>
public readonly record struct ModelInput(int Height, int Channels, float Mean, float Std, bool Invert);

/// <summary>Name, shape and location of one tensor in a container.</summary>
/// <param name="Name">The tensor name.</param>
/// <param name="Shape">Dimensions, row-major.</param>
/// <param name="Elements">Product of the dimensions.</param>
public sealed record TensorInfo(string Name, IReadOnlyList<int> Shape, long Elements)
{
    internal long DataOffset { get; init; }
}

/// <summary>
/// A parsed and fully validated model container (versions 0 and 1, see <c>docs/spec/model-container-v1.md</c>). Parsing is
/// strict: every length and offset is checked against the file and against <see cref="OcrLimits"/> before anything is
/// allocated from it, and problems surface as <see cref="OcrModelException"/> or <see cref="OcrLimitException"/>.
/// </summary>
public sealed class ModelContainer
{
    internal const string NoCalibrationNote = "The model file is container version 0, which carries no calibration: calibrated confidence equals raw confidence.";

    private const int HeaderSize = 16;
    private const int Alignment = 64;
    private static ReadOnlySpan<byte> Magic => "ONEOCRM\0"u8;

    private readonly byte[] _data;
    private readonly long _dataStart;
    private readonly Dictionary<string, TensorInfo> _byName;

    private ModelContainer(
        byte[] data, long dataStart, ManifestDto manifest, List<TensorInfo> tensors, Dictionary<string, TensorInfo> byName,
        uint formatVersion, WordCalibration? calibration)
    {
        _data = data;
        _dataStart = dataStart;
        _byName = byName;
        Architecture = manifest.Architecture;
        Name = manifest.Name;
        Version = manifest.Version;
        License = manifest.License;
        Input = new ModelInput(manifest.Input.Height, manifest.Input.Channels, manifest.Input.Mean, manifest.Input.Std, manifest.Input.Invert);
        Charset = manifest.Charset;
        Tensors = tensors;
        ConfigElement = manifest.Config;
        FormatVersion = formatVersion;
        Calibration = calibration;
        Diagnostics = calibration is null
            ? [NoCalibrationNote]
            : [];
    }

    /// <summary>The layer graph the weights belong to, for example <c>crnn-v0</c>.</summary>
    public string Architecture { get; }

    /// <summary>A human-readable model name.</summary>
    public string Name { get; }

    /// <summary>The model's own version string.</summary>
    public string Version { get; }

    /// <summary>The model's licence expression.</summary>
    public string License { get; }

    /// <summary>Input preprocessing parameters.</summary>
    public ModelInput Input { get; }

    /// <summary>Output classes in order; index 0 is the CTC blank.</summary>
    public IReadOnlyList<string> Charset { get; }

    /// <summary>The container format version of the file, 0 or 1.</summary>
    public uint FormatVersion { get; }

    /// <summary>The word-confidence calibration; null for a version 0 file.</summary>
    public WordCalibration? Calibration { get; }

    /// <summary>Notes about the file worth showing to a user, for example that a version 0 file has no calibration.</summary>
    public IReadOnlyList<string> Diagnostics { get; }

    /// <summary>All tensors, in manifest order.</summary>
    public IReadOnlyList<TensorInfo> Tensors { get; }

    internal JsonElement ConfigElement { get; }

    /// <summary>Reads and validates a container from a file.</summary>
    /// <param name="path">The file to read.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The validated container.</returns>
    /// <exception cref="OcrModelException">The file is malformed.</exception>
    /// <exception cref="OcrLimitException">The file exceeds <paramref name="limits"/>.</exception>
    public static ModelContainer Load(string path, OcrLimits? limits = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1, FileOptions.SequentialScan);
        return Load(stream, limits);
    }

    /// <summary>Reads and validates a container from a stream, reading at most <see cref="OcrLimits.MaxModelFileBytes"/> bytes.</summary>
    /// <param name="stream">The stream, positioned at the start of the container.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The validated container.</returns>
    /// <exception cref="OcrModelException">The data is malformed.</exception>
    /// <exception cref="OcrLimitException">The data exceeds <paramref name="limits"/>.</exception>
    public static ModelContainer Load(Stream stream, OcrLimits? limits = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        limits ??= OcrLimits.Default;
        long cap = limits.MaxModelFileBytes;
        byte[] buffer;
        int length;
        if (stream.CanSeek)
        {
            long remaining = stream.Length - stream.Position;
            if (remaining > cap)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxModelFileBytes), cap, remaining);
            }

            buffer = new byte[remaining];
            stream.ReadExactly(buffer);
            length = buffer.Length;
        }
        else
        {
            // Unknown size: grow geometrically, never past the limit.
            buffer = new byte[Math.Min(cap, 1 << 16)];
            length = 0;
            while (true)
            {
                if (length == buffer.Length)
                {
                    if (buffer.Length >= cap)
                    {
                        // Full at the limit: the stream must be at its end, else it is over the limit.
                        if (stream.ReadByte() >= 0)
                        {
                            throw new OcrLimitException(nameof(OcrLimits.MaxModelFileBytes), cap, cap + 1);
                        }

                        break;
                    }

                    Array.Resize(ref buffer, (int)Math.Min(cap, (long)buffer.Length * 2));
                }

                int n = stream.Read(buffer, length, buffer.Length - length);
                if (n == 0)
                {
                    break;
                }

                length += n;
            }
        }

        return Parse(buffer.AsSpan(0, length), limits, buffer.Length == length ? buffer : null);
    }

    /// <summary>Validates a container held in memory. The bytes are copied.</summary>
    /// <param name="data">The whole file.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The validated container.</returns>
    /// <exception cref="OcrModelException">The data is malformed.</exception>
    /// <exception cref="OcrLimitException">The data exceeds <paramref name="limits"/>.</exception>
    public static ModelContainer Parse(ReadOnlySpan<byte> data, OcrLimits? limits = null) =>
        Parse(data, limits ?? OcrLimits.Default, null);

    private static ModelContainer Parse(ReadOnlySpan<byte> data, OcrLimits limits, byte[]? owned)
    {
        if (data.Length > limits.MaxModelFileBytes)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxModelFileBytes), limits.MaxModelFileBytes, data.Length);
        }

        if (data.Length < HeaderSize)
        {
            throw new OcrModelException("The file is shorter than the 16-byte container header.");
        }

        if (!data[..8].SequenceEqual(Magic))
        {
            throw new OcrModelException("Not a model container: the magic bytes are wrong.");
        }

        uint version = BinaryPrimitives.ReadUInt32LittleEndian(data[8..]);
        if (version > 1)
        {
            throw new OcrModelException($"Unsupported container version {version}; this reader understands versions 0 and 1.");
        }

        uint manifestLength = BinaryPrimitives.ReadUInt32LittleEndian(data[12..]);
        if (manifestLength > limits.MaxManifestBytes)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxManifestBytes), limits.MaxManifestBytes, manifestLength);
        }

        long manifestEnd = HeaderSize + (long)manifestLength;
        if (manifestEnd > data.Length)
        {
            throw new OcrModelException("The manifest extends past the end of the file.");
        }

        long dataStart = AlignUp(manifestEnd);
        if (dataStart > data.Length)
        {
            throw new OcrModelException("The file ends inside the padding after the manifest.");
        }

        RequireZero(data[(int)manifestEnd..(int)dataStart], "the padding after the manifest");

        ManifestDto manifest = ParseManifest(data.Slice(HeaderSize, (int)manifestLength), limits);
        long dataLength = data.Length - dataStart;
        List<TensorInfo> tensors = ValidateManifest(manifest, limits, dataLength, out Dictionary<string, TensorInfo> byName);
        CheckLayout(data, dataStart, tensors);
        WordCalibration? calibration = null;
        if (version == 1)
        {
            calibration = ParseCalibration(manifest, limits);
        }

        byte[] copy = owned ?? data.ToArray();
        return new ModelContainer(copy, dataStart, manifest, tensors, byName, version, calibration);
    }

    /// <summary>Whether the container has a tensor with this name.</summary>
    /// <param name="name">The tensor name.</param>
    /// <returns><see langword="true"/> when present.</returns>
    public bool HasTensor(string name) => _byName.ContainsKey(name);

    /// <summary>Finds a tensor by name.</summary>
    /// <param name="name">The tensor name.</param>
    /// <returns>The tensor, or <see langword="null"/> when absent.</returns>
    public TensorInfo? FindTensor(string name) => _byName.GetValueOrDefault(name);

    /// <summary>Widens a tensor's fp16 values to fp32 (exact).</summary>
    /// <param name="name">The tensor name.</param>
    /// <returns>The values, row-major.</returns>
    /// <exception cref="OcrModelException">There is no tensor with that name.</exception>
    public float[] ReadTensor(string name)
    {
        if (!_byName.TryGetValue(name, out TensorInfo? info))
        {
            throw new OcrModelException($"The model has no tensor named '{name}'.");
        }

        var result = new float[info.Elements];
        ReadOnlySpan<byte> bytes = _data.AsSpan((int)(_dataStart + info.DataOffset), checked((int)info.Elements * 2));
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(bytes[(i * 2)..]);
        }

        return result;
    }

    private static long AlignUp(long value) => (value + Alignment - 1) / Alignment * Alignment;

    private static void RequireZero(ReadOnlySpan<byte> bytes, string where)
    {
        foreach (byte b in bytes)
        {
            if (b != 0)
            {
                throw new OcrModelException($"Non-zero bytes in {where}.");
            }
        }
    }

    private static ManifestDto ParseManifest(ReadOnlySpan<byte> json, OcrLimits limits)
    {
        if (json.StartsWith(new byte[] { 0xEF, 0xBB, 0xBF }))
        {
            throw new OcrModelException("The manifest starts with a byte order mark.");
        }

        if (!Utf8.IsValid(json))
        {
            throw new OcrModelException("The manifest is not valid UTF-8.");
        }

        var options = new JsonSerializerOptions
        {
            MaxDepth = limits.MaxJsonDepth,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            AllowDuplicateProperties = false,
            AllowTrailingCommas = false,
            ReadCommentHandling = JsonCommentHandling.Disallow,
        };
        try
        {
            ManifestDto? manifest = JsonSerializer.Deserialize(json, new ManifestJsonContext(options).ManifestDto);
            return manifest ?? throw new OcrModelException("The manifest is JSON null.");
        }
        catch (JsonException ex)
        {
            throw new OcrModelException("The manifest is not valid: " + ex.Message, ex);
        }
    }

    private static WordCalibration ParseCalibration(ManifestDto manifest, OcrLimits limits)
    {
        if (manifest.Calibration is not { ValueKind: JsonValueKind.Object } element)
        {
            throw new OcrModelException("A version 1 manifest needs a 'calibration' object.");
        }

        try
        {
            CalibrationDto dto = element.Deserialize(ManifestJsonContext.Default.CalibrationDto)
                ?? throw new OcrModelException("The calibration is JSON null.");
            return WordCalibration.FromDto(dto, limits);
        }
        catch (JsonException ex)
        {
            throw new OcrModelException("The calibration is not valid: " + ex.Message, ex);
        }
    }

    private static List<TensorInfo> ValidateManifest(
        ManifestDto m, OcrLimits limits, long dataLength, out Dictionary<string, TensorInfo> byName)
    {
        if (m.Input.Height < 1 || m.Input.Channels < 1)
        {
            throw new OcrModelException("The input height and channel count must be positive.");
        }

        if (m.Input.Height > limits.MaxInputHeight)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxInputHeight), limits.MaxInputHeight, m.Input.Height);
        }

        if (!float.IsFinite(m.Input.Mean) || !float.IsFinite(m.Input.Std) || m.Input.Std <= 0)
        {
            throw new OcrModelException("The input mean must be finite and the standard deviation finite and positive.");
        }

        if (m.Charset.Count == 0)
        {
            throw new OcrModelException("The charset is empty.");
        }

        if (m.Charset.Count > limits.MaxCharsetEntries)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxCharsetEntries), limits.MaxCharsetEntries, m.Charset.Count);
        }

        var seen = new HashSet<string>(m.Charset.Count, StringComparer.Ordinal);
        for (int i = 0; i < m.Charset.Count; i++)
        {
            string entry = m.Charset[i];
            if (entry is null)
            {
                throw new OcrModelException($"Charset entry {i} is null.");
            }

            if (entry.Length == 0 && i > 0)
            {
                throw new OcrModelException($"Charset entry {i} is empty.");
            }

            if (entry.Length > limits.MaxCharsetEntryLength)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxCharsetEntryLength), limits.MaxCharsetEntryLength, entry.Length);
            }

            if (!seen.Add(entry))
            {
                throw new OcrModelException($"Charset entry {i} duplicates an earlier entry.");
            }
        }

        if (m.Tensors.Count > limits.MaxTensorCount)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxTensorCount), limits.MaxTensorCount, m.Tensors.Count);
        }

        var result = new List<TensorInfo>(m.Tensors.Count);
        byName = new Dictionary<string, TensorInfo>(m.Tensors.Count, StringComparer.Ordinal);
        foreach (TensorDto t in m.Tensors)
        {
            if (t is null || t.Name is null)
            {
                throw new OcrModelException("A tensor entry is null or has no name.");
            }

            if (t.Name.Length == 0)
            {
                throw new OcrModelException("A tensor has an empty name.");
            }

            if (t.Name.Length > limits.MaxTensorNameLength)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxTensorNameLength), limits.MaxTensorNameLength, t.Name.Length);
            }

            if (t.Dtype != "f16")
            {
                throw new OcrModelException($"Tensor '{t.Name}' has dtype '{t.Dtype}'; version 0 allows only f16.");
            }

            if (t.Shape.Count == 0)
            {
                throw new OcrModelException($"Tensor '{t.Name}' has no dimensions.");
            }

            if (t.Shape.Count > limits.MaxTensorRank)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxTensorRank), limits.MaxTensorRank, t.Shape.Count);
            }

            long elements = 1;
            var shape = new int[t.Shape.Count];
            for (int i = 0; i < shape.Length; i++)
            {
                long d = t.Shape[i];
                if (d < 1 || d > int.MaxValue)
                {
                    throw new OcrModelException($"Tensor '{t.Name}' has dimension {d}; dimensions must be positive.");
                }

                shape[i] = (int)d;
                // Both factors are within 2^31 and the running product within the limit, so the product fits in a long.
                elements *= d;
                if (elements > limits.MaxTensorElements)
                {
                    throw new OcrLimitException(nameof(OcrLimits.MaxTensorElements), limits.MaxTensorElements, elements);
                }
            }

            if (t.Length != elements * 2)
            {
                throw new OcrModelException($"Tensor '{t.Name}': length {t.Length} does not match {elements} f16 elements.");
            }

            if (t.Offset < 0 || t.Offset % Alignment != 0)
            {
                throw new OcrModelException($"Tensor '{t.Name}': offset {t.Offset} is negative or not a multiple of {Alignment}.");
            }

            if (t.Offset > dataLength || t.Length > dataLength - t.Offset)
            {
                throw new OcrModelException($"Tensor '{t.Name}' extends past the end of the file.");
            }

            var info = new TensorInfo(t.Name, shape, elements) { DataOffset = t.Offset };
            if (!byName.TryAdd(t.Name, info))
            {
                throw new OcrModelException($"Duplicate tensor name '{t.Name}'.");
            }

            result.Add(info);
        }

        return result;
    }

    // Tensors must not overlap, the gaps between them must be zero, and the file ends where the last tensor ends.
    private static void CheckLayout(ReadOnlySpan<byte> data, long dataStart, List<TensorInfo> tensors)
    {
        var order = tensors.ToArray();
        Array.Sort(order, (a, b) => a.DataOffset.CompareTo(b.DataOffset));
        long cursor = 0;
        foreach (TensorInfo t in order)
        {
            if (t.DataOffset < cursor)
            {
                throw new OcrModelException($"Tensor '{t.Name}' overlaps another tensor.");
            }

            RequireZero(data[(int)(dataStart + cursor)..(int)(dataStart + t.DataOffset)], "the padding between tensors");
            cursor = t.DataOffset + t.Elements * 2;
        }

        if (dataStart + cursor != data.Length)
        {
            throw new OcrModelException("The file has bytes after the last tensor.");
        }
    }

}
