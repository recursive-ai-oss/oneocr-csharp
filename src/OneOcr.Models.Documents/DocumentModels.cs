using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using OneOcr.Documents;

namespace OneOcr.Models.Documents;

/// <summary>One document model the package pins: what it is, which version, and the SHA-256 of its file.</summary>
/// <param name="Name">The architecture, for example <c>docextract-v0</c>.</param>
/// <param name="Role">What the model does; <c>extractor</c> for the invoice, receipt and key-value model.</param>
/// <param name="Version">The weights version.</param>
/// <param name="Bytes">The file size in bytes.</param>
/// <param name="Sha256">The lower-case hexadecimal SHA-256 of the file.</param>
/// <param name="Asset">The file name the weights are published under.</param>
public sealed record BundledDocumentModel(string Name, string Role, string Version, long Bytes, string Sha256, string Asset);

/// <summary>
/// The document extraction weights that ship with the package. The pin (version, size, SHA-256) is known whenever the release has one;
/// the file itself is embedded when the package was built with the pinned file, and is checked against the pin every time it is read.
/// A build without weights reports that plainly instead of failing at load time with a parse error.
/// </summary>
public static class DocumentModels
{
    private static readonly Lazy<BundledDocumentModel?> s_extractor = new(() => ReadPin("extractor"));

    /// <summary>The extraction model this release pins, or <see langword="null"/> when the release pins none.</summary>
    public static BundledDocumentModel? Extractor => s_extractor.Value;

    /// <summary>Whether the extraction model file is embedded in this build of the package.</summary>
    public static bool HasExtractor => Extractor is { } pin && typeof(DocumentModels).Assembly.GetManifestResourceInfo(pin.Asset) is not null;

    /// <summary>Loads the bundled extraction model.</summary>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The extractor.</returns>
    /// <exception cref="OcrModelException">The package carries no weights, or they do not match their pin.</exception>
    public static DocumentExtractor LoadExtractor(OcrLimits? limits = null)
    {
        if (!TryReadExtractor(out byte[] data, out string? problem))
        {
            throw new OcrModelException(problem!);
        }

        return DocumentExtractor.Load(data, limits);
    }

    /// <summary>Loads the bundled extraction model and sets it as <see cref="DocumentExtractor.Default"/>. May be called once.</summary>
    /// <exception cref="OcrModelException">The package carries no weights, or they do not match their pin.</exception>
    /// <exception cref="InvalidOperationException">A default extractor was already configured.</exception>
    public static void Configure() => DocumentExtractor.ConfigureDefault(LoadExtractor());

    /// <summary>Reads the embedded extraction model and checks it against the pin.</summary>
    /// <param name="data">The container bytes when the result is <see langword="true"/>.</param>
    /// <param name="reason">A sentence saying why no model could be returned when the result is <see langword="false"/>; otherwise null.</param>
    /// <returns>True when <paramref name="data"/> holds the model.</returns>
    public static bool TryReadExtractor(out byte[] data, out string? reason)
    {
        data = [];
        reason = null;
        if (Extractor is not { } pin)
        {
            reason = "This release of RecursiveAI.OneOcr.Models.Documents pins no extraction model yet.";
            return false;
        }

        using Stream? stream = typeof(DocumentModels).Assembly.GetManifestResourceStream(pin.Asset);
        if (stream is null)
        {
            reason = $"This build of the package does not contain the extraction model {pin.Asset}; it was built without the weights file.";
            return false;
        }

        if (stream.Length != pin.Bytes)
        {
            reason = string.Create(CultureInfo.InvariantCulture, $"The embedded model is {stream.Length} bytes; the pin says {pin.Bytes}.");
            return false;
        }

        byte[] buffer = new byte[checked((int)pin.Bytes)];
        stream.ReadExactly(buffer);
        string actual = Convert.ToHexStringLower(SHA256.HashData(buffer));
        if (actual != pin.Sha256)
        {
            reason = $"The embedded model hashes to {actual}, but the pin is {pin.Sha256}.";
            return false;
        }

        data = buffer;
        return true;
    }

    private static BundledDocumentModel? ReadPin(string role)
    {
        using Stream stream = typeof(DocumentModels).Assembly.GetManifestResourceStream("models.json")
            ?? throw new InvalidOperationException("The models manifest is missing from the package.");
        using JsonDocument doc = JsonDocument.Parse(stream);
        foreach (JsonElement m in doc.RootElement.GetProperty("models").EnumerateArray())
        {
            if (m.GetProperty("role").GetString() == role)
            {
                return new BundledDocumentModel(
                    m.GetProperty("name").GetString()!, role, m.GetProperty("version").GetString()!,
                    m.GetProperty("bytes").GetInt64(), m.GetProperty("sha256").GetString()!, m.GetProperty("asset").GetString()!);
            }
        }

        return null;
    }
}
