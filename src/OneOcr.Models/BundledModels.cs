using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;

namespace OneOcr.Models;

/// <summary>One model the package pins: what it is, which version, and the SHA-256 of its file.</summary>
/// <param name="Name">The architecture, for example <c>crnn-v0</c>.</param>
/// <param name="Role">What the model does, for example <c>recognizer</c>.</param>
/// <param name="Version">The weights version.</param>
/// <param name="Bytes">The file size in bytes.</param>
/// <param name="Sha256">The lower-case hexadecimal SHA-256 of the file.</param>
/// <param name="Asset">The file name the weights are published under.</param>
public sealed record BundledModel(string Name, string Role, string Version, long Bytes, string Sha256, string Asset);

/// <summary>
/// The weights that ship with the package. The pin (version, size, SHA-256) is always known; the file itself is embedded when the
/// package was built with the pinned file, and is checked against the pin every time it is read.
/// </summary>
public static class BundledModels
{
    private static readonly Lazy<BundledModel> s_recognizer = new(() => ReadPin("recognizer"));
    private static readonly Lazy<BundledModel> s_detector = new(() => ReadPin("detector"));

    /// <summary>The recognizer this release pins.</summary>
    public static BundledModel Recognizer => s_recognizer.Value;

    /// <summary>The page detector this release pins.</summary>
    public static BundledModel Detector => s_detector.Value;

    /// <summary>Whether the recognizer file is embedded in this build of the package.</summary>
    public static bool HasRecognizer => typeof(BundledModels).Assembly.GetManifestResourceInfo(Recognizer.Asset) is not null;

    /// <summary>Whether the detector file is embedded in this build of the package.</summary>
    public static bool HasDetector => typeof(BundledModels).Assembly.GetManifestResourceInfo(Detector.Asset) is not null;

    /// <summary>Reads the embedded recognizer and checks it against the pin.</summary>
    /// <param name="data">The container bytes when the result is <see langword="null"/>.</param>
    /// <returns><see langword="null"/> on success; otherwise a sentence saying why no model could be returned.</returns>
    public static string? TryReadRecognizer(out byte[] data) => TryRead(Recognizer, out data);

    /// <summary>Reads the embedded page detector and checks it against the pin.</summary>
    /// <param name="data">The container bytes when the result is <see langword="null"/>.</param>
    /// <returns><see langword="null"/> on success; otherwise a sentence saying why no model could be returned.</returns>
    public static string? TryReadDetector(out byte[] data) => TryRead(Detector, out data);

    private static string? TryRead(BundledModel pin, out byte[] data)
    {
        data = [];
        Assembly assembly = typeof(BundledModels).Assembly;
        using Stream? stream = assembly.GetManifestResourceStream(pin.Asset);
        if (stream is null)
        {
            return $"This build of the models package carries no {pin.Role} weights; it was built without the pinned model file.";
        }

        // The pin names the size, so a resource of another size is rejected before it is buffered.
        if (stream.Length != pin.Bytes)
        {
            return string.Create(CultureInfo.InvariantCulture, $"The embedded model is {stream.Length} bytes; the pin says {pin.Bytes}.");
        }

        byte[] buffer = new byte[checked((int)pin.Bytes)];
        stream.ReadExactly(buffer);
        string actual = Convert.ToHexStringLower(SHA256.HashData(buffer));
        if (actual != pin.Sha256)
        {
            return $"The embedded model hashes to {actual}, but the pin is {pin.Sha256}.";
        }

        data = buffer;
        return null;
    }

    private static BundledModel ReadPin(string role)
    {
        using Stream stream = typeof(BundledModels).Assembly.GetManifestResourceStream("models.json")
            ?? throw new InvalidOperationException("The models manifest is missing from the package.");
        using JsonDocument doc = JsonDocument.Parse(stream);
        foreach (JsonElement m in doc.RootElement.GetProperty("models").EnumerateArray())
        {
            if (m.GetProperty("role").GetString() == role)
            {
                return new BundledModel(
                    m.GetProperty("name").GetString()!, role, m.GetProperty("version").GetString()!,
                    m.GetProperty("bytes").GetInt64(), m.GetProperty("sha256").GetString()!, m.GetProperty("asset").GetString()!);
            }
        }

        throw new InvalidOperationException($"The models manifest pins no {role}.");
    }
}
