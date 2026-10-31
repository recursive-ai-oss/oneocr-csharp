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
    private static readonly Lazy<BundledModel> s_languageModel = new(() => ReadPin("language-model"));
    private static readonly Lazy<BundledModel> s_wordFrequencies = new(() => ReadPin("word-frequencies"));

    /// <summary>The recognizer this release pins.</summary>
    public static BundledModel Recognizer => s_recognizer.Value;

    /// <summary>The page detector this release pins.</summary>
    public static BundledModel Detector => s_detector.Value;

    /// <summary>The character language model this release pins; the engine uses it to decide where the spaces between words go.</summary>
    public static BundledModel LanguageModel => s_languageModel.Value;

    /// <summary>The word frequency table this release pins; the engine uses it to take back spaces the network put inside words.</summary>
    public static BundledModel WordFrequencies => s_wordFrequencies.Value;

    /// <summary>Whether the recognizer file is embedded in this build of the package.</summary>
    public static bool HasRecognizer => typeof(BundledModels).Assembly.GetManifestResourceInfo(Recognizer.Asset) is not null;

    /// <summary>Whether the detector file is embedded in this build of the package.</summary>
    public static bool HasDetector => typeof(BundledModels).Assembly.GetManifestResourceInfo(Detector.Asset) is not null;

    /// <summary>Whether the language model file is embedded in this build of the package.</summary>
    public static bool HasLanguageModel => typeof(BundledModels).Assembly.GetManifestResourceInfo(LanguageModel.Asset) is not null;

    /// <summary>Whether the word frequency file is embedded in this build of the package.</summary>
    public static bool HasWordFrequencies => typeof(BundledModels).Assembly.GetManifestResourceInfo(WordFrequencies.Asset) is not null;

    /// <summary>Reads the embedded word frequency table and checks it against the pin.</summary>
    /// <param name="data">The file bytes when the result is <see langword="true"/>.</param>
    /// <param name="reason">A sentence saying why no table could be returned when the result is <see langword="false"/>; otherwise null.</param>
    /// <returns>True when <paramref name="data"/> holds the table.</returns>
    public static bool TryReadWordFrequencies(out byte[] data, out string? reason) => TryRead(WordFrequencies, out data, out reason);

    /// <summary>Reads the embedded language model and checks it against the pin.</summary>
    /// <param name="data">The model bytes when the result is <see langword="true"/>.</param>
    /// <param name="reason">A sentence saying why no model could be returned when the result is <see langword="false"/>; otherwise null.</param>
    /// <returns>True when <paramref name="data"/> holds the model.</returns>
    public static bool TryReadLanguageModel(out byte[] data, out string? reason) => TryRead(LanguageModel, out data, out reason);

    /// <summary>Reads the embedded recognizer and checks it against the pin.</summary>
    /// <param name="data">The container bytes when the result is <see langword="true"/>.</param>
    /// <param name="reason">A sentence saying why no model could be returned when the result is <see langword="false"/>; otherwise null.</param>
    /// <returns>True when <paramref name="data"/> holds the model.</returns>
    public static bool TryReadRecognizer(out byte[] data, out string? reason) => TryRead(Recognizer, out data, out reason);

    /// <summary>Reads the embedded page detector and checks it against the pin.</summary>
    /// <param name="data">The container bytes when the result is <see langword="true"/>.</param>
    /// <param name="reason">A sentence saying why no model could be returned when the result is <see langword="false"/>; otherwise null.</param>
    /// <returns>True when <paramref name="data"/> holds the model.</returns>
    public static bool TryReadDetector(out byte[] data, out string? reason) => TryRead(Detector, out data, out reason);

    /// <summary>Reads the embedded recognizer and checks it against the pin.</summary>
    /// <returns>The model container bytes.</returns>
    /// <exception cref="InvalidOperationException">This build of the package carries no recognizer file, or it does not match its pin.</exception>
    public static byte[] ReadRecognizer() =>
        TryRead(Recognizer, out byte[] data, out string? reason) ? data : throw new InvalidOperationException(reason);

    /// <summary>
    /// Writes the embedded recognizer to a file: the starting model for fine-tuning (<c>OcrTrainerOptions.ModelPath</c>), or a file to
    /// inspect. The bytes are checked against the pin first.
    /// </summary>
    /// <param name="path">The file to create; an existing file is replaced.</param>
    /// <exception cref="ArgumentNullException"><paramref name="path"/> is null.</exception>
    /// <exception cref="InvalidOperationException">This build of the package carries no recognizer file, or it does not match its pin.</exception>
    public static void ExportRecognizer(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        File.WriteAllBytes(path, ReadRecognizer());
    }

    private static bool TryRead(BundledModel pin, out byte[] data, out string? reason)
    {
        data = [];
        reason = null;
        Assembly assembly = typeof(BundledModels).Assembly;
        using Stream? stream = assembly.GetManifestResourceStream(pin.Asset);
        if (stream is null)
        {
            reason = $"This build of the models package carries no {pin.Role} weights; it was built without the pinned model file.";
            return false;
        }

        // The pin names the size, so a resource of another size is rejected before it is buffered.
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
