using System.Buffers.Binary;
using OneOcr.Training;

namespace OneOcr.Fuzz;

/// <summary>
/// The files a training work directory holds and reads back on resume: label files, the reviewed-lines file and the checkpoint.
/// The first byte picks the file, the rest is its content. Only <see cref="OcrFormatException"/> and
/// <see cref="OcrLimitException"/> may come out (a checkpoint of another run is the one documented <see cref="InvalidOperationException"/>).
/// </summary>
internal static class TrainingFilesTarget
{
    private static readonly OcrLimits Limits = new()
    {
        MaxModelFileBytes = 1 << 16, MaxLinesPerPage = 64, MaxLineWidth = 256, MaxInputHeight = 64, MaxJsonDepth = 8, MaxWordsPerPage = 64,
    };

    private static readonly int[] TensorLengths = [6, 3];
    private static readonly byte[] Fingerprint = Checkpoint.Fingerprint(["fuzz"]);

    private static readonly string Dir = Path.Combine(Path.GetTempPath(), "oneocr-fuzz-train-" + Environment.ProcessId);

    /// <summary>Inputs that were accepted (progress statistic for local runs).</summary>
    public static long Parsed;

    public static void Run(ReadOnlySpan<byte> data)
    {
        if (data.Length == 0)
        {
            return;
        }

        Directory.CreateDirectory(Dir);
        byte[] body = data[1..].ToArray();
        try
        {
            switch (data[0] & 3)
            {
                case 0:
                    Labels(body);
                    break;
                case 1:
                    Reviewed(body);
                    break;
                case 2:
                    Resume(body);
                    break;
                default:
                    ExtractionLabels(body);
                    break;
            }
        }
        catch (OcrFormatException)
        {
        }
        catch (OcrLimitException)
        {
        }
    }

    private static void Labels(byte[] body)
    {
        var store = new WorkStore(Dir, Limits);
        string file = Path.Combine(Dir, "labels", "doc.json");
        File.WriteAllBytes(file, body);
        try
        {
            store.LoadAll();
            Parsed++;
        }
        finally
        {
            File.Delete(file);
        }
    }

    private static void Reviewed(byte[] body)
    {
        var store = new WorkStore(Dir, Limits);
        File.WriteAllBytes(store.ReviewedPath, body);
        try
        {
            store.LoadReviewed();
            Parsed++;
        }
        finally
        {
            File.Delete(store.ReviewedPath);
        }
    }

    private static void Resume(byte[] body)
    {
        string path = Path.Combine(Dir, "checkpoint.bin");
        File.WriteAllBytes(path, body);
        try
        {
            _ = Checkpoint.TryLoad(path, Fingerprint, TensorLengths);
            Parsed++;
        }
        catch (InvalidOperationException)
        {
            // Documented: the checkpoint belongs to a different run.
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static void ExtractionLabels(byte[] body)
    {
        string dir = Path.Combine(Dir, "extract");
        Directory.CreateDirectory(dir);
        string file = Path.Combine(dir, "doc.json");
        File.WriteAllBytes(file, body);
        try
        {
            _ = ExtractionTrainer.ReadLabelFile(file, Limits, out _);
            Parsed++;
        }
        finally
        {
            File.Delete(file);
        }
    }

    public static void WriteSeeds(string dir)
    {
        Directory.CreateDirectory(dir);
        void W(string name, int kind, byte[] content) => File.WriteAllBytes(Path.Combine(dir, name + ".seed"), [(byte)kind, .. content]);
        W("labels", 0, System.Text.Encoding.UTF8.GetBytes(
            "{\"format\":\"oneocr.labels\",\"version\":1,\"id\":\"doc\",\"document\":\"scan.png\",\"height\":2,\"costUsd\":0.01,\"calls\":1,\"lines\":"
            + "[{\"index\":0,\"x\":1,\"y\":2,\"width\":3,\"height\":2,\"local\":\"Tota1\",\"confidence\":0.5,\"llm\":\"Total\",\"state\":0,\"cer\":0.2,\"cropWidth\":3,\"crop\":\"AAECAwQF\"}]}"));
        W("reviewed", 1, System.Text.Encoding.UTF8.GetBytes("{\"format\":\"oneocr.reviewed\",\"version\":1,\"items\":[{\"id\":\"doc-0\",\"text\":\"Total \\u00e9\"}]}"));
        W("extract", 3, System.Text.Encoding.UTF8.GetBytes(
            "{\"format\":\"oneocr.extraction-labels\",\"version\":1,\"id\":\"doc\",\"document\":\"a.png\",\"width\":100,\"height\":50,\"usable\":true,"
            + "\"words\":[\"Total\",\"12.50\"],\"boxes\":[1,2,30,12,40,2,70,12],\"fields\":[{\"name\":\"total\",\"value\":\"12.50\",\"words\":[1]}],\"notes\":[]}"));

        var cp = new byte[8 + 4 + 32 + 8 + 8 + 8 + 4 + (TensorLengths.Length * 4) + (3 * 9 * 4)];
        "ONEOCRC\0"u8.CopyTo(cp);
        BinaryPrimitives.WriteUInt32LittleEndian(cp.AsSpan(8), 1);
        Fingerprint.CopyTo(cp, 12);
        BinaryPrimitives.WriteInt32LittleEndian(cp.AsSpan(68), TensorLengths.Length);
        int pos = 72;
        foreach (int n in TensorLengths)
        {
            BinaryPrimitives.WriteInt32LittleEndian(cp.AsSpan(pos), n);
            pos += 4 + (3 * n * 4);
        }

        W("checkpoint", 2, cp);
    }
}
