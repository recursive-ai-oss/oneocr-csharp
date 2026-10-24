using System.Text;
using System.Text.Json;
using OneImage;

namespace OneOcr.Training;

/// <summary>What <see cref="LineReview.Export"/> wrote.</summary>
/// <param name="Items">Lines in the review file.</param>
/// <param name="ReviewFile">The JSON file to edit: set each item's <c>text</c> to the line as it should be read.</param>
/// <param name="CropsDirectory">The folder of line images the items point to.</param>
public sealed record ReviewExport(int Items, string ReviewFile, string CropsDirectory);

/// <summary>What <see cref="LineReview.Import(string, string, OcrLimits?)"/> did with a review file.</summary>
/// <param name="Accepted">Corrections stored (new or replacing an earlier correction of the same line).</param>
/// <param name="Replaced">Of those, corrections that replaced an earlier one.</param>
/// <param name="UnknownIds">Corrections whose id names no line of this work directory; they were ignored.</param>
/// <param name="Unreviewed">Items in the file without a <c>text</c>; they stay in the queue.</param>
/// <param name="Diagnostics">Notes, one per problem (at most 20 ids are listed).</param>
public sealed record ReviewImportResult(int Accepted, int Replaced, int UnknownIds, int Unreviewed, IReadOnlyList<string> Diagnostics);

/// <summary>
/// The review queue as a data API: the lines the local reader and the language model could not agree on, exported as JSON plus
/// a picture of each line, and the corrections imported back. There is no user interface; edit the file or build one on this.
/// A corrected line trains the next run, and it is also the reference text the held-out report uses for that line.
/// </summary>
public static class LineReview
{
    private const int MaxListedIds = 20;

    /// <summary>
    /// Writes <c>review.json</c> and a <c>crops/</c> folder of PNG line images into <paramref name="outputDirectory"/>: one item for
    /// each labelled line the two readings disagreed on or could not be matched, except lines already corrected (unless
    /// <paramref name="includeReviewed"/>). Items carry <c>id</c>, <c>document</c>, <c>box</c> (x, y, width, height on the page),
    /// <c>local</c> (the local reading), <c>llm</c> (the language model's matching text, or null), <c>state</c>, <c>confidence</c>,
    /// <c>crop</c> and <c>text</c> (null until reviewed). Existing files of the same names are overwritten.
    /// </summary>
    /// <param name="workDirectory">The run's work directory.</param>
    /// <param name="outputDirectory">Where to write; created if missing.</param>
    /// <param name="includeReviewed">Also export lines that already have a correction, with it filled in.</param>
    /// <param name="limits">Size limits; null means the defaults.</param>
    /// <returns>What was written.</returns>
    /// <exception cref="OcrFormatException">A work file is damaged.</exception>
    public static ReviewExport Export(string workDirectory, string outputDirectory, bool includeReviewed = false, OcrLimits? limits = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDirectory);
        limits ??= OcrLimits.Default;
        var store = new WorkStore(workDirectory, limits);
        Dictionary<string, string> reviewed = store.LoadReviewed();
        string cropsDir = Path.Combine(outputDirectory, "crops");
        Directory.CreateDirectory(cropsDir);

        using var ms = new MemoryStream();
        int count = 0;
        using (var w = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = true }))
        {
            w.WriteStartObject();
            w.WriteString("format", ReviewImport.Format);
            w.WriteNumber("version", 1);
            w.WriteStartArray("items");
            foreach (DocLabelsDto doc in store.LoadAll())
            {
                foreach (LineLabelDto line in doc.Lines.Where(static l => l.State != LineAgreement.Agreed))
                {
                    string id = WorkStore.LineId(doc.Id, line.Index);
                    bool isReviewed = reviewed.TryGetValue(id, out string? text);
                    if (isReviewed && !includeReviewed)
                    {
                        continue;
                    }

                    var pixels = new Gray8[line.Crop.Length];
                    for (int i = 0; i < pixels.Length; i++)
                    {
                        pixels[i] = new Gray8(line.Crop[i]);
                    }

                    string cropName = id + ".png";
                    Image.FromPixels<Gray8>(pixels, line.CropWidth, doc.Height).Save(Path.Combine(cropsDir, cropName));

                    w.WriteStartObject();
                    w.WriteString("id", id);
                    w.WriteString("document", doc.Document);
                    w.WriteStartArray("box");
                    w.WriteNumberValue(line.X);
                    w.WriteNumberValue(line.Y);
                    w.WriteNumberValue(line.Width);
                    w.WriteNumberValue(line.Height);
                    w.WriteEndArray();
                    w.WriteString("local", line.Local);
                    if (line.Llm is null)
                    {
                        w.WriteNull("llm");
                    }
                    else
                    {
                        w.WriteString("llm", line.Llm);
                    }

                    w.WriteString("state", line.State == LineAgreement.Disagreed ? "disagreed" : "unaligned");
                    w.WriteNumber("confidence", line.Confidence);
                    w.WriteString("crop", "crops/" + cropName);
                    if (isReviewed)
                    {
                        w.WriteString("text", text);
                    }
                    else
                    {
                        w.WriteNull("text");
                    }

                    w.WriteEndObject();
                    count++;
                }
            }

            w.WriteEndArray();
            w.WriteEndObject();
        }

        string file = Path.Combine(outputDirectory, "review.json");
        WorkStore.WriteAtomic(file, ms.ToArray());
        return new ReviewExport(count, file, cropsDir);
    }

    /// <summary>
    /// Reads a review file (the one <see cref="Export"/> wrote, with <c>text</c> filled in for the lines you checked) and stores its
    /// corrections in the work directory. The file is untrusted input: its size, item count, nesting and text length are checked
    /// against <paramref name="limits"/> before use, an item's id is only ever matched against the work directory's own lines, and
    /// no other member of an item is used. An item without <c>text</c> (or with <c>null</c>) stays in the queue; an empty
    /// <c>text</c> says the line has no text. The text is stored with whitespace normalized. Never touches the network.
    /// </summary>
    /// <param name="workDirectory">The run's work directory.</param>
    /// <param name="reviewFile">The file to read.</param>
    /// <param name="limits">Size limits; null means the defaults.</param>
    /// <returns>What was accepted and what was ignored.</returns>
    /// <exception cref="OcrLimitException">The file or its content is over <see cref="OcrLimits.MaxReviewImportBytes"/>, <see cref="OcrLimits.MaxReviewItems"/> or <see cref="OcrLimits.MaxReviewTextLength"/>.</exception>
    /// <exception cref="OcrFormatException">The file is not a review file.</exception>
    public static ReviewImportResult Import(string workDirectory, string reviewFile, OcrLimits? limits = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(reviewFile);
        limits ??= OcrLimits.Default;
        using var stream = new FileStream(reviewFile, FileMode.Open, FileAccess.Read, FileShare.Read, 1, FileOptions.SequentialScan);
        if (stream.Length > limits.MaxReviewImportBytes)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxReviewImportBytes), limits.MaxReviewImportBytes, stream.Length);
        }

        byte[] bytes = new byte[stream.Length];
        stream.ReadExactly(bytes);
        return Import(workDirectory, bytes, limits);
    }

    /// <summary>Like <see cref="Import(string, string, OcrLimits?)"/> for a review file already in memory.</summary>
    /// <param name="workDirectory">The run's work directory.</param>
    /// <param name="json">The file's bytes (UTF-8 JSON).</param>
    /// <param name="limits">Size limits; null means the defaults.</param>
    /// <returns>What was accepted and what was ignored.</returns>
    /// <exception cref="OcrLimitException">The data or its content is over a review limit.</exception>
    /// <exception cref="OcrFormatException">The data is not a review file.</exception>
    public static ReviewImportResult Import(string workDirectory, ReadOnlySpan<byte> json, OcrLimits? limits = null)
    {
        limits ??= OcrLimits.Default;
        ReviewFile file = ReviewImport.Parse(json, limits);
        var store = new WorkStore(workDirectory, limits);
        var known = new HashSet<string>(StringComparer.Ordinal);
        foreach (DocLabelsDto doc in store.LoadAll())
        {
            foreach (LineLabelDto l in doc.Lines)
            {
                known.Add(WorkStore.LineId(doc.Id, l.Index));
            }
        }

        Dictionary<string, string> reviewed = store.LoadReviewed();
        int accepted = 0, replaced = 0, unknown = 0;
        var notes = new List<string>();
        foreach (ReviewCorrection c in file.Corrections)
        {
            if (!known.Contains(c.Id))
            {
                unknown++;
                if (unknown <= MaxListedIds)
                {
                    notes.Add($"'{c.Id}' is not a line of this work directory; ignored.");
                }

                continue;
            }

            if (reviewed.ContainsKey(c.Id))
            {
                replaced++;
            }

            reviewed[c.Id] = TextMetrics.Normalize(c.Text);
            accepted++;
        }

        if (accepted > 0)
        {
            store.SaveReviewed(reviewed);
        }

        return new ReviewImportResult(accepted, replaced, unknown, file.Unreviewed, notes);
    }
}
