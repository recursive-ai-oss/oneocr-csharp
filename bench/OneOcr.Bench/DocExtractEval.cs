using System.Globalization;
using System.Text;
using System.Text.Json;
using OneOcr;
using OneOcr.Documents;

/// <summary>
/// <c>OneOcr.Bench docextract-eval --model FILE --pages DIR --out FILE.jsonl [--degree N]</c>: reads each <c>*.page.json</c> in DIR,
/// runs the document extractor and writes what it assembled (document type, header fields, line items, tax rows, key-value pairs, with
/// word indices) as one JSON line per page. The training project scores these lines with the same metric code as its own run, so the two
/// implementations are compared on the same words.
/// </summary>
internal static class DocExtractEval
{
    public static int Run(Dictionary<string, string> o)
    {
        if (!o.TryGetValue("model", out string? model) || !o.TryGetValue("pages", out string? dir) || !o.TryGetValue("out", out string? output))
        {
            Console.Error.WriteLine("docextract-eval needs --model FILE --pages DIR --out FILE");
            return 2;
        }

        int degree = o.TryGetValue("degree", out string? d) ? int.Parse(d, CultureInfo.InvariantCulture) : 1;
        DocumentExtractor extractor = DocumentExtractor.Load(model);
        string[] files = Directory.GetFiles(dir, "*.page.json");
        Array.Sort(files, StringComparer.Ordinal);
        using var writer = new StreamWriter(output, false, new UTF8Encoding(false));
        int pages = 0;
        foreach (string file in files)
        {
            string id = Path.GetFileName(file)[..^".page.json".Length];
            OcrPage page = OcrPageJson.Read(File.ReadAllBytes(file))[0];
            DocumentExtraction ex = extractor.Extract(page, new DocumentExtractionOptions { MaxDegreeOfParallelism = degree });
            using var ms = new MemoryStream();
            using (var w = new Utf8JsonWriter(ms))
            {
                w.WriteStartObject();
                w.WriteString("id", id);
                w.WriteNumber("words", page.Words.Count);
                w.WriteString("documentType", ex.DocumentType.ToString().ToLowerInvariant());
                w.WritePropertyName("fields");
                WriteFields(w, ex.Fields);
                w.WritePropertyName("lineItems");
                WriteGroups(w, ex.LineItems);
                w.WritePropertyName("taxBreakdown");
                WriteGroups(w, ex.TaxRows);
                w.WritePropertyName("keyValues");
                w.WriteStartArray();
                foreach (OcrKeyValue kv in ex.KeyValues)
                {
                    w.WriteStartObject();
                    WriteSide(w, "key", kv.KeyWordIndices, page);
                    WriteSide(w, "value", kv.ValueWordIndices, page);
                    w.WriteEndObject();
                }

                w.WriteEndArray();
                w.WriteEndObject();
            }

            writer.Write(Encoding.UTF8.GetString(ms.ToArray()));
            writer.Write('\n');
            pages++;
        }

        Console.WriteLine($"{pages} pages -> {output}");
        return 0;
    }

    private static void WriteSide(Utf8JsonWriter w, string name, IReadOnlyList<int> words, OcrPage page)
    {
        w.WritePropertyName(name);
        w.WriteStartObject();
        w.WriteString("text", string.Join(' ', words.Select(i => page.Words[i].Text)));
        w.WritePropertyName("words");
        w.WriteStartArray();
        foreach (int i in words)
        {
            w.WriteNumberValue(i);
        }

        w.WriteEndArray();
        w.WriteEndObject();
    }

    private static void WriteGroups(Utf8JsonWriter w, IReadOnlyList<ExtractedGroup> groups)
    {
        w.WriteStartArray();
        foreach (ExtractedGroup g in groups)
        {
            w.WriteStartObject();
            w.WritePropertyName("fields");
            WriteFields(w, g.Fields);
            w.WriteEndObject();
        }

        w.WriteEndArray();
    }

    private static void WriteFields(Utf8JsonWriter w, IReadOnlyList<ExtractedField> fields)
    {
        w.WriteStartArray();
        foreach (ExtractedField f in fields)
        {
            w.WriteStartObject();
            w.WriteString("id", f.Id);
            w.WriteString("text", f.Text);
            w.WritePropertyName("words");
            w.WriteStartArray();
            foreach (int i in f.WordIndices)
            {
                w.WriteNumberValue(i);
            }

            w.WriteEndArray();
            w.WriteNumber("confidence", f.Confidence);
            w.WriteEndObject();
        }

        w.WriteEndArray();
    }
}
