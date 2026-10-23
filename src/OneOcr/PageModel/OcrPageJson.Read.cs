using System.Text.Json;

namespace OneOcr;

public static partial class OcrPageJson
{
    // ---- reading ------------------------------------------------------------------------------------------------------------

    /// <summary>Reads pages from UTF-8 JSON text.</summary>
    /// <param name="json">The document.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The pages.</returns>
    /// <exception cref="OcrFormatException">The document is not valid <c>oneocr.page</c> version 1 JSON.</exception>
    /// <exception cref="OcrLimitException">The document, or something it declares, exceeds <paramref name="limits"/>.</exception>
    public static IReadOnlyList<OcrPage> Read(ReadOnlySpan<byte> json, OcrLimits? limits = null)
    {
        limits ??= OcrLimits.Default;
        if (json.Length > limits.MaxPageJsonBytes)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxPageJsonBytes), limits.MaxPageJsonBytes, json.Length);
        }

        if (json.StartsWith("\uFEFF"u8))
        {
            throw new OcrFormatException("The document starts with a byte order mark.");
        }

        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(
                json.ToArray(),
                new JsonDocumentOptions { MaxDepth = limits.MaxJsonDepth, AllowTrailingCommas = false, CommentHandling = JsonCommentHandling.Disallow, AllowDuplicateProperties = false });
        }
        catch (JsonException ex)
        {
            throw new OcrFormatException("The document is not valid JSON: " + ex.Message, ex);
        }

        using (doc)
        {
            return new Reader(limits).ReadRoot(doc.RootElement);
        }
    }

    /// <summary>Reads pages from UTF-8 JSON text in a string.</summary>
    /// <param name="json">The document.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The pages.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="json"/> is null.</exception>
    /// <exception cref="OcrFormatException">The document is not valid <c>oneocr.page</c> version 1 JSON.</exception>
    /// <exception cref="OcrLimitException">The document, or something it declares, exceeds <paramref name="limits"/>.</exception>
    public static IReadOnlyList<OcrPage> Parse(string json, OcrLimits? limits = null)
    {
        ArgumentNullException.ThrowIfNull(json);
        limits ??= OcrLimits.Default;
        // A UTF-8 encoding is at most three bytes per UTF-16 code unit; refuse before encoding anything larger than the limit.
        if (json.Length > limits.MaxPageJsonBytes)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxPageJsonBytes), limits.MaxPageJsonBytes, json.Length);
        }

        return Read(System.Text.Encoding.UTF8.GetBytes(json), limits);
    }

    /// <summary>Reads pages from a stream, taking at most <see cref="OcrLimits.MaxPageJsonBytes"/> bytes.</summary>
    /// <param name="input">The stream, positioned at the start of the document.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The pages.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="input"/> is null.</exception>
    /// <exception cref="OcrFormatException">The document is not valid <c>oneocr.page</c> version 1 JSON.</exception>
    /// <exception cref="OcrLimitException">The document, or something it declares, exceeds <paramref name="limits"/>.</exception>
    public static IReadOnlyList<OcrPage> Read(Stream input, OcrLimits? limits = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        limits ??= OcrLimits.Default;
        long cap = limits.MaxPageJsonBytes;
        var ms = new MemoryStream();
        var chunk = new byte[1 << 16];
        while (true)
        {
            int n = input.Read(chunk, 0, chunk.Length);
            if (n == 0)
            {
                break;
            }

            if (ms.Length + n > cap)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxPageJsonBytes), cap, ms.Length + n);
            }

            ms.Write(chunk, 0, n);
        }

        return Read(ms.GetBuffer().AsSpan(0, (int)ms.Length), limits);
    }

    private sealed class Reader(OcrLimits limits)
    {
        private int _wordsOnPage;
        private int _charsOnPage;

        public List<OcrPage> ReadRoot(JsonElement root)
        {
            Object(root, "$");
            if (Str(Req(root, "format", "$"), "$.format") != Format)
            {
                throw new OcrFormatException($"The format member must be \"{Format}\".");
            }

            long version = Int(Req(root, "version", "$"), "$.version");
            if (version != Version)
            {
                throw new OcrFormatException($"Unsupported {Format} version {version}; this reader understands version {Version}.");
            }

            long pageCount = Int(Req(root, "pageCount", "$"), "$.pageCount");
            JsonElement pagesEl = Req(root, "pages", "$");
            Array(pagesEl, "$.pages");
            int n = pagesEl.GetArrayLength();
            if (n > limits.MaxPagesPerCall)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxPagesPerCall), limits.MaxPagesPerCall, n);
            }

            if (pageCount != n)
            {
                throw new OcrFormatException($"pageCount is {pageCount} but pages has {n} entries.");
            }

            var pages = new List<OcrPage>(n);
            int i = 0;
            foreach (JsonElement p in pagesEl.EnumerateArray())
            {
                pages.Add(ReadPage(p, $"$.pages[{i++}]"));
            }

            return pages;
        }

        private OcrPage ReadPage(JsonElement e, string path)
        {
            Object(e, path);
            _wordsOnPage = 0;
            _charsOnPage = 0;
            int index = IntRange(Req(e, "index", path), path + ".index", 0, int.MaxValue);
            double width = Positive(Req(e, "width", path), path + ".width");
            double height = Positive(Req(e, "height", path), path + ".height");
            double upi = Positive(Req(e, "unitsPerInch", path), path + ".unitsPerInch");
            OcrMatrix? toPdf = null;
            if (e.TryGetProperty("toPdf", out JsonElement m))
            {
                Array(m, path + ".toPdf");
                if (m.GetArrayLength() != 6)
                {
                    throw new OcrFormatException($"{path}.toPdf must have 6 numbers.");
                }

                double[] v = new double[6];
                int k = 0;
                foreach (JsonElement x in m.EnumerateArray())
                {
                    v[k] = Num(x, $"{path}.toPdf[{k}]");
                    k++;
                }

                toPdf = new OcrMatrix(v[0], v[1], v[2], v[3], v[4], v[5]);
            }

            var diagnostics = new List<string>();
            JsonElement diag = Req(e, "diagnostics", path);
            Array(diag, path + ".diagnostics");
            CheckCount(diag.GetArrayLength(), limits.MaxPageElements, nameof(OcrLimits.MaxPageElements));
            foreach (JsonElement d in diag.EnumerateArray())
            {
                diagnostics.Add(Text(d, path + ".diagnostics"));
            }

            Text(Req(e, "text", path), path + ".text");

            JsonElement blocksEl = Req(e, "blocks", path);
            Array(blocksEl, path + ".blocks");
            CheckCount(blocksEl.GetArrayLength(), limits.MaxBlocksPerPage, nameof(OcrLimits.MaxBlocksPerPage));
            var blocks = new List<OcrBlock>(blocksEl.GetArrayLength());
            int bi = 0;
            foreach (JsonElement b in blocksEl.EnumerateArray())
            {
                blocks.Add(ReadBlock(b, $"{path}.blocks[{bi++}]"));
            }

            int lines = 0;
            foreach (OcrBlock b in blocks)
            {
                lines += b.Lines.Count;
            }

            CheckCount(lines, limits.MaxLinesPerPage, nameof(OcrLimits.MaxLinesPerPage));
            int totalWords = _wordsOnPage;

            var regions = new List<OcrRegion>();
            JsonElement regionsEl = Req(e, "regions", path);
            Array(regionsEl, path + ".regions");
            CheckCount(regionsEl.GetArrayLength(), limits.MaxPageElements, nameof(OcrLimits.MaxPageElements));
            int ri = 0;
            foreach (JsonElement r in regionsEl.EnumerateArray())
            {
                regions.Add(ReadRegion(r, $"{path}.regions[{ri++}]", blocks.Count));
            }

            var tables = new List<OcrTable>();
            JsonElement tablesEl = Req(e, "tables", path);
            Array(tablesEl, path + ".tables");
            CheckCount(tablesEl.GetArrayLength(), limits.MaxPageElements, nameof(OcrLimits.MaxPageElements));
            int ti = 0;
            foreach (JsonElement t in tablesEl.EnumerateArray())
            {
                tables.Add(ReadTable(t, $"{path}.tables[{ti++}]", blocks.Count, totalWords));
            }

            var keyValues = new List<OcrKeyValue>();
            JsonElement kvEl = Req(e, "keyValues", path);
            Array(kvEl, path + ".keyValues");
            CheckCount(kvEl.GetArrayLength(), limits.MaxPageElements, nameof(OcrLimits.MaxPageElements));
            int ki = 0;
            foreach (JsonElement kv in kvEl.EnumerateArray())
            {
                keyValues.Add(ReadKeyValue(kv, $"{path}.keyValues[{ki++}]", totalWords));
            }

            return new OcrPage(index, width, height, upi, blocks, regions, tables, keyValues, diagnostics, toPdf);
        }

        private OcrBlock ReadBlock(JsonElement e, string path)
        {
            Object(e, path);
            OcrBlockRole role = Str(Req(e, "role", path), path + ".role") switch
            {
                "body" => OcrBlockRole.Body,
                "header" => OcrBlockRole.Header,
                "footer" => OcrBlockRole.Footer,
                _ => throw new OcrFormatException($"{path}.role must be \"body\", \"header\" or \"footer\"."),
            };
            bool table = Bool(Req(e, "table", path), path + ".table");
            OcrBox box = Box(Req(e, "bounds", path), path + ".bounds");
            JsonElement linesEl = Req(e, "lines", path);
            Array(linesEl, path + ".lines");
            CheckCount(linesEl.GetArrayLength(), limits.MaxLinesPerPage, nameof(OcrLimits.MaxLinesPerPage));
            var lines = new List<OcrLine>(linesEl.GetArrayLength());
            int li = 0;
            foreach (JsonElement l in linesEl.EnumerateArray())
            {
                lines.Add(ReadLine(l, $"{path}.lines[{li++}]"));
            }

            return new OcrBlock(role, table, box, lines, Conf(e, path), Source(e, path), Quad(e, path));
        }

        private OcrLine ReadLine(JsonElement e, string path)
        {
            Object(e, path);
            bool rtl = Str(Req(e, "direction", path), path + ".direction") switch
            {
                "ltr" => false,
                "rtl" => true,
                _ => throw new OcrFormatException($"{path}.direction must be \"ltr\" or \"rtl\"."),
            };
            bool vertical = Bool(Req(e, "vertical", path), path + ".vertical");
            OcrBox box = Box(Req(e, "bounds", path), path + ".bounds");
            JsonElement cellsEl = Req(e, "cells", path);
            Array(cellsEl, path + ".cells");
            CheckCount(cellsEl.GetArrayLength(), limits.MaxWordsPerPage, nameof(OcrLimits.MaxWordsPerPage));
            string text = Text(Req(e, "text", path), path + ".text");
            JsonElement wordsEl = Req(e, "words", path);
            Array(wordsEl, path + ".words");
            int wordCount = wordsEl.GetArrayLength();
            _wordsOnPage = checked(_wordsOnPage + wordCount);
            CheckCount(_wordsOnPage, limits.MaxWordsPerPage, nameof(OcrLimits.MaxWordsPerPage));
            var words = new List<OcrWord>(wordCount);
            int wi = 0;
            foreach (JsonElement w in wordsEl.EnumerateArray())
            {
                words.Add(ReadWord(w, $"{path}.words[{wi++}]"));
            }

            var cells = new List<int>(cellsEl.GetArrayLength());
            int previous = -1;
            foreach (JsonElement c in cellsEl.EnumerateArray())
            {
                int v = IntRange(c, path + ".cells", 0, int.MaxValue);
                if (v <= previous || v >= wordCount || (cells.Count == 0 && v != 0))
                {
                    throw new OcrFormatException($"{path}.cells must start at 0 and strictly increase to less than the word count.");
                }

                cells.Add(v);
                previous = v;
            }

            if (wordCount > 0 && cells.Count == 0)
            {
                throw new OcrFormatException($"{path}.cells is empty but the line has words.");
            }

            if (wordCount == 0 && cells.Count != 0)
            {
                throw new OcrFormatException($"{path}.cells must be empty for a line without words.");
            }

            return new OcrLine(text, words, cells, box, Conf(e, path), Source(e, path), rtl, vertical, Quad(e, path));
        }

        private OcrWord ReadWord(JsonElement e, string path)
        {
            Object(e, path);
            string text = Text(Req(e, "text", path), path + ".text");
            if (text.Length == 0)
            {
                throw new OcrFormatException($"{path}.text is empty.");
            }

            OcrBox box = Box(Req(e, "bounds", path), path + ".bounds");
            if (e.TryGetProperty("visible", out JsonElement vis))
            {
                Bool(vis, path + ".visible");
            }

            float raw = Prob(Req(e, "rawConfidence", path), path + ".rawConfidence");
            var chars = new List<OcrChar>();
            if (e.TryGetProperty("chars", out JsonElement charsEl))
            {
                Array(charsEl, path + ".chars");
                int n = charsEl.GetArrayLength();
                _charsOnPage = checked(_charsOnPage + n);
                CheckCount(_charsOnPage, limits.MaxCharsPerPage, nameof(OcrLimits.MaxCharsPerPage));
                chars.Capacity = n;
                int ci = 0;
                foreach (JsonElement c in charsEl.EnumerateArray())
                {
                    string cp = $"{path}.chars[{ci++}]";
                    Object(c, cp);
                    string ct = Text(Req(c, "text", cp), cp + ".text");
                    if (ct.Length == 0)
                    {
                        throw new OcrFormatException($"{cp}.text is empty.");
                    }

                    chars.Add(new OcrChar(ct, Box(Req(c, "bounds", cp), cp + ".bounds"), Prob(Req(c, "confidence", cp), cp + ".confidence"), Source(c, cp), Bool(Req(c, "approximate", cp), cp + ".approximate"), Quad(c, cp)));
                }
            }

            return new OcrWord(text, box, Prob(Req(e, "confidence", path), path + ".confidence"), raw, Source(e, path), chars, Quad(e, path));
        }

        private OcrRegion ReadRegion(JsonElement e, string path, int blockCount)
        {
            Object(e, path);
            OcrRegionRole role = Str(Req(e, "role", path), path + ".role") switch
            {
                "text" => OcrRegionRole.Text,
                "title" => OcrRegionRole.Title,
                "list" => OcrRegionRole.List,
                "table" => OcrRegionRole.Table,
                "figure" => OcrRegionRole.Figure,
                "header" => OcrRegionRole.Header,
                "footer" => OcrRegionRole.Footer,
                "formField" => OcrRegionRole.FormField,
                "checkbox" => OcrRegionRole.Checkbox,
                _ => throw new OcrFormatException($"{path}.role is not a known region role."),
            };
            return new OcrRegion(role, Box(Req(e, "bounds", path), path + ".bounds"), Prob(Req(e, "confidence", path), path + ".confidence"), Source(e, path), Indices(Req(e, "blocks", path), path + ".blocks", blockCount), Quad(e, path));
        }

        private OcrTable ReadTable(JsonElement e, string path, int blockCount, int wordCount)
        {
            Object(e, path);
            OcrBox box = Box(Req(e, "bounds", path), path + ".bounds");
            int rows = IntRange(Req(e, "rows", path), path + ".rows", 1, limits.MaxPageElements);
            int columns = IntRange(Req(e, "columns", path), path + ".columns", 1, limits.MaxPageElements);
            if ((long)rows * columns > limits.MaxPageElements)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxPageElements), limits.MaxPageElements, (long)rows * columns);
            }

            JsonElement cellsEl = Req(e, "cells", path);
            Array(cellsEl, path + ".cells");
            CheckCount(cellsEl.GetArrayLength(), limits.MaxPageElements, nameof(OcrLimits.MaxPageElements));
            var cells = new List<OcrTableCell>(cellsEl.GetArrayLength());
            int ci = 0;
            foreach (JsonElement c in cellsEl.EnumerateArray())
            {
                string cp = $"{path}.cells[{ci++}]";
                Object(c, cp);
                int row = IntRange(Req(c, "row", cp), cp + ".row", 0, rows - 1);
                int col = IntRange(Req(c, "column", cp), cp + ".column", 0, columns - 1);
                int rs = IntRange(Req(c, "rowSpan", cp), cp + ".rowSpan", 1, rows - row);
                int cs = IntRange(Req(c, "columnSpan", cp), cp + ".columnSpan", 1, columns - col);
                cells.Add(new OcrTableCell(row, col, rs, cs, Box(Req(c, "bounds", cp), cp + ".bounds"), Indices(Req(c, "words", cp), cp + ".words", wordCount)));
            }

            int block = IntRange(Req(e, "block", path), path + ".block", -1, blockCount - 1);
            return new OcrTable(box, rows, columns, cells, Prob(Req(e, "confidence", path), path + ".confidence"), Source(e, path), block, Quad(e, path));
        }

        private OcrKeyValue ReadKeyValue(JsonElement e, string path, int wordCount)
        {
            Object(e, path);
            return new OcrKeyValue(
                Indices(Req(e, "key", path), path + ".key", wordCount), Indices(Req(e, "value", path), path + ".value", wordCount),
                Box(Req(e, "bounds", path), path + ".bounds"), Prob(Req(e, "confidence", path), path + ".confidence"), Source(e, path));
        }

        // ---- value helpers: every one throws OcrFormatException with the path of the member ----

        private static void CheckCount(long actual, long limit, string name)
        {
            if (actual > limit)
            {
                throw new OcrLimitException(name, limit, actual);
            }
        }

        private static JsonElement Req(JsonElement e, string name, string path) =>
            e.TryGetProperty(name, out JsonElement v) ? v : throw new OcrFormatException($"{path} has no member \"{name}\".");

        private static void Object(JsonElement e, string path)
        {
            if (e.ValueKind != JsonValueKind.Object)
            {
                throw new OcrFormatException($"{path} must be an object.");
            }
        }

        private static void Array(JsonElement e, string path)
        {
            if (e.ValueKind != JsonValueKind.Array)
            {
                throw new OcrFormatException($"{path} must be an array.");
            }
        }

        private static bool Bool(JsonElement e, string path) =>
            e.ValueKind is JsonValueKind.True or JsonValueKind.False ? e.GetBoolean() : throw new OcrFormatException($"{path} must be true or false.");

        private static string Str(JsonElement e, string path)
        {
            if (e.ValueKind != JsonValueKind.String)
            {
                throw new OcrFormatException($"{path} must be a string.");
            }

            return e.GetString() ?? throw new OcrFormatException($"{path} must be a string.");
        }

        private string Text(JsonElement e, string path)
        {
            string s = Str(e, path);
            if (s.Length > limits.MaxPageStringLength)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxPageStringLength), limits.MaxPageStringLength, s.Length);
            }

            return s;
        }

        private static double Num(JsonElement e, string path)
        {
            if (e.ValueKind != JsonValueKind.Number || !e.TryGetDouble(out double v) || !double.IsFinite(v))
            {
                throw new OcrFormatException($"{path} must be a finite number.");
            }

            return v;
        }

        private static double Positive(JsonElement e, string path)
        {
            double v = Num(e, path);
            return v > 0 ? v : throw new OcrFormatException($"{path} must be positive.");
        }

        private static long Int(JsonElement e, string path) =>
            e.ValueKind == JsonValueKind.Number && e.TryGetInt64(out long v) ? v : throw new OcrFormatException($"{path} must be an integer.");

        private static int IntRange(JsonElement e, string path, int min, int max)
        {
            long v = Int(e, path);
            return v >= min && v <= max ? (int)v : throw new OcrFormatException($"{path} must be an integer from {min} to {max}.");
        }

        private static float Prob(JsonElement e, string path)
        {
            double v = Num(e, path);
            return v >= 0 && v <= 1 ? (float)v : throw new OcrFormatException($"{path} must be between 0 and 1.");
        }

        private static float Conf(JsonElement e, string path) => Prob(Req(e, "confidence", path), path + ".confidence");

        private static OcrSource Source(JsonElement e, string path) =>
            Str(Req(e, "source", path), path + ".source") switch
            {
                "ocr" => OcrSource.Ocr,
                "textLayer" => OcrSource.TextLayer,
                "llm" => OcrSource.Llm,
                _ => throw new OcrFormatException($"{path}.source must be \"ocr\", \"textLayer\" or \"llm\"."),
            };

        private static OcrBox Box(JsonElement e, string path)
        {
            Array(e, path);
            if (e.GetArrayLength() != 4)
            {
                throw new OcrFormatException($"{path} must have 4 numbers.");
            }

            double[] v = new double[4];
            int i = 0;
            foreach (JsonElement x in e.EnumerateArray())
            {
                v[i] = Num(x, $"{path}[{i}]");
                i++;
            }

            if (v[0] > v[2] || v[1] > v[3])
            {
                throw new OcrFormatException($"{path} must be [left, top, right, bottom] with left <= right and top <= bottom.");
            }

            return new OcrBox(v[0], v[1], v[2], v[3]);
        }

        private static OcrQuad? Quad(JsonElement e, string path)
        {
            if (!e.TryGetProperty("quad", out JsonElement q))
            {
                return null;
            }

            Array(q, path + ".quad");
            if (q.GetArrayLength() != 8)
            {
                throw new OcrFormatException($"{path}.quad must have 8 numbers.");
            }

            double[] v = new double[8];
            int i = 0;
            foreach (JsonElement x in q.EnumerateArray())
            {
                v[i] = Num(x, $"{path}.quad[{i}]");
                i++;
            }

            return new OcrQuad(new OcrPoint(v[0], v[1]), new OcrPoint(v[2], v[3]), new OcrPoint(v[4], v[5]), new OcrPoint(v[6], v[7]));
        }

        // An array of indices into a sequence of `count` elements; its length is checked against the limits before the list is made.
        private List<int> Indices(JsonElement e, string path, int count)
        {
            Array(e, path);
            CheckCount(e.GetArrayLength(), limits.MaxWordsPerPage, nameof(OcrLimits.MaxWordsPerPage));
            var list = new List<int>(e.GetArrayLength());
            foreach (JsonElement x in e.EnumerateArray())
            {
                long v = Int(x, path);
                if (v < 0 || v >= count)
                {
                    throw new OcrFormatException($"{path} refers to element {v}, which does not exist (there are {count}).");
                }

                list.Add((int)v);
            }

            return list;
        }
    }
}
