// The writer follows PdfExtractionJson in the OnePDF library (src/OnePdf/Text/Extraction/PdfExtractionJson.cs): the member
// names and order, UTF-8 without a byte-order mark, JSON escaping and nothing more, and numbers rounded half to even and
// written in invariant form without trailing zeros. What differs is documented in docs/spec/oneocr-page-v1.md: y runs down,
// boxes are [left, top, right, bottom], and the members that OnePDF has no use for are additive.

using System.Buffers;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace OneOcr;

/// <summary>Options for <see cref="OcrPageJson"/>.</summary>
public sealed class OcrPageJsonOptions
{
    /// <summary>Write the characters of every word. Default true.</summary>
    public bool IncludeChars { get; init; } = true;

    /// <summary>Indent the output. Default false.</summary>
    public bool Indented { get; init; }
}

/// <summary>
/// Reads and writes the <c>oneocr.page</c> JSON form of a list of pages (<c>docs/spec/oneocr-page-v1.md</c>). The writer is
/// deterministic: the same pages give the same bytes on every platform. The reader checks every member and every size against
/// <see cref="OcrLimits"/> before it builds anything of that size, and reports problems as <see cref="OcrFormatException"/> or
/// <see cref="OcrLimitException"/>.
/// </summary>
public static partial class OcrPageJson
{
    /// <summary>The value of the <c>version</c> member.</summary>
    public const int Version = 1;

    /// <summary>The value of the <c>format</c> member.</summary>
    public const string Format = "oneocr.page";

    // ---- writing ------------------------------------------------------------------------------------------------------------

    /// <summary>Serializes pages to a JSON string.</summary>
    /// <param name="pages">The pages, normally in page order.</param>
    /// <param name="options">Output options; null for the defaults.</param>
    /// <returns>The JSON text.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="pages"/> is null.</exception>
    public static string ToJson(IReadOnlyList<OcrPage> pages, OcrPageJsonOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(pages);
        var buffer = new ArrayBufferWriter<byte>(16 * 1024);
        WriteTo(buffer, pages, options ?? new OcrPageJsonOptions());
        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    /// <summary>Serializes pages as UTF-8 JSON to a stream; the stream stays open.</summary>
    /// <param name="output">The destination.</param>
    /// <param name="pages">The pages, normally in page order.</param>
    /// <param name="options">Output options; null for the defaults.</param>
    /// <exception cref="ArgumentNullException">An argument is null.</exception>
    public static void Write(Stream output, IReadOnlyList<OcrPage> pages, OcrPageJsonOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(pages);
        var buffer = new ArrayBufferWriter<byte>(16 * 1024);
        WriteTo(buffer, pages, options ?? new OcrPageJsonOptions());
        output.Write(buffer.WrittenSpan);
    }

    private static void WriteTo(ArrayBufferWriter<byte> buffer, IReadOnlyList<OcrPage> pages, OcrPageJsonOptions options)
    {
        using var w = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = options.Indented, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping });
        w.WriteStartObject();
        w.WriteString("format", Format);
        w.WriteNumber("version", Version);
        w.WriteNumber("pageCount", pages.Count);
        w.WriteStartArray("pages");
        foreach (OcrPage page in pages)
        {
            WritePage(w, page, options);
        }

        w.WriteEndArray();
        w.WriteEndObject();
        w.Flush();
    }

    private static void WritePage(Utf8JsonWriter w, OcrPage p, OcrPageJsonOptions o)
    {
        w.WriteStartObject();
        w.WriteNumber("index", p.Index);
        Num(w, "width", p.Width, 3);
        Num(w, "height", p.Height, 3);
        Num(w, "unitsPerInch", p.UnitsPerInch, 3);
        if (p.ToPdf is OcrMatrix m)
        {
            w.WriteStartArray("toPdf");
            foreach (double v in new[] { m.A, m.B, m.C, m.D, m.E, m.F })
            {
                RawNum(w, v, 6);
            }

            w.WriteEndArray();
        }

        if (p.Orientation != 0)
        {
            w.WriteNumber("orientation", p.Orientation);
        }

        if (p.SkewDegrees != 0)
        {
            Num(w, "skew", p.SkewDegrees, 3);
        }

        w.WriteStartArray("diagnostics");
        foreach (string d in p.Diagnostics)
        {
            w.WriteStringValue(Clean(d));
        }

        w.WriteEndArray();
        w.WriteString("text", Clean(p.Text));
        w.WriteStartArray("blocks");
        foreach (OcrBlock b in p.Blocks)
        {
            w.WriteStartObject();
            w.WriteString("role", b.Role switch { OcrBlockRole.Header => "header", OcrBlockRole.Footer => "footer", _ => "body" });
            w.WriteBoolean("table", b.IsTable);
            Box(w, "bounds", b.Box);
            w.WriteStartArray("lines");
            foreach (OcrLine l in b.Lines)
            {
                WriteLine(w, l, o);
            }

            w.WriteEndArray();
            Common(w, b.Source, b.Confidence, b.Quad);
            w.WriteEndObject();
        }

        w.WriteEndArray();
        w.WriteStartArray("regions");
        foreach (OcrRegion r in p.Regions)
        {
            w.WriteStartObject();
            w.WriteString("role", RegionRoleName(r.Role));
            Box(w, "bounds", r.Box);
            Num(w, "confidence", r.Confidence, 4);
            w.WriteString("source", SourceName(r.Source));
            Ints(w, "blocks", r.BlockIndices);
            Quad(w, r.Quad);
            w.WriteEndObject();
        }

        w.WriteEndArray();
        w.WriteStartArray("tables");
        foreach (OcrTable t in p.Tables)
        {
            w.WriteStartObject();
            Box(w, "bounds", t.Box);
            w.WriteNumber("rows", t.Rows);
            w.WriteNumber("columns", t.Columns);
            w.WriteStartArray("cells");
            foreach (OcrTableCell c in t.Cells)
            {
                w.WriteStartObject();
                w.WriteNumber("row", c.Row);
                w.WriteNumber("column", c.Column);
                w.WriteNumber("rowSpan", c.RowSpan);
                w.WriteNumber("columnSpan", c.ColumnSpan);
                Box(w, "bounds", c.Box);
                Ints(w, "words", c.WordIndices);
                w.WriteEndObject();
            }

            w.WriteEndArray();
            Num(w, "confidence", t.Confidence, 4);
            w.WriteString("source", SourceName(t.Source));
            w.WriteNumber("block", t.BlockIndex);
            Quad(w, t.Quad);
            w.WriteEndObject();
        }

        w.WriteEndArray();
        w.WriteStartArray("keyValues");
        foreach (OcrKeyValue kv in p.KeyValues)
        {
            w.WriteStartObject();
            Ints(w, "key", kv.KeyWordIndices);
            Ints(w, "value", kv.ValueWordIndices);
            Box(w, "bounds", kv.Box);
            Num(w, "confidence", kv.Confidence, 4);
            w.WriteString("source", SourceName(kv.Source));
            w.WriteEndObject();
        }

        w.WriteEndArray();
        w.WriteEndObject();
    }

    private static void WriteLine(Utf8JsonWriter w, OcrLine l, OcrPageJsonOptions o)
    {
        w.WriteStartObject();
        w.WriteString("direction", l.IsRightToLeft ? "rtl" : "ltr");
        w.WriteBoolean("vertical", l.IsVertical);
        Box(w, "bounds", l.Box);
        Ints(w, "cells", l.Cells);
        w.WriteString("text", Clean(l.Text));
        w.WriteStartArray("words");
        foreach (OcrWord word in l.Words)
        {
            w.WriteStartObject();
            w.WriteString("text", Clean(word.Text));
            Box(w, "bounds", word.Box);
            w.WriteBoolean("visible", true);
            w.WriteString("source", SourceName(word.Source));
            Num(w, "confidence", word.Confidence, 4);
            Num(w, "rawConfidence", word.RawConfidence, 4);
            if (o.IncludeChars && word.Chars.Count > 0)
            {
                w.WriteStartArray("chars");
                foreach (OcrChar c in word.Chars)
                {
                    w.WriteStartObject();
                    w.WriteString("text", Clean(c.Text));
                    Box(w, "bounds", c.Box);
                    Num(w, "confidence", c.Confidence, 4);
                    w.WriteString("source", SourceName(c.Source));
                    w.WriteBoolean("approximate", c.BoxIsApproximate);
                    Quad(w, c.Quad);
                    w.WriteEndObject();
                }

                w.WriteEndArray();
            }

            Quad(w, word.Quad);
            w.WriteEndObject();
        }

        w.WriteEndArray();
        Common(w, l.Source, l.Confidence, l.Quad);
        w.WriteEndObject();
    }

    private static void Common(Utf8JsonWriter w, OcrSource source, float confidence, OcrQuad? quad)
    {
        w.WriteString("source", SourceName(source));
        Num(w, "confidence", confidence, 4);
        Quad(w, quad);
    }

    // Unpaired surrogates (a model file's charset can hold them) are not valid JSON text; they are written as U+FFFD.
    private static string Clean(string text)
    {
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (char.IsHighSurrogate(c) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                i++;
            }
            else if (char.IsSurrogate(c))
            {
                var sb = new StringBuilder(text.Length);
                for (int j = 0; j < text.Length; j++)
                {
                    if (char.IsHighSurrogate(text[j]) && j + 1 < text.Length && char.IsLowSurrogate(text[j + 1]))
                    {
                        sb.Append(text[j]).Append(text[j + 1]);
                        j++;
                    }
                    else
                    {
                        sb.Append(char.IsSurrogate(text[j]) ? '\uFFFD' : text[j]);
                    }
                }

                return sb.ToString();
            }
        }

        return text;
    }

    private static string SourceName(OcrSource s) => s switch { OcrSource.TextLayer => "textLayer", OcrSource.Llm => "llm", _ => "ocr" };

    private static string RegionRoleName(OcrRegionRole r) => r switch
    {
        OcrRegionRole.Title => "title",
        OcrRegionRole.List => "list",
        OcrRegionRole.Table => "table",
        OcrRegionRole.Figure => "figure",
        OcrRegionRole.Header => "header",
        OcrRegionRole.Footer => "footer",
        OcrRegionRole.FormField => "formField",
        OcrRegionRole.Checkbox => "checkbox",
        _ => "text",
    };

    private static void Box(Utf8JsonWriter w, string name, OcrBox b)
    {
        w.WriteStartArray(name);
        RawNum(w, b.Left, 3);
        RawNum(w, b.Top, 3);
        RawNum(w, b.Right, 3);
        RawNum(w, b.Bottom, 3);
        w.WriteEndArray();
    }

    private static void Quad(Utf8JsonWriter w, OcrQuad? q)
    {
        if (q is not OcrQuad quad)
        {
            return;
        }

        w.WriteStartArray("quad");
        foreach (OcrPoint p in new[] { quad.P0, quad.P1, quad.P2, quad.P3 })
        {
            RawNum(w, p.X, 3);
            RawNum(w, p.Y, 3);
        }

        w.WriteEndArray();
    }

    private static void Ints(Utf8JsonWriter w, string name, IReadOnlyList<int> values)
    {
        w.WriteStartArray(name);
        foreach (int v in values)
        {
            w.WriteNumberValue(v);
        }

        w.WriteEndArray();
    }

    private static void Num(Utf8JsonWriter w, string name, double v, int decimals)
    {
        Span<byte> buf = stackalloc byte[64];
        int n = FormatNum(v, decimals, buf);
        w.WritePropertyName(name);
        w.WriteRawValue(buf[..n], skipInputValidation: true);
    }

    private static void RawNum(Utf8JsonWriter w, double v, int decimals)
    {
        Span<byte> buf = stackalloc byte[64];
        int n = FormatNum(v, decimals, buf);
        w.WriteRawValue(buf[..n], skipInputValidation: true);
    }

    /// <summary>
    /// Writes <paramref name="v"/> rounded to <paramref name="decimals"/> places (half to even), invariant form, no exponent, no
    /// trailing zeros; zero (also negative zero) and non-finite values give <c>0</c>. Values whose rounded magnitude does not
    /// fit a 64-bit count of units (about 9e15 / 10^decimals) are clamped to it.
    /// </summary>
    internal static int FormatNum(double v, int decimals, Span<byte> dst)
    {
        if (!double.IsFinite(v))
        {
            dst[0] = (byte)'0';
            return 1;
        }

        long scale = decimals == 3 ? 1000 : decimals == 4 ? 10000 : 1000000;
        double limit = 9.0e15 / scale;
        double r = Math.Round(Math.Clamp(v, -limit, limit), decimals, MidpointRounding.ToEven);
        long k = (long)Math.Round(r * scale);
        if (k == 0)
        {
            dst[0] = (byte)'0';
            return 1;
        }

        int pos = 0;
        ulong a = (ulong)Math.Abs(k);
        if (k < 0)
        {
            dst[pos++] = (byte)'-';
        }

        (ulong whole, ulong frac) = (a / (ulong)scale, a % (ulong)scale);
        whole.TryFormat(dst[pos..], out int written, default, System.Globalization.CultureInfo.InvariantCulture);
        pos += written;
        if (frac != 0)
        {
            dst[pos++] = (byte)'.';
            Span<byte> digits = stackalloc byte[decimals];
            for (int i = decimals - 1; i >= 0; i--)
            {
                digits[i] = (byte)('0' + (frac % 10));
                frac /= 10;
            }

            int len = decimals;
            while (digits[len - 1] == (byte)'0')
            {
                len--;
            }

            digits[..len].CopyTo(dst[pos..]);
            pos += len;
        }

        return pos;
    }
}
