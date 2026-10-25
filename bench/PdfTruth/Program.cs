using System;
using System.Text.Json;
using OnePdf;
using OnePdf.Text;

// PdfTruth <file.pdf> [pages]   pages = 1-based, comma separated, ranges allowed: 3,7,10-12 (default: all)
// Writes JSON to stdout: { "pageCount": n, "pages": [ { "page": k, "width", "height", "content": "...", "words": [ { "t", "b": [l, b, r, t], "v" } ] } ] }
// "content" is the page text in content-stream order; "words" follow the reading-order analysis. Boxes are PDF points, y up.
if (args.Length < 1)
{
    Console.Error.WriteLine("usage: PdfTruth file.pdf [pages]");
    return 2;
}

using PdfDocument doc = PdfDocument.Open(args[0]);
int count = doc.PageCount;
var wanted = new System.Collections.Generic.List<int>();
if (args.Length >= 2)
{
    foreach (string part in args[1].Split(','))
    {
        string[] ab = part.Split('-');
        int a = int.Parse(ab[0], System.Globalization.CultureInfo.InvariantCulture);
        int b = ab.Length > 1 ? int.Parse(ab[1], System.Globalization.CultureInfo.InvariantCulture) : a;
        for (int p = a; p <= b && p <= count; p++)
        {
            wanted.Add(p);
        }
    }
}
else
{
    for (int p = 1; p <= count; p++)
    {
        wanted.Add(p);
    }
}

var content = new PdfTextOptions { Mode = PdfTextMode.ContentOrder, RetainGlyphs = true };
var reading = new PdfTextOptions { Mode = PdfTextMode.ReadingOrder, RetainGlyphs = true };

using var stream = Console.OpenStandardOutput();
using var w = new Utf8JsonWriter(stream);
w.WriteStartObject();
w.WriteNumber("pageCount", count);
w.WriteStartArray("pages");
foreach (int p in wanted)
{
    w.WriteStartObject();
    w.WriteNumber("page", p);
    try
    {
        PdfPage page = doc.Pages[p - 1];
        PdfPageText c = page.ExtractText(content);
        PdfPageText r = page.ExtractText(reading);
        w.WriteNumber("width", r.Width);
        w.WriteNumber("height", r.Height);
        w.WriteString("content", c.Text);
        w.WriteStartArray("words");
        foreach (PdfTextWord word in r.Words)
        {
            w.WriteStartObject();
            w.WriteString("t", word.Text);
            w.WriteStartArray("b");
            w.WriteNumberValue(word.Bounds.Left);
            w.WriteNumberValue(word.Bounds.Bottom);
            w.WriteNumberValue(word.Bounds.Right);
            w.WriteNumberValue(word.Bounds.Top);
            w.WriteEndArray();
            w.WriteBoolean("v", word.IsVisible);
            w.WriteEndObject();
        }

        w.WriteEndArray();
    }
    catch (Exception e) when (e is not OutOfMemoryException)
    {
        w.WriteString("error", e.GetType().Name + ": " + e.Message);
    }

    w.WriteEndObject();
}

w.WriteEndArray();
w.WriteEndObject();
w.Flush();
return 0;
