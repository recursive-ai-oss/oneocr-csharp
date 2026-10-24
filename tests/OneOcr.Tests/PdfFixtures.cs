using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Globalization;
using System.Threading;
using System.IO.Compression;
using System.Text;
using OneImage;
using OneImage.Pdf;

namespace OneOcr.Tests;

/// <summary>A minimal PDF writer for test documents: numbered objects, a cross-reference table, nothing else.</summary>
internal sealed class PdfBuilder
{
    private readonly List<byte[]?> _objects = [];

    public int Reserve()
    {
        _objects.Add(null);
        return _objects.Count;
    }

    public int Add(string body) => Add(Encoding.Latin1.GetBytes(body));

    public int Add(byte[] body)
    {
        _objects.Add(body);
        return _objects.Count;
    }

    public void Set(int id, string body) => _objects[id - 1] = Encoding.Latin1.GetBytes(body);

    public int AddStream(string dictionaryEntries, byte[] data)
    {
        using var ms = new MemoryStream();
        ms.Write(Encoding.Latin1.GetBytes($"<< {dictionaryEntries} /Length {data.Length} >>\nstream\n"));
        ms.Write(data);
        ms.Write(Encoding.Latin1.GetBytes("\nendstream"));
        return Add(ms.ToArray());
    }

    public byte[] Build(int root)
    {
        using var ms = new MemoryStream();
        ms.Write("%PDF-1.5\n%\u00e2\u00e3\u00cf\u00d3\n"u8);
        var offsets = new long[_objects.Count];
        for (int i = 0; i < _objects.Count; i++)
        {
            offsets[i] = ms.Position;
            ms.Write(Encoding.Latin1.GetBytes(string.Create(CultureInfo.InvariantCulture, $"{i + 1} 0 obj\n")));
            ms.Write(_objects[i] ?? throw new InvalidOperationException("object not set"));
            ms.Write("\nendobj\n"u8);
        }

        long xref = ms.Position;
        var sb = new StringBuilder();
        sb.Append(CultureInfo.InvariantCulture, $"xref\n0 {_objects.Count + 1}\n0000000000 65535 f \n");
        foreach (long o in offsets)
        {
            sb.Append(CultureInfo.InvariantCulture, $"{o:D10} 00000 n \n");
        }

        sb.Append(CultureInfo.InvariantCulture, $"trailer\n<< /Size {_objects.Count + 1} /Root {root} 0 R >>\nstartxref\n{xref}\n%%EOF\n");
        ms.Write(Encoding.Latin1.GetBytes(sb.ToString()));
        return ms.ToArray();
    }
}

/// <summary>What a fixture page should read as: words with the source each must come from.</summary>
internal sealed record PageTruth(string Category, string Description, IReadOnlyList<(OcrSource Source, string Text)> Lines)
{
    public IEnumerable<(OcrSource Source, string Word)> Words =>
        Lines.SelectMany(l => l.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(w => (l.Source, w)));
}

/// <summary>A baseline position: a text line placed at known user-space coordinates, for checking the box round trip.</summary>
internal sealed record PlacedLine(string Text, double X, double Baseline, double Size);

/// <summary>
/// Generated test PDFs with invented text: text-only, scanned, mixed, an invisible OCR layer with wrong words, text whose character
/// map is garbled and text whose character map is wrong. Scans are drawn by rendering a text page with the PDF bridge, so no font
/// or image file is needed.
/// </summary>
internal static class PdfFixtures
{
    public static readonly string[] BodyA =
    [
        "Quarterly harbor report for the northern district",
        "Cargo volumes rose by twelve percent over spring",
        "Warehouse seven reopened after the winter repairs",
        "Crane operators finished training on schedule",
        "Customs clearance now takes under two hours",
        "The ferry timetable changes on the first of June",
        "Fuel deliveries arrive every Tuesday morning",
        "Maintenance windows are posted at the east gate",
    ];

    public static readonly string[] BodyB =
    [
        "Meeting notes from the orchard cooperative",
        "Apple harvest begins in the third week of September",
        "Pickers are assigned to rows by lot number",
        "Cold storage holds four hundred crates",
        "Trucks depart from the packing shed at dawn",
        "Pears ripen later than expected this year",
        "Volunteers receive lunch and a small basket",
        "The next board vote is scheduled for Friday",
    ];

    public static readonly string[] Image4 =
    [
        "Figure caption inside the scanned insert",
        "Reading room hours Monday to Saturday",
        "Silent study on the second floor only",
        "Ask at the desk for archive access forms",
    ];

    public static readonly string[] Heading =
    [
        "Mixed page with a pasted scan",
        "Printed text above and below the picture",
        "Section three continues after the insert",
    ];

    public static readonly string[] Below =
    [
        "Text below the scan is born digital",
        "It must keep the text layer as its source",
    ];

    public static readonly string[] WrongLayer =
    [
        "Zebra quartz vexing ninth pylon omega",
        "Jumbo flask crypt width quiz vapor lemon",
        "Fjord nymph glib quack waltz exit prism",
        "Sphinx of black quartz judge my vow today",
        "Pack my box with five dozen liquor jugs now",
        "How vexingly quick daft zebras jump over it",
        "Bright vixens jump dozy fowl quack loudly",
        "Crazy Fredrick bought many very exquisite pearls",
    ];

    public static readonly string[] Garbled =
    [
        "Shipping labels carry the sorting code",
        "Printed totals match the ledger entries",
        "Pallets are wrapped before they are stacked",
        "Loose items travel in the blue crates",
        "Inspectors sign the form at the loading dock",
        "Returns are logged by the afternoon shift",
    ];

    public static readonly string[] WrongMap =
    [
        "Library catalog update for the winter term",
        "New atlases arrived from the printing house",
        "Borrowing limits remain at six volumes each",
        "Late fees were removed for children under ten",
        "Volunteers shelve returned books every evening",
        "The reading circle meets in the glass annex",
    ];

    /// <summary>
    /// The sample document: every page category once. Page order: 0 text only, 1 scan only, 2 mixed, 3 invisible wrong layer over a scan,
    /// 4 garbled character map on alternate lines, 5 wrong character map on every line.
    /// </summary>
    public static (byte[] Pdf, PageTruth[] Truth, PlacedLine[][] Placed) Sample()
    {
        var pdf = new PdfBuilder();
        int catalog = pdf.Reserve(), pages = pdf.Reserve();
        int helv = pdf.Add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
        int garbledFont = AddMappedFont(pdf, c => char.IsLower((char)c) ? (char)(0xE000 + c) : (char)c);
        int wrongFont = AddMappedFont(pdf, c => c is >= 'a' and <= 'z' ? (char)('a' + ((c - 'a' + 7) % 26)) : (char)c);
        string fonts = $"/Font << /F1 {helv} 0 R /F2 {garbledFont} 0 R /F3 {wrongFont} 0 R >>";

        var kids = new List<int>();
        var truth = new List<PageTruth>();
        var placed = new List<PlacedLine[]>();

        // 0: text only.
        {
            var lines = new List<PlacedLine>();
            var content = new StringBuilder();
            for (int i = 0; i < BodyA.Length; i++)
            {
                lines.Add(new PlacedLine(BodyA[i], 72, 700 - (22 * i), 12));
                content.Append(Show("F1", 12, 72, 700 - (22 * i), BodyA[i]));
            }

            kids.Add(AddPage(pdf, pages, "0 0 612 792", fonts, content.ToString()));
            truth.Add(new PageTruth("text only", "Helvetica lines, visible", BodyA.Select(l => (OcrSource.TextLayer, l)).ToArray()));
            placed.Add([.. lines]);
        }

        // 1: scan only.
        {
            (byte[] raster, int w, int h) = RenderLines(BodyB, 612, 792, 72, 12, 22, 92);
            int im = AddGrayImage(pdf, raster, w, h);
            kids.Add(AddPage(pdf, pages, "0 0 612 792", $"/XObject << /Im1 {im} 0 R >>", "q 612 0 0 792 0 0 cm /Im1 Do Q\n"));
            truth.Add(new PageTruth("scan only", "image of text, no text layer", BodyB.Select(l => (OcrSource.Ocr, l)).ToArray()));
            placed.Add([]);
        }

        // 2: mixed.
        {
            (byte[] raster, int w, int h) = RenderLines(Image4, 400, 160, 10, 11, 24, 28);
            int im = AddGrayImage(pdf, raster, w, h);
            var content = new StringBuilder("q 400 0 0 160 100 400 cm /Im1 Do Q\n");
            var lines = new List<PlacedLine>();
            for (int i = 0; i < Heading.Length; i++)
            {
                lines.Add(new PlacedLine(Heading[i], 72, 730 - (24 * i), 14));
                content.Append(Show("F1", 14, 72, 730 - (24 * i), Heading[i]));
            }

            for (int i = 0; i < Below.Length; i++)
            {
                lines.Add(new PlacedLine(Below[i], 72, 350 - (22 * i), 12));
                content.Append(Show("F1", 12, 72, 350 - (22 * i), Below[i]));
            }

            kids.Add(AddPage(pdf, pages, "0 0 612 792", $"{fonts} /XObject << /Im1 {im} 0 R >>", content.ToString()));
            truth.Add(new PageTruth(
                "mixed", "text above and below, scanned insert in the middle",
                [.. Heading.Select(l => (OcrSource.TextLayer, l)), .. Image4.Select(l => (OcrSource.Ocr, l)), .. Below.Select(l => (OcrSource.TextLayer, l))]));
            placed.Add([.. lines]);
        }

        // 3: invisible, wrong OCR layer over a scan.
        {
            (byte[] raster, int w, int h) = RenderLines(BodyA, 612, 792, 72, 12, 22, 92);
            int im = AddGrayImage(pdf, raster, w, h);
            var content = new StringBuilder("q 612 0 0 792 0 0 cm /Im1 Do Q\n");
            for (int i = 0; i < WrongLayer.Length; i++)
            {
                content.Append(Show("F1", 12, 72, 700 - (22 * i), WrongLayer[i], renderMode: 3));
            }

            kids.Add(AddPage(pdf, pages, "0 0 612 792", $"{fonts} /XObject << /Im1 {im} 0 R >>", content.ToString()));
            truth.Add(new PageTruth("invisible layer", "scan with an invisible layer of wrong words", BodyA.Select(l => (OcrSource.Ocr, l)).ToArray()));
            placed.Add([]);
        }

        // 4: garbled character map on lines 0, 2, 4.
        {
            var content = new StringBuilder();
            var lines = new List<(OcrSource, string)>();
            for (int i = 0; i < Garbled.Length; i++)
            {
                bool bad = i % 2 == 0;
                content.Append(Show(bad ? "F2" : "F1", 12, 72, 700 - (22 * i), Garbled[i]));
                lines.Add((bad ? OcrSource.Ocr : OcrSource.TextLayer, Garbled[i]));
            }

            kids.Add(AddPage(pdf, pages, "0 0 612 792", fonts, content.ToString()));
            truth.Add(new PageTruth("garbled map", "alternate lines map to private-use code points", lines));
            placed.Add([]);
        }

        // 5: wrong (but plausible) character map on every line.
        {
            var content = new StringBuilder();
            for (int i = 0; i < WrongMap.Length; i++)
            {
                content.Append(Show("F3", 12, 72, 700 - (22 * i), WrongMap[i]));
            }

            kids.Add(AddPage(pdf, pages, "0 0 612 792", fonts, content.ToString()));
            truth.Add(new PageTruth("wrong map", "ToUnicode shifts letters, text looks plausible", WrongMap.Select(l => (OcrSource.Ocr, l)).ToArray()));
            placed.Add([]);
        }

        pdf.Set(pages, $"<< /Type /Pages /Count {kids.Count} /Kids [{string.Join(' ', kids.Select(k => $"{k} 0 R"))}] >>");
        pdf.Set(catalog, $"<< /Type /Catalog /Pages {pages} 0 R >>");
        return (pdf.Build(catalog), [.. truth], [.. placed]);
    }

    /// <summary>One page of Helvetica lines with the given page box attributes (CropBox, Rotate, UserUnit), for the geometry checks.</summary>
    public static (byte[] Pdf, PlacedLine[] Lines) GeometryPage(string pageEntries, double size = 12)
    {
        var pdf = new PdfBuilder();
        int catalog = pdf.Reserve(), pages = pdf.Reserve();
        int helv = pdf.Add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
        var content = new StringBuilder();
        var lines = new List<PlacedLine>();
        string[] texts = ["Corner anchor alpha", "Middle anchor beta", "Lower anchor gamma"];
        double[][] spots = [[100, 650], [250, 400], [160, 150]];
        for (int i = 0; i < texts.Length; i++)
        {
            lines.Add(new PlacedLine(texts[i], spots[i][0], spots[i][1], size));
            content.Append(Show("F1", size, spots[i][0], spots[i][1], texts[i]));
        }

        int page = AddPage(pdf, pages, "0 0 612 792", $"/Font << /F1 {helv} 0 R >>", content.ToString(), pageEntries);
        pdf.Set(pages, $"<< /Type /Pages /Count 1 /Kids [{page} 0 R] >>");
        pdf.Set(catalog, $"<< /Type /Catalog /Pages {pages} 0 R >>");
        return (pdf.Build(catalog), [.. lines]);
    }

    /// <summary>A text-only page with a title, a paragraph, a sub-heading, a dashed list, a numbered list and a page number.</summary>
    public static byte[] MarkdownPage()
    {
        var pdf = new PdfBuilder();
        int catalog = pdf.Reserve(), pages = pdf.Reserve();
        int helv = pdf.Add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
        var content = new StringBuilder();
        content.Append(Show("F1", 24, 72, 720, "Harbor Report"));
        string[] paragraph = ["Cargo volumes rose by twelve percent over", "the spring months and the cranes finished", "their training ahead of schedule."];
        for (int i = 0; i < paragraph.Length; i++)
        {
            content.Append(Show("F1", 12, 72, 685 - (15 * i), paragraph[i]));
        }

        content.Append(Show("F1", 16, 72, 615, "Cranes"));
        string[] bullets = ["- Crane operators", "- Warehouse seven", "- Customs clearance"];
        for (int i = 0; i < bullets.Length; i++)
        {
            content.Append(Show("F1", 12, 72, 585 - (18 * i), bullets[i]));
        }

        string[] steps = ["1. Open the east gate", "2. Sign the log"];
        for (int i = 0; i < steps.Length; i++)
        {
            content.Append(Show("F1", 12, 72, 510 - (18 * i), steps[i]));
        }

        content.Append(Show("F1", 12, 300, 30, "7"));
        int page = AddPage(pdf, pages, "0 0 612 792", $"/Font << /F1 {helv} 0 R >>", content.ToString());
        pdf.Set(pages, $"<< /Type /Pages /Count 1 /Kids [{page} 0 R] >>");
        pdf.Set(catalog, $"<< /Type /Catalog /Pages {pages} 0 R >>");
        return pdf.Build(catalog);
    }

    /// <summary>Small valid files to start fuzzing from: a text page, a tiny scan, a page with a garbled map, and a rotated cropped page.</summary>
    public static IEnumerable<(string Name, byte[] Bytes)> Seeds()
    {
        yield return ("text-rotated", GeometryPage("/CropBox [50 60 500 700] /Rotate 90").Pdf);
        yield return ("text-plain", GeometryPage("").Pdf);

        var pdf = new PdfBuilder();
        int catalog = pdf.Reserve(), pages = pdf.Reserve();
        int helv = pdf.Add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
        int garbledFont = AddMappedFont(pdf, c => char.IsLower((char)c) ? (char)(0xE000 + c) : (char)c);
        (byte[] raster, int w, int h) = RenderLines(["Small insert text", "second small line"], 200, 70, 6, 10, 18, 22);
        int im = AddGrayImage(pdf, raster, w, h);
        string content = "q 200 0 0 70 40 300 cm /Im1 Do Q\n" + Show("F1", 12, 40, 700, "Plain visible line") + Show("F2", 12, 40, 660, "garbled words here")
            + Show("F1", 12, 40, 620, "invisible words", renderMode: 3);
        int page = AddPage(pdf, pages, "0 0 612 792", $"/Font << /F1 {helv} 0 R /F2 {garbledFont} 0 R >> /XObject << /Im1 {im} 0 R >>", content);
        pdf.Set(pages, $"<< /Type /Pages /Count 1 /Kids [{page} 0 R] >>");
        pdf.Set(catalog, $"<< /Type /Catalog /Pages {pages} 0 R >>");
        yield return ("mixed-small", pdf.Build(catalog));
    }

    private static string Show(string font, double size, double x, double y, string text, int renderMode = 0) =>
        string.Create(CultureInfo.InvariantCulture, $"BT /{font} {size} Tf {renderMode} Tr {x} {y} Td ({text}) Tj ET\n");

    private static int AddPage(PdfBuilder pdf, int pages, string mediaBox, string resources, string content, string extra = "")
    {
        int stream = pdf.AddStream("", Encoding.Latin1.GetBytes(content));
        return pdf.Add($"<< /Type /Page /Parent {pages} 0 R /MediaBox [{mediaBox}] /Resources << {resources} >> /Contents {stream} 0 R {extra} >>");
    }

    private static int AddGrayImage(PdfBuilder pdf, byte[] gray, int width, int height)
    {
        using var ms = new MemoryStream();
        using (var z = new ZLibStream(ms, CompressionLevel.Fastest, leaveOpen: true))
        {
            z.Write(gray);
        }

        return pdf.AddStream(
            string.Create(CultureInfo.InvariantCulture, $"/Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode"),
            ms.ToArray());
    }

    // A font with a ToUnicode map: code c (0x20..0x7E) means map(c). It is not embedded; the glyphs are Helvetica's.
    private static int AddMappedFont(PdfBuilder pdf, Func<int, char> map)
    {
        var cmap = new StringBuilder();
        cmap.Append("/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CMapName /Test def /CMapType 2 def\n");
        cmap.Append("1 begincodespacerange <00> <FF> endcodespacerange\n95 beginbfchar\n");
        for (int c = 0x20; c <= 0x7E; c++)
        {
            cmap.Append(CultureInfo.InvariantCulture, $"<{c:X2}> <{(int)map(c):X4}>\n");
        }

        cmap.Append("endbfchar endcmap CMapName currentdict /CMap defineresource pop end end\n");
        int toUnicode = pdf.AddStream("", Encoding.Latin1.GetBytes(cmap.ToString()));
        return pdf.Add($"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding /ToUnicode {toUnicode} 0 R >>");
    }

    /// <summary>Draws text lines on a page of the given size with the PDF bridge at 300 DPI and returns the 8-bit gray raster.</summary>
    private static (byte[] Gray, int Width, int Height) RenderLines(
        string[] lines, double width, double height, double left, double size, double step, double firstBaseline)
    {
        var pdf = new PdfBuilder();
        int catalog = pdf.Reserve(), pages = pdf.Reserve();
        int helv = pdf.Add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
        var content = new StringBuilder();
        // Baselines are measured from the top of the picture: firstBaseline, then step apart.
        for (int i = 0; i < lines.Length; i++)
        {
            content.Append(Show("F1", size, left, height - firstBaseline - (step * i), lines[i]));
        }

        string mediaBox = string.Create(CultureInfo.InvariantCulture, $"0 0 {width} {height}");
        int page = AddPage(pdf, pages, mediaBox, $"/Font << /F1 {helv} 0 R >>", content.ToString());
        pdf.Set(pages, $"<< /Type /Pages /Count 1 /Kids [{page} 0 R] >>");
        pdf.Set(catalog, $"<< /Type /Catalog /Pages {pages} 0 R >>");
        var render = new PdfRenderOptions { Dpi = 300 };
        var configuration = new ImageConfiguration().Add(new PdfFormat(render));
        Image image = Image.Load(pdf.Build(catalog), new LoadOptions { Configuration = configuration });
        ImageFrame frame = image.Frames[0];
        Gray8[] pixels = frame.ToArray<Gray8>(1, CancellationToken.None);
        var gray = new byte[pixels.Length];
        for (int i = 0; i < gray.Length; i++)
        {
            gray[i] = pixels[i].V;
        }

        return (gray, frame.Width, frame.Height);
    }
}
