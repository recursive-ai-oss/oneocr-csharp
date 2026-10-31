namespace OneOcr.Tests;

/// <summary>A page that uses every member of the page model; shared by the tests and the fuzz seed writer.</summary>
public static class SamplePage
{
    public static OcrPage Build()
    {
        var chars = new List<OcrChar>
        {
            new("h", new OcrBox(10, 20, 15.5, 30), 0.9f),
            new("i", new OcrBox(15.5, 20, 20, 30), 0.8f, OcrSource.Ocr, false, new OcrQuad(new OcrPoint(15.5, 20), new OcrPoint(20, 20), new OcrPoint(20, 30), new OcrPoint(15.5, 30))),
        };
        var w1 = new OcrWord("hi", new OcrBox(10, 20, 20, 30), 0.85f, 0.9f, OcrSource.Ocr, chars);
        var w2 = new OcrWord("there \"x\" \u00e9\U0001F600", new OcrBox(30, 20, 80, 30.0005), 0.5f, 0.55f, OcrSource.TextLayer, []);
        var w3 = new OcrWord("cell", new OcrBox(200, 20, 240, 30), 1f, 1f, OcrSource.Llm, []);
        var line = new OcrLine("hi there \"x\" \u00e9\U0001F600\tcell", [w1, w2, w3], [0, 2], new OcrBox(10, 20, 240, 30), 0.7f, OcrSource.Ocr);
        var rtl = new OcrLine("x", [new OcrWord("x", new OcrBox(1, 40, 5, 50), 0.1f, 0.1f, OcrSource.Ocr, [])], [0], new OcrBox(1, 40, 5, 50), 0.1f, OcrSource.Ocr, true, true,
            new OcrQuad(new OcrPoint(1, 40), new OcrPoint(5, 40), new OcrPoint(5, 50), new OcrPoint(1, 50)));
        var b1 = new OcrBlock(OcrBlockRole.Body, true, new OcrBox(10, 20, 240, 30), [line], 0.7f);
        var b2 = new OcrBlock(OcrBlockRole.Footer, false, new OcrBox(1, 40, 5, 50), [rtl], 0.1f, OcrSource.TextLayer);
        var table = new OcrTable(new OcrBox(10, 20, 240, 30), 1, 2, [new OcrTableCell(0, 0, 1, 1, new OcrBox(10, 20, 80, 30), [0, 1]), new OcrTableCell(0, 1, 1, 1, new OcrBox(200, 20, 240, 30), [2])], 0.75f, OcrSource.Ocr, 0);
        var region = new OcrRegion(OcrRegionRole.FormField, new OcrBox(0, 0, 300, 100), 0.6f, OcrSource.Llm, [0, 1]);
        var kv = new OcrKeyValue([0], [1, 2], new OcrBox(10, 20, 240, 30), 0.3f, OcrSource.Llm);
        return new OcrPage(2, 612, 792, 72, [b1, b2], [region], [table], [kv], [new OcrDiagnostic("OCR901", OcrDiagnosticSeverity.Warning, "note \"one\"")], new OcrMatrix(1, 0, 0, -1, 0, 792));
    }
}
