using System.Buffers.Binary;
using OneImage;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Tests;

public class ReportCalibrationTests
{
    private static readonly TestModels.Shape Shape = TestModels.Small;

    private const string Decision =
        "{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Latin\":{\"x\":[0.0,0.5,1.0],\"y\":[0.0,0.25,1.0]}}}";

    private const string Report =
        "{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Latin\":{\"x\":[0.0,0.4,1.0],\"y\":[0.05,0.5,0.97]},\"Cyrillic\":{\"x\":[0.5],\"y\":[0.5]}}}";

    private static byte[] File(string? report) => TestModels.RandomContainerV1(Shape, 3, Decision, report);

    [Fact]
    public void AMemberIsReadWithTheSameRulesAsCalibration()
    {
        ModelContainer withReport = ModelContainer.Parse(File(Report));
        Assert.NotNull(withReport.ReportCalibration);
        Assert.True(withReport.ReportCalibration!.HasCurve(OcrScript.Latin));
        Assert.False(withReport.ReportCalibration.HasCurve(OcrScript.Greek));
        Assert.Equal(0.5f, withReport.ReportCalibration.Apply(OcrScript.Cyrillic, 0.9f));
        Assert.Equal(0.5f, withReport.ReportCalibration.Apply(OcrScript.Latin, 0.4f));

        // The decision curve is untouched by the member.
        Assert.Equal(0.25f, withReport.Calibration!.Apply(OcrScript.Latin, 0.5f));
        Assert.Empty(withReport.Diagnostics);
    }

    [Fact]
    public void FilesWithoutTheMemberLoadAsBefore()
    {
        ModelContainer c = ModelContainer.Parse(File(null));
        Assert.Null(c.ReportCalibration);
        Assert.NotNull(c.Calibration);
        Assert.Null(LineRecognizer.FromContainer(c).ReportCalibration);

        // A version 0 header ignores the member like any other member that version does not name.
        byte[] asV0 = File(Report);
        BinaryPrimitives.WriteUInt32LittleEndian(asV0.AsSpan(8), 0);
        Assert.Null(ModelContainer.Parse(asV0).ReportCalibration);
    }

    [Theory]
    [InlineData("{\"raw\":\"other\",\"scripts\":{}}")]
    [InlineData("{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Klingon\":{\"x\":[0],\"y\":[0]}}}")]
    [InlineData("{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Latin\":{\"x\":[0,0],\"y\":[0,1]}}}")]
    [InlineData("{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Latin\":{\"x\":[0,1],\"y\":[1,0]}}}")]
    [InlineData("{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Latin\":{\"x\":[0,2],\"y\":[0,1]}}}")]
    [InlineData("{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Latin\":{\"x\":[],\"y\":[]}}}")]
    [InlineData("{\"raw\":\"word-mean-max-run\"}")]
    [InlineData("[]")]
    [InlineData("3")]
    public void AMalformedMemberIsRejected(string report)
    {
        Assert.Throws<OcrModelException>(() => ModelContainer.Parse(File(report)));
    }

    [Fact]
    public void TooManyKnotsIsALimitError()
    {
        string x = string.Join(",", Enumerable.Range(0, 65).Select(i => (i / 64.0).ToString("R", System.Globalization.CultureInfo.InvariantCulture)));
        string report = "{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Latin\":{\"x\":[" + x + "],\"y\":[" + x + "]}}}";
        Assert.Throws<OcrLimitException>(() => ModelContainer.Parse(File(report)));
    }

    [Fact]
    public void JsonNullMeansAbsent()
    {
        Assert.Null(ModelContainer.Parse(File("null")).ReportCalibration);
    }

    private static Image Page()
    {
        var (px, _) = EngineTests.SyntheticPage(900, 420, [60, 130, 200, 270], 14, 8);
        return Image.FromPixels<Gray8>(px.Select(b => new Gray8(b)).ToArray(), 900, 420);
    }

    private static string Words(OcrPage page) =>
        string.Join("|", page.Words.Select(w => FormattableString.Invariant($"{w.Text} {w.Box.Left} {w.Box.Top} {w.Box.Right} {w.Box.Bottom} {BitConverter.SingleToInt32Bits(w.RawConfidence):x8}")));

    private static string Hex(float v) => BitConverter.SingleToInt32Bits(v).ToString("x8", System.Globalization.CultureInfo.InvariantCulture);

    private static string WithConfidence(OcrPage page) =>
        string.Join("|", page.Blocks.SelectMany(b => b.Lines.Select(l => Hex(b.Confidence) + " " + Hex(l.Confidence) + " " + string.Join(",", l.Words.Select(w => Hex(w.Confidence))))));

    [Fact]
    public void OnlyTheReportedConfidenceChanges()
    {
        Image page = Page();
        var plain = OcrEngine.Create(ModelContainer.Parse(File(null)), null, KernelPath.Scalar);
        var reporting = OcrEngine.Create(ModelContainer.Parse(File(Report)), null, KernelPath.Scalar);
        OcrPage a = plain.Ocr(page, new OcrOptions { MaxDegreeOfParallelism = 1 });
        OcrPage b = reporting.Ocr(page, new OcrOptions { MaxDegreeOfParallelism = 1 });

        // Text, boxes and raw confidences are those of the file without the member; so is everything the page model says about structure.
        Assert.Equal(a.ToText(), b.ToText());
        Assert.Equal(Words(a), Words(b));
        Assert.Equal(a.Blocks.Count, b.Blocks.Count);
        Assert.Equal(a.Diagnostics, b.Diagnostics);
        Assert.NotEmpty(b.Words);

        ModelContainer report = ModelContainer.Parse(File(Report));
        foreach (OcrWord w in b.Words)
        {
            Assert.Equal(report.ReportCalibration!.Apply(ScriptDetector.Of(w.Text), w.RawConfidence), w.Confidence);
        }

        Assert.NotEqual(WithConfidence(a), WithConfidence(b));

        // A line's confidence is its words' confidences weighted by length, as the reader computes it.
        foreach (OcrLine line in b.Blocks.SelectMany(x => x.Lines))
        {
            double sum = 0, len = 0;
            foreach (OcrWord w in line.Words)
            {
                sum += w.Confidence * (double)w.Text.Length;
                len += w.Text.Length;
            }

            Assert.Equal((float)(sum / len), line.Confidence);
        }
    }

    [Fact]
    public void ReportedConfidenceIsBitIdenticalOnEveryPathAndDegree()
    {
        Image page = Page();
        ModelContainer container = ModelContainer.Parse(File(Report));
        string baseline = WithConfidence(OcrEngine.Create(container, null, KernelPath.Scalar).Ocr(page, new OcrOptions { MaxDegreeOfParallelism = 1 }));
        string json = OcrPageJson.ToJson([OcrEngine.Create(container, null, KernelPath.Scalar).Ocr(page, new OcrOptions { MaxDegreeOfParallelism = 1 })]);
        foreach (KernelPath path in Dispatch.Available())
        {
            var engine = OcrEngine.Create(container, null, path);
            foreach (int degree in new[] { 1, 4 })
            {
                OcrPage r = engine.Ocr(page, new OcrOptions { MaxDegreeOfParallelism = degree });
                Assert.Equal(baseline, WithConfidence(r));
                Assert.Equal(json, OcrPageJson.ToJson([r]));
            }
        }
    }
}
