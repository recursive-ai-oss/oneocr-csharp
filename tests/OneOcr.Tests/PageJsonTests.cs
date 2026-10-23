using System.Text;
using OneImage;
using OneOcr.Kernels;
using OneOcr.Layout;
using OneOcr.Model;

namespace OneOcr.Tests;

public class PageJsonTests
{
    private static OcrPage Sample() => SamplePage.Build();

    [Fact]
    public void RoundTripsToTheSameBytes()
    {
        OcrPage page = Sample();
        string json = OcrPageJson.ToJson([page, new OcrPage(3, 100, 50, 300, [])]);
        IReadOnlyList<OcrPage> back = OcrPageJson.Parse(json);
        Assert.Equal(2, back.Count);
        Assert.Equal(json, OcrPageJson.ToJson(back));
        Assert.Equal(page.Text, back[0].Text);
        Assert.Equal(page.ToText(), back[0].ToText());
        Assert.Equal(page.Words.Count, back[0].Words.Count);
        Assert.Equal(page.ToPdf, back[0].ToPdf);
        Assert.Equal(page.Tables[0].Cells[1].WordIndices, back[0].Tables[0].Cells[1].WordIndices);
        Assert.Equal(page.KeyValues[0].ValueWordIndices, back[0].KeyValues[0].ValueWordIndices);
        Assert.True(back[0].Blocks[1].Lines[0].IsRightToLeft);
        Assert.NotNull(back[0].Blocks[1].Lines[0].Quad);
        Assert.Equal(OcrRegionRole.FormField, back[0].Regions[0].Role);
        Assert.Empty(back[1].Blocks);

        // The stream forms agree with the string form; indentation changes nothing but whitespace.
        var ms = new MemoryStream();
        OcrPageJson.Write(ms, [page, new OcrPage(3, 100, 50, 300, [])]);
        Assert.Equal(json, Encoding.UTF8.GetString(ms.ToArray()));
        Assert.Equal(json, OcrPageJson.ToJson(OcrPageJson.Read(new MemoryStream(ms.ToArray())), null));
        string indented = OcrPageJson.ToJson([page], new OcrPageJsonOptions { Indented = true });
        Assert.Equal(OcrPageJson.ToJson([page]), OcrPageJson.ToJson(OcrPageJson.Parse(indented)));
        Assert.DoesNotContain("\"chars\"", OcrPageJson.ToJson([page], new OcrPageJsonOptions { IncludeChars = false }));
    }

    [Fact]
    public void MemberOrderAndRounding()
    {
        string json = OcrPageJson.ToJson([Sample()]);
        Assert.StartsWith("{\"format\":\"oneocr.page\",\"version\":1,\"pageCount\":1,\"pages\":[{\"index\":2,\"width\":612,\"height\":792,\"unitsPerInch\":72,\"toPdf\":[1,0,0,-1,0,792],\"diagnostics\":", json);
        // Word bounds in OnePDF order (text, bounds) come before the additive members.
        Assert.Contains("{\"text\":\"hi\",\"bounds\":[10,20,20,30],\"visible\":true,\"source\":\"ocr\",\"confidence\":0.85,\"rawConfidence\":0.9,\"chars\":[", json);
        Assert.Contains("[30,20,80,30]", json); // 30.0005 rounds half to even at three decimals (30.0005 is a hair under the tie in binary)
        Assert.Contains("\u00e9\\uD83D\\uDE00", json); // as OnePDF: the relaxed encoder keeps BMP text, escapes surrogate pairs
    }

    [Theory]
    [InlineData(0.0005, 3, "0")]
    [InlineData(0.0015, 3, "0.002")]
    [InlineData(0.0625, 3, "0.062")]
    [InlineData(0.1875, 3, "0.188")]
    [InlineData(2.5, 0 + 3, "2.5")]
    [InlineData(-0.0001, 3, "0")]
    [InlineData(-1.25, 3, "-1.25")]
    [InlineData(1234567.1, 3, "1234567.1")]
    [InlineData(0.12346, 4, "0.1235")]
    [InlineData(double.NaN, 3, "0")]
    [InlineData(double.PositiveInfinity, 3, "0")]
    [InlineData(100, 6, "100")]
    [InlineData(0.0000005, 6, "0")]
    public void NumberFormatting(double v, int decimals, string expected)
    {
        Span<byte> buf = stackalloc byte[64];
        int n = OcrPageJson.FormatNum(v, decimals, buf);
        Assert.Equal(expected, Encoding.ASCII.GetString(buf[..n]));
    }

    [Fact]
    public void LoneSurrogatesAreWrittenAsReplacementCharacters()
    {
        var w = new OcrWord("a\uD800b", new OcrBox(0, 0, 1, 1), 1, 1, OcrSource.Ocr, []);
        var page = new OcrPage(0, 10, 10, 72, [new OcrBlock(OcrBlockRole.Body, false, new OcrBox(0, 0, 1, 1), [new OcrLine("a\uD800b", [w], [0], new OcrBox(0, 0, 1, 1), 1, OcrSource.Ocr)], 1)]);
        string json = OcrPageJson.ToJson([page]);
        Assert.Contains("a\uFFFDb", json);
        Assert.Equal("a\uFFFDb", OcrPageJson.Parse(json)[0].Words[0].Text);
    }

    // ---- rejection ----

    private static readonly string Valid = OcrPageJson.ToJson([Sample()]);

    private static void Format(string json) => Assert.Throws<OcrFormatException>(() => OcrPageJson.Parse(json));

    private static string Edit(string from, string to)
    {
        Assert.Contains(from, Valid);
        return Valid.Replace(from, to);
    }

    [Theory]
    [InlineData("\"format\":\"oneocr.page\"", "\"format\":\"onepdf.extraction\"")]
    [InlineData("\"version\":1", "\"version\":2")]
    [InlineData("\"version\":1", "\"version\":\"1\"")]
    [InlineData("\"version\":1", "\"version\":1.5")]
    [InlineData("\"pageCount\":1", "\"pageCount\":2")]
    [InlineData("\"width\":612", "\"width\":-1")]
    [InlineData("\"width\":612", "\"width\":0")]
    [InlineData("\"width\":612", "\"width\":\"612\"")]
    [InlineData("\"role\":\"footer\"", "\"role\":\"sidebar\"")]
    [InlineData("\"direction\":\"rtl\"", "\"direction\":\"up\"")]
    [InlineData("\"source\":\"llm\"", "\"source\":\"human\"")]
    [InlineData("\"bounds\":[10,20,20,30]", "\"bounds\":[20,20,10,30]")]     // right < left
    [InlineData("\"bounds\":[10,20,20,30]", "\"bounds\":[10,20,20]")]
    [InlineData("\"bounds\":[10,20,20,30]", "\"bounds\":[10,20,20,\"x\"]")]
    [InlineData("\"bounds\":[10,20,20,30]", "\"bounds\":[10,20,20,1e999]")]
    [InlineData("\"confidence\":0.85", "\"confidence\":1.5")]
    [InlineData("\"confidence\":0.85", "\"confidence\":null")]
    [InlineData("\"rawConfidence\":0.9", "\"rawConfidence\":-0.1")]
    [InlineData("\"cells\":[0,2]", "\"cells\":[1,2]")]            // must start at 0
    [InlineData("\"cells\":[0,2]", "\"cells\":[0,3]")]            // past the last word
    [InlineData("\"cells\":[0,2]", "\"cells\":[0,0]")]            // not increasing
    [InlineData("\"cells\":[0,2]", "\"cells\":[]")]               // words but no cells
    [InlineData("\"words\":[0,1]", "\"words\":[0,9]")]            // bad word index
    [InlineData("\"words\":[0,1]", "\"words\":[-1]")]
    [InlineData("\"block\":0", "\"block\":5")]
    [InlineData("\"blocks\":[0,1]", "\"blocks\":[0,2]")]
    [InlineData("\"rowSpan\":1", "\"rowSpan\":2")]                // leaves the table
    [InlineData("\"columns\":2", "\"columns\":0")]
    [InlineData("\"key\":[0]", "\"key\":[7]")]
    [InlineData("\"toPdf\":[1,0,0,-1,0,792]", "\"toPdf\":[1,0,0]")]
    [InlineData("\"quad\":[1,40,5,40,5,50,1,50]", "\"quad\":[1,40,5]")]
    [InlineData("\"approximate\":true", "\"approximate\":1")]
    [InlineData("\"text\":\"cell\"", "\"text\":\"\"")]               // empty word
    [InlineData("\"blocks\":[{", "\"blocks\":{")]
    [InlineData("\"pages\":[", "\"extra\":1,\"pages\":[1,")]
    public void RejectsMalformedDocuments(string from, string to)
    {
        // The first occurrence only, so each case damages one member.
        int i = Valid.IndexOf(from, StringComparison.Ordinal);
        Assert.True(i >= 0, from);
        Format(Valid[..i] + to + Valid[(i + from.Length)..]);
    }

    [Fact]
    public void RejectsBadStructure()
    {
        Format("");
        Format("[]");
        Format("null");
        Format("{}");
        Format("{\"format\":\"oneocr.page\",\"version\":1,\"pageCount\":0}");
        Format(Valid + " x");
        Format(Valid[..^1]);
        Format(Valid[..^1] + ",}");                                  // trailing comma
        Format("\uFEFF" + Valid);
        Format(Valid.Replace("\"format\":\"oneocr.page\",", "\"format\":\"oneocr.page\",\"format\":\"oneocr.page\","));
        Assert.Throws<OcrFormatException>(() => OcrPageJson.Read(new byte[] { 0xEF, 0xBB, 0xBF, (byte)'{', (byte)'}' }));
        Assert.Throws<OcrFormatException>(() => OcrPageJson.Read(new byte[] { 0xFF, 0xFE, 0x00 }));
        // Invalid UTF-8 inside a string (found by the fuzz target) is a format error, not an InvalidOperationException.
        byte[] bad = Encoding.UTF8.GetBytes(Valid);
        int at = Array.IndexOf(bad, (byte)'h');
        bad[at] = 0xE5;
        Assert.Throws<OcrFormatException>(() => OcrPageJson.Read(bad));
        Format(Valid.Replace("\"text\":\"cell\"", "\"text\":\"\\uDE00\""));
        Assert.Throws<ArgumentNullException>(() => OcrPageJson.Parse(null!));
        Assert.Throws<ArgumentNullException>(() => OcrPageJson.ToJson(null!));
    }

    [Fact]
    public void UnknownMembersAreIgnored()
    {
        string extended = Valid.Replace("\"index\":2,", "\"index\":2,\"futureMember\":{\"a\":[1,2]},");
        Assert.Equal(Valid, OcrPageJson.ToJson(OcrPageJson.Parse(extended)));
    }

    [Fact]
    public void LimitsAreCheckedBeforeBuilding()
    {
        // Depth: nesting far beyond MaxJsonDepth is refused by the parser before any page is built.
        Assert.Throws<OcrFormatException>(() => OcrPageJson.Parse("{\"a\":" + new string('[', 5000) + new string(']', 5000) + "}"));
        Assert.Throws<OcrFormatException>(() => OcrPageJson.Parse(Valid, new OcrLimits { MaxJsonDepth = 3 }));

        Assert.Throws<OcrLimitException>(() => OcrPageJson.Parse(Valid, new OcrLimits { MaxPageJsonBytes = Valid.Length - 1 }));
        Assert.Throws<OcrLimitException>(() => OcrPageJson.Read(new MemoryStream(Encoding.UTF8.GetBytes(Valid)), new OcrLimits { MaxPageJsonBytes = 100 }));
        Assert.Throws<OcrLimitException>(() => OcrPageJson.Parse(Valid, new OcrLimits { MaxWordsPerPage = 2 }));
        Assert.Throws<OcrLimitException>(() => OcrPageJson.Parse(Valid, new OcrLimits { MaxCharsPerPage = 1 }));
        Assert.Throws<OcrLimitException>(() => OcrPageJson.Parse(Valid, new OcrLimits { MaxBlocksPerPage = 1 }));
        Assert.Throws<OcrLimitException>(() => OcrPageJson.Parse(Valid, new OcrLimits { MaxPageStringLength = 3 }));
        Assert.Throws<OcrLimitException>(() => OcrPageJson.Parse(Valid, new OcrLimits { MaxPagesPerCall = 0 }));

        // A crafted table that declares an enormous grid is refused without allocating it.
        string huge = Edit("\"rows\":1,\"columns\":2", "\"rows\":60000,\"columns\":60000");
        Assert.Throws<OcrLimitException>(() => OcrPageJson.Parse(huge));

        // Many words across many lines are counted as they are read: a page of 300k tiny words exceeds MaxWordsPerPage.
        var sb = new StringBuilder();
        sb.Append("{\"format\":\"oneocr.page\",\"version\":1,\"pageCount\":1,\"pages\":[{\"index\":0,\"width\":1,\"height\":1,\"unitsPerInch\":72,\"diagnostics\":[],\"text\":\"\",\"blocks\":[{\"role\":\"body\",\"table\":false,\"bounds\":[0,0,1,1],\"lines\":[{\"direction\":\"ltr\",\"vertical\":false,\"bounds\":[0,0,1,1],\"cells\":[0],\"text\":\"\",\"words\":[");
        for (int i = 0; i < 300_000; i++)
        {
            sb.Append(i == 0 ? "" : ",").Append("{\"text\":\"a\",\"bounds\":[0,0,1,1],\"source\":\"ocr\",\"confidence\":1,\"rawConfidence\":1}");
        }

        sb.Append("],\"source\":\"ocr\",\"confidence\":1}],\"source\":\"ocr\",\"confidence\":1}],\"regions\":[],\"tables\":[],\"keyValues\":[]}]}");
        Assert.Throws<OcrLimitException>(() => OcrPageJson.Parse(sb.ToString(), new OcrLimits { MaxPageJsonBytes = 1L << 30 }));
    }

    // ---- pages from the engine ----

    private static Image Columns(out int lineCount)
    {
        // Two columns of three text-like lines each, with a wide gutter; the right column starts higher than the left.
        int w = 900, h = 420;
        var px = new Gray8[w * h];
        Array.Fill(px, new Gray8(255));
        var rng = new Random(5);
        lineCount = 0;
        foreach ((int x0, int x1, int top0) in new[] { (60, 380, 60), (520, 840, 40) })
        {
            for (int row = 0; row < 3; row++)
            {
                int top = top0 + (row * 80);
                lineCount++;
                for (int x = x0; x < x1; x += 10)
                {
                    int bw = 6 + rng.Next(3);
                    bool asc = rng.Next(3) == 0;
                    for (int y = top + (asc ? 0 : 8); y < top + 30; y++)
                    {
                        for (int xx = x; xx < Math.Min(x + bw, x1); xx++)
                        {
                            px[(y * w) + xx] = new Gray8(20);
                        }
                    }
                }
            }
        }

        return Image.FromPixels<Gray8>(px, w, h);
    }

    [Fact]
    public void ColumnsAreReadWholeLeftThenRight()
    {
        Image page = Columns(out _);
        var engine = OcrEngine.Create(ModelContainer.Parse(TestModels.RandomContainerV1(TestModels.Small, 5, "{\"raw\":\"word-mean-max-run\",\"scripts\":{}}")), null, KernelPath.Scalar);
        OcrPage result = engine.Ocr(page);
        var lines = result.Blocks.SelectMany(b => b.Lines).Where(l => l.Words.Count > 0).ToList();
        // Whatever the random model reads, the lines it does read must run through the left column before the right one.
        Assert.NotEmpty(lines);
        int firstRight = lines.FindIndex(l => l.Box.Left > 450);
        if (firstRight >= 0)
        {
            Assert.All(lines.Skip(firstRight), l => Assert.True(l.Box.Left > 450));
            Assert.All(lines.Take(firstRight), l => Assert.True(l.Box.Right < 450));
        }

        // Whole-page paths agree: degrees and JSON.
        Assert.Equal(OcrPageJson.ToJson([result]), OcrPageJson.ToJson([engine.Ocr(page, new OcrOptions { MaxDegreeOfParallelism = 1 })]));
    }

    [Fact]
    public void ColumnFinderFindsBothColumnsAndOneColumnPagesTakeTheLineFinderPath()
    {
        Image two = Columns(out int expected);
        var gray = new Gray8[two.Width * two.Height];
        two.CopyPixelsTo<Gray8>(gray, new ImageRect(0, 0, two.Width, two.Height), 1);
        var masks = new BufferPool<byte>(1L << 26, 1);
        var strips = StripFinder.Find(gray, two.Width, two.Height, 4096, masks, 1, out bool cut, out _, out _);
        Assert.True(cut);
        Assert.Equal(expected, strips.Count);
        Assert.True(strips.Select(s => s.Leaf).Distinct().Count() >= 2);

        // A single column: exactly what the plain line finder says.
        var (px, _) = EngineTests.SyntheticPage(900, 420, [60, 130, 200, 270], 14, 8);
        List<LineBox> plain = LineFinder.Find(px, 900, 420, 4096);
        Gray8[] pg = px.Select(b => new Gray8(b)).ToArray();
        var fromStrips = StripFinder.Find(pg, 900, 420, 4096, masks, 1, out bool usedRegions, out _, out _).Select(s => s.Box).ToList();
        Assert.False(usedRegions);
        Assert.Equal(plain, fromStrips);
    }
}
