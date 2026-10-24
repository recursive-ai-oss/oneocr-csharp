using System.Text;
using OneOcr.Training;

namespace OneOcr.Tests;

/// <summary>Reproducers of inputs a fuzz campaign found, kept under <c>testdata/fuzz</c>. Each must end in a typed error.</summary>
public class FuzzRegressionTests
{
    private static byte[] Repro(string name) => File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "testdata", "fuzz", name));

    // JsonDocument and Utf8JsonReader accept an escaped lone surrogate while parsing and fail later, when a string or property name is
    // read or compared, with a plain InvalidOperationException.
    [Theory]
    [InlineData("page-lone-low-surrogate-name.json")]
    [InlineData("page-lone-high-surrogate-value.json")]
    public void Page_json_with_a_lone_surrogate_escape_is_a_format_error(string file) =>
        Assert.Throws<OcrFormatException>(() => OcrPageJson.Read(Repro(file)));

    [Fact]
    public void Review_file_with_a_lone_surrogate_escape_is_a_format_error() =>
        Assert.Throws<OcrFormatException>(() => ReviewImport.Parse(Repro("review-lone-low-surrogate-name.json"), OcrLimits.Default));

    [Theory]
    [InlineData("\\ud83d\\ude00", true)]
    [InlineData("\\\\ud800", true)] // an escaped backslash followed by text that looks like an escape
    [InlineData("\\u00e9 \\n \\\"", true)]
    [InlineData("\\ud800", false)]
    [InlineData("\\ud800\\u0041", false)]
    [InlineData("\\ude00\\ud800", false)]
    [InlineData("x\\ud83d", false)]
    [InlineData("\\\\\\ud800", false)] // escaped backslash, then a real lone surrogate escape
    public void Lone_surrogate_scan_accepts_pairs_and_rejects_everything_else(string body, bool ok)
    {
        byte[] json = Encoding.UTF8.GetBytes("\"" + body + "\"");
        if (ok)
        {
            JsonEscapes.RejectLoneSurrogates(json, "t");
        }
        else
        {
            Assert.Throws<OcrFormatException>(() => JsonEscapes.RejectLoneSurrogates(json, "t"));
        }
    }
}
