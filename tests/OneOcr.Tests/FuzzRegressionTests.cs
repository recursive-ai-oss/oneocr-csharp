using System.Text;
using OneOcr.Documents;
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

    // DateTime.DaysInMonth throws for year 0, so a date written with the year 0000 escaped the parser as ArgumentOutOfRangeException.
    [Fact]
    public void A_date_with_year_zero_is_not_a_date_and_does_not_throw()
    {
        string text = Encoding.Unicode.GetString(Repro("normalization-year-zero.utf16"));
        foreach (DateOrder order in new[] { DateOrder.Auto, DateOrder.DayFirst, DateOrder.MonthFirst })
        {
            Assert.Empty(ValueNormalize.ParseDate(text, order));
            Assert.Empty(ValueNormalize.ParseDate("00000101", order));
            Assert.Empty(ValueNormalize.ParseDate("1 Jan 0000", order));
        }

        Assert.Single(ValueNormalize.ParseDate("2024-02-29", DateOrder.Auto));
    }
}
