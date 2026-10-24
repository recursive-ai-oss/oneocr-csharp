using System.Text;
using OneOcr.Training;

namespace OneOcr.Tests;

public class ReviewImportTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "oneocr-review-" + Guid.NewGuid().ToString("N"));

    public ReviewImportTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try
        {
            Directory.Delete(_root, true);
        }
        catch (IOException)
        {
        }
    }

    private static byte[] B(string s) => Encoding.UTF8.GetBytes(s);

    private static string Doc(string items) => "{\"format\":\"oneocr.review\",\"version\":1,\"items\":" + items + "}";

    [Fact]
    public void Corrections_ids_and_unreviewed_items_are_read_and_unused_members_skipped()
    {
        ReviewFile f = ReviewImport.Parse(B(Doc("[{\"id\":\"a-1\",\"text\":\"Total due\",\"box\":[1,2,3,4],\"x\":{\"y\":[1]}},{\"id\":\"a-2\",\"text\":null},{\"id\":\"a-3\"},{\"id\":\"a-4\",\"text\":\"\"}]")), OcrLimits.Default);
        Assert.Equal([new ReviewCorrection("a-1", "Total due"), new ReviewCorrection("a-4", "")], f.Corrections);
        Assert.Equal(2, f.Unreviewed);
    }

    [Theory]
    [InlineData("")]
    [InlineData("[]")]
    [InlineData("{\"format\":\"other\",\"version\":1,\"items\":[]}")]
    [InlineData("{\"format\":\"oneocr.review\",\"version\":2,\"items\":[]}")]
    [InlineData("{\"format\":\"oneocr.review\",\"version\":1}")]
    [InlineData("{\"format\":\"oneocr.review\",\"version\":1,\"items\":{}}")]
    [InlineData("{\"format\":\"oneocr.review\",\"version\":1,\"items\":[1]}")]
    [InlineData("{\"format\":\"oneocr.review\",\"version\":1,\"items\":[{\"text\":\"x\"}]}")]
    [InlineData("{\"format\":\"oneocr.review\",\"version\":1,\"items\":[{\"id\":\"a\",\"text\":5}]}")]
    [InlineData("{\"format\":\"oneocr.review\",\"version\":1,\"items\":[{\"id\":\"a\",\"text\":\"tab\\there\"}]}")]
    [InlineData("{\"format\":\"oneocr.review\",\"version\":1,\"items\":[{\"id\":\"a\",\"text\":\"lone \\ud800 surrogate\"}]}")]
    [InlineData("{\"format\":\"oneocr.review\",\"version\":1,\"items\":[]} trailing")]
    [InlineData("{\"format\":\"oneocr.review\",\"version\":1,\"items\":[],\"items\":[]}")]
    [InlineData("{\"format\":\"oneocr.review\",\"version\":1,\"items\":[{\"id\":\"")]
    public void Malformed_review_files_are_format_errors(string json)
    {
        Assert.Throws<OcrFormatException>(() => ReviewImport.Parse(B(json), OcrLimits.Default));
    }

    [Fact]
    public void Bytes_that_are_not_utf8_are_a_format_error_not_a_crash()
    {
        // Found by the fuzz target: invalid UTF-8 inside a string used to surface as InvalidOperationException.
        byte[] bad = [.. B("{\"format\":\"oneocr.review\",\"version\":1,\"items\":[{\"id\":\"a"), 0xD4, .. B("\",\"text\":\"x\"}]}")];
        Assert.Throws<OcrFormatException>(() => ReviewImport.Parse(bad, OcrLimits.Default));
        byte[] badFormat = [.. B("{\"format\":\"oneocr"), 0xD4, .. B("\"}")];
        Assert.Throws<OcrFormatException>(() => ReviewImport.Parse(badFormat, OcrLimits.Default));
    }

    [Fact]
    public void Size_item_count_text_length_and_nesting_are_checked_before_they_cost_anything()
    {
        var small = new OcrLimits { MaxReviewImportBytes = 100, MaxReviewItems = 3, MaxReviewTextLength = 8, MaxJsonDepth = 6 };
        Assert.Throws<OcrLimitException>(() => ReviewImport.Parse(new byte[101], small));
        string four = Doc("[{\"id\":\"a\"},{\"id\":\"b\"},{\"id\":\"c\"},{\"id\":\"d\"}]");
        Assert.True(B(four).Length <= 100);
        Assert.Throws<OcrLimitException>(() => ReviewImport.Parse(B(four), small));
        Assert.Throws<OcrLimitException>(() => ReviewImport.Parse(B(Doc("[{\"id\":\"a\",\"text\":\"123456789\"}]")), small));
        OcrFormatException deep = Assert.Throws<OcrFormatException>(() => ReviewImport.Parse(B(Doc("[{\"id\":\"a\",\"x\":[[[[[[[[1]]]]]]]]}]")), new OcrLimits { MaxReviewImportBytes = 1000, MaxReviewItems = 3, MaxReviewTextLength = 8, MaxJsonDepth = 6 }));
        Assert.NotNull(deep);

        // A long id is not an id this tool issued.
        Assert.Throws<OcrFormatException>(() => ReviewImport.Parse(B(Doc("[{\"id\":\"" + new string('x', 129) + "\"}]")), OcrLimits.Default));

        // A text whose escaped form is far longer than the limit is refused from its encoded length, before it is decoded.
        string huge = Doc("[{\"id\":\"a\",\"text\":\"" + new string('x', 200_000) + "\"}]");
        Assert.Throws<OcrLimitException>(() => ReviewImport.Parse(B(huge), OcrLimits.Default));
    }

    [Fact]
    public void A_file_over_the_import_limit_is_refused_without_being_read_into_memory()
    {
        string work = Path.Combine(_root, "work");
        string file = Path.Combine(_root, "big.json");
        using (var fs = new FileStream(file, FileMode.Create))
        {
            fs.SetLength(2_000_000); // a sparse file of zeros: reading it would cost 2 MB, refusing it costs nothing
        }

        var limits = new OcrLimits { MaxReviewImportBytes = 1_000_000 };
        Assert.Throws<OcrLimitException>(() => LineReview.Import(work, file, limits));
    }

    [Fact]
    public void Ids_that_name_no_line_are_ignored_and_counted()
    {
        string work = Path.Combine(_root, "work");
        ReviewImportResult r = LineReview.Import(work, B(Doc("[{\"id\":\"ffff-1\",\"text\":\"x\"},{\"id\":\"../../etc/passwd\",\"text\":\"y\"}]")));
        Assert.Equal(0, r.Accepted);
        Assert.Equal(2, r.UnknownIds);
        Assert.False(File.Exists(Path.Combine(work, "reviewed.json")));
    }
}
