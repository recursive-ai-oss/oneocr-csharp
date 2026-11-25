using System.Text;
using OneOcr.Pdf;

namespace OneOcr.Tests;

public class ActualTextMarkerTests
{
    private const string Open = "/Span <</ActualText ()>> BDC ";

    private static string Mark(string content, (double, double, double, double) box, out int marked)
    {
        byte[]? r = ActualTextMarker.Mark(Encoding.Latin1.GetBytes(content), [box], out marked);
        return r is null ? content : Encoding.Latin1.GetString(r);
    }

    [Fact]
    public void WrapsOnlyTheShowsThatStartInTheBox()
    {
        // Two lines 22 pt apart; the box is the first line's extent.
        string content = "BT /F1 12 Tf 72 700 Td (garbled) Tj 0 -22 Td (fine) Tj ET";
        string result = Mark(content, (72, 698, 200, 712), out int n);
        Assert.Equal(1, n);
        Assert.Equal($"BT /F1 12 Tf 72 700 Td {Open}(garbled) Tj EMC 0 -22 Td (fine) Tj ET", result);
    }

    [Fact]
    public void FollowsTheTransformationAndTextMatrices()
    {
        string content = "q 2 0 0 2 10 10 cm BT 1 0 0 1 100 100 Tm [(a) -20 (b)] TJ ET Q BT 1 0 0 1 210 210 Tm (c) Tj ET";
        // (a b) starts at 2*100+10 = 210; so does (c) in default space; both are in the box.
        string result = Mark(content, (205, 205, 300, 230), out int n);
        Assert.Equal(2, n);
        Assert.Equal(2, result.Split(Open).Length - 1);
        // Outside it: nothing changes.
        Assert.Equal(content, Mark(content, (0, 0, 50, 50), out int none));
        Assert.Equal(0, none);
    }

    [Fact]
    public void QuoteOperatorsMoveTheLineFirstAndInvisibleTextIsLeftAlone()
    {
        string content = "BT /F1 12 Tf 14 TL 72 700 Td 3 Tr (hidden) Tj 0 Tr (x) Tj T* (next) ' 1 2 (third) \" ET";
        // The first line (y 700) holds invisible text, never wrapped, and a visible show after it.
        string first = Mark(content, (70, 699, 100, 701), out int n);
        Assert.Equal(1, n);
        Assert.Contains($"{Open}(x) Tj EMC", first, StringComparison.Ordinal);
        Assert.DoesNotContain($"{Open}(hidden)", first, StringComparison.Ordinal);

        // ' and " start one line (14) down from the line before.
        string quote = Mark(content, (70, 671, 100, 673), out int q);
        Assert.Equal(1, q);
        Assert.Contains($"{Open}(next) ' EMC", quote, StringComparison.Ordinal);
        string dquote = Mark(content, (70, 657, 100, 659), out int d);
        Assert.Equal(1, d);
        Assert.Contains($"{Open}1 2 (third) \" EMC", dquote, StringComparison.Ordinal);
    }

    [Fact]
    public void OriginalBytesSurviveExactly()
    {
        string content = "q 1 0 0 1 0 0 cm BT /F1 9 Tf 10 10 Td (a \\) b) Tj ET Q % (comment\nBI /W 1 /H 1 /BPC 8 /CS /G ID \u0001 EI\nBT 1 0 0 1 10 10 Tm <4142> Tj ET";
        string result = Mark(content, (0, 0, 100, 100), out int n);
        Assert.Equal(2, n);
        Assert.Equal(content, result.Replace(Open, string.Empty, StringComparison.Ordinal).Replace(" EMC", string.Empty, StringComparison.Ordinal));
    }

    [Fact]
    public void GivesUpOrIgnoresBadInput()
    {
        Assert.Null(ActualTextMarker.Mark(Encoding.ASCII.GetBytes("BT (a) Tj ET"), [], out int n0));
        Assert.Equal(0, n0);
        Assert.Null(ActualTextMarker.Mark(Encoding.ASCII.GetBytes(string.Concat(Enumerable.Repeat("q ", 100_000)) + "BT (a) Tj ET"), [(-1e9, -1e9, 1e9, 1e9)], out _));
        Assert.Null(ActualTextMarker.Mark(Encoding.ASCII.GetBytes("BI /W 1 ID no end"), [(-1e9, -1e9, 1e9, 1e9)], out _));
        // Non-numeric and non-finite operands never throw.
        string odd = "BT 1 0 0 1 x y Tm (a) Tj 1e999 2 Td (b) Tj ET [ ( Tj";
        _ = ActualTextMarker.Mark(Encoding.ASCII.GetBytes(odd), [(-1e9, -1e9, 1e9, 1e9)], out _);
        byte[] big = new byte[1 << 20];
        new Random(5).NextBytes(big);
        _ = ActualTextMarker.Mark(big, [(-1e9, -1e9, 1e9, 1e9)], out _);
    }
}
