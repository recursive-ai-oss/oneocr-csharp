using System.Text;
using OneOcr.Pdf;
using OnePdf;

namespace OneOcr.Tests;

/// <summary>The content-stream rewrite that removes a third-party invisible layer, and the font the layer is written in.</summary>
public class InvisibleTextRemoverTests
{
    private static string Run(string content, out int removed)
    {
        byte[]? result = InvisibleTextRemover.Remove(Encoding.Latin1.GetBytes(content), out removed);
        return result is null ? "<unchanged>" : Encoding.Latin1.GetString(result);
    }

    private static string Norm(string s) => string.Join(' ', s.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

    [Fact]
    public void CutsInvisibleShowsAndKeepsEverythingElse()
    {
        const string Content = "q 1 0 0 1 0 0 cm /Im1 Do Q\nBT /F1 12 Tf 3 Tr 72 700 Td (hidden words) Tj 0 -14 Td [(more) -20 (hidden)] TJ ET\nBT 0 Tr /F1 12 Tf 72 600 Td (visible) Tj ET\n";
        string output = Run(Content, out int removed);
        Assert.Equal(2, removed);
        Assert.Equal("q 1 0 0 1 0 0 cm /Im1 Do Q BT /F1 12 Tf 3 Tr 72 700 Td 0 -14 Td ET BT 0 Tr /F1 12 Tf 72 600 Td (visible) Tj ET", Norm(output));
    }

    [Fact]
    public void RenderModeFollowsTheGraphicsState()
    {
        // The mode set inside q ... Q does not leak out; the show after Q is visible again.
        string output = Run("BT 0 Tr /F1 10 Tf q 3 Tr (a) Tj Q 10 0 Td (b) Tj ET", out int removed);
        Assert.Equal(1, removed);
        Assert.Contains("(b) Tj", output, StringComparison.Ordinal);
        Assert.DoesNotContain("(a)", output, StringComparison.Ordinal);
    }

    [Fact]
    public void QuoteOperatorsKeepTheirLineMoves()
    {
        string output = Run("BT 3 Tr /F1 10 Tf 12 TL 10 700 Td (a) ' 2 1 (b) \" ET", out int removed);
        Assert.Equal(2, removed);
        Assert.Equal("BT 3 Tr /F1 10 Tf 12 TL 10 700 Td T* 2 Tw 1 Tc T* ET", Norm(output));
    }

    [Fact]
    public void StringsAndCommentsAreNotOperators()
    {
        string output = Run("BT 3 Tr (a ) Tj (x\\) y) Tj % (y) Tj\n(z) Tj ET", out int removed);
        Assert.Equal(3, removed);
        Assert.DoesNotContain("(z)", output, StringComparison.Ordinal);
    }

    [Fact]
    public void InlineImageDataIsNeverReadAsContent()
    {
        // The image data contains bytes that look like a show; it is copied as it is.
        string content = "BT 3 Tr (gone) Tj ET q 4 0 0 4 0 0 cm BI /W 2 /H 2 /BPC 8 /CS /G ID (x) Tj\n EI Q BT 0 Tr (kept) Tj ET";
        string output = Run(content, out int removed);
        Assert.Equal(1, removed);
        Assert.Contains("ID (x) Tj", output, StringComparison.Ordinal);
        Assert.Contains("(kept) Tj", output, StringComparison.Ordinal);
    }

    [Fact]
    public void VisibleTextContinuingAfterACutShowGivesUp()
    {
        // Cutting the first show would move the second one: the page is left as it was.
        Assert.Equal("<unchanged>", Run("BT 3 Tr (a) Tj 0 Tr (b) Tj ET", out int removed));
        Assert.Equal(0, removed);
        // With a positioning operator in between it is safe.
        Assert.Equal(1, RunCount("BT 3 Tr (a) Tj 0 Tr 10 0 Td (b) Tj ET"));
    }

    private static int RunCount(string content)
    {
        Run(content, out int removed);
        return removed;
    }

    [Fact]
    public void NothingInvisibleMeansNoChange()
    {
        Assert.Equal("<unchanged>", Run("BT /F1 12 Tf (visible) Tj ET", out int removed));
        Assert.Equal(0, removed);
    }

    [Fact]
    public void HostileInputNeitherThrowsNorHangs()
    {
        var random = new Random(20261125);
        string[] pieces = ["BT", "ET", "q", "Q", "3 Tr", "0 Tr", "(", ")", "((", "\\", "[", "]", "<", ">", "<<", ">>", "Tj", "TJ", "'", "\"", "BI", "ID", "EI", "%", "\n", " ", "/N", "12", "-.5", "Td", "T*", "\u00FF"];
        for (int n = 0; n < 3000; n++)
        {
            var sb = new StringBuilder();
            int count = random.Next(1, 60);
            for (int i = 0; i < count; i++)
            {
                sb.Append(pieces[random.Next(pieces.Length)]).Append(random.Next(4) == 0 ? "" : " ");
            }

            byte[]? result = InvisibleTextRemover.Remove(Encoding.Latin1.GetBytes(sb.ToString()), out int removed);
            Assert.True(result is null == (removed == 0));
        }

        // A very deep q nesting is refused, not followed.
        Assert.Null(InvisibleTextRemover.Remove(Encoding.ASCII.GetBytes(string.Concat(Enumerable.Repeat("q ", 100_000)) + "BT 3 Tr (a) Tj ET"), out _));
        // Long runs of garbage stay linear.
        byte[] big = new byte[8_000_000];
        random.NextBytes(big);
        InvisibleTextRemover.Remove(big, out _);
    }
}

/// <summary>The glyphless font.</summary>
public class GlyphlessFontTests
{
    [Fact]
    public void OnePdfEmbedsAndSubsetsIt()
    {
        using var doc = PdfDocument.CreateEmpty();
        PdfPage page = doc.InsertBlankPage(0, 200, 100);
        var writer = new PdfTextLayerWriter(doc);
        OcrPage model = SearchablePdfWriterTests.Page(["A\u00E9\u0394\u0416 x"], 0, 300, 200);
        using (PdfPageContentBuilder b = page.BeginContent())
        {
            Assert.Equal(2, writer.Draw(b, model));
            b.Commit();
        }

        using var ms = new MemoryStream();
        doc.Save(ms, new PdfSaveOptions { DocumentId = new byte[16] });
        // The file holds a subset of the font (a few glyphs), not the 60,000-glyph original.
        Assert.True(ms.Length < 8_000, ms.Length.ToString());
        Assert.Contains("OneOcrText", Encoding.Latin1.GetString(ms.ToArray()), StringComparison.Ordinal);
    }

    [Fact]
    public void EveryCharacterOfTheRecognizerComesBackAsItself()
    {
        // Latin (with every accent the charset has), Greek, Cyrillic and common punctuation and symbols, in words of ten.
        var chars = new List<string>();
        foreach ((int from, int to) in new[] { (0x21, 0x7E), (0xA1, 0x24F), (0x370, 0x3FF), (0x400, 0x4FF), (0x2010, 0x2044), (0x20A0, 0x20BF), (0x2190, 0x2199) })
        {
            for (int c = from; c <= to; c++)
            {
                if (char.GetUnicodeCategory((char)c) is not (System.Globalization.UnicodeCategory.OtherNotAssigned or System.Globalization.UnicodeCategory.Control or System.Globalization.UnicodeCategory.Format or System.Globalization.UnicodeCategory.SpaceSeparator))
                {
                    chars.Add(((char)c).ToString());
                }
            }
        }

        string[] words = [.. chars.Chunk(10).Select(static c => string.Concat(c))];
        OcrPage model = SearchablePdfWriterTests.Page([string.Join(' ', words.Take(words.Length / 2)), string.Join(' ', words.Skip(words.Length / 2))], 0, 40_000, 1_600);
        byte[] pdf = SearchablePdfWriterTests.Write(model);
        using var doc = PdfDocument.Open(pdf);
        var options = new OnePdf.Text.PdfTextOptions { RetainGlyphs = true };
        string got = string.Concat(doc.Pages[0].ExtractText(options).Glyphs.Select(static g => g.Text));
        Assert.Equal(string.Concat(words), got);
    }

    [Fact]
    public void CharactersWithoutAGlyphBecomeTheReplacementCharacter()
    {
        Assert.Equal(0xFFFD, GlyphlessFont.Map(new Rune(0x1F600)));
        Assert.Equal(0xFFFD, GlyphlessFont.Map(new Rune(0x00AD - 0x00AD + 0x9F)));
        Assert.Equal('a', GlyphlessFont.Map(new Rune('a')));
        Assert.True(GlyphlessFont.IsCovered(0xFFFD));
        Assert.False(GlyphlessFont.IsCovered(0xFFFE));
    }

    [Fact]
    public void FontBytesAreFixed()
    {
        // Same bytes every time; no clock or randomness in the file.
        byte[] a = GlyphlessFont.Bytes.ToArray();
        byte[] b = System.Security.Cryptography.SHA256.HashData(a);
        Assert.Equal(b, System.Security.Cryptography.SHA256.HashData(GlyphlessFont.Bytes.Span));
        Assert.InRange(a.Length, 200_000, 400_000);
    }
}
