using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Tests;

/// <summary>The letter O read as 0 inside identifiers and dates, and nowhere else.</summary>
public class DigitContextTests
{
    [Theory]
    [InlineData("ACC-71O3O5", "ACC-710305")]
    [InlineData("2020-04-O7", "2020-04-07")]
    [InlineData("ORD-86928O", "ORD-869280")]
    [InlineData("1O,OOO", "1O,OOO")] // one digit: too few to tell
    [InlineData("1,2O0,3OO", "1,200,300")]
    [InlineData("O2", "O2")] // too few digits
    [InlineData("CO2", "CO2")] // another letter in the run
    [InlineData("COVID-19", "COVID-19")]
    [InlineData("ACC71O3O5", "ACC71O3O5")] // not separated from the letters by punctuation
    [InlineData("OO-123", "OO-123")] // a run of O's alone has no digit
    [InlineData("Option", "Option")]
    [InlineData("2o19-O1-1O", "2o19-01-10")] // the run with a lower-case o is left; the others are numbers with an O
    [InlineData("100%", "100%")]
    [InlineData("", "")]
    public void OnlyRunsOfDigitsAndOAreChanged(string read, string expected) => Assert.Equal(expected, DigitContext.Fix(read));

    [Fact]
    public void ApplyChangesTheWordAndItsCharactersAndKeepsEverythingElse()
    {
        static RecognizedWord Word(string text) => new(
            text, 0, 10, 0.8f, 0.9f, OcrScript.Latin, text.Length, [.. text.Select((c, i) => new RecognizedChar(c.ToString(), i, i + 1, 0.5f))]);

        var line = new LineWords("x", 0.9f, [Word("ref"), Word("ACC-71O3O5")]);

        LineWords fixedLine = DigitContext.Apply(line);

        Assert.Equal("ref", fixedLine.Words[0].Text);
        Assert.Equal(line.Words[0], fixedLine.Words[0]);
        RecognizedWord w = fixedLine.Words[1];
        Assert.Equal("ACC-710305", w.Text);
        Assert.Equal("ACC-710305", string.Concat(w.Chars.Select(c => c.Text)));
        Assert.Equal(0.8f, w.Confidence);
        Assert.Equal(line.Words[1].Chars[5].X0, w.Chars[5].X0);
        var plain = new LineWords("x", 0.9f, [Word("ref"), Word("order")]);
        Assert.Same(plain, DigitContext.Apply(plain));
    }

    [Fact]
    public void AWordWhoseCharactersDoNotSpellItIsLeftAlone()
    {
        var word = new RecognizedWord("ACC-71O3O5", 0, 10, 0.8f, 0.9f, OcrScript.Latin, 3, []);
        var line = new LineWords("x", 0.9f, [word]);

        Assert.Same(line, DigitContext.Apply(line));
    }
}
