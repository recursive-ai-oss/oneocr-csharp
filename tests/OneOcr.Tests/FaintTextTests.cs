using OneImage;
using OneOcr.Detection;
using OneOcr.Model;

namespace OneOcr.Tests;

public sealed class FaintTextTests
{
    private const int N = 48;

    private static readonly DetectorPostprocess.FaintRule Rule = new(0.02, 110, 6, 1.5, 0.9);

    // A text map with one ordinary line (probability 0.5) and one faint stretch (probability about 0.05, under the ordinary threshold).
    private static float[] Map()
    {
        var prob = new float[N * N];
        Array.Fill(prob, -8f);
        for (int x = 4; x < 14; x++)
        {
            prob[(6 * N) + x] = prob[(7 * N) + x] = 0f;
        }

        for (int x = 4; x < 20; x++)
        {
            prob[(30 * N) + x] = prob[(31 * N) + x] = -2.94f;
        }

        return prob;
    }

    private static Gray8[] Page(byte background, byte ink = 0)
    {
        var gray = new Gray8[4 * N * N];
        Array.Fill(gray, new Gray8(background));
        if (ink > 0)
        {
            // Ink columns past the end of the detected cells: the faint line is followed to them.
            for (int y = 60; y < 64; y++)
            {
                for (int x = 40; x < 60; x += 2)
                {
                    gray[(y * 2 * N) + x] = new Gray8(ink);
                }
            }
        }

        return gray;
    }

    private static PageDetection Decode(float[] prob, Gray8[] gray, DetectorPostprocess.Options options, double? sigma = null) =>
        DetectorPostprocess.Decode(
            prob, N, N, new float[N * N * 16], N, N, 16, TestDetectorModels.AllClasses, new bool[8], new float[4], 1.0, 2 * N, 2 * N, OcrLimits.Default, options, sigma,
            gray, 2 * N, 2 * N);

    private static DetectorPostprocess.Options WithFaint => new(0.3, 1.0, 0.6, 0.85, Faint: Rule);

    [Fact]
    public void ADarkBackgroundKeepsTheFaintLineAfterTheOrdinaryOnes()
    {
        PageDetection d = Decode(Map(), Page(60, 80), WithFaint);
        Assert.Equal(2, d.Lines.Count);
        Assert.Equal(0f, d.Lines[0].MinConfidence);
        Assert.Equal(0.9f, d.Lines[1].MinConfidence);
        // The ordinary line is where it was; the faint one is below it and followed to the ink columns on its right.
        Assert.True(d.Lines[1].Top > d.Lines[0].Bottom);
        Assert.True(d.Lines[1].Right > 40);
    }

    [Fact]
    public void WithoutTheRuleOrOnABrightBackgroundNothingIsAdded()
    {
        PageDetection plain = Decode(Map(), Page(60), DetectorPostprocess.Options.Default);
        PageDetection bright = Decode(Map(), Page(200), WithFaint);
        PageDetection off = Decode(Map(), Page(60), DetectorPostprocess.Options.Default with { Faint = null });
        Assert.Single(plain.Lines);
        Assert.Single(bright.Lines);
        Assert.Single(off.Lines);
        Assert.Equal(plain.Lines[0], off.Lines[0]);
        Assert.Equal(plain.Lines[0], Decode(Map(), Page(60), WithFaint).Lines[0]);
    }

    [Fact]
    public void ACameraPageThatTheNoiseRuleSelectsGetsNoFaintLines()
    {
        DetectorPostprocess.Options options = WithFaint with { RuleSigmaMin = 0.1235, RuleJunkMax = 1e6, RuleTextThreshold = 0.2 };
        PageDetection d = Decode(Map(), Page(60), options, sigma: 0.5);
        Assert.True(d.RuleApplied);
        Assert.Single(d.Lines);
        Assert.Equal(2, Decode(Map(), Page(60), options, sigma: 0.05).Lines.Count);
    }

    [Fact]
    public void ACandidateNeedsItsCellsAndItsAspect()
    {
        float[] prob = Map();
        for (int x = 20; x < 24; x++)
        {
            // Too few cells: 4 of the 6 needed.
            prob[(30 * N) + x] = prob[(31 * N) + x] = -8f;
        }

        Assert.Equal(2, Decode(prob, Page(60), WithFaint).Lines.Count);
        Assert.Single(Decode(prob, Page(60), WithFaint with { Faint = Rule with { MinCells = 40 } }).Lines);
        Assert.Single(Decode(prob, Page(60), WithFaint with { Faint = Rule with { MinAspect = 20 } }).Lines);
    }

    [Fact]
    public void TheFringeOfAnOrdinaryLineIsNeverACandidate()
    {
        // A halo of probability 0.05 around the ordinary line: all of it is within a cell of the line or overlaps its box.
        var prob = new float[N * N];
        Array.Fill(prob, -8f);
        for (int y = 4; y < 10; y++)
        {
            for (int x = 2; x < 16; x++)
            {
                prob[(y * N) + x] = -2.94f;
            }
        }

        for (int x = 4; x < 14; x++)
        {
            prob[(6 * N) + x] = prob[(7 * N) + x] = 0f;
        }

        Assert.Single(Decode(prob, Page(60), WithFaint).Lines);
    }

    private static DbNetModel Load(string? postprocessJson) =>
        DbNetModel.FromContainer(ModelContainer.Parse(TestDetectorModels.Random(TestDetectorModels.V1Small, 1, postprocessJson: postprocessJson)), OcrLimits.Default);

    [Fact]
    public void TheManifestMembersLoadAsTheRule()
    {
        Assert.Null(Load(null).Postprocess.Faint);
        Assert.Null(Load("{\"text_threshold\":0.3}").Postprocess.Faint);
        Assert.Equal(Rule, Load(TestDetectorModels.FaintRuleJson).Postprocess.Faint);
    }

    [Theory]
    [InlineData("{\"faint_text_threshold\":0.02}")]
    [InlineData("{\"faint_text_threshold\":0.02,\"faint_bg_max\":110,\"faint_min_cells\":6,\"faint_min_aspect\":1.5}")]
    [InlineData("{\"faint_bg_max\":110,\"faint_min_cells\":6,\"faint_min_aspect\":1.5,\"faint_min_confidence\":0.9}")]
    // Out of range.
    [InlineData("{\"faint_text_threshold\":0,\"faint_bg_max\":110,\"faint_min_cells\":6,\"faint_min_aspect\":1.5,\"faint_min_confidence\":0.9}")]
    [InlineData("{\"faint_text_threshold\":1,\"faint_bg_max\":110,\"faint_min_cells\":6,\"faint_min_aspect\":1.5,\"faint_min_confidence\":0.9}")]
    [InlineData("{\"faint_text_threshold\":0.02,\"faint_bg_max\":256,\"faint_min_cells\":6,\"faint_min_aspect\":1.5,\"faint_min_confidence\":0.9}")]
    [InlineData("{\"faint_text_threshold\":0.02,\"faint_bg_max\":-1,\"faint_min_cells\":6,\"faint_min_aspect\":1.5,\"faint_min_confidence\":0.9}")]
    [InlineData("{\"faint_text_threshold\":0.02,\"faint_bg_max\":110,\"faint_min_cells\":0,\"faint_min_aspect\":1.5,\"faint_min_confidence\":0.9}")]
    [InlineData("{\"faint_text_threshold\":0.02,\"faint_bg_max\":110,\"faint_min_cells\":5000,\"faint_min_aspect\":1.5,\"faint_min_confidence\":0.9}")]
    [InlineData("{\"faint_text_threshold\":0.02,\"faint_bg_max\":110,\"faint_min_cells\":6,\"faint_min_aspect\":-1,\"faint_min_confidence\":0.9}")]
    [InlineData("{\"faint_text_threshold\":0.02,\"faint_bg_max\":110,\"faint_min_cells\":6,\"faint_min_aspect\":1.5,\"faint_min_confidence\":1.5}")]
    // Wrong type.
    [InlineData("{\"faint_text_threshold\":\"0.02\",\"faint_bg_max\":110,\"faint_min_cells\":6,\"faint_min_aspect\":1.5,\"faint_min_confidence\":0.9}")]
    [InlineData("{\"faint_text_threshold\":0.02,\"faint_bg_max\":110.5,\"faint_min_cells\":6,\"faint_min_aspect\":1.5,\"faint_min_confidence\":0.9}")]
    [InlineData("{\"faint_text_threshold\":0.02,\"faint_bg_max\":110,\"faint_min_cells\":[6],\"faint_min_aspect\":1.5,\"faint_min_confidence\":0.9}")]
    [InlineData("{\"faint_text_threshold\":0.02,\"faint_bg_max\":110,\"faint_min_cells\":6,\"faint_min_aspect\":true,\"faint_min_confidence\":0.9}")]
    public void ABadSetOfMembersIsRejected(string postprocessJson) =>
        Assert.Throws<OcrModelException>(() => Load(postprocessJson));
}
