using OneOcr.Detection;
using OneOcr.Kernels;
using OneOcr.Model;

namespace OneOcr.Tests;

/// <summary>The detector's loader and its limits: bad files are typed errors, and sizes are checked before anything is allocated.</summary>
public class DetectorLoaderTests
{
    private static DbNetModel Load(byte[] file, OcrLimits? limits = null) => DbNetModel.FromContainer(ModelContainer.Parse(file, limits), limits ?? OcrLimits.Default);

    [Fact]
    public void TheShippedSizeLoadsWithItsArchitecture()
    {
        DbNetModel model = Load(TestDetectorModels.Random(TestDetectorModels.Default, 1));
        Assert.Equal(8, model.Classes);
        Assert.Equal([true, true, true, true, true, true, false, false], model.LayoutTrained.ToArray());
        Assert.Equal(4, model.Stages.Length);
        Assert.Equal([2, 3, 4, 3], model.Stages.Select(s => s.Length).ToArray());
        Assert.Equal(80, model.Stages[1][0].OutPadded);
        Assert.Equal(64, model.Lateral[0].OutPadded);
        long parameters = TestDetectorModels.RandomTensors(TestDetectorModels.Default, 1).Sum(t => (long)t.Values.Length);
        Assert.Equal(309_077, parameters);
    }

    [Fact]
    public void TheTrainedListComesFromTheManifest()
    {
        DbNetModel model = Load(TestDetectorModels.Random(TestDetectorModels.Small, 1, "[\"checkbox\",\"text\"]"));
        Assert.Equal([true, false, false, false, false, false, false, true], model.LayoutTrained.ToArray());
    }

    [Fact]
    public void MalformedConfigurationsAreModelErrors()
    {
        TestDetectorModels.Shape s = TestDetectorModels.Small;
        List<TestTensor> tensors = TestDetectorModels.RandomTensors(s, 1);
        string[] charset = TestDetectorModels.Charset(s);
        string good = TestDetectorModels.ConfigJson(s);
        string[] bad =
        [
            good.Replace("\"dbnet-lite\"", "\"other\"", StringComparison.Ordinal),
            good.Replace("\"widths\":[16,24,40,56]", "\"widths\":[16,24,40]", StringComparison.Ordinal),
            good.Replace("\"depths\":[1,2,1,1]", "\"depths\":[1,0,1,1]", StringComparison.Ordinal),
            good.Replace("\"classes\":8", "\"classes\":7", StringComparison.Ordinal),
            good.Replace("\"text\",\"title\"", "\"text\",\"text\"", StringComparison.Ordinal),
            good.Replace("\"layout_blocks\":2", "\"layout_blocks\":0", StringComparison.Ordinal),
            good.Replace("\"widths\":[16,24,40,56]", "\"widths\":null", StringComparison.Ordinal),
            good.Replace("\"fpn\":20", "\"fpn\":0", StringComparison.Ordinal),
        ];
        foreach (string config in bad)
        {
            byte[] file = TestContainerWriter.Write("dbnet-lite-v0", 0, charset, config, tensors, calibrationJson: "{\"raw\":\"word-mean-max-run\",\"scripts\":{}}");
            Assert.Throws<OcrModelException>(() => Load(file));
        }
    }

    [Fact]
    public void V1LoadsItsBranchAndTwoScaleOrientation()
    {
        DbNetModel model = Load(TestDetectorModels.Random(TestDetectorModels.V1Default, 1));
        Assert.Equal(4, model.Layout16Blocks.Length);
        Assert.Equal(160, model.Layout16Blocks[0].Out);
        Assert.Equal(48, model.Layout16Blocks[0].In);
        Assert.Equal(96, model.LayoutMix!.Out);
        Assert.True(model.OrientFromTwoScales);
        Assert.Equal(4 * (176 + 288), model.OrientWeights.Length);
        // A v1 file with both switches off is the v0 graph.
        DbNetModel plain = Load(TestDetectorModels.Random(TestDetectorModels.V1Small with { Layout16Blocks = 0, Orient16 = false }, 1));
        Assert.Empty(plain.Layout16Blocks);
        Assert.Null(plain.LayoutMix);
        Assert.False(plain.OrientFromTwoScales);
    }

    [Fact]
    public void V1MembersAreValidatedBeforeTensorsAreRead()
    {
        TestDetectorModels.Shape s = TestDetectorModels.V1Small;
        List<TestTensor> tensors = TestDetectorModels.RandomTensors(s, 1);
        string[] charset = TestDetectorModels.Charset(s);
        string good = TestDetectorModels.ConfigJson(s);
        string[] bad =
        [
            good.Replace("\"layout16_blocks\":2", "\"layout16_blocks\":-1", StringComparison.Ordinal),
            good.Replace("\"layout16\":18", "\"layout16\":0", StringComparison.Ordinal),
            good.Replace("\"layout16\":18,", string.Empty, StringComparison.Ordinal),
            good.Replace("\"orient16\":true", "\"orient16\":\"yes\"", StringComparison.Ordinal),
        ];
        foreach (string config in bad)
        {
            Assert.NotEqual(good, config);
            byte[] file = TestContainerWriter.Write("dbnet-lite-v1", 0, charset, config, tensors, calibrationJson: "{\"raw\":\"word-mean-max-run\",\"scripts\":{}}");
            Assert.Throws<OcrModelException>(() => Load(file));
        }

        // The v1 switches in a v0 file are an error rather than silently ignored.
        byte[] v0 = TestContainerWriter.Write("dbnet-lite-v0", 0, charset, good, tensors, calibrationJson: "{\"raw\":\"word-mean-max-run\",\"scripts\":{}}");
        Assert.Throws<OcrModelException>(() => Load(v0));

        // Branch or orientation tensors with the wrong shape.
        List<TestTensor> mixWrong = tensors.Select(t => t.Name == "layout_mix.weight" ? new TestTensor { Name = t.Name, Shape = [t.Shape[1], t.Shape[0], 1, 1], Values = t.Values } : t).ToList();
        Assert.Throws<OcrModelException>(() => Load(TestDetectorModels.Random(s, 1, tensors: mixWrong)));
        List<TestTensor> orientWrong = tensors.Select(t => t.Name == "orient.weight" ? new TestTensor { Name = t.Name, Shape = [4, s.Widths[3]], Values = t.Values.Take(4 * s.Widths[3]).ToArray() } : t).ToList();
        Assert.Throws<OcrModelException>(() => Load(TestDetectorModels.Random(s, 1, tensors: orientWrong)));
        Assert.Throws<OcrModelException>(() => Load(TestDetectorModels.Random(s, 1, tensors: tensors.Where(t => t.Name != "layout16.1.pw.weight").ToList())));
    }

    [Fact]
    public void V1BranchSizesAreLimited()
    {
        TestDetectorModels.Shape wide = TestDetectorModels.V1Small with { Layout16 = 100_000 };
        byte[] file = TestContainerWriter.Write(
            "dbnet-lite-v1", 0, TestDetectorModels.Charset(wide), TestDetectorModels.ConfigJson(wide),
            TestDetectorModels.RandomTensors(TestDetectorModels.V1Small, 1).Take(2).ToList(), calibrationJson: "{\"raw\":\"word-mean-max-run\",\"scripts\":{}}");
        Assert.Equal(nameof(OcrLimits.MaxLayerWidth), Assert.Throws<OcrLimitException>(() => Load(file)).LimitName);
        TestDetectorModels.Shape deep = TestDetectorModels.V1Small with { Layout16Blocks = 1_000_000 };
        file = TestContainerWriter.Write(
            "dbnet-lite-v1", 0, TestDetectorModels.Charset(deep), TestDetectorModels.ConfigJson(deep),
            TestDetectorModels.RandomTensors(TestDetectorModels.V1Small, 1).Take(2).ToList(), calibrationJson: "{\"raw\":\"word-mean-max-run\",\"scripts\":{}}");
        Assert.Equal(nameof(OcrLimits.MaxConvBlocks), Assert.Throws<OcrLimitException>(() => Load(file)).LimitName);
        // Branch blocks count toward the total block limit with the backbone and the layout head.
        byte[] small = TestDetectorModels.Random(TestDetectorModels.V1Small, 1);
        Assert.Equal(nameof(OcrLimits.MaxConvBlocks), Assert.Throws<OcrLimitException>(() => Load(small, new OcrLimits { MaxConvBlocks = 8 })).LimitName);
        // The stride-16 buffers are checked against the activation limit before any is taken.
        DbNetModel model = Load(small);
        Assert.Equal(nameof(OcrLimits.MaxActivationElements), Assert.Throws<OcrLimitException>(() =>
            DbNetForward.CheckInput(model, 64, 64, new OcrLimits { MaxActivationElements = 100 })).LimitName);
    }

    [Fact]
    public void MissingAndMisshapenTensorsAreModelErrors()
    {
        TestDetectorModels.Shape s = TestDetectorModels.Small;
        List<TestTensor> all = TestDetectorModels.RandomTensors(s, 1);
        byte[] missing = TestDetectorModels.Random(s, 1, tensors: all.Where(t => t.Name != "smooth4.pw.bias").ToList());
        Assert.Throws<OcrModelException>(() => Load(missing));
        List<TestTensor> reshaped = all.Select(t => t.Name == "lat1.weight" ? new TestTensor { Name = t.Name, Shape = [t.Shape[1], t.Shape[0], 1, 1], Values = t.Values } : t).ToList();
        Assert.Throws<OcrModelException>(() => Load(TestDetectorModels.Random(s, 1, tensors: reshaped)));
        // A container that declares the detector's input height 0 for another architecture is rejected by the container itself.
        Assert.Throws<OcrModelException>(() => ModelContainer.Parse(TestContainerWriter.Write("crnn-v0", 0, ["<blank>", "a"], "{}", [])));
    }

    [Fact]
    public void WidthsBeyondTheLimitsAreRejectedBeforeAnyTensorIsRead()
    {
        TestDetectorModels.Shape huge = TestDetectorModels.Small with { Widths = [16, 24, 40, 100_000] };
        // Only a tensor or two exist: the width check must fire first.
        byte[] file = TestContainerWriter.Write(
            "dbnet-lite-v0", 0, TestDetectorModels.Charset(huge), TestDetectorModels.ConfigJson(huge),
            TestDetectorModels.RandomTensors(TestDetectorModels.Small, 1).Take(2).ToList(), calibrationJson: "{\"raw\":\"word-mean-max-run\",\"scripts\":{}}");
        var ex = Assert.Throws<OcrLimitException>(() => Load(file));
        Assert.Equal(nameof(OcrLimits.MaxLayerWidth), ex.LimitName);
        TestDetectorModels.Shape deep = TestDetectorModels.Small with { Depths = [1, 1_000_000, 1, 1] };
        file = TestContainerWriter.Write(
            "dbnet-lite-v0", 0, TestDetectorModels.Charset(deep), TestDetectorModels.ConfigJson(deep),
            TestDetectorModels.RandomTensors(TestDetectorModels.Small, 1).Take(2).ToList(), calibrationJson: "{\"raw\":\"word-mean-max-run\",\"scripts\":{}}");
        Assert.Equal(nameof(OcrLimits.MaxConvBlocks), Assert.Throws<OcrLimitException>(() => Load(file)).LimitName);
        // The same real model is fine until its own limits are lowered.
        byte[] small = TestDetectorModels.Random(TestDetectorModels.Small, 1);
        Assert.Equal(nameof(OcrLimits.MaxLayerWidth), Assert.Throws<OcrLimitException>(() => Load(small, new OcrLimits { MaxLayerWidth = 30 })).LimitName);
    }

    [Fact]
    public void InputSizesAreCheckedBeforeBuffersAreTaken()
    {
        DbNetModel model = Load(TestDetectorModels.Random(TestDetectorModels.Small, 1));
        var pool = new BufferPool<float>(1 << 20, 4);
        float[] pixels = new float[64 * 64];
        Assert.Throws<ArgumentException>(() => DbNetForward.Run(model, KernelPath.Scalar, 1, pool, pixels, 40, 64, OcrLimits.Default, null, CancellationToken.None));
        Assert.Throws<ArgumentException>(() => DbNetForward.Run(model, KernelPath.Scalar, 1, pool, new float[10], 64, 64, OcrLimits.Default, null, CancellationToken.None));
        var side = Assert.Throws<OcrLimitException>(() =>
            DbNetForward.CheckInput(model, 32, 8192, OcrLimits.Default));
        Assert.Equal(nameof(OcrLimits.MaxDetectorSide), side.LimitName);
        var activation = Assert.Throws<OcrLimitException>(() =>
            DbNetForward.Run(model, KernelPath.Scalar, 1, pool, pixels, 64, 64, new OcrLimits { MaxActivationElements = 1000 }, null, CancellationToken.None));
        Assert.Equal(nameof(OcrLimits.MaxActivationElements), activation.LimitName);
        // A page whose detection would pass the line limit: the limit is an error, not a truncation.
        var prob = new float[40 * 40];
        Array.Fill(prob, -5f);
        for (int y = 0; y < 40; y += 2)
        {
            for (int x = 0; x < 40; x++)
            {
                prob[(y * 40) + x] = 3f;
            }
        }

        var lines = Assert.Throws<OcrLimitException>(() => DetectorPostprocess.Decode(
            prob, 40, 40, new float[5 * 5 * 16], 5, 5, 16, TestDetectorModels.AllClasses, new bool[8], new float[4], 1.0, 80, 80, new OcrLimits { MaxLinesPerPage = 3, LineCandidateFloor = 0 }));
        Assert.Equal(nameof(OcrLimits.MaxLinesPerPage), lines.LimitName);
        var regions = Assert.Throws<OcrLimitException>(() => DetectorPostprocess.Decode(
            new float[4], 2, 2, Enumerable.Repeat(3f, 4 * 4 * 16).ToArray(), 4, 4, 16, TestDetectorModels.AllClasses, Enumerable.Repeat(true, 8).ToArray(), new float[4], 1.0, 8, 8,
            new OcrLimits { MaxPageElements = 3 }));
        Assert.Equal(nameof(OcrLimits.MaxPageElements), regions.LimitName);
    }

    [Fact]
    public void LabellingIsIterativeSoALargeMapNeverDeepensTheStack()
    {
        // One snake-shaped component over a quarter million cells: a recursive fill would overflow the stack.
        const int n = 512;
        var prob = new float[n * n];
        Array.Fill(prob, -5f);
        for (int y = 0; y < n; y += 2)
        {
            for (int x = 0; x < n; x++)
            {
                prob[(y * n) + x] = 4f;
            }

            if (y + 2 < n)
            {
                prob[((y + 1) * n) + ((y / 2) % 2 == 0 ? n - 1 : 0)] = 4f;
            }
        }

        PageDetection d = DetectorPostprocess.Decode(prob, n, n, new float[16], 1, 1, 16, TestDetectorModels.AllClasses, new bool[8], new float[4], 1.0, 2 * n, 2 * n, OcrLimits.Default);
        Assert.Single(d.Lines);
    }

    [Fact]
    public void ThresholdsAndExpansionsFollowTheOptions()
    {
        // One text line of 4 x 2 cells (probability 0.5) and one layout cell (probability 0.5) in the first class.
        var prob = new float[8 * 8];
        Array.Fill(prob, -6f);
        for (int x = 2; x < 6; x++)
        {
            prob[(3 * 8) + x] = 0f;
            prob[(4 * 8) + x] = 0f;
        }

        var layout = new float[8 * 8 * 16];
        Array.Fill(layout, -6f);
        layout[((3 * 8) + 3) * 16] = 0f;

        PageDetection Run(DetectorPostprocess.Options? o) => DetectorPostprocess.Decode(
            prob, 8, 8, layout, 8, 8, 16, TestDetectorModels.AllClasses, new bool[8].Select(_ => true).ToArray(), new float[4], 1.0, 128, 128, OcrLimits.Default, o);

        PageDetection byDefault = Run(null);
        Assert.Single(byDefault.Lines);
        Assert.Empty(byDefault.Regions); // 0.5 is under the default layout threshold of 0.6

        PageDetection looser = Run(new DetectorPostprocess.Options(0.3, 0.5, 0.4, 0.5));
        Assert.Single(looser.Regions);
        Assert.True(looser.Lines[0].Right - looser.Lines[0].Left < byDefault.Lines[0].Right - byDefault.Lines[0].Left);

        Assert.Empty(Run(new DetectorPostprocess.Options(0.6, 1.0, 0.6, 0.85)).Lines);
    }

    [Fact]
    public void OrientationIsTheSoftmaxWinner()
    {
        Assert.Equal(1, DetectorPostprocess.Orientation([0.1f, 3f, 0f, 0f]).Quarter);
        (int quarter, float confidence) = DetectorPostprocess.Orientation([1f, 1f, 1f, 1f]);
        Assert.Equal(0, quarter);
        Assert.InRange(confidence, 0.2499f, 0.2501f);
        Assert.True(DetectorPostprocess.Orientation([0f, 0f, 9f, 0f]).Confidence > DetectorPostprocess.OrientationConfidenceMinimum);
    }
}
