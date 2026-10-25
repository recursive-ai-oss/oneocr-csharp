using System.Text.Json;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Detection;

/// <summary>A depthwise 3x3 convolution followed by a 1x1 convolution, each with folded batch norm and ReLU.</summary>
internal sealed class DwBlock
{
    public required int In { get; init; }
    public required int Out { get; init; }

    /// <summary><see cref="In"/> rounded up to a multiple of 16: the channel stride of the block's input.</summary>
    public int InPadded => DetectKernels.Padded(In);

    /// <summary><see cref="Out"/> rounded up to a multiple of 16: the channel stride of the block's output.</summary>
    public int OutPadded => DetectKernels.Padded(Out);

    /// <summary>Depthwise weights as <c>[tap][InPadded]</c>, taps in (ky, kx) order, padding zero.</summary>
    public required float[] DepthwiseWeights { get; init; }

    /// <summary>Depthwise bias, <see cref="InPadded"/> values, padding zero.</summary>
    public required float[] DepthwiseBias { get; init; }

    /// <summary>The 1x1 convolution.</summary>
    public required PointwiseConv Pointwise { get; init; }
}

/// <summary>A 1x1 convolution: a matrix product over channels. Weights are <c>[Out][InPadded]</c> with zero padding in the inputs.</summary>
internal sealed class PointwiseConv
{
    public required int In { get; init; }
    public required int Out { get; init; }
    public int InPadded => DetectKernels.Padded(In);
    public int OutPadded => DetectKernels.Padded(Out);
    public required float[] Weights { get; init; }
    public required float[] Bias { get; init; }
}

/// <summary>
/// The validated, widened weights of a <c>dbnet-lite-v0</c> or <c>dbnet-lite-v1</c> page detector (docs/spec/detector-v0.md, detector-v1.md): shapes are checked against the
/// configuration and widths against <see cref="OcrLimits"/> before anything is allocated from them.
/// </summary>
internal sealed class DbNetModel
{
    /// <summary>Layout class names that have no training labels unless the model's configuration says otherwise.</summary>
    private static readonly string[] UntrainedByDefault = ["form_field", "checkbox"];

    public required ModelInputView Input { get; init; }

    public required int StemChannels { get; init; }

    /// <summary>Stem weights as <c>[tap][StemPadded]</c>.</summary>
    public required float[] StemWeights { get; init; }

    public required float[] StemBias { get; init; }

    public int StemPadded => DetectKernels.Padded(StemChannels);

    /// <summary>The four stages' blocks; the first block of each stage has stride 2.</summary>
    public required DwBlock[][] Stages { get; init; }

    /// <summary>The 1x1 lateral convolutions on the four stage outputs (no ReLU).</summary>
    public required PointwiseConv[] Lateral { get; init; }

    public required DwBlock Smooth8 { get; init; }
    public required DwBlock Smooth4 { get; init; }
    public required DwBlock[] LayoutBlocks { get; init; }

    /// <summary>The stride-16 context blocks of a <c>dbnet-lite-v1</c> layout head, on p16 (empty for v0 graphs and v1 files without the branch).</summary>
    public DwBlock[] Layout16Blocks { get; init; } = [];

    /// <summary>The 1x1 convolution (no ReLU) that takes the stride-16 branch to the layout width before it is upsampled and added; null without the branch.</summary>
    public PointwiseConv? LayoutMix { get; init; }

    /// <summary>Whether the orientation logits read the channel means of stage 2 followed by those of stage 3 (v1 <c>orient16</c>).</summary>
    public bool OrientFromTwoScales { get; init; }

    public required PointwiseConv LayoutOut { get; init; }
    public required DwBlock Text4 { get; init; }
    public required PointwiseConv StemProjection { get; init; }
    public required DwBlock Text2 { get; init; }
    public required PointwiseConv ProbOut { get; init; }

    /// <summary>Orientation classifier weights <c>[4][channels of stage 3]</c>.</summary>
    public required float[] OrientWeights { get; init; }

    public required float[] OrientBias { get; init; }

    /// <summary>Layout class names in channel order.</summary>
    public required IReadOnlyList<string> LayoutClasses { get; init; }

    /// <summary>Per layout class, whether the model was trained on it; untrained classes are never reported.</summary>
    public required IReadOnlyList<bool> LayoutTrained { get; init; }

    /// <summary>Post-processing values from <c>config.postprocess</c>, the defaults where the model gives none.</summary>
    public DetectorPostprocess.Options Postprocess { get; init; } = DetectorPostprocess.Options.Default;

    public int Classes => LayoutClasses.Count;

    private static double CheckProbability(double v, string name) =>
        v > 0 && v < 1 ? v : throw new OcrModelException($"config.postprocess.{name} must be strictly between 0 and 1.");

    private static double CheckExpand(double v, string name) =>
        v > 0 && v <= 4 ? v : throw new OcrModelException($"config.postprocess.{name} must be in (0, 4].");

    public static DbNetModel FromContainer(ModelContainer container, OcrLimits limits)
    {
        ArgumentNullException.ThrowIfNull(container);
        ArgumentNullException.ThrowIfNull(limits);
        if (container.Architecture is not ("dbnet-lite-v0" or "dbnet-lite-v1"))
        {
            throw new OcrModelException($"Unknown architecture '{container.Architecture}'; this reader runs 'dbnet-lite-v0' and 'dbnet-lite-v1'.");
        }

        DbNetConfigDto config;
        try
        {
            config = container.ConfigElement.Deserialize(ManifestJsonContext.Default.DbNetConfigDto)
                ?? throw new OcrModelException("The model configuration is null.");
        }
        catch (JsonException ex)
        {
            throw new OcrModelException("The model configuration is not valid: " + ex.Message, ex);
        }

        if (config is null || config.Widths is null || config.Depths is null || config.LayoutClasses is null)
        {
            throw new OcrModelException("The model configuration lacks widths, depths or layout_classes.");
        }

        if (config.Arch != "dbnet-lite")
        {
            throw new OcrModelException($"config.arch is '{config.Arch}'; expected 'dbnet-lite'.");
        }

        if (container.Input.Channels != 1 || container.Input.Height != 0)
        {
            throw new OcrModelException("dbnet-lite reads grayscale input of any height: input.channels must be 1 and input.height 0.");
        }

        if (config.Widths.Count != 4 || config.Depths.Count != 4)
        {
            throw new OcrModelException("config.widths and config.depths must each have four entries.");
        }

        CheckWidth(config.Stem, limits);
        CheckWidth(config.Fpn, limits);
        CheckWidth(config.Text, limits);
        CheckWidth(config.Layout, limits);
        CheckWidth(config.Classes, limits);
        long blocks = 0;
        for (int i = 0; i < 4; i++)
        {
            CheckWidth(config.Widths[i], limits);
            if (config.Depths[i] < 1 || config.Depths[i] > limits.MaxConvBlocks)
            {
                throw config.Depths[i] < 1
                    ? new OcrModelException("config.depths entries must be at least 1.")
                    : new OcrLimitException(nameof(OcrLimits.MaxConvBlocks), limits.MaxConvBlocks, config.Depths[i]);
            }

            blocks += config.Depths[i];
        }

        if (config.LayoutBlocks < 1 || config.LayoutBlocks > limits.MaxConvBlocks)
        {
            throw config.LayoutBlocks < 1
                ? new OcrModelException("config.layout_blocks must be at least 1.")
                : new OcrLimitException(nameof(OcrLimits.MaxConvBlocks), limits.MaxConvBlocks, config.LayoutBlocks);
        }

        blocks += config.LayoutBlocks;

        // The v1 switches: absent or 0/false is exactly the v0 graph, which a v0 file may not leave.
        int layout16Blocks = config.Layout16Blocks ?? 0;
        bool orient16 = config.Orient16 ?? false;
        if ((layout16Blocks != 0 || orient16 || config.Layout16 is not null) && container.Architecture != "dbnet-lite-v1")
        {
            throw new OcrModelException("layout16_blocks, layout16 and orient16 belong to architecture 'dbnet-lite-v1'.");
        }

        if (layout16Blocks < 0 || layout16Blocks > limits.MaxConvBlocks)
        {
            throw layout16Blocks < 0
                ? new OcrModelException("config.layout16_blocks must not be negative.")
                : new OcrLimitException(nameof(OcrLimits.MaxConvBlocks), limits.MaxConvBlocks, layout16Blocks);
        }

        if (layout16Blocks > 0)
        {
            CheckWidth(config.Layout16 ?? 0, limits);
            blocks += layout16Blocks;
        }
        if (blocks > limits.MaxConvBlocks)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxConvBlocks), limits.MaxConvBlocks, blocks);
        }

        if (config.LayoutClasses.Count != config.Classes)
        {
            throw new OcrModelException($"config.classes is {config.Classes} but layout_classes has {config.LayoutClasses.Count} entries.");
        }

        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (string name in config.LayoutClasses)
        {
            if (string.IsNullOrEmpty(name) || !names.Add(name))
            {
                throw new OcrModelException("config.layout_classes entries must be non-empty and unique.");
            }

            if (name.Length > limits.MaxCharsetEntryLength)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxCharsetEntryLength), limits.MaxCharsetEntryLength, name.Length);
            }
        }

        bool[] trained = new bool[config.Classes];
        if (config.LayoutTrained is null)
        {
            for (int i = 0; i < trained.Length; i++)
            {
                trained[i] = Array.IndexOf(UntrainedByDefault, config.LayoutClasses[i]) < 0;
            }
        }
        else
        {
            foreach (string name in config.LayoutTrained)
            {
                int index = config.LayoutClasses.IndexOf(name);
                if (index < 0)
                {
                    throw new OcrModelException($"config.layout_trained names '{name}', which is not in layout_classes.");
                }

                trained[index] = true;
            }
        }

        DetectorPostprocess.Options post = DetectorPostprocess.Options.Default;
        if (config.Postprocess is { } pp)
        {
            post = new DetectorPostprocess.Options(
                CheckProbability(pp.TextThreshold ?? post.TextThreshold, "text_threshold"),
                CheckExpand(pp.TextExpand ?? post.TextExpand, "text_expand"),
                CheckProbability(pp.LayoutThreshold ?? post.LayoutThreshold, "layout_threshold"),
                CheckExpand(pp.LayoutExpand ?? post.LayoutExpand, "layout_expand"));
            if (pp.RuleSigmaMin is not null || pp.RuleJunkMax is not null || pp.RuleTextThreshold is not null)
            {
                if (pp.RuleSigmaMin is null || pp.RuleJunkMax is null || pp.RuleTextThreshold is null)
                {
                    throw new OcrModelException("config.postprocess rule_sigma_min, rule_junk_max and rule_text_threshold must be given together.");
                }

                if (!double.IsFinite(pp.RuleSigmaMin.Value) || pp.RuleSigmaMin.Value < 0 || !double.IsFinite(pp.RuleJunkMax.Value) || pp.RuleJunkMax.Value < 0)
                {
                    throw new OcrModelException("config.postprocess rule_sigma_min and rule_junk_max must be finite and not negative.");
                }

                post = post with
                {
                    RuleSigmaMin = pp.RuleSigmaMin,
                    RuleJunkMax = pp.RuleJunkMax,
                    RuleTextThreshold = CheckProbability(pp.RuleTextThreshold.Value, "rule_text_threshold"),
                };
            }
        }

        int stem = config.Stem;
        int stemP = DetectKernels.Padded(stem);
        float[] stemRaw = ReadChecked(container, "stem.weight", stem, 1, 3, 3);
        var stemWeights = new float[checked(9 * stemP)];
        for (int o = 0; o < stem; o++)
        {
            for (int tap = 0; tap < 9; tap++)
            {
                stemWeights[(tap * stemP) + o] = stemRaw[(o * 9) + tap];
            }
        }

        var stages = new DwBlock[4][];
        int cin = stem;
        for (int i = 0; i < 4; i++)
        {
            stages[i] = new DwBlock[config.Depths[i]];
            for (int j = 0; j < stages[i].Length; j++)
            {
                stages[i][j] = ReadBlock(container, $"stage{i}.{j}", cin, config.Widths[i]);
                cin = config.Widths[i];
            }
        }

        var lateral = new PointwiseConv[4];
        for (int i = 0; i < 4; i++)
        {
            lateral[i] = ReadPointwise(container, $"lat{i}", config.Widths[i], config.Fpn);
        }

        var layoutBlocks = new DwBlock[config.LayoutBlocks];
        cin = config.Fpn;
        for (int j = 0; j < layoutBlocks.Length; j++)
        {
            layoutBlocks[j] = ReadBlock(container, $"layout.{j}", cin, config.Layout);
            cin = config.Layout;
        }

        var layout16 = new DwBlock[layout16Blocks];
        cin = config.Fpn;
        for (int j = 0; j < layout16.Length; j++)
        {
            layout16[j] = ReadBlock(container, $"layout16.{j}", cin, config.Layout16!.Value);
            cin = config.Layout16.Value;
        }

        PointwiseConv? layoutMix = layout16Blocks > 0 ? ReadPointwise(container, "layout_mix", config.Layout16!.Value, config.Layout) : null;
        int last = config.Widths[3] + (orient16 ? config.Widths[2] : 0);
        return new DbNetModel
        {
            Input = new ModelInputView(container.Input.Mean, container.Input.Std, container.Input.Invert),
            StemChannels = stem,
            StemWeights = stemWeights,
            StemBias = Pad(ReadChecked(container, "stem.bias", stem), stemP),
            Stages = stages,
            Lateral = lateral,
            Smooth8 = ReadBlock(container, "smooth8", config.Fpn, config.Fpn),
            Smooth4 = ReadBlock(container, "smooth4", config.Fpn, config.Fpn),
            LayoutBlocks = layoutBlocks,
            Layout16Blocks = layout16,
            LayoutMix = layoutMix,
            OrientFromTwoScales = orient16,
            LayoutOut = ReadPointwise(container, "layout_out", config.Layout, config.Classes),
            Text4 = ReadBlock(container, "text4", config.Fpn, config.Text),
            StemProjection = ReadPointwise(container, "stem_proj", stem, config.Text),
            Text2 = ReadBlock(container, "text2", config.Text, config.Text),
            ProbOut = ReadPointwise(container, "prob_out", config.Text, 1),
            OrientWeights = ReadChecked(container, "orient.weight", 4, last),
            OrientBias = ReadChecked(container, "orient.bias", 4),
            LayoutClasses = config.LayoutClasses,
            LayoutTrained = trained,
            Postprocess = post,
        };
    }

    private static DwBlock ReadBlock(ModelContainer c, string path, int cin, int cout)
    {
        int cinP = DetectKernels.Padded(cin);
        float[] raw = ReadChecked(c, $"{path}.dw.weight", cin, 1, 3, 3);
        var weights = new float[checked(9 * cinP)];
        for (int k = 0; k < cin; k++)
        {
            for (int tap = 0; tap < 9; tap++)
            {
                weights[(tap * cinP) + k] = raw[(k * 9) + tap];
            }
        }

        return new DwBlock
        {
            In = cin,
            Out = cout,
            DepthwiseWeights = weights,
            DepthwiseBias = Pad(ReadChecked(c, $"{path}.dw.bias", cin), cinP),
            Pointwise = ReadPointwise(c, $"{path}.pw", cin, cout),
        };
    }

    private static PointwiseConv ReadPointwise(ModelContainer c, string path, int cin, int cout)
    {
        int cinP = DetectKernels.Padded(cin);
        float[] raw = ReadChecked(c, $"{path}.weight", cout, cin, 1, 1);
        var weights = new float[checked(cout * cinP)];
        for (int o = 0; o < cout; o++)
        {
            Array.Copy(raw, (long)o * cin, weights, (long)o * cinP, cin);
        }

        return new PointwiseConv { In = cin, Out = cout, Weights = weights, Bias = ReadChecked(c, $"{path}.bias", cout) };
    }

    private static float[] Pad(float[] values, int length)
    {
        var padded = new float[length];
        values.CopyTo(padded, 0);
        return padded;
    }

    private static void CheckWidth(int value, OcrLimits limits)
    {
        if (value < 1)
        {
            throw new OcrModelException("Layer sizes must be positive.");
        }

        if (value > limits.MaxLayerWidth)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxLayerWidth), limits.MaxLayerWidth, value);
        }
    }

    private static float[] ReadChecked(ModelContainer c, string name, params int[] shape)
    {
        TensorInfo info = c.FindTensor(name) ?? throw new OcrModelException($"The model has no tensor named '{name}'.");
        if (!info.Shape.SequenceEqual(shape))
        {
            throw new OcrModelException($"Tensor '{name}' has shape [{string.Join(", ", info.Shape)}]; expected [{string.Join(", ", shape)}].");
        }

        return c.ReadTensor(name);
    }
}
