using System.Text.Json;
using OneOcr.Model;

namespace OneOcr.Recognition;

/// <summary>One convolution block of a <c>crnn-v0</c> model, with batch normalization already folded in.</summary>
internal sealed class ConvBlock
{
    public required int In { get; init; }
    public required int Out { get; init; }
    public required int KernelH { get; init; }
    public required int KernelW { get; init; }
    public required int PadH { get; init; }
    public required int PadW { get; init; }
    public required int PoolH { get; init; }
    public required int PoolW { get; init; }

    /// <summary>Input length of one dot product, <c>In * KernelH * KernelW</c>.</summary>
    public required int K { get; init; }

    /// <summary><see cref="K"/> rounded up to a multiple of 16 so no dot product has a tail; the padding is zero.</summary>
    public required int KPadded { get; init; }

    /// <summary>Weights as <c>[Out][KPadded]</c> in (in, ky, kx) order.</summary>
    public required float[] Weights { get; init; }

    public required float[] Bias { get; init; }
}

internal sealed class LstmDirection
{
    /// <summary><c>[4 * hidden][in]</c>.</summary>
    public required float[] WeightIh { get; init; }

    /// <summary><c>[4 * hidden][hidden]</c>.</summary>
    public required float[] WeightHh { get; init; }

    /// <summary><c>bias_ih + bias_hh</c>, added once in fp32 at load.</summary>
    public required float[] Bias { get; init; }
}

internal sealed class LstmLayer
{
    public required int In { get; init; }
    public required LstmDirection Forward { get; init; }
    public required LstmDirection Reverse { get; init; }
}

/// <summary>
/// The validated, widened weights of a <c>crnn-v0</c> recognizer (docs/spec/model-container-v0.md): fp16 tensors become fp32
/// here, shapes are checked against the configuration, and the heights are checked to reduce to exactly one row.
/// </summary>
internal sealed class CrnnModel
{
    public required ModelInput Input { get; init; }
    public required IReadOnlyList<string> Charset { get; init; }
    public required ConvBlock[] Conv { get; init; }
    public required int Hidden { get; init; }
    public required LstmLayer[] Lstm { get; init; }
    public required float[] FcWeight { get; init; }
    public required float[] FcBias { get; init; }

    /// <summary>Channels of the last convolution block, the feature size of each sequence step.</summary>
    public int FeatureSize => Conv[^1].Out;

    public int Classes => Charset.Count;

    public static CrnnModel FromContainer(ModelContainer container, OcrLimits limits)
    {
        ArgumentNullException.ThrowIfNull(container);
        ArgumentNullException.ThrowIfNull(limits);
        if (container.Architecture != "crnn-v0")
        {
            throw new OcrModelException($"Unknown architecture '{container.Architecture}'; this reader runs 'crnn-v0'.");
        }

        CrnnConfigDto config;
        try
        {
            config = container.ConfigElement.Deserialize(ManifestJsonContext.Default.CrnnConfigDto)
                ?? throw new OcrModelException("The model configuration is null.");
        }
        catch (JsonException ex)
        {
            throw new OcrModelException("The model configuration is not valid: " + ex.Message, ex);
        }

        if (container.Input.Channels != 1)
        {
            throw new OcrModelException("crnn-v0 reads grayscale input; input.channels must be 1.");
        }

        if (config.Conv.Count == 0)
        {
            throw new OcrModelException("crnn-v0 needs at least one convolution block.");
        }

        if (config.Conv.Count > limits.MaxConvBlocks)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxConvBlocks), limits.MaxConvBlocks, config.Conv.Count);
        }

        if (config.Lstm.Layers < 1 || config.Lstm.Hidden < 1)
        {
            throw new OcrModelException("The LSTM needs at least one layer and one hidden unit.");
        }

        if (config.Lstm.Layers > limits.MaxLstmLayers)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxLstmLayers), limits.MaxLstmLayers, config.Lstm.Layers);
        }

        CheckWidth(config.Lstm.Hidden, limits);
        if (config.Classes != container.Charset.Count)
        {
            throw new OcrModelException($"config.classes is {config.Classes} but the charset has {container.Charset.Count} entries.");
        }

        var blocks = new ConvBlock[config.Conv.Count];
        int channels = container.Input.Channels;
        int height = container.Input.Height;
        for (int i = 0; i < blocks.Length; i++)
        {
            ConvDto d = config.Conv[i];
            CheckWidth(d.Out, limits);
            if (d.Kernel.Count != 2 || d.Padding.Count != 2 || (d.Pool is not null && d.Pool.Count != 2))
            {
                throw new OcrModelException($"Convolution block {i}: kernel, padding and pool must have two entries.");
            }

            int kh = CheckKernel(d.Kernel[0], i, limits), kw = CheckKernel(d.Kernel[1], i, limits);
            int ph = d.Padding[0], pw = d.Padding[1];
            if (ph < 0 || pw < 0 || ph > limits.MaxKernelSize || pw > limits.MaxKernelSize)
            {
                throw new OcrModelException($"Convolution block {i}: padding must be between 0 and {limits.MaxKernelSize}.");
            }

            int poolH = 1, poolW = 1;
            if (d.Pool is not null)
            {
                poolH = CheckKernel(d.Pool[0], i, limits);
                poolW = CheckKernel(d.Pool[1], i, limits);
            }

            // Output height of the convolution, then of the pool (a trailing partial window is dropped).
            height = height + (2 * ph) - kh + 1;
            if (height < 1)
            {
                throw new OcrModelException($"Convolution block {i} leaves no rows for the declared input height.");
            }

            height /= poolH;
            if (height < 1)
            {
                throw new OcrModelException($"The pooling of block {i} leaves no rows for the declared input height.");
            }

            int k = checked(channels * kh * kw);
            int kPadded = (k + 15) & ~15;
            float[] weight = ReadChecked(container, $"conv.{i}.weight", d.Out, channels, kh, kw);
            float[] padded = new float[checked(d.Out * kPadded)];
            for (int o = 0; o < d.Out; o++)
            {
                Array.Copy(weight, (long)o * k, padded, (long)o * kPadded, k);
            }

            blocks[i] = new ConvBlock
            {
                In = channels, Out = d.Out, KernelH = kh, KernelW = kw, PadH = ph, PadW = pw, PoolH = poolH, PoolW = poolW,
                K = k, KPadded = kPadded, Weights = padded, Bias = ReadChecked(container, $"conv.{i}.bias", d.Out),
            };
            channels = d.Out;
        }

        if (height != 1)
        {
            throw new OcrModelException($"The convolution stack leaves {height} rows for input height {container.Input.Height}; it must leave exactly 1.");
        }

        int hidden = config.Lstm.Hidden;
        var layers = new LstmLayer[config.Lstm.Layers];
        int inSize = channels;
        for (int l = 0; l < layers.Length; l++)
        {
            layers[l] = new LstmLayer
            {
                In = inSize,
                Forward = ReadDirection(container, l, string.Empty, hidden, inSize),
                Reverse = ReadDirection(container, l, "_reverse", hidden, inSize),
            };
            inSize = 2 * hidden;
        }

        return new CrnnModel
        {
            Input = container.Input,
            Charset = container.Charset,
            Conv = blocks,
            Hidden = hidden,
            Lstm = layers,
            FcWeight = ReadChecked(container, "fc.weight", config.Classes, 2 * hidden),
            FcBias = ReadChecked(container, "fc.bias", config.Classes),
        };
    }

    private static LstmDirection ReadDirection(ModelContainer c, int layer, string suffix, int hidden, int inSize)
    {
        float[] bih = ReadChecked(c, $"lstm.bias_ih_l{layer}{suffix}", 4 * hidden);
        float[] bhh = ReadChecked(c, $"lstm.bias_hh_l{layer}{suffix}", 4 * hidden);
        var bias = new float[bih.Length];
        for (int i = 0; i < bias.Length; i++)
        {
            bias[i] = bih[i] + bhh[i];
        }

        return new LstmDirection
        {
            WeightIh = ReadChecked(c, $"lstm.weight_ih_l{layer}{suffix}", 4 * hidden, inSize),
            WeightHh = ReadChecked(c, $"lstm.weight_hh_l{layer}{suffix}", 4 * hidden, hidden),
            Bias = bias,
        };
    }

    private static int CheckKernel(int value, int block, OcrLimits limits)
    {
        if (value < 1 || value > limits.MaxKernelSize)
        {
            throw new OcrModelException($"Convolution block {block}: kernel and pool sizes must be between 1 and {limits.MaxKernelSize}.");
        }

        return value;
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
