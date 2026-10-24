using System.Buffers.Binary;
using System.Text.Json;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Training;

/// <summary>
/// Writes a fine-tuned <c>crnn-v0</c> model as a container (docs/spec/model-container-v1.md). The result has the same manifest as the model it
/// started from — architecture, configuration, charset, input settings, tensor names and shapes, calibration — with the weights
/// replaced by the trained ones, rounded to fp16 once. The in-memory weights are in the runtime's layouts (convolution taps in
/// (ky, kx, channel) order with zero padding, the two LSTM biases already summed), so they are mapped back here: the combined
/// LSTM bias is stored as <c>bias_ih</c> and <c>bias_hh</c> becomes zero, which reads back as the same sum.
/// </summary>
internal static class ContainerWriter
{
    public static byte[] Write(ModelContainer baseModel, CrnnModel trained, string provenanceNote)
    {
        ArgumentNullException.ThrowIfNull(baseModel);
        ArgumentNullException.ThrowIfNull(trained);

        var values = new float[baseModel.Tensors.Count][];
        for (int i = 0; i < values.Length; i++)
        {
            TensorInfo t = baseModel.Tensors[i];
            values[i] = TrainedValues(baseModel, trained, t);
        }

        var offsets = new long[values.Length];
        long cursor = 0;
        for (int i = 0; i < values.Length; i++)
        {
            offsets[i] = (cursor + 63) / 64 * 64;
            cursor = offsets[i] + (values[i].Length * 2L);
        }

        byte[] manifest = Manifest(baseModel, values, offsets, provenanceNote);
        long dataStart = (16 + (long)manifest.Length + 63) / 64 * 64;
        var file = new byte[checked(dataStart + cursor)];
        "ONEOCRM"u8.CopyTo(file);
        BinaryPrimitives.WriteUInt32LittleEndian(file.AsSpan(8), baseModel.FormatVersion);
        BinaryPrimitives.WriteUInt32LittleEndian(file.AsSpan(12), (uint)manifest.Length);
        manifest.CopyTo(file, 16);
        for (int i = 0; i < values.Length; i++)
        {
            Span<byte> dst = file.AsSpan((int)(dataStart + offsets[i]));
            for (int j = 0; j < values[i].Length; j++)
            {
                BinaryPrimitives.WriteHalfLittleEndian(dst[(j * 2)..], (Half)values[i][j]);
            }
        }

        return file;
    }

    private static float[] TrainedValues(ModelContainer baseModel, CrnnModel m, TensorInfo t)
    {
        string name = t.Name;
        if (name.StartsWith("conv.", StringComparison.Ordinal) && TryIndex(name, "conv.", out int ci, out string rest) && ci < m.Conv.Length)
        {
            ConvBlock b = m.Conv[ci];
            if (rest == "bias")
            {
                return b.Bias;
            }

            if (rest == "weight")
            {
                var w = new float[b.Out * b.In * b.KernelH * b.KernelW];
                for (int o = 0; o < b.Out; o++)
                {
                    for (int ic = 0; ic < b.In; ic++)
                    {
                        for (int ky = 0; ky < b.KernelH; ky++)
                        {
                            for (int kx = 0; kx < b.KernelW; kx++)
                            {
                                w[((((o * b.In) + ic) * b.KernelH) + ky) * b.KernelW + kx] = b.Weights[(o * b.KPadded) + (((ky * b.KernelW) + kx) * b.In) + ic];
                            }
                        }
                    }
                }

                return w;
            }
        }

        if (name == "fc.weight")
        {
            return m.FcWeight;
        }

        if (name == "fc.bias")
        {
            return m.FcBias;
        }

        if (name.StartsWith("lstm.", StringComparison.Ordinal))
        {
            for (int l = 0; l < m.Lstm.Length; l++)
            {
                for (int d = 0; d < 2; d++)
                {
                    string sfx = d == 0 ? string.Empty : "_reverse";
                    LstmDirection dir = d == 0 ? m.Lstm[l].Forward : m.Lstm[l].Reverse;
                    if (name == $"lstm.weight_ih_l{l}{sfx}")
                    {
                        return dir.WeightIh;
                    }

                    if (name == $"lstm.weight_hh_l{l}{sfx}")
                    {
                        return dir.WeightHh;
                    }

                    if (name == $"lstm.bias_ih_l{l}{sfx}")
                    {
                        return dir.Bias;
                    }

                    if (name == $"lstm.bias_hh_l{l}{sfx}")
                    {
                        return new float[dir.Bias.Length];
                    }
                }
            }
        }

        // A tensor this writer does not know is carried over as it was.
        return baseModel.ReadTensor(name);
    }

    private static bool TryIndex(string name, string prefix, out int index, out string rest)
    {
        index = -1;
        rest = "";
        int dot = name.IndexOf('.', prefix.Length);
        if (dot < 0 || !int.TryParse(name.AsSpan(prefix.Length, dot - prefix.Length), System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out index))
        {
            return false;
        }

        rest = name[(dot + 1)..];
        return true;
    }

    private static byte[] Manifest(ModelContainer c, float[][] values, long[] offsets, string note)
    {
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms))
        {
            w.WriteStartObject();
            w.WriteString("architecture", c.Architecture);
            w.WriteString("name", c.Name);
            w.WriteString("version", c.Version);
            w.WriteString("license", c.License);
            w.WriteStartObject("input");
            w.WriteNumber("height", c.Input.Height);
            w.WriteNumber("channels", c.Input.Channels);
            w.WriteNumber("mean", c.Input.Mean);
            w.WriteNumber("std", c.Input.Std);
            w.WriteBoolean("invert", c.Input.Invert);
            w.WriteEndObject();
            w.WriteStartArray("charset");
            foreach (string s in c.Charset)
            {
                w.WriteStringValue(s);
            }

            w.WriteEndArray();
            w.WritePropertyName("config");
            c.ConfigElement.WriteTo(w);
            w.WriteStartArray("tensors");
            for (int i = 0; i < values.Length; i++)
            {
                w.WriteStartObject();
                w.WriteString("name", c.Tensors[i].Name);
                w.WriteString("dtype", "f16");
                w.WriteStartArray("shape");
                foreach (int d in c.Tensors[i].Shape)
                {
                    w.WriteNumberValue(d);
                }

                w.WriteEndArray();
                w.WriteNumber("offset", offsets[i]);
                w.WriteNumber("length", values[i].Length * 2L);
                w.WriteEndObject();
            }

            w.WriteEndArray();
            if (c.CalibrationElement is { } cal)
            {
                w.WritePropertyName("calibration");
                cal.WriteTo(w);
            }

            w.WriteStartObject("provenance");
            w.WriteString("fine_tuned", note);
            w.WriteEndObject();
            w.WriteEndObject();
        }

        return ms.ToArray();
    }
}
