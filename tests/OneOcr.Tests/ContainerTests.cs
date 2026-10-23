using System.Buffers.Binary;
using System.Text;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Tests;

public class ContainerTests
{
    private static readonly TestModels.Shape Shape = TestModels.Small;
    private static readonly byte[] Valid = TestModels.RandomContainer(Shape, seed: 1);

    private static int DataStart(byte[] file) => (16 + (int)BinaryPrimitives.ReadUInt32LittleEndian(file.AsSpan(12)) + 63) / 64 * 64;

    /// <summary>Applies a text edit to the manifest and reassembles a file with the same tensor data.</summary>
    private static byte[] EditManifest(byte[] file, Func<string, string> edit)
    {
        int m = (int)BinaryPrimitives.ReadUInt32LittleEndian(file.AsSpan(12));
        string json = Encoding.UTF8.GetString(file, 16, m);
        byte[] edited = Encoding.UTF8.GetBytes(edit(json));
        return Reassemble(file, edited);
    }

    private static byte[] Reassemble(byte[] file, byte[] manifest)
    {
        byte[] data = file[DataStart(file)..];
        int start = (16 + manifest.Length + 63) / 64 * 64;
        var result = new byte[start + data.Length];
        file.AsSpan(0, 8).CopyTo(result);
        BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(12), (uint)manifest.Length);
        manifest.CopyTo(result, 16);
        data.CopyTo(result, start);
        return result;
    }

    private static void AssertRejected(byte[] file, OcrLimits? limits = null)
    {
        Exception ex = Assert.ThrowsAny<Exception>(() => ModelContainer.Parse(file, limits));
        Assert.True(ex is OcrModelException or OcrLimitException, $"unexpected {ex.GetType().Name}: {ex.Message}");
    }

    [Fact]
    public void ReadsAValidFileAndWidensHalfPrecisionExactly()
    {
        var c = ModelContainer.Parse(Valid);
        Assert.Equal("crnn-v0", c.Architecture);
        Assert.Equal(32, c.Input.Height);
        Assert.Equal(Shape.Classes, c.Charset.Count);
        Assert.Equal(5 * 2 + (Shape.LstmLayers * 8) + 2, c.Tensors.Count);
        var source = TestModels.RandomTensors(Shape, 1);
        foreach (TestTensor t in source)
        {
            float[] read = c.ReadTensor(t.Name);
            Assert.Equal(t.Values.Length, read.Length);
            for (int i = 0; i < read.Length; i++)
            {
                Assert.Equal((float)(Half)t.Values[i], read[i]);
            }
        }

        Assert.NotNull(LineRecognizer.FromContainer(c));
    }

    [Fact]
    public void RoundTripsThroughFilesAndNonSeekableStreams()
    {
        string path = Path.Combine(Path.GetTempPath(), $"oneocr-test-{Guid.NewGuid():N}.onm");
        try
        {
            File.WriteAllBytes(path, Valid);
            Assert.Equal(ModelContainer.Parse(Valid).Tensors.Count, ModelContainer.Load(path).Tensors.Count);
            using var forward = new ForwardOnlyStream(Valid);
            Assert.Equal("crnn-v0", ModelContainer.Load(forward).Architecture);
            using var big = new ForwardOnlyStream(new byte[100_000]);
            Assert.Throws<OcrLimitException>(() => ModelContainer.Load(big, new OcrLimits { MaxModelFileBytes = 70_000 }));
            Assert.Throws<OcrLimitException>(() => ModelContainer.Load(new MemoryStream(new byte[100_000]), new OcrLimits { MaxModelFileBytes = 1000 }));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void RejectsEveryTruncation()
    {
        for (int length = 0; length < Valid.Length; length += length < 4096 ? 1 : 997)
        {
            AssertRejected(Valid[..length]);
        }

        AssertRejected(Valid[..^1]);
    }

    [Fact]
    public void RejectsHeaderAndLayoutErrors()
    {
        byte[] bad = (byte[])Valid.Clone();
        bad[0] ^= 1;
        AssertRejected(bad);

        bad = (byte[])Valid.Clone();
        bad[8] = 2;
        AssertRejected(bad);

        bad = (byte[])Valid.Clone();
        BinaryPrimitives.WriteUInt32LittleEndian(bad.AsSpan(12), uint.MaxValue);
        Assert.Throws<OcrLimitException>(() => ModelContainer.Parse(bad));

        bad = (byte[])Valid.Clone();
        BinaryPrimitives.WriteUInt32LittleEndian(bad.AsSpan(12), (uint)Valid.Length);
        AssertRejected(bad);

        AssertRejected([.. Valid, 0]);

        // A non-zero byte in the padding after the manifest.
        int m = (int)BinaryPrimitives.ReadUInt32LittleEndian(Valid.AsSpan(12));
        if (16 + m < DataStart(Valid))
        {
            bad = (byte[])Valid.Clone();
            bad[16 + m] = 7;
            AssertRejected(bad);
        }

        // A non-zero byte in the padding between tensors: conv.0.weight is 144 bytes and the next tensor starts at 192.
        bad = (byte[])Valid.Clone();
        bad[DataStart(Valid) + 150] = 1;
        AssertRejected(bad);
    }

    [Fact]
    public void RejectsManifestErrors()
    {
        // Unknown architecture parses as a container; the recognizer builder rejects it.
        var unknown = ModelContainer.Parse(EditManifest(Valid, j => j.Replace("\"crnn-v0\"", "\"crnn-v9\"")));
        Assert.Throws<OcrModelException>(() => LineRecognizer.FromContainer(unknown));

        AssertRejected(EditManifest(Valid, j => j.Replace("\"f16\"", "\"f32\"")));
        AssertRejected(EditManifest(Valid, j => "\uFEFF" + j));
        AssertRejected(EditManifest(Valid, j => j + " x"));
        AssertRejected(EditManifest(Valid, j => j.Replace("\"charset\"", "\"charse\"")));
        AssertRejected(EditManifest(Valid, j => j.Replace("\"offset\":0", "\"offset\":1")));
        AssertRejected(EditManifest(Valid, j => j.Replace("\"conv.0.weight\"", "\"conv.0.bias\"")));
        AssertRejected(EditManifest(Valid, j => j.Replace("\"std\":0.5", "\"std\":0")));
        AssertRejected(EditManifest(Valid, j => j.Replace("\"height\":32", "\"height\":0")));
        AssertRejected(EditManifest(Valid, j => j.Replace("\"architecture\"", "\"architecture\":\"a\",\"architecture\"")));
        AssertRejected(EditManifest(Valid, j => j.Replace("\"charset\":[", "\"charset\":[\"a\",\"a\",")));
        AssertRejected(EditManifest(Valid, j => j.Replace("\"provenance\":{\"note\":\"test\"}", "\"provenance\":" + new string('[', 100) + new string(']', 100))));
        AssertRejected(Reassemble(Valid, [0x7B, 0xFF, 0x7D]));
        AssertRejected(Reassemble(Valid, "null"u8.ToArray()));
        AssertRejected(Reassemble(Valid, []));
    }

    [Fact]
    public void RejectsOverlapAndOverflowingOffsets()
    {
        // Two tensors at the same offset.
        var tensors = new List<TestTensor>
        {
            new() { Name = "a", Shape = [4], Values = new float[4], Offset = 0 },
            new() { Name = "b", Shape = [4], Values = new float[4], Offset = 0 },
        };
        AssertRejected(TestContainerWriter.Write("crnn-v0", 32, ["<blank>", "a"], "{}", tensors));

        // Offset not a multiple of 64.
        tensors = [new() { Name = "a", Shape = [4], Values = new float[4], Offset = 8 }];
        AssertRejected(TestContainerWriter.Write("crnn-v0", 32, ["<blank>", "a"], "{}", tensors));

        // Offsets and lengths that would overflow 64-bit arithmetic when added.
        AssertRejected(EditManifest(Valid, j => j.Replace("\"offset\":0", "\"offset\":9223372036854775744")));
        AssertRejected(EditManifest(Valid, j => j.Replace("\"offset\":0", "\"offset\":-64")));
        AssertRejected(EditManifest(Valid, j => j.Replace("\"length\":144", "\"length\":9223372036854775807")));
    }

    [Fact]
    public void LimitsAreCheckedBeforeAllocating()
    {
        // Dimensions that would need terabytes if believed.
        Assert.Throws<OcrLimitException>(() => ModelContainer.Parse(EditManifest(Valid, j => j.Replace("\"shape\":[8,1,3,3]", "\"shape\":[2000000000,2000000000,3,3]"))));
        AssertRejected(EditManifest(Valid, j => j.Replace("\"shape\":[8,1,3,3]", "\"shape\":[8,-1,3,3]")));
        AssertRejected(EditManifest(Valid, j => j.Replace("\"shape\":[8,1,3,3]", "\"shape\":[8,1,3,3,1,1,1,1,1,1]")));
        Assert.Throws<OcrLimitException>(() => ModelContainer.Parse(Valid, new OcrLimits { MaxTensorCount = 3 }));
        Assert.Throws<OcrLimitException>(() => ModelContainer.Parse(Valid, new OcrLimits { MaxCharsetEntries = 5 }));
        Assert.Throws<OcrLimitException>(() => ModelContainer.Parse(Valid, new OcrLimits { MaxManifestBytes = 100 }));
        Assert.Throws<OcrLimitException>(() => ModelContainer.Parse(Valid, new OcrLimits { MaxTensorElements = 100 }));

        // Model sizes checked when building the recognizer.
        var c = ModelContainer.Parse(Valid);
        Assert.Throws<OcrLimitException>(() => LineRecognizer.FromContainer(c, new OcrLimits { MaxLayerWidth = 4 }));
        Assert.Throws<OcrLimitException>(() => LineRecognizer.FromContainer(c, new OcrLimits { MaxConvBlocks = 2 }));
        Assert.Throws<OcrModelException>(() => LineRecognizer.FromContainer(ModelContainer.Parse(EditManifest(Valid, j => j.Replace("\"hidden\":24", "\"hidden\":25")))));
    }

    [Fact]
    public void RandomCorruptionNeverEscapesTheTypedExceptions()
    {
        var rng = new Random(99);
        int ok = 0;
        for (int iteration = 0; iteration < 3000; iteration++)
        {
            byte[] mutated = (byte[])Valid.Clone();
            int flips = 1 + rng.Next(4);
            for (int f = 0; f < flips; f++)
            {
                // Concentrate on the header and manifest, where parsing decisions are made.
                int index = rng.Next(3) == 0 ? rng.Next(mutated.Length) : rng.Next(Math.Min(mutated.Length, DataStart(Valid)));
                mutated[index] = (byte)rng.Next(256);
            }

            try
            {
                ModelContainer c = ModelContainer.Parse(mutated);
                ok++;
                try
                {
                    LineRecognizer.FromContainer(c);
                }
                catch (OcrModelException)
                {
                }
                catch (OcrLimitException)
                {
                }
            }
            catch (OcrModelException)
            {
            }
            catch (OcrLimitException)
            {
            }
        }

        Assert.True(ok < 3000);
    }

    private sealed class ForwardOnlyStream(byte[] data) : Stream
    {
        private readonly MemoryStream _inner = new(data);
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) => _inner.Read(buffer, offset, count);
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
