using System.Buffers.Binary;

namespace OneOcr.Fuzz;

/// <summary>
/// Image file bytes through the reader of the resolution a file declares (PNG, JPEG and TIFF headers). It must never throw, whatever
/// the bytes; an answer must be a finite number in the accepted range, and the same bytes must always give the same answer.
/// </summary>
internal static class DensityTarget
{
    /// <summary>Inputs that declared a resolution (progress statistic for local runs).</summary>
    public static long Declared;

    public static void Run(ReadOnlySpan<byte> data)
    {
        double? dpi = ImageDensity.TryRead(data);
        if (dpi is double d)
        {
            Declared++;
            if (!double.IsFinite(d) || d < 10 || d > 10_000)
            {
                throw new InvalidOperationException("declared resolution outside the accepted range");
            }
        }

        if (dpi != ImageDensity.TryRead(data))
        {
            throw new InvalidOperationException("the same bytes gave two answers");
        }

        // A truncated file must be handled as well as a whole one.
        _ = ImageDensity.TryRead(data[..(data.Length / 2)]);
    }

    /// <summary>Writes small header-only files that declare a resolution in each supported format, and a few that declare none.</summary>
    public static void WriteSeeds(string dir)
    {
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(Path.Combine(dir, "png-300.seed"), Png(11811, 11811));
        File.WriteAllBytes(Path.Combine(dir, "png-none.seed"), Png(0, 0));
        File.WriteAllBytes(Path.Combine(dir, "jpeg-jfif-dpi.seed"), Jpeg(1, 200, 200));
        File.WriteAllBytes(Path.Combine(dir, "jpeg-jfif-aspect.seed"), Jpeg(0, 1, 1));
        File.WriteAllBytes(Path.Combine(dir, "tiff-le-300.seed"), Tiff(true, 2, 300));
        File.WriteAllBytes(Path.Combine(dir, "tiff-be-cm.seed"), Tiff(false, 3, 118));
        File.WriteAllBytes(Path.Combine(dir, "truncated.seed"), Png(11811, 11811)[..30]);
    }

    private static byte[] Png(uint ppuX, uint ppuY)
    {
        var ms = new MemoryStream();
        ms.Write([0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        Chunk(ms, "IHDR", [0, 0, 0, 8, 0, 0, 0, 8, 8, 0, 0, 0, 0]);
        var phys = new byte[9];
        BinaryPrimitives.WriteUInt32BigEndian(phys, ppuX);
        BinaryPrimitives.WriteUInt32BigEndian(phys.AsSpan(4), ppuY);
        phys[8] = 1;
        Chunk(ms, "pHYs", phys);
        Chunk(ms, "IDAT", [0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01]);
        Chunk(ms, "IEND", []);
        return ms.ToArray();
    }

    private static void Chunk(MemoryStream ms, string type, byte[] body)
    {
        Span<byte> n = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(n, (uint)body.Length);
        ms.Write(n);
        ms.Write(System.Text.Encoding.ASCII.GetBytes(type));
        ms.Write(body);
        ms.Write([0, 0, 0, 0]); // the reader does not check the checksum
    }

    private static byte[] Jpeg(byte units, ushort x, ushort y)
    {
        var app0 = new byte[]
        {
            0xFF, 0xE0, 0x00, 0x10, (byte)'J', (byte)'F', (byte)'I', (byte)'F', 0, 1, 1, units,
            (byte)(x >> 8), (byte)x, (byte)(y >> 8), (byte)y, 0, 0,
        };
        return [0xFF, 0xD8, .. app0, 0xFF, 0xD9];
    }

    private static byte[] Tiff(bool little, ushort unit, uint dpi)
    {
        var ms = new MemoryStream();
        void U16(int v) { Span<byte> b = stackalloc byte[2]; if (little) { BinaryPrimitives.WriteUInt16LittleEndian(b, (ushort)v); } else { BinaryPrimitives.WriteUInt16BigEndian(b, (ushort)v); } ms.Write(b); }
        void U32(uint v) { Span<byte> b = stackalloc byte[4]; if (little) { BinaryPrimitives.WriteUInt32LittleEndian(b, v); } else { BinaryPrimitives.WriteUInt32BigEndian(b, v); } ms.Write(b); }
        ms.Write(little ? "II"u8 : "MM"u8);
        U16(42);
        U32(8);
        U16(2); // entries: XResolution (282, rational) and ResolutionUnit (296)
        U16(282); U16(5); U32(1); U32(8 + 2 + 24 + 4);
        U16(296); U16(3); U32(1); U16(unit); U16(0);
        U32(0);
        U32(dpi); U32(1);
        return ms.ToArray();
    }
}
