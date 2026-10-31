using System.Buffers.Binary;

namespace OneOcr;

/// <summary>
/// Reads the resolution an image file declares (PNG <c>pHYs</c>, JPEG JFIF density, the first TIFF directory's <c>XResolution</c>) from
/// the start of the file, so that a page's <see cref="OcrPage.UnitsPerInch"/> is what the file says instead of an estimate from the size
/// of the text. Only a few header fields are read, every offset is checked against the buffer, and anything unexpected gives no answer.
/// </summary>
internal static class ImageDensity
{
    /// <summary>How much of the start of a file is looked at.</summary>
    public const int WindowBytes = 4 * 1024 * 1024;

    private const double MinDpi = 10;
    private const double MaxDpi = 10_000;

    /// <summary>The declared resolution in pixels per inch, or null when the file declares none or the data is not understood.</summary>
    public static double? TryRead(ReadOnlySpan<byte> data)
    {
        if (data.Length > WindowBytes)
        {
            data = data[..WindowBytes];
        }

        double? dpi = null;
        if (data.Length >= 8 && data[0] == 0x89 && data[1] == (byte)'P' && data[2] == (byte)'N' && data[3] == (byte)'G')
        {
            dpi = Png(data);
        }
        else if (data.Length >= 4 && data[0] == 0xFF && data[1] == 0xD8)
        {
            dpi = Jpeg(data);
        }
        else if (data.Length >= 8 && ((data[0] == (byte)'I' && data[1] == (byte)'I' && data[2] == 42 && data[3] == 0) || (data[0] == (byte)'M' && data[1] == (byte)'M' && data[2] == 0 && data[3] == 42)))
        {
            dpi = Tiff(data);
        }

        return dpi is double d && double.IsFinite(d) && d >= MinDpi && d <= MaxDpi ? d : null;
    }

    // Chunks are length, type, data, CRC; pHYs comes before the first IDAT.
    private static double? Png(ReadOnlySpan<byte> data)
    {
        int pos = 8;
        while (pos + 12 <= data.Length)
        {
            uint length = BinaryPrimitives.ReadUInt32BigEndian(data[pos..]);
            ReadOnlySpan<byte> type = data.Slice(pos + 4, 4);
            long body = pos + 8L;
            if (type.SequenceEqual("pHYs"u8))
            {
                if (length < 9 || body + 9 > data.Length)
                {
                    return null;
                }

                uint perMeter = BinaryPrimitives.ReadUInt32BigEndian(data[(int)body..]);
                return data[(int)body + 8] == 1 ? perMeter * 0.0254 : null;
            }

            if (type.SequenceEqual("IDAT"u8) || type.SequenceEqual("IEND"u8))
            {
                return null;
            }

            long next = body + length + 4;
            if (next > data.Length)
            {
                return null;
            }

            pos = (int)next;
        }

        return null;
    }

    // Segments are FF marker, length (including the two length bytes), data. JFIF is the first APP0.
    private static double? Jpeg(ReadOnlySpan<byte> data)
    {
        int pos = 2;
        while (pos + 4 <= data.Length)
        {
            if (data[pos] != 0xFF)
            {
                return null;
            }

            byte marker = data[pos + 1];
            if (marker == 0xFF)
            {
                pos++;
                continue;
            }

            if (marker == 0xDA || marker == 0xD9)
            {
                return null;
            }

            int length = BinaryPrimitives.ReadUInt16BigEndian(data[(pos + 2)..]);
            if (length < 2)
            {
                return null;
            }

            if (marker == 0xE0 && length >= 16 && pos + 2 + length <= data.Length && data.Slice(pos + 4, 5).SequenceEqual("JFIF\0"u8))
            {
                int units = data[pos + 11];
                int x = BinaryPrimitives.ReadUInt16BigEndian(data[(pos + 12)..]);
                return units switch { 1 => x, 2 => x * 2.54, _ => null };
            }

            pos += 2 + length;
        }

        return null;
    }

    // The first directory's XResolution (a rational at an offset) and ResolutionUnit (default inch).
    private static double? Tiff(ReadOnlySpan<byte> data)
    {
        bool little = data[0] == (byte)'I';
        long ifd = U32(data, little, 4);
        if (ifd < 8 || ifd + 2 > data.Length)
        {
            return null;
        }

        int count = U16(data, little, (int)ifd);
        double? resolution = null;
        int unit = 2;
        for (int i = 0; i < count; i++)
        {
            long entry = ifd + 2 + (12L * i);
            if (entry + 12 > data.Length)
            {
                return null;
            }

            int e = (int)entry;
            int tag = U16(data, little, e);
            int type = U16(data, little, e + 2);
            if (tag == 282 && type == 5 && U32(data, little, e + 4) == 1)
            {
                long at = U32(data, little, e + 8);
                if (at + 8 > data.Length)
                {
                    return null;
                }

                uint numerator = U32(data, little, (int)at), denominator = U32(data, little, (int)at + 4);
                resolution = denominator == 0 ? null : (double)numerator / denominator;
            }
            else if (tag == 296 && type == 3)
            {
                unit = U16(data, little, e + 8);
            }
        }

        return resolution is double r ? unit switch { 2 => r, 3 => r * 2.54, _ => null } : null;
    }

    private static uint U32(ReadOnlySpan<byte> data, bool little, int at) =>
        little ? BinaryPrimitives.ReadUInt32LittleEndian(data[at..]) : BinaryPrimitives.ReadUInt32BigEndian(data[at..]);

    private static int U16(ReadOnlySpan<byte> data, bool little, int at) =>
        little ? BinaryPrimitives.ReadUInt16LittleEndian(data[at..]) : BinaryPrimitives.ReadUInt16BigEndian(data[at..]);
}
