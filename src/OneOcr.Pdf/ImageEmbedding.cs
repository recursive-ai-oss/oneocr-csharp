using System.Buffers.Binary;
using System.IO.Compression;
using OneImage;
using OnePdf;

namespace OneOcr.Pdf;

/// <summary>
/// Puts decoded image frames into a PDF without losing a pixel: bilevel frames stay 1 bit, gray frames stay 8-bit gray, everything
/// else is 8-bit RGB. The samples are written as a PNG and handed to OnePDF, which copies the PNG's compressed data into the PDF
/// as it is (Flate with PNG predictors), so there is a single compression pass and no re-encoding of lossy data.
/// </summary>
internal static class ImageEmbedding
{
    private const int Band = 256;

    /// <summary>How a frame was stored, for the diagnostics.</summary>
    internal enum Kind
    {
        Bilevel,
        Gray8,
        Rgb8,
    }

    /// <summary>Reads <paramref name="frame"/> and embeds it in <paramref name="document"/>.</summary>
    public static PdfCreatedImage Embed(PdfDocument document, ImageFrame frame, out Kind kind, out byte[] encoded, CancellationToken cancellationToken)
    {
        int w = frame.Width, h = frame.Height;
        bool gray = frame.Format.ColorChannels == 1;
        byte[] png;
        if (gray)
        {
            byte[] samples = ReadGray(frame, w, h, cancellationToken);
            if (IsBilevel(samples))
            {
                kind = Kind.Bilevel;
                png = PngEncoder.Encode(w, h, bitDepth: 1, colorType: 0, Pack1(samples, w, h), cancellationToken);
            }
            else
            {
                kind = Kind.Gray8;
                png = PngEncoder.Encode(w, h, 8, 0, samples, cancellationToken);
            }
        }
        else
        {
            byte[] rgb = ReadRgb(frame, w, h, cancellationToken);
            if (IsNeutral(rgb))
            {
                byte[] samples = new byte[(long)w * h];
                for (long i = 0; i < samples.Length; i++)
                {
                    samples[i] = rgb[3 * i];
                }

                if (IsBilevel(samples))
                {
                    kind = Kind.Bilevel;
                    png = PngEncoder.Encode(w, h, 1, 0, Pack1(samples, w, h), cancellationToken);
                }
                else
                {
                    kind = Kind.Gray8;
                    png = PngEncoder.Encode(w, h, 8, 0, samples, cancellationToken);
                }
            }
            else
            {
                kind = Kind.Rgb8;
                png = PngEncoder.Encode(w, h, 8, 2, rgb, cancellationToken);
            }
        }

        encoded = png;
        return document.AddPngImage(png);
    }

    private static byte[] ReadGray(ImageFrame frame, int w, int h, CancellationToken ct)
    {
        long count = checked((long)w * h);
        byte[] samples = new byte[count <= Array.MaxLength ? (int)count : throw new OcrLimitException("ImagePixels", Array.MaxLength, count)];
        Gray8[] band = new Gray8[w * Math.Min(Band, h)];
        for (int y = 0; y < h; y += Band)
        {
            int rows = Math.Min(Band, h - y);
            frame.CopyPixelsTo(band.AsSpan(0, w * rows), new ImageRect(0, y, w, rows), 1, ct);
            for (int i = 0; i < w * rows; i++)
            {
                samples[(long)y * w + i] = band[i].V;
            }
        }

        return samples;
    }

    private static byte[] ReadRgb(ImageFrame frame, int w, int h, CancellationToken ct)
    {
        long total = checked(3L * w * h);
        byte[] samples = new byte[total <= Array.MaxLength ? (int)total : throw new OcrLimitException("ImagePixels", Array.MaxLength, total)];
        Rgb24[] band = new Rgb24[w * Math.Min(Band, h)];
        for (int y = 0; y < h; y += Band)
        {
            int rows = Math.Min(Band, h - y);
            frame.CopyPixelsTo(band.AsSpan(0, w * rows), new ImageRect(0, y, w, rows), 1, ct);
            long o = 3L * y * w;
            for (int i = 0; i < w * rows; i++)
            {
                samples[o++] = band[i].R;
                samples[o++] = band[i].G;
                samples[o++] = band[i].B;
            }
        }

        return samples;
    }

    private static bool IsBilevel(ReadOnlySpan<byte> samples)
    {
        foreach (byte s in samples)
        {
            if (s is not (0 or 255))
            {
                return false;
            }
        }

        return true;
    }

    private static bool IsNeutral(ReadOnlySpan<byte> rgb)
    {
        for (int i = 0; i + 2 < rgb.Length; i += 3)
        {
            if (rgb[i] != rgb[i + 1] || rgb[i] != rgb[i + 2])
            {
                return false;
            }
        }

        return true;
    }

    // 1 bit per sample, most significant bit first, rows padded to whole bytes; 255 is white, as in PDF DeviceGray.
    private static byte[] Pack1(ReadOnlySpan<byte> samples, int w, int h)
    {
        int stride = (w + 7) / 8;
        byte[] packed = new byte[checked(stride * h)];
        for (int y = 0; y < h; y++)
        {
            ReadOnlySpan<byte> row = samples.Slice(y * w, w);
            int o = y * stride;
            for (int x = 0; x < w; x++)
            {
                if (row[x] != 0)
                {
                    packed[o + (x >> 3)] |= (byte)(0x80 >> (x & 7));
                }
            }
        }

        return packed;
    }
}

/// <summary>A minimal PNG writer (ISO/IEC 15948) for the sample layouts the embedder produces. Rows pick the filter with the smallest sum of absolute values, the usual heuristic.</summary>
internal static class PngEncoder
{
    private static readonly uint[] s_crcTable = MakeCrcTable();

    public static byte[] Encode(int width, int height, int bitDepth, int colorType, byte[] rows, CancellationToken cancellationToken)
    {
        int channels = colorType == 2 ? 3 : 1;
        int stride = (width * channels * bitDepth + 7) / 8;
        int bpp = Math.Max(1, channels * bitDepth / 8);
        if (rows.Length != checked(stride * height))
        {
            throw new ArgumentException("Row data does not match the image size.", nameof(rows));
        }

        using var compressed = new MemoryStream();
        using (var z = new ZLibStream(compressed, CompressionLevel.SmallestSize, leaveOpen: true))
        {
            byte[][] candidate = [new byte[stride], new byte[stride], new byte[stride], new byte[stride], new byte[stride]];
            byte[] zero = new byte[stride];
            for (int y = 0; y < height; y++)
            {
                if ((y & 63) == 0)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                }

                ReadOnlySpan<byte> row = rows.AsSpan(y * stride, stride);
                ReadOnlySpan<byte> prior = y == 0 ? zero : rows.AsSpan((y - 1) * stride, stride);
                int best = 0;
                long bestScore = long.MaxValue;
                for (int f = 0; f < 5; f++)
                {
                    Filter(f, row, prior, bpp, candidate[f]);
                    long score = 0;
                    foreach (byte b in candidate[f])
                    {
                        score += b < 128 ? b : 256 - b;
                    }

                    if (score < bestScore)
                    {
                        bestScore = score;
                        best = f;
                    }
                }

                z.WriteByte((byte)best);
                z.Write(candidate[best]);
            }
        }

        using var png = new MemoryStream();
        png.Write([0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        Span<byte> ihdr = stackalloc byte[13];
        BinaryPrimitives.WriteUInt32BigEndian(ihdr, (uint)width);
        BinaryPrimitives.WriteUInt32BigEndian(ihdr[4..], (uint)height);
        ihdr[8] = (byte)bitDepth;
        ihdr[9] = (byte)colorType;
        Chunk(png, "IHDR"u8, ihdr);
        Chunk(png, "IDAT"u8, compressed.GetBuffer().AsSpan(0, (int)compressed.Length));
        Chunk(png, "IEND"u8, []);
        return png.ToArray();
    }

    private static void Filter(int type, ReadOnlySpan<byte> row, ReadOnlySpan<byte> prior, int bpp, Span<byte> output)
    {
        for (int i = 0; i < row.Length; i++)
        {
            int left = i >= bpp ? row[i - bpp] : 0;
            int up = prior[i];
            int upLeft = i >= bpp ? prior[i - bpp] : 0;
            int predictor = type switch
            {
                0 => 0,
                1 => left,
                2 => up,
                3 => (left + up) >> 1,
                _ => Paeth(left, up, upLeft),
            };
            output[i] = (byte)(row[i] - predictor);
        }
    }

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c;
        int pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }

    private static void Chunk(Stream output, ReadOnlySpan<byte> type, ReadOnlySpan<byte> data)
    {
        Span<byte> head = stackalloc byte[8];
        BinaryPrimitives.WriteUInt32BigEndian(head, (uint)data.Length);
        type.CopyTo(head[4..]);
        output.Write(head);
        output.Write(data);
        uint crc = 0xFFFFFFFF;
        foreach (byte b in head[4..])
        {
            crc = s_crcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
        }

        foreach (byte b in data)
        {
            crc = s_crcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
        }

        Span<byte> tail = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(tail, ~crc);
        output.Write(tail);
    }

    private static uint[] MakeCrcTable()
    {
        uint[] table = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++)
            {
                c = (c & 1) != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1;
            }

            table[n] = c;
        }

        return table;
    }
}
