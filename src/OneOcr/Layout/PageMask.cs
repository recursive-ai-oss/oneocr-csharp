using System.Runtime.CompilerServices;
using System.Buffers.Binary;

namespace OneOcr.Layout;

/// <summary>
/// The page's ink mask at two bits per pixel (0 background, 1 ink, 2 ruling line), four pixels to a byte, rows padded to whole
/// bytes. A page-sized mask with one byte per pixel is as large as the page itself; packed it is a quarter of that, and the
/// finders unpack one row at a time into a small scratch row. Rows never share a byte, so threads may write different rows at
/// once; threads that write the same rows must keep to column ranges that start at multiples of four.
/// </summary>
internal sealed class PageMask
{
    public const byte Background = 0;
    public const byte Ink = 1;
    public const byte Rule = 2;

    // Four packed values to four bytes, in memory order.
    private static readonly uint[] Unpack = BuildUnpackTable();

    private readonly byte[] _data;

    public PageMask(byte[] storage, int width, int height)
    {
        Width = width;
        Height = height;
        Stride = (width + 3) / 4;
        if (storage.Length < (long)Stride * height)
        {
            throw new ArgumentException("The storage is too small for the mask.", nameof(storage));
        }

        _data = storage;
    }

    public int Width { get; }

    public int Height { get; }

    /// <summary>Bytes per packed row.</summary>
    public int Stride { get; }

    /// <summary>Bytes needed for a mask of this size.</summary>
    public static long BytesFor(int width, int height) => (long)((width + 3) / 4) * height;

    private static uint[] BuildUnpackTable()
    {
        var table = new uint[256];
        for (int b = 0; b < 256; b++)
        {
            table[b] = (uint)((b & 3) | (((b >> 2) & 3) << 8) | (((b >> 4) & 3) << 16) | (((b >> 6) & 3) << 24));
        }

        return table;
    }

    public byte Get(int x, int y) => (byte)((_data[(y * Stride) + (x >> 2)] >> ((x & 3) * 2)) & 3);

    /// <summary>Sets a pixel to <paramref name="value"/> (0..2), whatever it was.</summary>
    public void Set(int x, int y, byte value)
    {
        int i = (y * Stride) + (x >> 2);
        int shift = (x & 3) * 2;
        _data[i] = (byte)((_data[i] & ~(3 << shift)) | (value << shift));
    }

    /// <summary>Sets pixels <paramref name="from"/> up to but not including <paramref name="to"/> of row <paramref name="y"/>.</summary>
    public void SetRun(int y, int from, int to, byte value)
    {
        for (int x = from; x < to; x++)
        {
            Set(x, y, value);
        }
    }

    /// <summary>Stores a row of 0/1 values (one byte per pixel) as the whole row, replacing it.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void StoreInkRow(int y, ReadOnlySpan<byte> ink)
    {
        Span<byte> row = _data.AsSpan(y * Stride, Stride);
        int x = 0;
        int i = 0;
        for (; x + 4 <= ink.Length; x += 4, i++)
        {
            row[i] = (byte)(ink[x] | (ink[x + 1] << 2) | (ink[x + 2] << 4) | (ink[x + 3] << 6));
        }

        if (x < ink.Length)
        {
            int b = 0;
            for (int k = 0; x + k < ink.Length; k++)
            {
                b |= ink[x + k] << (2 * k);
            }

            row[i] = (byte)b;
        }
    }

    /// <summary>Unpacks <paramref name="count"/> pixels of row <paramref name="y"/> from column <paramref name="x"/> into <paramref name="destination"/>, one byte each.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void ReadRow(int y, int x, int count, Span<byte> destination)
    {
        ReadOnlySpan<byte> row = _data.AsSpan(y * Stride, Stride);
        int i = 0;
        while (i < count && ((x + i) & 3) != 0)
        {
            destination[i] = (byte)((row[(x + i) >> 2] >> (((x + i) & 3) * 2)) & 3);
            i++;
        }

        for (; i + 4 <= count; i += 4)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(destination[i..], Unpack[row[(x + i) >> 2]]);
        }

        for (; i < count; i++)
        {
            destination[i] = (byte)((row[(x + i) >> 2] >> (((x + i) & 3) * 2)) & 3);
        }
    }
}
