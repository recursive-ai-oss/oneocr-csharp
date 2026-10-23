using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using OneImage;

namespace OneOcr.Kernels;

/// <summary>Views over pixel arrays that spare a copy; the only place pixel structs are reinterpreted.</summary>
internal static class PixelSpans
{
    /// <summary>The bytes of a run of <see cref="Gray8"/> pixels (one byte each, no padding), without copying.</summary>
    /// <param name="pixels">The pixels.</param>
    /// <returns>The same memory as bytes, one per pixel.</returns>
    public static ReadOnlySpan<byte> AsBytes(ReadOnlySpan<Gray8> pixels)
    {
        // Gray8 is a single byte; this guards against a layout change in the imaging package.
        if (!Gray8IsOneByte)
        {
            throw new InvalidOperationException("Gray8 is not one byte per pixel.");
        }

        return MemoryMarshal.Cast<Gray8, byte>(pixels);
    }

    /// <summary>Whether <see cref="Gray8"/> is laid out as one byte per pixel, which <see cref="AsBytes"/> relies on.</summary>
    public static bool Gray8IsOneByte => Unsafe.SizeOf<Gray8>() == 1;
}
