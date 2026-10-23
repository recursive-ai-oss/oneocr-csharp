using System.Runtime.Intrinsics.X86;

namespace OneOcr.Kernels;

/// <summary>The instruction-set path a kernel runs on. All paths produce bit-identical results; scalar is the reference.</summary>
internal enum KernelPath
{
    /// <summary>Portable code using fused multiply-add one element at a time.</summary>
    Scalar = 0,

    /// <summary>256-bit vectors with FMA.</summary>
    Avx2 = 1,
}

internal static class Dispatch
{
    /// <summary>Whether the AVX2 path can run on this machine (it needs FMA too).</summary>
    public static bool Avx2Available => Avx2.IsSupported && Fma.IsSupported;

    /// <summary>The fastest path this machine supports. Chosen once per call by the callers, never per element.</summary>
    public static KernelPath Best => Avx2Available ? KernelPath.Avx2 : KernelPath.Scalar;

    public static KernelPath Require(KernelPath path)
    {
        if (path == KernelPath.Avx2 && !Avx2Available)
        {
            throw new PlatformNotSupportedException("The AVX2 kernel path needs AVX2 and FMA.");
        }

        return path;
    }
}
