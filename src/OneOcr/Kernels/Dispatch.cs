using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace OneOcr.Kernels;

/// <summary>The instruction-set path a kernel runs on. All paths produce bit-identical results; scalar is the reference.</summary>
internal enum KernelPath
{
    /// <summary>Portable code using fused multiply-add one element at a time (exact, also without hardware FMA).</summary>
    Scalar = 0,

    /// <summary>128-bit vectors with FMA (x86 FMA3 or Arm NEON).</summary>
    Vector128 = 1,

    /// <summary>256-bit vectors with FMA.</summary>
    Avx2 = 2,

    /// <summary>512-bit vectors (AVX-512F).</summary>
    Avx512 = 3,
}

internal static class Dispatch
{
    /// <summary>Whether the AVX-512 path can run on this machine.</summary>
    public static bool Avx512Available => Avx512F.IsSupported && Fma.IsSupported;

    /// <summary>Whether the AVX2 path can run on this machine (it needs FMA too).</summary>
    public static bool Avx2Available => Avx2.IsSupported && Fma.IsSupported;

    /// <summary>Whether the 128-bit path can run: it needs hardware fused multiply-add, otherwise scalar is as good.</summary>
    public static bool Vector128Available => Vector128.IsHardwareAccelerated && (Fma.IsSupported || AdvSimd.IsSupported);

    /// <summary>Whether a path can run on this machine.</summary>
    public static bool IsAvailable(KernelPath path) => path switch
    {
        KernelPath.Scalar => true,
        KernelPath.Vector128 => Vector128Available,
        KernelPath.Avx2 => Avx2Available,
        KernelPath.Avx512 => Avx512Available,
        _ => false,
    };

    /// <summary>The fastest path this machine supports. Chosen once per call by the callers, never per element.</summary>
    public static KernelPath Best =>
        Avx512Available ? KernelPath.Avx512 : Avx2Available ? KernelPath.Avx2 : Vector128Available ? KernelPath.Vector128 : KernelPath.Scalar;

    /// <summary>Every path this machine can run, scalar first.</summary>
    public static KernelPath[] Available() =>
        Enum.GetValues<KernelPath>().Where(IsAvailable).ToArray();

    public static KernelPath Require(KernelPath path)
    {
        if (!IsAvailable(path))
        {
            throw new PlatformNotSupportedException($"The {path} kernel path is not supported on this machine.");
        }

        return path;
    }
}
