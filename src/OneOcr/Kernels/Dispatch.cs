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

    /// <summary>
    /// Whether the CPU itself offers AVX2 and FMA, asked with CPUID rather than through <c>IsSupported</c>: a NativeAOT binary compiled
    /// for a lower x86-64 level answers <c>false</c> to the latter on every machine. False on other architectures.
    /// </summary>
    public static bool CpuHasAvx2Fma()
    {
        if (!X86Base.IsSupported)
        {
            return false;
        }

        (int maxLeaf, _, _, _) = X86Base.CpuId(0, 0);
        if (maxLeaf < 7)
        {
            return false;
        }

        // Leaf 1 ECX: bit 12 FMA, bit 27 OSXSAVE (the OS saves the wide registers), bit 28 AVX. Leaf 7 EBX: bit 5 AVX2.
        (_, _, int ecx1, _) = X86Base.CpuId(1, 0);
        (_, int ebx7, _, _) = X86Base.CpuId(7, 0);
        const int fma = 1 << 12, osxsave = 1 << 27, avx = 1 << 28, avx2 = 1 << 5;
        return (ecx1 & (fma | osxsave | avx)) == (fma | osxsave | avx) && (ebx7 & avx2) != 0;
    }

    /// <summary>A sentence for the user when <paramref name="used"/> is the scalar path on a CPU that could run AVX2; otherwise null.</summary>
    public static string? SlowPathNotice(KernelPath used)
    {
        if (used != KernelPath.Scalar || !CpuHasAvx2Fma())
        {
            return null;
        }

        return "The scalar kernel path is running although this CPU supports AVX2 and FMA; OCR is far slower (about 100 times for a NativeAOT binary) " +
            "but gives the same text. A NativeAOT app published below x86-64-v3 does this: set <IlcInstructionSet>x86-64-v3</IlcInstructionSet> " +
            "(or x86-64-v4 for AVX-512) in the app project. The RecursiveAI.OneOcr package sets v3 for x64 publishes unless you set your own. " +
            "Turning the instruction sets off with DOTNET_EnableAVX2=0 or DOTNET_EnableHWIntrinsic=0 has the same effect.";
    }
}
