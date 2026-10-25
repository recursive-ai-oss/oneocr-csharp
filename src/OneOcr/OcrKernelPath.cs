namespace OneOcr;

/// <summary>The instruction-set path the engine's kernels run on. Every path gives bit-identical results; they differ in speed.</summary>
public enum OcrKernelPath
{
    /// <summary>Portable code, one fused multiply-add at a time. Exact but slow; used when the CPU or the compiled binary lacks the vector instructions.</summary>
    Scalar = 0,

    /// <summary>128-bit vectors with FMA (x86 FMA3 or Arm NEON).</summary>
    Vector128 = 1,

    /// <summary>256-bit vectors with FMA.</summary>
    Avx2 = 2,

    /// <summary>512-bit vectors (AVX-512F).</summary>
    Avx512 = 3,
}
