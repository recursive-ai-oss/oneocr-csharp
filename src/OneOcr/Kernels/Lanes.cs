using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace OneOcr.Kernels;

/// <summary>
/// A vector of fp32 lanes with the few operations the channel-parallel kernels need. Every kernel written over this interface
/// treats each lane as an independent output (one channel), so its result does not depend on the width: the scalar
/// implementation (one lane) and the 128-, 256- and 512-bit ones produce the same bits. Fused multiply-add is exact and
/// single-rounded on all of them.
/// </summary>
internal interface ILanes<TSelf>
    where TSelf : struct, ILanes<TSelf>
{
    /// <summary>Number of fp32 lanes.</summary>
    static abstract int Width { get; }

    /// <summary>All lanes +0.</summary>
    static abstract TSelf Zero { get; }

    /// <summary>The same value in every lane.</summary>
    static abstract TSelf Broadcast(float value);

    /// <summary>Loads <see cref="Width"/> values; no alignment needed.</summary>
    static abstract unsafe TSelf Load(float* source);

    /// <summary>Stores <see cref="Width"/> values; no alignment needed.</summary>
    unsafe void Store(float* destination);

    /// <summary><c>a * b + c</c> with a single rounding.</summary>
    static abstract TSelf Fma(TSelf a, TSelf b, TSelf c);

    /// <summary><c>a + b</c>.</summary>
    static abstract TSelf Add(TSelf a, TSelf b);

    /// <summary><c>v &gt; 0 ? v : 0</c> per lane (so NaN and -0 become +0).</summary>
    static abstract TSelf Relu(TSelf value);
}

internal readonly struct Lanes1 : ILanes<Lanes1>
{
    private readonly float _v;

    private Lanes1(float v) => _v = v;

    public static int Width => 1;

    public static Lanes1 Zero => default;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 Broadcast(float value) => new(value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe Lanes1 Load(float* source) => new(*source);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void Store(float* destination) => *destination = _v;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 Fma(Lanes1 a, Lanes1 b, Lanes1 c) => new(MathF.FusedMultiplyAdd(a._v, b._v, c._v));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 Add(Lanes1 a, Lanes1 b) => new(a._v + b._v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 Relu(Lanes1 value) => new(value._v > 0f ? value._v : 0f);
}

internal readonly struct Lanes4 : ILanes<Lanes4>
{
    private readonly Vector128<float> _v;

    private Lanes4(Vector128<float> v) => _v = v;

    public static int Width => 4;

    public static Lanes4 Zero => default;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 Broadcast(float value) => new(Vector128.Create(value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe Lanes4 Load(float* source) => new(Vector128.Load(source));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void Store(float* destination) => _v.Store(destination);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 Fma(Lanes4 a, Lanes4 b, Lanes4 c) => new(Vector128.FusedMultiplyAdd(a._v, b._v, c._v));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 Add(Lanes4 a, Lanes4 b) => new(a._v + b._v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 Relu(Lanes4 value) => new(Vector128.GreaterThan(value._v, Vector128<float>.Zero) & value._v);
}

internal readonly struct Lanes8 : ILanes<Lanes8>
{
    private readonly Vector256<float> _v;

    private Lanes8(Vector256<float> v) => _v = v;

    public static int Width => 8;

    public static Lanes8 Zero => default;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes8 Broadcast(float value) => new(Vector256.Create(value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe Lanes8 Load(float* source) => new(Vector256.Load(source));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void Store(float* destination) => _v.Store(destination);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes8 Fma(Lanes8 a, Lanes8 b, Lanes8 c) => new(Vector256.FusedMultiplyAdd(a._v, b._v, c._v));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes8 Add(Lanes8 a, Lanes8 b) => new(a._v + b._v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes8 Relu(Lanes8 value) => new(Vector256.GreaterThan(value._v, Vector256<float>.Zero) & value._v);
}

internal readonly struct Lanes16 : ILanes<Lanes16>
{
    private readonly Vector512<float> _v;

    private Lanes16(Vector512<float> v) => _v = v;

    public static int Width => 16;

    public static Lanes16 Zero => default;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes16 Broadcast(float value) => new(Vector512.Create(value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe Lanes16 Load(float* source) => new(Vector512.Load(source));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void Store(float* destination) => _v.Store(destination);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes16 Fma(Lanes16 a, Lanes16 b, Lanes16 c) => new(Vector512.FusedMultiplyAdd(a._v, b._v, c._v));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes16 Add(Lanes16 a, Lanes16 b) => new(a._v + b._v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes16 Relu(Lanes16 value) => new(Vector512.GreaterThan(value._v, Vector512<float>.Zero) & value._v);
}
