using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace OneOcr.Kernels;

// Vector loops of the element-wise kernels at the three widths (same code, vector type changed). Each returns how many
// elements it handled; the caller finishes the rest with the scalar code, which gives the same bits per element.

internal static unsafe class ElementwiseV128
{
    public const int Lanes = 4;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]

    public static int LstmCell(float* pz, float* pc, float* ph, int hidden)
    {
        int j = 0;
        for (; j + 4 <= hidden; j += 4)
        {
            Vector128<float> i = FastMathV128.Sigmoid(Vector128.Load(pz + j));
            Vector128<float> f = FastMathV128.Sigmoid(Vector128.Load(pz + hidden + j));
            Vector128<float> g = FastMathV128.Tanh(Vector128.Load(pz + (2 * hidden) + j));
            Vector128<float> o = FastMathV128.Sigmoid(Vector128.Load(pz + (3 * hidden) + j));
            Vector128<float> cNew = Vector128.FusedMultiplyAdd(f, Vector128.Load(pc + j), Vector128.Multiply(i, g));
            Vector128.Store(cNew, pc + j);
            Vector128.Store(Vector128.Multiply(o, FastMathV128.Tanh(cNew)), ph + j);
        }

        return j;
    }

    /// <summary>Adds exp(x[k] - shift) into the 16 partials p (element k into partial k mod 16) for whole blocks of 16 elements.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int SumExpBlocks(float* px, int n, float shift, float* p)
    {
        Vector128<float> s = Vector128.Create(shift);
        int k = 0;
        for (; k + 16 <= n; k += 16)
        {
            for (int c = 0; c < 16; c += 4)
            {
                Vector128.Store(Vector128.Add(Vector128.Load(p + c), FastMathV128.Exp(Vector128.Subtract(Vector128.Load(px + k + c), s))), p + c);
            }
        }

        return k;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]

    public static int Apply(int function, float* px, float* py, int n)
    {
        int k = 0;
        for (; k + 4 <= n; k += 4)
        {
            Vector128<float> v = Vector128.Load(px + k);
            Vector128<float> r = function switch { 0 => FastMathV128.Exp(v), 1 => FastMathV128.Sigmoid(v), _ => FastMathV128.Tanh(v) };
            Vector128.Store(r, py + k);
        }

        return k;
    }
}

internal static unsafe class ElementwiseV256
{
    public const int Lanes = 8;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]

    public static int LstmCell(float* pz, float* pc, float* ph, int hidden)
    {
        int j = 0;
        for (; j + 8 <= hidden; j += 8)
        {
            Vector256<float> i = FastMathV256.Sigmoid(Vector256.Load(pz + j));
            Vector256<float> f = FastMathV256.Sigmoid(Vector256.Load(pz + hidden + j));
            Vector256<float> g = FastMathV256.Tanh(Vector256.Load(pz + (2 * hidden) + j));
            Vector256<float> o = FastMathV256.Sigmoid(Vector256.Load(pz + (3 * hidden) + j));
            Vector256<float> cNew = Vector256.FusedMultiplyAdd(f, Vector256.Load(pc + j), Vector256.Multiply(i, g));
            Vector256.Store(cNew, pc + j);
            Vector256.Store(Vector256.Multiply(o, FastMathV256.Tanh(cNew)), ph + j);
        }

        return j;
    }

    /// <summary>Adds exp(x[k] - shift) into the 16 partials p (element k into partial k mod 16) for whole blocks of 16 elements.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int SumExpBlocks(float* px, int n, float shift, float* p)
    {
        Vector256<float> s = Vector256.Create(shift);
        int k = 0;
        for (; k + 16 <= n; k += 16)
        {
            for (int c = 0; c < 16; c += 8)
            {
                Vector256.Store(Vector256.Add(Vector256.Load(p + c), FastMathV256.Exp(Vector256.Subtract(Vector256.Load(px + k + c), s))), p + c);
            }
        }

        return k;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]

    public static int Apply(int function, float* px, float* py, int n)
    {
        int k = 0;
        for (; k + 8 <= n; k += 8)
        {
            Vector256<float> v = Vector256.Load(px + k);
            Vector256<float> r = function switch { 0 => FastMathV256.Exp(v), 1 => FastMathV256.Sigmoid(v), _ => FastMathV256.Tanh(v) };
            Vector256.Store(r, py + k);
        }

        return k;
    }
}

internal static unsafe class ElementwiseV512
{
    public const int Lanes = 16;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]

    public static int LstmCell(float* pz, float* pc, float* ph, int hidden)
    {
        int j = 0;
        for (; j + 16 <= hidden; j += 16)
        {
            Vector512<float> i = FastMathV512.Sigmoid(Vector512.Load(pz + j));
            Vector512<float> f = FastMathV512.Sigmoid(Vector512.Load(pz + hidden + j));
            Vector512<float> g = FastMathV512.Tanh(Vector512.Load(pz + (2 * hidden) + j));
            Vector512<float> o = FastMathV512.Sigmoid(Vector512.Load(pz + (3 * hidden) + j));
            Vector512<float> cNew = Vector512.FusedMultiplyAdd(f, Vector512.Load(pc + j), Vector512.Multiply(i, g));
            Vector512.Store(cNew, pc + j);
            Vector512.Store(Vector512.Multiply(o, FastMathV512.Tanh(cNew)), ph + j);
        }

        return j;
    }

    /// <summary>Adds exp(x[k] - shift) into the 16 partials p (element k into partial k mod 16) for whole blocks of 16 elements.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int SumExpBlocks(float* px, int n, float shift, float* p)
    {
        Vector512<float> s = Vector512.Create(shift);
        int k = 0;
        for (; k + 16 <= n; k += 16)
        {
            for (int c = 0; c < 16; c += 16)
            {
                Vector512.Store(Vector512.Add(Vector512.Load(p + c), FastMathV512.Exp(Vector512.Subtract(Vector512.Load(px + k + c), s))), p + c);
            }
        }

        return k;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]

    public static int Apply(int function, float* px, float* py, int n)
    {
        int k = 0;
        for (; k + 16 <= n; k += 16)
        {
            Vector512<float> v = Vector512.Load(px + k);
            Vector512<float> r = function switch { 0 => FastMathV512.Exp(v), 1 => FastMathV512.Sigmoid(v), _ => FastMathV512.Tanh(v) };
            Vector512.Store(r, py + k);
        }

        return k;
    }
}
