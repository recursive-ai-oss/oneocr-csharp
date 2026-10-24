namespace OneOcr.Training;

/// <summary>Settings of <see cref="AdamW"/>. Defaults follow Loshchilov and Hutter 2019 and PyTorch's AdamW.</summary>
internal sealed record AdamWOptions
{
    public double LearningRate { get; init; } = 1e-3;
    public double Beta1 { get; init; } = 0.9;
    public double Beta2 { get; init; } = 0.999;
    public double Epsilon { get; init; } = 1e-8;

    /// <summary>Decoupled weight decay, applied to the tensors flagged for decay (weight matrices, not biases).</summary>
    public double WeightDecay { get; init; } = 0.01;

    internal void Validate()
    {
        if (!(LearningRate >= 0) || !(Beta1 >= 0 && Beta1 < 1) || !(Beta2 >= 0 && Beta2 < 1) || !(Epsilon > 0) || !(WeightDecay >= 0))
        {
            throw new ArgumentException("AdamW settings are out of range.");
        }
    }
}

/// <summary>
/// AdamW (Loshchilov and Hutter 2019, "Decoupled weight decay regularization", algorithm 2) on fp32 master weights. The update is
/// element-wise, written as separately rounded fp32 operations in the order of docs/spec/backward-numerics.md (the order of PyTorch's
/// single-tensor implementation), so it is identical on every path and at every degree of parallelism. Bias-correction powers are
/// carried as running double products.
/// </summary>
internal sealed class AdamW
{
    private readonly float[][] _parameters;
    private readonly bool[] _decay;
    private readonly float[][] _m;
    private readonly float[][] _v;
    private double _beta1Power = 1.0;
    private double _beta2Power = 1.0;

    /// <summary>Creates the optimizer over tensors it updates in place; <paramref name="decay"/> flags which receive weight decay.</summary>
    public AdamW(float[][] parameters, bool[] decay, AdamWOptions options)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(decay);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        if (decay.Length != parameters.Length)
        {
            throw new ArgumentException("One decay flag per tensor is required.", nameof(decay));
        }

        Options = options;
        _parameters = parameters;
        _decay = decay;
        _m = new float[parameters.Length][];
        _v = new float[parameters.Length][];
        for (int i = 0; i < parameters.Length; i++)
        {
            _m[i] = new float[parameters[i].Length];
            _v[i] = new float[parameters[i].Length];
        }
    }

    public AdamWOptions Options { get; }

    /// <summary>The number of steps taken.</summary>
    public long Steps { get; private set; }

    /// <summary>Takes one step with the given gradients (same shapes as the tensors) and learning rate.</summary>
    public void Step(float[][] gradients, double learningRate)
    {
        if (gradients.Length != _parameters.Length)
        {
            throw new ArgumentException("One gradient per tensor is required.", nameof(gradients));
        }

        Steps++;
        _beta1Power *= Options.Beta1;
        _beta2Power *= Options.Beta2;
        double correction1 = 1.0 - _beta1Power;
        double correction2 = 1.0 - _beta2Power;
        float beta2 = (float)Options.Beta2;
        float oneMinusBeta1 = (float)(1.0 - Options.Beta1), oneMinusBeta2 = (float)(1.0 - Options.Beta2);
        float stepSize = (float)(learningRate / correction1);
        float sqrtCorrection2 = (float)Math.Sqrt(correction2);
        float epsilon = (float)Options.Epsilon;
        float decayFactor = (float)(1.0 - (learningRate * Options.WeightDecay));

        for (int i = 0; i < _parameters.Length; i++)
        {
            float[] p = _parameters[i], g = gradients[i], m = _m[i], v = _v[i];
            if (g.Length != p.Length)
            {
                throw new ArgumentException("A gradient has the wrong length.", nameof(gradients));
            }

            bool decays = _decay[i];
            for (int j = 0; j < p.Length; j++)
            {
                float w = decays ? p[j] * decayFactor : p[j];
                float gj = g[j];
                float mj = m[j] + (oneMinusBeta1 * (gj - m[j]));
                float vj = (v[j] * beta2) + (oneMinusBeta2 * (gj * gj));
                m[j] = mj;
                v[j] = vj;
                float denom = (MathF.Sqrt(vj) / sqrtCorrection2) + epsilon;
                p[j] = w - (stepSize * (mj / denom));
            }
        }
    }

    /// <summary>
    /// Restores the state saved by a checkpoint: the step count and the moments. The bias-correction powers are rebuilt by the same
    /// running products as an uninterrupted run, so the next step is bit-identical to the one that would have followed.
    /// </summary>
    public void Restore(long steps, float[][] m, float[][] v)
    {
        if (steps < 0 || m.Length != _m.Length || v.Length != _v.Length)
        {
            throw new ArgumentException("The saved state does not match the tensors.");
        }

        for (int i = 0; i < _m.Length; i++)
        {
            if (m[i].Length != _m[i].Length || v[i].Length != _v[i].Length)
            {
                throw new ArgumentException("The saved state does not match the tensors.");
            }

            m[i].CopyTo(_m[i], 0);
            v[i].CopyTo(_v[i], 0);
        }

        Steps = steps;
        _beta1Power = 1.0;
        _beta2Power = 1.0;
        for (long s = 0; s < steps; s++)
        {
            _beta1Power *= Options.Beta1;
            _beta2Power *= Options.Beta2;
        }
    }

    /// <summary>Copies of the first and second moment estimates, for tests and checkpoints.</summary>
    public (float[][] M, float[][] V) Moments() => (_m.Select(a => (float[])a.Clone()).ToArray(), _v.Select(a => (float[])a.Clone()).ToArray());
}
