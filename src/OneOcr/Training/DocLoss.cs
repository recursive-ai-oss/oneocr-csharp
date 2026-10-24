using OneOcr.Decoding;

namespace OneOcr.Training;

/// <summary>
/// Cross-entropy losses of the extraction model, in double precision with the library's own <see cref="DetMath"/> exponential and
/// logarithm, so loss and gradient are the same bits everywhere (as for CTC in docs/spec/backward-numerics.md).
/// </summary>
internal static class DocLoss
{
    /// <summary>
    /// Cross-entropy over <paramref name="rows"/> rows of <paramref name="classes"/> logits, PyTorch's <c>reduction='mean'</c> with
    /// <c>ignore_index</c> and optional class weights: <c>Σ w[t] · nll / Σ w[t]</c> over the rows whose target is not negative. A row
    /// with a negative target contributes neither to the loss nor to the gradient. <paramref name="dLogits"/> receives
    /// <c>scale · w[t] / Σ w · (softmax - onehot)</c> rounded once to binary32 (zero rows for ignored ones); when no row counts the loss
    /// is 0 and every gradient is zero.
    /// </summary>
    /// <param name="logits">The logits, row-major <c>[rows, classes]</c> with the given row stride.</param>
    /// <param name="ld">Row stride of <paramref name="logits"/>.</param>
    /// <param name="rows">Number of rows.</param>
    /// <param name="classes">Number of classes (columns).</param>
    /// <param name="targets">Target class per row, negative to ignore the row.</param>
    /// <param name="classWeights">Weight per class; empty means all ones.</param>
    /// <param name="scale">Factor applied to the gradient (the term's weight over the batch size).</param>
    /// <param name="dLogits">The gradient, contiguous <c>[rows, classes]</c>.</param>
    /// <returns>The mean loss before <paramref name="scale"/>.</returns>
    public static double CrossEntropy(
        ReadOnlySpan<float> logits, int ld, int rows, int classes, ReadOnlySpan<int> targets, ReadOnlySpan<float> classWeights, double scale, Span<float> dLogits)
    {
        if (targets.Length < rows || dLogits.Length < (long)rows * classes || (!classWeights.IsEmpty && classWeights.Length < classes) || ld < classes)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        dLogits[..(rows * classes)].Clear();
        double weightSum = 0;
        for (int i = 0; i < rows; i++)
        {
            if (targets[i] >= 0)
            {
                if (targets[i] >= classes)
                {
                    throw new ArgumentOutOfRangeException(nameof(targets), "A target is outside the classes.");
                }

                weightSum += classWeights.IsEmpty ? 1.0 : classWeights[targets[i]];
            }
        }

        if (!(weightSum > 0))
        {
            return 0;
        }

        double total = 0;
        for (int i = 0; i < rows; i++)
        {
            int t = targets[i];
            if (t < 0)
            {
                continue;
            }

            ReadOnlySpan<float> z = logits.Slice(i * ld, classes);
            double m = z[0];
            for (int k = 1; k < classes; k++)
            {
                if (z[k] > m)
                {
                    m = z[k];
                }
            }

            double s = 0;
            for (int k = 0; k < classes; k++)
            {
                s += DetMath.Exp(z[k] - m);
            }

            double logS = DetMath.Log(s);
            double w = classWeights.IsEmpty ? 1.0 : classWeights[t];
            total += w * (logS - (z[t] - m));
            double factor = scale * w / weightSum;
            Span<float> g = dLogits.Slice(i * classes, classes);
            for (int k = 0; k < classes; k++)
            {
                double p = DetMath.Exp((z[k] - m) - logS);
                g[k] = (float)(factor * (k == t ? p - 1.0 : p));
            }
        }

        return total / weightSum;
    }
}
