using System.Buffers;
using OneOcr.Decoding;

namespace OneOcr.Training;

/// <summary>
/// Connectionist temporal classification loss and its gradient with respect to the logits (Graves et al. 2006, sections 3.1-3.2 and
/// 4.1): the forward-backward recursions over the blank-extended label sequence, in the log domain. The class at index 0 is the
/// blank. All arithmetic is double precision scalar code using <see cref="DetMath"/>'s exp and log, summed in a fixed order, so the
/// result does not depend on the kernel path or the platform (docs/spec/backward-numerics.md).
/// </summary>
internal static class CtcLoss
{
    /// <summary>
    /// Returns the negative log likelihood of <paramref name="target"/> given the logits <c>[steps][classes]</c> (softmax is applied
    /// here), and writes <paramref name="scale"/> times its gradient with respect to the logits to <paramref name="grad"/>. When no
    /// alignment exists (fewer steps than the label needs, counting a blank between repeated labels) the loss is +infinity and the
    /// gradient is zero, which is PyTorch's <c>zero_infinity</c> rule once the caller replaces the infinite loss by 0.
    /// </summary>
    /// <exception cref="ArgumentException">A target label is the blank or outside the class range, or a buffer is too small.</exception>
    public static double Compute(ReadOnlySpan<float> logits, int steps, int classes, ReadOnlySpan<int> target, double scale, Span<float> grad)
    {
        if (steps < 1 || classes < 2 || logits.Length < (long)steps * classes || grad.Length < (long)steps * classes)
        {
            throw new ArgumentException("The logits or gradient buffer does not match steps x classes.");
        }

        foreach (int label in target)
        {
            if (label < 1 || label >= classes)
            {
                throw new ArgumentException("Target labels must be in [1, classes); class 0 is the blank.");
            }
        }

        int len = target.Length;
        int ext = (2 * len) + 1;
        double[] lp = ArrayPool<double>.Shared.Rent(steps * classes);
        double[] alpha = ArrayPool<double>.Shared.Rent(steps * ext);
        double[] beta = ArrayPool<double>.Shared.Rent(steps * ext);
        double[] acc = ArrayPool<double>.Shared.Rent(classes);
        int[] lab = ArrayPool<int>.Shared.Rent(ext);
        try
        {
            for (int s = 0; s < ext; s++)
            {
                lab[s] = (s & 1) == 0 ? 0 : target[s >> 1];
            }

            // log softmax, classes summed in ascending order.
            for (int t = 0; t < steps; t++)
            {
                ReadOnlySpan<float> row = logits.Slice(t * classes, classes);
                double max = row[0];
                for (int k = 1; k < classes; k++)
                {
                    if (row[k] > max)
                    {
                        max = row[k];
                    }
                }

                double sum = 0;
                for (int k = 0; k < classes; k++)
                {
                    sum += DetMath.Exp(row[k] - max);
                }

                double logSum = DetMath.Log(sum);
                for (int k = 0; k < classes; k++)
                {
                    lp[(t * classes) + k] = (row[k] - max) - logSum;
                }
            }

            const double NegInf = double.NegativeInfinity;
            for (int i = 0; i < steps * ext; i++)
            {
                alpha[i] = NegInf;
                beta[i] = NegInf;
            }

            alpha[0] = lp[0];
            if (ext > 1)
            {
                alpha[1] = lp[lab[1]];
            }

            for (int t = 1; t < steps; t++)
            {
                for (int s = 0; s < ext; s++)
                {
                    double a = alpha[((t - 1) * ext) + s];
                    if (s >= 1)
                    {
                        a = DetMath.LogAdd(a, alpha[((t - 1) * ext) + s - 1]);
                    }

                    if (s >= 2 && lab[s] != 0 && lab[s] != lab[s - 2])
                    {
                        a = DetMath.LogAdd(a, alpha[((t - 1) * ext) + s - 2]);
                    }

                    alpha[(t * ext) + s] = a + lp[(t * classes) + lab[s]];
                }
            }

            double logP = alpha[((steps - 1) * ext) + ext - 1];
            if (ext > 1)
            {
                logP = DetMath.LogAdd(logP, alpha[((steps - 1) * ext) + ext - 2]);
            }

            grad.Slice(0, steps * classes).Clear();
            if (double.IsNegativeInfinity(logP) || double.IsNaN(logP))
            {
                return double.PositiveInfinity;
            }

            // beta includes the emission at its own step, like alpha, so alpha * beta counts that emission twice (divided out below).
            beta[((steps - 1) * ext) + ext - 1] = lp[((steps - 1) * classes) + lab[ext - 1]];
            if (ext > 1)
            {
                beta[((steps - 1) * ext) + ext - 2] = lp[((steps - 1) * classes) + lab[ext - 2]];
            }

            for (int t = steps - 2; t >= 0; t--)
            {
                for (int s = 0; s < ext; s++)
                {
                    double b = beta[((t + 1) * ext) + s];
                    if (s + 1 < ext)
                    {
                        b = DetMath.LogAdd(b, beta[((t + 1) * ext) + s + 1]);
                    }

                    if (s + 2 < ext && lab[s + 2] != 0 && lab[s + 2] != lab[s])
                    {
                        b = DetMath.LogAdd(b, beta[((t + 1) * ext) + s + 2]);
                    }

                    beta[(t * ext) + s] = b + lp[(t * classes) + lab[s]];
                }
            }

            for (int t = 0; t < steps; t++)
            {
                Array.Fill(acc, NegInf, 0, classes);
                for (int s = 0; s < ext; s++)
                {
                    int k = lab[s];
                    acc[k] = DetMath.LogAdd(acc[k], alpha[(t * ext) + s] + beta[(t * ext) + s]);
                }

                for (int k = 0; k < classes; k++)
                {
                    double l = lp[(t * classes) + k];
                    double occupancy = DetMath.Exp(acc[k] - l - logP);
                    grad[(t * classes) + k] = (float)(scale * (DetMath.Exp(l) - occupancy));
                }
            }

            return -logP;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(lp);
            ArrayPool<double>.Shared.Return(alpha);
            ArrayPool<double>.Shared.Return(beta);
            ArrayPool<double>.Shared.Return(acc);
            ArrayPool<int>.Shared.Return(lab);
        }
    }
}
