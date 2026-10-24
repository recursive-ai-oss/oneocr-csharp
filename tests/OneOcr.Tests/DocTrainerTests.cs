using System.Globalization;
using OneOcr.Documents;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Training;

namespace OneOcr.Tests;

/// <summary>
/// The document-extraction training step on small random models: the training forward pass is the library's forward pass bit for bit,
/// loss, gradients and weights are bit-identical on every kernel path and at every degree, the backward kernels agree across paths on
/// awkward sizes, and the optimizer learns.
/// </summary>
public class DocTrainerTests
{
    private static readonly TestDocModels.Shape Shape = TestDocModels.Small;

    private static DocExtractModel Load(int seed = 5) =>
        DocExtractModel.FromContainer(ModelContainer.Parse(TestDocModels.RandomContainer(Shape, seed), OcrLimits.Default), OcrLimits.Default);

    private static List<DocExample> Examples(DocExtractModel m, int count, int seed, bool pointers)
    {
        var list = new List<DocExample>();
        var rng = new Random(seed);
        for (int e = 0; e < count; e++)
        {
            int words = 5 + rng.Next(m.MaxWords - 4);
            (string[] texts, OcrBox[] boxes, double w, double h) = TestDocModels.Page(words, seed * 31 + e);
            DocFeatures f = DocFeatures.Build(m, texts, boxes, w, h);
            int[] Targets(int classes, double ignore) => [.. Enumerable.Range(0, words).Select(_ => rng.NextDouble() < ignore ? -1 : rng.Next(classes))];
            list.Add(new DocExample(f, new DocTargets(Targets(m.Tags, 0.2), Targets(m.Roles, 0.3), rng.Next(m.DocTypes), pointers ? Targets(words, 0.3) : null, pointers ? Targets(words, 0.5) : null)));
        }

        return list;
    }

    private static DocTrainSettings AllLosses(double backbone) => new()
    {
        BackboneScale = backbone,
        Weights = new DocLossWeights { Tag = 1f, Role = 0.5f, Doc = 0.25f, Item = 0.75f, KeyValue = 1.25f },
        Adam = new AdamWOptions { LearningRate = 1e-3 },
        MaxGradNorm = 5,
    };

    private static void AssertBits(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
            {
                Assert.Fail($"{what}: element {i} is {actual[i]:R}, expected {expected[i]:R}");
            }
        }
    }

    [Fact]
    public void TrainingForwardIsTheLibraryForwardBitForBit()
    {
        DocExtractModel m = Load();
        foreach (KernelPath path in Dispatch.Available())
        {
            var trainer = new DocTrainer(m, path, AllLosses(1.0));
            foreach (DocExample ex in Examples(m, 3, 9, false))
            {
                DocWindowOutputs w = DocForward.RunWindow(m, ex.Features, 0, ex.Features.Count, path, 1, null, CancellationToken.None);
                (float[] tags, float[] roles, float[] doc, float[] rel) = trainer.ForwardForTests(ex.Features);
                AssertBits(w.Tags, tags, $"{path} tags");
                AssertBits(w.Roles, roles, $"{path} roles");
                AssertBits(w.Doc, doc, $"{path} doc");
                AssertBits(w.Rel, rel, $"{path} rel");
            }
        }
    }

    private static (double[] Losses, float[][] Grads, float[][] Weights) Run(KernelPath path, int degree, DocTrainSettings settings, int steps, bool pointers)
    {
        DocExtractModel m = Load();
        var trainer = new DocTrainer(m, path, settings);
        var losses = new List<double>();
        var grads = new List<float[]>();
        for (int s = 0; s < steps; s++)
        {
            List<DocExample> batch = Examples(m, 5, 100 + s, pointers);
            losses.Add(trainer.Step(batch, degree));
            grads.AddRange(trainer.Gradients.Select(static g => (float[])g.Clone()));
        }

        return ([.. losses], [.. grads], [.. trainer.Parameters.Select(static p => (float[])p.Clone())]);
    }

    [Theory]
    [InlineData(1.0, true)]
    [InlineData(0.1, true)]
    [InlineData(0.0, false)]
    public void StepsAreBitIdenticalOnEveryPathAndDegree(double backbone, bool pointers)
    {
        DocTrainSettings settings = AllLosses(backbone);
        (double[] refLoss, float[][] refGrads, float[][] refWeights) = Run(KernelPath.Scalar, 1, settings, 3, pointers);
        foreach (KernelPath path in Dispatch.Available())
        {
            foreach (int degree in new[] { 1, 4 })
            {
                (double[] loss, float[][] grads, float[][] weights) = Run(path, degree, settings, 3, pointers);
                for (int s = 0; s < refLoss.Length; s++)
                {
                    Assert.Equal(BitConverter.DoubleToInt64Bits(refLoss[s]), BitConverter.DoubleToInt64Bits(loss[s]));
                }

                for (int i = 0; i < refGrads.Length; i++)
                {
                    AssertBits(refGrads[i], grads[i], $"{path}/{degree} gradient {i}");
                }

                for (int i = 0; i < refWeights.Length; i++)
                {
                    AssertBits(refWeights[i], weights[i], $"{path}/{degree} weights {i}");
                }
            }
        }
    }

    [Fact]
    public void TagHeadGradientDoesNotDependOnWhetherTheBackboneTrains()
    {
        DocExtractModel a = Load(), b = Load();
        var head = new DocTrainer(a, Dispatch.Best, new DocTrainSettings { BackboneScale = 0, MaxGradNorm = 0 });
        var full = new DocTrainer(b, Dispatch.Best, new DocTrainSettings { BackboneScale = 0.5, MaxGradNorm = 0 });
        double l1 = head.ComputeGradients(Examples(a, 4, 3, false));
        double l2 = full.ComputeGradients(Examples(b, 4, 3, false));
        Assert.Equal(l1, l2);
        Assert.Equal(["tag_head.weight", "tag_head.bias"], head.Names);
        for (int i = 0; i < head.Names.Count; i++)
        {
            int j = full.Names.ToList().IndexOf(head.Names[i]);
            AssertBits(head.Gradients[i], full.Gradients[j], head.Names[i]);
        }
    }

    [Fact]
    public void TheOptimizerLearnsABatch()
    {
        DocExtractModel m = Load();
        var trainer = new DocTrainer(m, Dispatch.Best, new DocTrainSettings
        {
            BackboneScale = 0.5, Adam = new AdamWOptions { LearningRate = 3e-3 }, Weights = new DocLossWeights { Tag = 1f, Item = 1f, KeyValue = 1f, Role = 1f, Doc = 1f },
        });
        List<DocExample> batch = Examples(m, 4, 21, true);
        double first = trainer.Evaluate(batch);
        for (int s = 0; s < 60; s++)
        {
            trainer.Step(batch, 2);
        }

        double last = trainer.Evaluate(batch);
        Assert.True(last < first * 0.5, $"loss {first.ToString("F3", CultureInfo.InvariantCulture)} -> {last.ToString("F3", CultureInfo.InvariantCulture)}");
    }

    [Fact]
    public void ReferenceBackwardKernelsAgreeAcrossPathsOnAwkwardSizes()
    {
        var rng = new Random(11);
        float[] Rand(int n, float scale = 1f) => [.. Enumerable.Range(0, n).Select(_ => ((float)rng.NextDouble() - 0.5f) * 2 * scale)];
        foreach (int n in new[] { 1, 3, 15, 16, 17, 33, 50 })
        {
            float[] p = Rand(n * n), dP = Rand(n * n), dA = Rand(n * 5), v = Rand(n * 5), q = Rand(n * 5), k = Rand(n * 5), dS = Rand(n * n);
            float[] dRawRef = new float[n * n], dBiasRef = new float[n * n], dPRef = new float[n * n], dVRef = new float[n * 5], dQRef = new float[n * 5], dKRef = new float[n * 5];
            void Run(KernelPath path, float[] dRaw, float[] dBias, float[] dPp, float[] dV, float[] dQ, float[] dK)
            {
                DocBackward.SoftmaxRowsBackward(path, p, dP, n, n, 5.6f, dRaw, dBias);
                DocBackward.ContextBackward(path, p, dA, 5, v, 5, n, 5, dPp, dV, 5);
                DocBackward.ScoresBackward(path, dS, q, 5, k, 5, n, 5, dQ, 5, dK, 5);
            }

            Run(KernelPath.Scalar, dRawRef, dBiasRef, dPRef, dVRef, dQRef, dKRef);
            foreach (KernelPath path in Dispatch.Available())
            {
                float[] dRaw = new float[n * n], dBias = new float[n * n], dPp = new float[n * n], dV = new float[n * 5], dQ = new float[n * 5], dK = new float[n * 5];
                Run(path, dRaw, dBias, dPp, dV, dQ, dK);
                AssertBits(dRawRef, dRaw, $"{path} n={n} softmax raw");
                AssertBits(dBiasRef, dBias, $"{path} n={n} softmax bias");
                AssertBits(dPRef, dPp, $"{path} n={n} dP");
                AssertBits(dVRef, dV, $"{path} n={n} dV");
                AssertBits(dQRef, dQ, $"{path} n={n} dQ");
                AssertBits(dKRef, dK, $"{path} n={n} dK");
            }
        }

        foreach (int d in new[] { 1, 7, 16, 37, 50 })
        {
            int rows = 9;
            float[] x = Rand(rows * d, 2f), w = Rand(d), b = Rand(d), dy = Rand(rows * d);
            float[]? refDx = null, refDw = null, refDb = null;
            foreach (KernelPath path in Dispatch.Available().OrderBy(static p => p != KernelPath.Scalar))
            {
                float[] y = new float[rows * d], xhat = new float[rows * d], inv = new float[rows], dx = new float[rows * d], dw = new float[d], db = new float[d];
                DocBackward.LayerNormForward(path, x, y, rows, d, w, b, xhat, inv);
                DocBackward.LayerNormBackward(path, dy, xhat, inv, w, rows, d, dx, dw, db);
                if (refDx is null)
                {
                    (refDx, refDw, refDb) = (dx, dw, db);
                    continue;
                }

                AssertBits(refDx, dx, $"{path} layer norm {d} dx");
                AssertBits(refDw, dw, $"{path} layer norm {d} dw");
                AssertBits(refDb, db, $"{path} layer norm {d} db");
            }
        }
    }

    [Fact]
    public void CrossEntropyIgnoresMaskedRowsAndHandlesEmptyTargets()
    {
        float[] z = [1f, 2f, 3f, 0f, 0f, 0f];
        var d = new float[6];
        Assert.Equal(0.0, DocLoss.CrossEntropy(z, 3, 2, 3, [-1, -1], default, 1.0, d));
        Assert.All(d, static v => Assert.Equal(0f, v));
        double loss = DocLoss.CrossEntropy(z, 3, 2, 3, [2, -1], default, 1.0, d);
        Assert.True(loss > 0 && loss < 1);
        Assert.Equal(0f, d[3]);
        Assert.Equal(0f, d[4]);
        Assert.Equal(0f, d[5]);
        Assert.True(d[2] < 0 && d[0] > 0 && d[1] > 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => DocLoss.CrossEntropy(z, 3, 2, 3, [3, -1], default, 1.0, d));
    }
}
