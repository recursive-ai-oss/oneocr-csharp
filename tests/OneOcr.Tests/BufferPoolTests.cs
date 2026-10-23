using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Tests;

public class BufferPoolTests
{
    [Fact]
    public void ReusesTheSmallestFreeArrayThatFits()
    {
        var pool = new BufferPool<float>(1L << 30, sizeof(float));
        float[] big = pool.Rent(1_000_000);
        float[] small = pool.Rent(100_000);
        pool.Return(big);
        pool.Return(small);

        Assert.Same(small, pool.Rent(90_000));
        Assert.Same(big, pool.Rent(200_000));
        Assert.NotSame(big, pool.Rent(200_000));
    }

    [Fact]
    public void SizesAreAtLeastTheRequestAndRoundToAGrid()
    {
        var pool = new BufferPool<byte>(0, 1);
        Assert.True(pool.Rent(1).Length >= 1);
        Assert.Empty(pool.Rent(0));
        int big = pool.Rent((1 << 20) + 1).Length;
        Assert.Equal(0, big % (1 << 16));
        Assert.True(big >= (1 << 20) + 1);
    }

    [Fact]
    public void DropsArraysBeyondTheByteBudget()
    {
        var pool = new BufferPool<float>(1_966_080, sizeof(float));
        float[] a = pool.Rent(300_000);
        float[] b = pool.Rent(300_000);
        pool.Return(a);
        pool.Return(b);

        Assert.Same(a, pool.Rent(300_000));
        Assert.NotSame(b, pool.Rent(300_000));
    }

    [Fact]
    public void PooledBuffersLeaveTextAndConfidenceUnchanged()
    {
        // The same line read twice through one recognizer (the second time from reused, dirty buffers) gives identical results.
        var r = new LineRecognizer(CrnnModel.FromContainer(ModelContainer.Parse(TestModels.RandomContainer(TestModels.Small, 3)), OcrLimits.Default), OcrLimits.Default, KernelPath.Scalar);
        byte[] image = TestModels.RandomImage(r.InputHeight, 300, 5);
        RecognizedLine first = r.Recognize(image, 300, 1);
        byte[] narrower = TestModels.RandomImage(r.InputHeight, 200, 6);
        _ = r.Recognize(narrower, 200, 1);
        Assert.Equal(first, r.Recognize(image, 300, 1));
    }
}
