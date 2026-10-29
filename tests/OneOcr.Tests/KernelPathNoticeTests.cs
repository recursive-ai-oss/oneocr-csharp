using OneOcr.Kernels;
using OneOcr.Model;

namespace OneOcr.Tests;

public class KernelPathNoticeTests
{
    [Fact]
    public void Scalar_on_a_cpu_with_avx2_and_fma_is_reported_and_other_paths_are_not()
    {
        string? notice = Dispatch.SlowPathNotice(KernelPath.Scalar);
        Assert.Equal(Dispatch.CpuHasAvx2Fma(), notice is not null);
        if (notice is not null)
        {
            Assert.Contains("x86-64-v3", notice);
        }

        foreach (KernelPath path in new[] { KernelPath.Vector128, KernelPath.Avx2, KernelPath.Avx512 })
        {
            Assert.Null(Dispatch.SlowPathNotice(path));
        }
    }

    [Fact]
    public void A_runtime_that_runs_avx2_agrees_with_cpuid()
    {
        // CPUID can say yes when the runtime has the instruction sets switched off, never the other way round.
        if (Dispatch.Avx2Available)
        {
            Assert.True(Dispatch.CpuHasAvx2Fma());
        }
    }

    [Fact]
    public void The_engine_reports_the_path_it_runs_and_a_notice_only_for_scalar()
    {
        var shape = new TestModels.Shape(8, [(4, 2, 2), (4, 2, 2), (8, 2, 1)], 8, 2, 12);
        OcrEngine best = OcrEngine.Load(new MemoryStream(TestModels.RandomContainer(shape, 1)));
        Assert.Equal((OcrKernelPath)(int)Dispatch.Best, best.KernelPath);
        Assert.Equal(best.KernelPath == OcrKernelPath.Scalar && Dispatch.CpuHasAvx2Fma(), best.PerformanceNotice is not null);

        OcrEngine scalar = OcrEngine.Create(ModelContainer.Parse(TestModels.RandomContainer(shape, 1)), null, KernelPath.Scalar);
        Assert.Equal(OcrKernelPath.Scalar, scalar.KernelPath);
        Assert.Equal(Dispatch.CpuHasAvx2Fma(), scalar.PerformanceNotice is not null);
    }
}
