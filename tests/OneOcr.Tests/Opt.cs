using OneOcr.Documents;

namespace OneOcr.Tests;

/// <summary>Short option builders for the tests.</summary>
internal static class Opt
{
    public static OcrOptions Degree(int degree) => new() { MaxDegreeOfParallelism = degree };

    public static DocumentExtractionOptions Doc(int degree) => new() { MaxDegreeOfParallelism = degree };
}
