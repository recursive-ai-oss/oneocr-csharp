namespace OneOcr.Documents;

/// <summary>Per-call options of the extraction methods of <see cref="DocumentExtractor"/>. Everything has a default.</summary>
public sealed class DocumentExtractionOptions
{
    /// <summary>Most threads to use; 0 (the default) means min(4, cores). The result does not depend on it.</summary>
    public int MaxDegreeOfParallelism { get; init; }

    /// <summary>
    /// For <c>Extract&lt;T&gt;</c>: whether a property of <c>T</c> that no class of the model maps to is an error. Default false: such a
    /// property keeps its default and a diagnostic (<see cref="OcrDiagnosticCodes.PropertyUnmapped"/>) says so. When no property of
    /// <c>T</c> maps at all the call always throws, since nothing could be read.
    /// </summary>
    public bool Strict { get; init; }

    internal void Validate()
    {
        if (MaxDegreeOfParallelism < 0)
        {
            throw new ArgumentOutOfRangeException("options", MaxDegreeOfParallelism, "MaxDegreeOfParallelism must not be negative.");
        }
    }
}
