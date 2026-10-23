namespace OneOcr;

/// <summary>Base class of the typed exceptions the library throws for bad input, as opposed to caller mistakes
/// (those are <see cref="ArgumentException"/>) and I/O failures (passed through unchanged).</summary>
public abstract class OcrException : Exception
{
    /// <summary>Creates the exception with a message.</summary>
    /// <param name="message">What was wrong.</param>
    protected OcrException(string message) : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the failure that caused it.</summary>
    /// <param name="message">What was wrong.</param>
    /// <param name="innerException">The underlying failure.</param>
    protected OcrException(string message, Exception innerException) : base(message, innerException)
    {
    }
}

/// <summary>An input that is not in the format it claims to be: a page-model JSON document with a wrong member, type or value.</summary>
public sealed class OcrFormatException : OcrException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="message">What is wrong, naming the member where known.</param>
    public OcrFormatException(string message) : base(message)
    {
    }

    /// <summary>Creates the exception with the error that caused it.</summary>
    /// <param name="message">What is wrong.</param>
    /// <param name="innerException">The underlying parse error.</param>
    public OcrFormatException(string message, Exception innerException) : base(message, innerException)
    {
    }
}

/// <summary>A model file is malformed, truncated, inconsistent, or describes an architecture this version cannot run.</summary>
public sealed class OcrModelException : OcrException
{
    /// <summary>Creates the exception with a message.</summary>
    /// <param name="message">What was wrong with the model file.</param>
    public OcrModelException(string message) : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the failure that caused it.</summary>
    /// <param name="message">What was wrong with the model file.</param>
    /// <param name="innerException">The underlying failure, for example a JSON parse error.</param>
    public OcrModelException(string message, Exception innerException) : base(message, innerException)
    {
    }
}

/// <summary>Input is well formed but exceeds one of the configured <see cref="OcrLimits"/>. Thrown before the work the limit guards.</summary>
public sealed class OcrLimitException : OcrException
{
    /// <summary>Creates the exception for a limit.</summary>
    /// <param name="limitName">The name of the <see cref="OcrLimits"/> property that was exceeded.</param>
    /// <param name="limit">The configured bound.</param>
    /// <param name="actual">The declared or running value that exceeded it.</param>
    public OcrLimitException(string limitName, long limit, long actual)
        : base($"{limitName} exceeded: {actual} > {limit}.")
    {
        LimitName = limitName;
        Limit = limit;
        Actual = actual;
    }

    /// <summary>The name of the <see cref="OcrLimits"/> property that was exceeded.</summary>
    public string LimitName { get; }

    /// <summary>The configured bound.</summary>
    public long Limit { get; }

    /// <summary>The value that exceeded the bound.</summary>
    public long Actual { get; }
}
