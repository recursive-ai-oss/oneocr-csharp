using System.Runtime.ExceptionServices;

namespace OneOcr;

internal static class ParallelRun
{
    /// <summary>
    /// Runs <paramref name="body"/> for 0..count-1 on up to <paramref name="degree"/> threads. A failure surfaces as the body's own
    /// exception (the first one if several threads fail), not wrapped in an <see cref="AggregateException"/>, so callers see the typed
    /// exceptions the library documents.
    /// </summary>
    public static void For(int count, int degree, Action<int> body, CancellationToken cancellationToken)
    {
        if (degree <= 1 || count <= 1)
        {
            for (int i = 0; i < count; i++)
            {
                body(i);
            }

            return;
        }

        try
        {
            Parallel.For(0, count, new ParallelOptions { MaxDegreeOfParallelism = degree, CancellationToken = cancellationToken }, body);
        }
        catch (AggregateException ex) when (ex.InnerExceptions.Count > 0)
        {
            ExceptionDispatchInfo.Capture(ex.InnerExceptions[0]).Throw();
            throw;
        }
    }
}
