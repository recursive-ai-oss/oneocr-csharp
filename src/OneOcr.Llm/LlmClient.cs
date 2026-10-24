using System.Globalization;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;
using OneImage;

namespace OneOcr.Llm;

/// <summary>
/// Calls one <see cref="LlmProvider"/> under a policy (<see cref="LlmOptions"/>): offline switch, redaction, cost estimate and
/// bounds, concurrency limit, bounded retries, result cache. Give it the page image and the page model read from it; it sends
/// the image plus the compact word list (id, text, box) and returns the model's structured answer as JSON. Thread-safe.
/// Everything that can reach the network is asynchronous; nothing in the local OCR methods ever calls this class.
/// </summary>
public sealed class LlmClient : IDisposable
{
    private readonly LlmProvider _provider;
    private readonly LlmOptions _options;
    private readonly SemaphoreSlim _gate;
    private readonly object _ledger = new();
    private decimal _spent;
    private decimal _reserved;

    /// <summary>Creates a client.</summary>
    /// <param name="provider">The API to call.</param>
    /// <param name="options">Policy; default <see cref="LlmOptions"/> with nothing bounded and no cache.</param>
    public LlmClient(LlmProvider provider, LlmOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(provider);
        _provider = provider;
        _options = options ?? new LlmOptions();
        ArgumentOutOfRangeException.ThrowIfLessThan(_options.MaxConcurrency, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(_options.MaxRetries, 0);
        ArgumentOutOfRangeException.ThrowIfLessThan(_options.MaxOutputTokens, 1);
        _gate = new SemaphoreSlim(_options.MaxConcurrency);
    }

    /// <summary>Releases the concurrency gate. In-flight calls should be finished first.</summary>
    public void Dispose() => _gate.Dispose();

    /// <summary>The provider this client calls.</summary>
    public LlmProvider Provider => _provider;

    /// <summary>USD spent through this client so far, by token counts the providers reported (estimates where they reported none) at the table price. Calls to unpriced models add nothing.</summary>
    public decimal SpentUsd
    {
        get
        {
            lock (_ledger)
            {
                return _spent;
            }
        }
    }

    /// <summary>
    /// Works out what a call would cost at most without sending it: shrinks and encodes the images (the same work a real call
    /// does), builds the prompt and prices the worst case. Not blocked by <see cref="LlmOptions.Offline"/>.
    /// </summary>
    /// <param name="pages">The pages, one or more.</param>
    /// <param name="instruction">What to extract.</param>
    /// <param name="jsonSchema">The JSON Schema of the answer, an object schema.</param>
    /// <param name="schemaName">Name for the schema: letters, digits, <c>_</c> and <c>-</c>, up to 64 characters. Default <c>extraction</c>.</param>
    /// <returns>The estimate.</returns>
    public LlmEstimate Estimate(IReadOnlyList<LlmPageInput> pages, string instruction, string jsonSchema, string schemaName = "extraction")
    {
        Prepared p = Prepare(pages, instruction, jsonSchema, schemaName, CancellationToken.None);
        return p.Estimate;
    }

    /// <summary>
    /// Asks the model about one page and returns its answer as JSON for a raw schema.
    /// </summary>
    /// <param name="image">The page image.</param>
    /// <param name="page">The page model read from it.</param>
    /// <param name="instruction">What to extract, in the caller's words. Trusted text.</param>
    /// <param name="jsonSchema">The JSON Schema of the answer: an object schema, within <see cref="OcrLimits.MaxLlmSchemaBytes"/> and <see cref="OcrLimits.MaxLlmJsonDepth"/>.</param>
    /// <param name="schemaName">Name for the schema. Default <c>extraction</c>.</param>
    /// <param name="cancellationToken">Cancels the call, including waits between retries.</param>
    /// <returns>The answer and what it cost.</returns>
    /// <exception cref="OcrProviderException">Offline, unreachable, an error status, or an answer that is not what the API promises.</exception>
    /// <exception cref="OcrBudgetException">The call could cost more than <see cref="LlmOptions.MaxCostPerCallUsd"/> or <see cref="LlmOptions.SpendCapUsd"/> allows.</exception>
    /// <exception cref="OcrLimitException">An <see cref="OcrLimits"/> bound was exceeded: image pixels, request, response or schema size, JSON depth.</exception>
    /// <exception cref="ArgumentException">The schema is not a JSON object schema, or cannot be expressed for this provider.</exception>
    public Task<LlmExtraction> ExtractAsync(
        Image image, OcrPage page, string instruction, string jsonSchema, string schemaName = "extraction", CancellationToken cancellationToken = default) =>
        ExtractAsync([new LlmPageInput(image, page)], instruction, jsonSchema, schemaName, cancellationToken);

    /// <summary>Asks the model about several pages together and returns its answer as JSON for a raw schema. Word ids run on across pages.</summary>
    /// <param name="pages">The pages, between one and <see cref="OcrLimits.MaxPagesPerCall"/>.</param>
    /// <param name="instruction">What to extract. Trusted text.</param>
    /// <param name="jsonSchema">The JSON Schema of the answer.</param>
    /// <param name="schemaName">Name for the schema. Default <c>extraction</c>.</param>
    /// <param name="cancellationToken">Cancels the call.</param>
    /// <returns>The answer and what it cost.</returns>
    /// <exception cref="OcrProviderException">See the single-page overload.</exception>
    /// <exception cref="OcrBudgetException">See the single-page overload.</exception>
    /// <exception cref="OcrLimitException">See the single-page overload.</exception>
    public async Task<LlmExtraction> ExtractAsync(
        IReadOnlyList<LlmPageInput> pages, string instruction, string jsonSchema, string schemaName = "extraction", CancellationToken cancellationToken = default)
    {
        ThrowIfOffline();
        cancellationToken.ThrowIfCancellationRequested();
        Prepared prepared = Prepare(pages, instruction, jsonSchema, schemaName, cancellationToken);
        return await SendAsync(prepared, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Asks the model about one page and reads the answer into <typeparamref name="T"/>. The schema comes from the source-generated type information, so no reflection is used.</summary>
    /// <typeparam name="T">The answer type. Use nullable members for values that may be missing: strict providers make every member required.</typeparam>
    /// <param name="image">The page image.</param>
    /// <param name="page">The page model read from it.</param>
    /// <param name="instruction">What to extract. Trusted text.</param>
    /// <param name="typeInfo">Type information from a <c>JsonSerializerContext</c>.</param>
    /// <param name="schemaName">Name for the schema. Default <c>extraction</c>.</param>
    /// <param name="cancellationToken">Cancels the call.</param>
    /// <returns>The value, the raw JSON and what the call cost.</returns>
    /// <exception cref="OcrProviderException">As the raw overload; also <see cref="LlmFailure.Malformed"/> when the answer does not deserialize.</exception>
    public Task<LlmExtraction<T>> ExtractAsync<T>(
        Image image, OcrPage page, string instruction, JsonTypeInfo<T> typeInfo, string schemaName = "extraction", CancellationToken cancellationToken = default) =>
        ExtractAsync([new LlmPageInput(image, page)], instruction, typeInfo, schemaName, cancellationToken);

    /// <summary>Asks the model about several pages and reads the answer into <typeparamref name="T"/>.</summary>
    /// <typeparam name="T">The answer type.</typeparam>
    /// <param name="pages">The pages.</param>
    /// <param name="instruction">What to extract. Trusted text.</param>
    /// <param name="typeInfo">Type information from a <c>JsonSerializerContext</c>.</param>
    /// <param name="schemaName">Name for the schema. Default <c>extraction</c>.</param>
    /// <param name="cancellationToken">Cancels the call.</param>
    /// <returns>The value, the raw JSON and what the call cost.</returns>
    /// <exception cref="OcrProviderException">See the single-page overload.</exception>
    public async Task<LlmExtraction<T>> ExtractAsync<T>(
        IReadOnlyList<LlmPageInput> pages, string instruction, JsonTypeInfo<T> typeInfo, string schemaName = "extraction", CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(typeInfo);
        LlmExtraction raw = await ExtractAsync(pages, instruction, SchemaTools.Export(typeInfo), schemaName, cancellationToken).ConfigureAwait(false);
        try
        {
            T? value = JsonSerializer.Deserialize(raw.Json, typeInfo);
            return value is null
                ? throw new OcrProviderException(_provider.Name, LlmFailure.Malformed, "the answer is null.")
                : new LlmExtraction<T>(value, raw);
        }
        catch (JsonException e)
        {
            throw new OcrProviderException(_provider.Name, LlmFailure.Malformed, _provider.Scrub("the answer does not fit the requested type: " + e.Message));
        }
    }

    private void ThrowIfOffline()
    {
        if (_options.Offline)
        {
            throw new OcrProviderException(_provider.Name, LlmFailure.Offline, "the client is offline; nothing was sent.");
        }
    }

    private sealed record Prepared(
        LlmCall Call, IReadOnlyList<LlmWordRef> Words, int Masked, LlmEstimate Estimate, LlmModelPrice? Price, string CacheKey, List<string> Warnings);

    private Prepared Prepare(IReadOnlyList<LlmPageInput> pages, string instruction, string jsonSchema, string schemaName, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(pages);
        ArgumentException.ThrowIfNullOrWhiteSpace(instruction);
        ArgumentNullException.ThrowIfNull(jsonSchema);
        OcrLimits limits = _options.Limits;
        if (pages.Count == 0)
        {
            throw new ArgumentException("At least one page is required.", nameof(pages));
        }

        if (pages.Count > limits.MaxPagesPerCall)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxPagesPerCall), limits.MaxPagesPerCall, pages.Count);
        }

        if (schemaName.Length is < 1 or > 64 || !schemaName.All(static c => char.IsAsciiLetterOrDigit(c) || c is '_' or '-'))
        {
            throw new ArgumentException("The schema name must be 1 to 64 letters, digits, '_' or '-'.", nameof(schemaName));
        }

        // Validates size, depth and shape; the provider receives the re-serialized form.
        string schema = SchemaTools.Parse(jsonSchema, limits).ToJsonString();

        var masks = new IReadOnlyList<OcrBox>[pages.Count];
        for (int i = 0; i < pages.Count; i++)
        {
            masks[i] = _options.Redactor?.Invoke(pages[i].Page, i) ?? [];
        }

        // Worst-case request size is checked on the image sizes before anything is encoded.
        var images = new List<PreparedImage>(pages.Count);
        long requestBytes = Encoding.UTF8.GetByteCount(instruction) + schema.Length;
        for (int i = 0; i < pages.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            LlmPageInput input = pages[i];
            string label = string.Create(CultureInfo.InvariantCulture, $"Page {i + 1} of {pages.Count}:");
            PreparedImage image = ImagePrep.Prepare(input.Image, label, _provider.ImageOptions, masks[i], input.Page.Width, input.Page.Height, limits, ct);
            requestBytes += (image.Data.Length * 4L / 3) + 64;
            if (requestBytes > limits.MaxLlmRequestBytes)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxLlmRequestBytes), limits.MaxLlmRequestBytes, requestBytes);
            }

            images.Add(image);
        }

        (string userText, IReadOnlyList<LlmWordRef> words, int masked) = Prompt.Build(instruction, pages, masks, limits);
        requestBytes += Encoding.UTF8.GetByteCount(userText);
        if (requestBytes > limits.MaxLlmRequestBytes)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxLlmRequestBytes), limits.MaxLlmRequestBytes, requestBytes);
        }

        var call = new LlmCall(Prompt.SystemPrompt, userText, images, schemaName, schema, _options.MaxOutputTokens);

        long inputTokens = LlmProvider.EstimateTextTokens(Encoding.UTF8.GetByteCount(call.System) + Encoding.UTF8.GetByteCount(userText) + schema.Length);
        foreach (PreparedImage image in images)
        {
            inputTokens += _provider.EstimateImageTokens(image.Width, image.Height) + 16;
        }

        var warnings = new List<string>();
        DateOnly today = DateOnly.FromDateTime(_options.TimeProvider.GetUtcNow().UtcDateTime);
        if (_options.PriceTable.IsStale(today))
        {
            warnings.Add(string.Create(CultureInfo.InvariantCulture,
                $"The price table is from {_options.PriceTable.AsOf:yyyy-MM-dd}, more than {LlmPriceTable.StaleAfterDays} days ago; costs and spend caps may be wrong. Pass a current LlmPriceTable in LlmOptions."));
        }

        LlmModelPrice? price = _options.PriceTable.TryFind(_provider.PriceKey, _provider.Model, out LlmModelPrice found) ? found : null;
        decimal? worst = price?.Cost(inputTokens, call.MaxOutputTokens);
        var estimate = new LlmEstimate(inputTokens, call.MaxOutputTokens, worst, warnings);
        return new Prepared(call, words, masked, estimate, price, CacheKey(call), warnings);
    }

    private string CacheKey(LlmCall call)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        void Field(ReadOnlySpan<byte> bytes)
        {
            Span<byte> len = stackalloc byte[8];
            System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(len, bytes.Length);
            hash.AppendData(len);
            hash.AppendData(bytes);
        }

        Field(Encoding.UTF8.GetBytes(_provider.Name));
        Field(Encoding.UTF8.GetBytes(_provider.Model));
        Field(Encoding.UTF8.GetBytes(call.System));
        Field(Encoding.UTF8.GetBytes(call.UserText));
        Field(Encoding.UTF8.GetBytes(call.SchemaName));
        Field(Encoding.UTF8.GetBytes(call.SchemaJson));
        Field(Encoding.UTF8.GetBytes(call.MaxOutputTokens.ToString(CultureInfo.InvariantCulture)));
        foreach (PreparedImage image in call.Images)
        {
            Field(image.Data);
        }

        return Convert.ToHexString(hash.GetHashAndReset());
    }

    private async Task<LlmExtraction> SendAsync(Prepared p, CancellationToken ct)
    {
        if (_options.Cache is { } cache && cache.TryGet(p.CacheKey, out LlmExtraction hit))
        {
            return new LlmExtraction(
                hit.Json, new LlmUsage(0, 0, true), 0m, fromCache: true, hit.Provider, hit.Model, hit.Words, hit.MaskedWords, p.Warnings);
        }

        // A bound that cannot be checked is refused rather than ignored.
        bool bounded = _options.MaxCostPerCallUsd is not null || _options.SpendCapUsd is not null;
        if (bounded && p.Estimate.WorstCaseCostUsd is null)
        {
            throw new OcrBudgetException(
                LlmBudgetKind.UnknownPrice,
                $"A cost bound is set but the price table has no entry for {_provider.PriceKey} model '{_provider.Model}'. Add one with LlmPriceTable.With (a local server costs 0).",
                null, null, SpentUsd);
        }

        decimal worst = p.Estimate.WorstCaseCostUsd ?? 0m;
        if (_options.MaxCostPerCallUsd is { } perCall && worst > perCall)
        {
            throw new OcrBudgetException(
                LlmBudgetKind.PerCall,
                string.Create(CultureInfo.InvariantCulture, $"The call could cost up to ${worst:0.######}, above MaxCostPerCallUsd ${perCall:0.######}."),
                perCall, worst, SpentUsd);
        }

        byte[] body = _provider.BuildBody(p.Call, _options.Limits.MaxLlmJsonDepth);
        if (body.Length > _options.Limits.MaxLlmRequestBytes)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxLlmRequestBytes), _options.Limits.MaxLlmRequestBytes, body.Length);
        }

        await _gate.WaitAsync(ct).ConfigureAwait(false);
        decimal reserved = 0m;
        try
        {
            if (_options.SpendCapUsd is { } cap)
            {
                lock (_ledger)
                {
                    if (_spent + _reserved + worst > cap)
                    {
                        throw new OcrBudgetException(
                            LlmBudgetKind.SpendCap,
                            string.Create(CultureInfo.InvariantCulture, $"Spent ${_spent:0.######} (+${_reserved:0.######} in flight); this call could cost up to ${worst:0.######}, past the cap of ${cap:0.######}."),
                            cap, worst, _spent);
                    }

                    _reserved += worst;
                    reserved = worst;
                }
            }

            ProviderReply reply = await SendWithRetriesAsync(body, ct).ConfigureAwait(false);

            bool reported = reply.InputTokens >= 0 && reply.OutputTokens >= 0;
            long inTok = reply.InputTokens >= 0 ? reply.InputTokens : p.Estimate.InputTokens;
            long outTok = reply.OutputTokens >= 0 ? reply.OutputTokens : LlmProvider.EstimateTextTokens(Encoding.UTF8.GetByteCount(reply.Json));
            decimal? cost = p.Price?.Cost(inTok, outTok);
            lock (_ledger)
            {
                _spent += cost ?? 0m;
            }

            string answer = ScrubAnswer(reply.Json);
            var result = new LlmExtraction(
                answer, new LlmUsage(inTok, outTok, reported), cost, fromCache: false, _provider.Name, _provider.Model, p.Words, p.Masked, p.Warnings);
            _options.Cache?.Set(p.CacheKey, result);
            return result;
        }
        finally
        {
            if (reserved > 0m)
            {
                lock (_ledger)
                {
                    _reserved -= reserved;
                }
            }

            _gate.Release();
        }
    }

    private async Task<ProviderReply> SendWithRetriesAsync(byte[] body, CancellationToken ct)
    {
        string name = _provider.Name;
        for (int attempt = 0; ; attempt++)
        {
            ct.ThrowIfCancellationRequested();
            TimeSpan? retryAfter = null;
            OcrProviderException failure;
            try
            {
                using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                cts.CancelAfter(_options.Timeout);
                using HttpRequestMessage request = _provider.CreateRequest(body);
                using HttpResponseMessage response = await _provider.Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cts.Token).ConfigureAwait(false);
                if (response.IsSuccessStatusCode)
                {
                    byte[] bytes = await ResponseReader.ReadAsync(response.Content, _options.Limits.MaxLlmResponseBytes, cts.Token).ConfigureAwait(false);
                    return Cleaned(() => _provider.ParseReply(bytes, _options.Limits));
                }

                // Error bodies are small; read at most 64 KiB of one for the message.
                byte[] errorBody = await ResponseReader.ReadAtMostAsync(response.Content, 64 * 1024, cts.Token).ConfigureAwait(false);
                int status = (int)response.StatusCode;
                failure = new OcrProviderException(name, LlmFailure.Status, _provider.Scrub($"HTTP {status}: {ProviderJson.ErrorMessage(errorBody)}"), status);
                if (!LlmProvider.IsTransient(response.StatusCode))
                {
                    throw failure;
                }

                retryAfter = RetryAfter(response);
            }
            catch (OperationCanceledException) when (!ct.IsCancellationRequested)
            {
                failure = new OcrProviderException(name, LlmFailure.Transport, "the request timed out.");
            }
            catch (Exception e) when (e is HttpRequestException or IOException)
            {
                // The inner exception is not kept: its text can carry whatever the server or a proxy sent back.
                failure = new OcrProviderException(name, LlmFailure.Transport, _provider.Scrub("the request failed: " + e.Message));
            }

            if (attempt >= _options.MaxRetries)
            {
                throw failure;
            }

            await Task.Delay(Backoff(attempt, retryAfter), _options.TimeProvider, ct).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Runs a parser and re-throws its provider errors with the key scrubbed out of the text and without the inner exception, so
    /// nothing a server echoed can reach a caller's log.
    /// </summary>
    private ProviderReply Cleaned(Func<ProviderReply> parse)
    {
        try
        {
            return parse();
        }
        catch (OcrProviderException e)
        {
            string prefix = e.Provider + ": ";
            string text = e.Message.StartsWith(prefix, StringComparison.Ordinal) ? e.Message[prefix.Length..] : e.Message;
            throw new OcrProviderException(e.Provider, e.Failure, _provider.Scrub(text), e.StatusCode);
        }
    }

    /// <summary>A model that echoes the key in its answer must not get it into the result or the cache.</summary>
    private string ScrubAnswer(string json)
    {
        string clean = _provider.ScrubKey(json);
        if (ReferenceEquals(clean, json) || clean == json)
        {
            return json;
        }

        try
        {
            ProviderJson.CheckDepth(Encoding.UTF8.GetBytes(clean), _options.Limits.MaxLlmJsonDepth);
        }
        catch (JsonException)
        {
            throw new OcrProviderException(_provider.Name, LlmFailure.Malformed, "the answer repeated the API key and could not be cleaned.");
        }

        return clean;
    }

    private TimeSpan Backoff(int attempt, TimeSpan? retryAfter)
    {
        double ms = Math.Min(_options.MaxRetryDelay.TotalMilliseconds, _options.RetryBaseDelay.TotalMilliseconds * Math.Pow(2, attempt));
        ms *= 1 + (Random.Shared.NextDouble() * 0.25);
        if (retryAfter is { } ra && ra.TotalMilliseconds > ms)
        {
            ms = ra.TotalMilliseconds;
        }

        return TimeSpan.FromMilliseconds(Math.Min(ms, _options.MaxRetryDelay.TotalMilliseconds));
    }

    private TimeSpan? RetryAfter(HttpResponseMessage response)
    {
        var header = response.Headers.RetryAfter;
        if (header?.Delta is { } delta)
        {
            return delta;
        }

        if (header?.Date is { } date)
        {
            TimeSpan wait = date - _options.TimeProvider.GetUtcNow();
            return wait > TimeSpan.Zero ? wait : null;
        }

        return null;
    }
}

/// <summary>Reads response bodies with the byte limit checked against the declared length first and then while reading.</summary>
internal static class ResponseReader
{
    public static async Task<byte[]> ReadAsync(HttpContent content, long max, CancellationToken ct)
    {
        if (content.Headers.ContentLength is { } declared && declared > max)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxLlmResponseBytes), max, declared);
        }

        using Stream stream = await content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        var ms = new MemoryStream(content.Headers.ContentLength is { } n ? (int)Math.Min(n, 1 << 20) : 4096);
        byte[] chunk = System.Buffers.ArrayPool<byte>.Shared.Rent(16 * 1024);
        try
        {
            int read;
            while ((read = await stream.ReadAsync(chunk, ct).ConfigureAwait(false)) > 0)
            {
                if (ms.Length + read > max)
                {
                    throw new OcrLimitException(nameof(OcrLimits.MaxLlmResponseBytes), max, ms.Length + read);
                }

                ms.Write(chunk, 0, read);
            }
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(chunk);
        }

        return ms.ToArray();
    }

    public static async Task<byte[]> ReadAtMostAsync(HttpContent content, int max, CancellationToken ct)
    {
        using Stream stream = await content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        byte[] buffer = new byte[max];
        int total = 0, read;
        while (total < max && (read = await stream.ReadAsync(buffer.AsMemory(total, max - total), ct).ConfigureAwait(false)) > 0)
        {
            total += read;
        }

        return buffer.AsSpan(0, total).ToArray();
    }
}
