namespace OneOcr.Llm;

/// <summary>
/// An opt-in, in-memory cache of answers, keyed by a SHA-256 of the provider, model, prompt, schema and the exact bytes of
/// the images that would be uploaded (after redaction). Thread-safe; the least recently used entry is dropped at capacity.
/// A cached answer costs nothing and sends nothing. Share one instance between clients only if their providers should
/// see each other's answers; the provider and model are part of the key.
/// </summary>
public sealed class LlmResultCache
{
    private readonly object _gate = new();
    private readonly Dictionary<string, LinkedListNode<(string Key, LlmExtraction Value)>> _map = new(StringComparer.Ordinal);
    private readonly LinkedList<(string Key, LlmExtraction Value)> _order = new();
    private readonly int _capacity;

    /// <summary>Creates a cache.</summary>
    /// <param name="capacity">Most answers kept. Default 256.</param>
    public LlmResultCache(int capacity = 256)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        _capacity = capacity;
    }

    /// <summary>Number of answers held.</summary>
    public int Count
    {
        get
        {
            lock (_gate)
            {
                return _map.Count;
            }
        }
    }

    /// <summary>Drops every answer.</summary>
    public void Clear()
    {
        lock (_gate)
        {
            _map.Clear();
            _order.Clear();
        }
    }

    internal bool TryGet(string key, out LlmExtraction value)
    {
        lock (_gate)
        {
            if (_map.TryGetValue(key, out var node))
            {
                _order.Remove(node);
                _order.AddFirst(node);
                value = node.Value.Value;
                return true;
            }
        }

        value = null!;
        return false;
    }

    internal void Set(string key, LlmExtraction value)
    {
        lock (_gate)
        {
            if (_map.TryGetValue(key, out var existing))
            {
                _order.Remove(existing);
                _map.Remove(key);
            }

            _map[key] = _order.AddFirst((key, value));
            while (_map.Count > _capacity)
            {
                var last = _order.Last!;
                _order.RemoveLast();
                _map.Remove(last.Value.Key);
            }
        }
    }
}
