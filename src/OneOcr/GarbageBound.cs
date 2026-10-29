namespace OneOcr;

/// <summary>
/// Keeps the garbage a page leaves behind from piling up on hosts whose collector waits a long time between collections.
/// </summary>
/// <remarks>
/// The runtime sizes its gen 0 budget from the processor cache, and on a machine with a large cache that is
/// hundreds of megabytes: a run over many pages then grows by what every page allocated and never collects. After a page, if the
/// process has allocated <see cref="Threshold"/> bytes since any collection, one gen 0 collection is run, which is what a
/// collector with a small budget would have done on its own. Where the runtime collects more often than that, this never fires.
/// </remarks>
internal static class GarbageBound
{
    // Roughly one page of ordinary size; a gen 0 collection over that much garbage costs well under a millisecond.
    internal const long Threshold = 4L << 20;

    private static long s_allocatedAtLastCollection;
    private static int s_collections;

    public static void AfterPage()
    {
        int collections = GC.CollectionCount(0);
        long allocated = GC.GetTotalAllocatedBytes(precise: false);
        if (collections != Volatile.Read(ref s_collections))
        {
            // The runtime collected on its own since the last look.
            Volatile.Write(ref s_collections, collections);
            Volatile.Write(ref s_allocatedAtLastCollection, allocated);
            return;
        }

        if (allocated - Volatile.Read(ref s_allocatedAtLastCollection) >= Threshold)
        {
            GC.Collect(0, GCCollectionMode.Forced, blocking: true, compacting: false);
            Volatile.Write(ref s_collections, GC.CollectionCount(0));
            Volatile.Write(ref s_allocatedAtLastCollection, allocated);
        }
    }
}
