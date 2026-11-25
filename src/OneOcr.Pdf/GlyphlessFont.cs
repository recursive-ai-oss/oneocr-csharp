using System.Buffers.Binary;
using System.Text;

namespace OneOcr.Pdf;

/// <summary>
/// Builds the font that carries the invisible text layer: a TrueType font whose glyphs are all empty, one glyph for every
/// code point of the Basic Multilingual Plane that can be written, every glyph 0.5 em wide.
/// </summary>
/// <remarks>
/// <para>
/// The text is never painted (render mode 3), so the layer needs a font only for the two things a PDF reader takes from it:
/// the code-to-Unicode map and the advance widths that place each character along its word. Empty outlines make the font a few
/// hundred kilobytes before the writer cuts it down to the glyphs a file uses (a few hundred bytes), keep it free of anyone's
/// design, and need no licence: the bytes come from this code. The font is embedded, as PDF/A requires of every font.
/// </para>
/// <para>
/// Layout of glyph ids: 0 is <c>.notdef</c>; 1..95 are U+0020..U+007E; then U+00A0..U+D7FF; then U+E000..U+FFFD. Surrogates, the
/// C0 and C1 controls and the non-characters U+FFFE and U+FFFF have no glyph. The byte output is fixed (no clock, no randomness),
/// so a document that embeds it is reproducible. See <c>docs/spec/searchable-pdf.md</c>.
/// </para>
/// </remarks>
internal static class GlyphlessFont
{
    /// <summary>The advance of every glyph, in thousandths of an em.</summary>
    public const int AdvanceUnits = 500;

    /// <summary>Distance of the glyph box above the baseline, in ems (OnePDF reads glyph boxes from 0.9 em above to 0.2 em below).</summary>
    public const double Ascent = 0.9;

    /// <summary>Distance of the glyph box below the baseline, in ems.</summary>
    public const double Descent = 0.2;

    private const int FirstAscii = 0x20;
    private const int LastAscii = 0x7E;
    private const int FirstHigh = 0xA0;
    private const int LastLow = 0xD7FF;
    private const int FirstPrivate = 0xE000;
    private const int LastBmp = 0xFFFD;

    private static readonly Lazy<byte[]> s_bytes = new(Build);

    /// <summary>The font file.</summary>
    public static ReadOnlyMemory<byte> Bytes => s_bytes.Value;

    /// <summary>
    /// The code point the font draws for <paramref name="rune"/>: itself when the font has a glyph for it, otherwise U+FFFD
    /// (controls, surrogates, non-characters and everything outside the Basic Multilingual Plane).
    /// </summary>
    /// <param name="rune">A character of the text.</param>
    /// <returns>A code point with a glyph.</returns>
    public static int Map(Rune rune)
    {
        int v = rune.Value;
        return IsCovered(v) ? v : 0xFFFD;
    }

    /// <summary>Whether the font has a glyph for <paramref name="codePoint"/>.</summary>
    /// <param name="codePoint">A Unicode scalar value.</param>
    /// <returns>True when covered.</returns>
    public static bool IsCovered(int codePoint) =>
        (codePoint >= FirstAscii && codePoint <= LastAscii) || (codePoint >= FirstHigh && codePoint <= LastLow) || (codePoint >= FirstPrivate && codePoint <= LastBmp);

    private static int GlyphCount =>
        1 + (LastAscii - FirstAscii + 1) + (LastLow - FirstHigh + 1) + (LastBmp - FirstPrivate + 1);

    private static byte[] Build()
    {
        int n = GlyphCount;
        var tables = new SortedDictionary<string, byte[]>(StringComparer.Ordinal)
        {
            ["OS/2"] = Os2(),
            ["cmap"] = Cmap(),
            ["glyf"] = new byte[4],
            ["head"] = Head(),
            ["hhea"] = Hhea(),
            ["hmtx"] = Hmtx(n),
            ["loca"] = new byte[2 * (n + 1)],
            ["maxp"] = Maxp(n),
            ["name"] = Name(),
            ["post"] = Post(),
        };

        int count = tables.Count;
        int dirEnd = 12 + 16 * count;
        int total = dirEnd;
        foreach (byte[] t in tables.Values)
        {
            total += (t.Length + 3) & ~3;
        }

        byte[] file = new byte[total];
        BinaryPrimitives.WriteUInt32BigEndian(file, 0x00010000);
        BinaryPrimitives.WriteUInt16BigEndian(file.AsSpan(4), (ushort)count);
        int entrySelector = 0;
        while ((2 << entrySelector) <= count)
        {
            entrySelector++;
        }

        int searchRange = 16 << entrySelector;
        BinaryPrimitives.WriteUInt16BigEndian(file.AsSpan(6), (ushort)searchRange);
        BinaryPrimitives.WriteUInt16BigEndian(file.AsSpan(8), (ushort)entrySelector);
        BinaryPrimitives.WriteUInt16BigEndian(file.AsSpan(10), (ushort)(count * 16 - searchRange));

        int pos = dirEnd, entry = 12, headOffset = 0;
        foreach ((string tag, byte[] data) in tables)
        {
            Encoding.ASCII.GetBytes(tag, file.AsSpan(entry, 4));
            BinaryPrimitives.WriteUInt32BigEndian(file.AsSpan(entry + 4), Checksum(data));
            BinaryPrimitives.WriteUInt32BigEndian(file.AsSpan(entry + 8), (uint)pos);
            BinaryPrimitives.WriteUInt32BigEndian(file.AsSpan(entry + 12), (uint)data.Length);
            data.CopyTo(file, pos);
            if (tag == "head")
            {
                headOffset = pos;
            }

            pos += (data.Length + 3) & ~3;
            entry += 16;
        }

        // head.checkSumAdjustment: 0xB1B0AFBA minus the sum of the whole file (checksum field zero while summing).
        uint whole = Checksum(file);
        BinaryPrimitives.WriteUInt32BigEndian(file.AsSpan(headOffset + 8), unchecked(0xB1B0AFBA - whole));
        return file;
    }

    private static uint Checksum(ReadOnlySpan<byte> data)
    {
        uint sum = 0;
        int i = 0;
        for (; i + 4 <= data.Length; i += 4)
        {
            sum = unchecked(sum + BinaryPrimitives.ReadUInt32BigEndian(data[i..]));
        }

        if (i < data.Length)
        {
            Span<byte> last = stackalloc byte[4];
            data[i..].CopyTo(last);
            sum = unchecked(sum + BinaryPrimitives.ReadUInt32BigEndian(last));
        }

        return sum;
    }

    private static byte[] Head()
    {
        byte[] t = new byte[54];
        BinaryPrimitives.WriteUInt32BigEndian(t, 0x00010000);          // version
        BinaryPrimitives.WriteUInt32BigEndian(t.AsSpan(4), 0x00010000); // fontRevision 1.0
        BinaryPrimitives.WriteUInt32BigEndian(t.AsSpan(12), 0x5F0F3CF5); // magic
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(16), 0);         // flags
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(18), 1000);      // unitsPerEm
        // created/modified stay zero: no clock in the output.
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(36), 0);          // xMin
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(38), -200);       // yMin
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(40), AdvanceUnits); // xMax
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(42), 900);        // yMax
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(44), 0);         // macStyle
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(46), 8);         // lowestRecPPEM
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(48), 2);          // fontDirectionHint
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(50), 0);          // indexToLocFormat: short offsets
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(52), 0);          // glyphDataFormat
        return t;
    }

    private static byte[] Hhea()
    {
        byte[] t = new byte[36];
        BinaryPrimitives.WriteUInt32BigEndian(t, 0x00010000);
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(4), 900);
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(6), -200);
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(8), 0);
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(10), AdvanceUnits);
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(12), 0);
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(14), 0);
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(16), AdvanceUnits);
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(18), 1);          // caretSlopeRise
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(34), 1);         // numberOfHMetrics: one record, the rest share its advance
        return t;
    }

    // One (advance, lsb) record, then a left side bearing per remaining glyph; every glyph has the same advance.
    private static byte[] Hmtx(int glyphs)
    {
        byte[] t = new byte[4 + 2 * (glyphs - 1)];
        BinaryPrimitives.WriteUInt16BigEndian(t, AdvanceUnits);
        return t;
    }

    private static byte[] Maxp(int glyphs)
    {
        byte[] t = new byte[32];
        BinaryPrimitives.WriteUInt32BigEndian(t, 0x00010000);
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(4), (ushort)glyphs);
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(14), 1);         // maxZones
        return t;
    }

    // Format 4, platform 3 (Windows), encoding 1 (Unicode BMP). Every segment maps a run of code points to consecutive glyph ids by idDelta alone.
    private static byte[] Cmap()
    {
        (int start, int end, int firstGlyph)[] segments =
        [
            (FirstAscii, LastAscii, 1),
            (FirstHigh, LastLow, 1 + (LastAscii - FirstAscii + 1)),
            (FirstPrivate, LastBmp, 1 + (LastAscii - FirstAscii + 1) + (LastLow - FirstHigh + 1)),
            (0xFFFF, 0xFFFF, 0),
        ];

        int segCount = segments.Length;
        int subLength = 16 + 8 * segCount;
        byte[] t = new byte[12 + subLength];
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(2), 1);              // one encoding record
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(4), 3);              // platform
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(6), 1);              // encoding
        BinaryPrimitives.WriteUInt32BigEndian(t.AsSpan(8), 12);             // offset of the subtable
        Span<byte> s = t.AsSpan(12);
        BinaryPrimitives.WriteUInt16BigEndian(s, 4);
        BinaryPrimitives.WriteUInt16BigEndian(s[2..], (ushort)subLength);
        BinaryPrimitives.WriteUInt16BigEndian(s[6..], (ushort)(segCount * 2));
        int entrySelector = 0;
        while ((2 << entrySelector) <= segCount)
        {
            entrySelector++;
        }

        int searchRange = 2 << entrySelector;
        BinaryPrimitives.WriteUInt16BigEndian(s[8..], (ushort)searchRange);
        BinaryPrimitives.WriteUInt16BigEndian(s[10..], (ushort)entrySelector);
        BinaryPrimitives.WriteUInt16BigEndian(s[12..], (ushort)(segCount * 2 - searchRange));
        int endCodes = 14, startCodes = endCodes + 2 * segCount + 2, deltas = startCodes + 2 * segCount, offsets = deltas + 2 * segCount;
        for (int i = 0; i < segCount; i++)
        {
            BinaryPrimitives.WriteUInt16BigEndian(s[(endCodes + 2 * i)..], (ushort)segments[i].end);
            BinaryPrimitives.WriteUInt16BigEndian(s[(startCodes + 2 * i)..], (ushort)segments[i].start);
            ushort delta = unchecked((ushort)(segments[i].firstGlyph - segments[i].start));
            if (i == segCount - 1)
            {
                delta = 1; // the terminating segment maps 0xFFFF to glyph 0 (0xFFFF + 1 wraps to 0)
            }

            BinaryPrimitives.WriteUInt16BigEndian(s[(deltas + 2 * i)..], delta);
            BinaryPrimitives.WriteUInt16BigEndian(s[(offsets + 2 * i)..], 0);
        }

        return t;
    }

    private static byte[] Os2()
    {
        byte[] t = new byte[96];
        BinaryPrimitives.WriteUInt16BigEndian(t, 4);                         // version
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(2), AdvanceUnits);     // xAvgCharWidth
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(4), 400);             // usWeightClass
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(6), 5);               // usWidthClass
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(8), 0);               // fsType: installable embedding
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(68), 900);             // sTypoAscender
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(70), -200);            // sTypoDescender
        BinaryPrimitives.WriteInt16BigEndian(t.AsSpan(72), 0);               // sTypoLineGap
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(74), 900);            // usWinAscent
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(76), 200);            // usWinDescent
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(64), FirstAscii);     // usFirstCharIndex
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(66), LastBmp);        // usLastCharIndex
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(62), 0x0040);         // fsSelection: regular
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(86), 0);              // sxHeight
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(90), 0);              // usDefaultChar
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(92), 0x20);           // usBreakChar
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(94), 1);              // usMaxContext
        return t;
    }

    private static byte[] Post()
    {
        byte[] t = new byte[32];
        BinaryPrimitives.WriteUInt32BigEndian(t, 0x00030000);                // format 3: no glyph names
        return t;
    }

    private static byte[] Name()
    {
        (int id, string text)[] records =
        [
            (0, "Copyright (c) Recursive AI, LLC. MIT OR Apache-2.0."),
            (1, "OneOcrText"),
            (2, "Regular"),
            (4, "OneOcrText Regular"),
            (6, "OneOcrText-Regular"),
        ];

        int stringsStart = 6 + 12 * records.Length;
        var strings = new List<byte[]>();
        int offset = 0;
        byte[] header = new byte[stringsStart];
        BinaryPrimitives.WriteUInt16BigEndian(header.AsSpan(2), (ushort)records.Length);
        BinaryPrimitives.WriteUInt16BigEndian(header.AsSpan(4), (ushort)stringsStart);
        for (int i = 0; i < records.Length; i++)
        {
            byte[] s = Encoding.BigEndianUnicode.GetBytes(records[i].text);
            Span<byte> r = header.AsSpan(6 + 12 * i);
            BinaryPrimitives.WriteUInt16BigEndian(r, 3);                      // platform: Windows
            BinaryPrimitives.WriteUInt16BigEndian(r[2..], 1);                 // encoding: Unicode BMP
            BinaryPrimitives.WriteUInt16BigEndian(r[4..], 0x0409);            // language: en-US
            BinaryPrimitives.WriteUInt16BigEndian(r[6..], (ushort)records[i].id);
            BinaryPrimitives.WriteUInt16BigEndian(r[8..], (ushort)s.Length);
            BinaryPrimitives.WriteUInt16BigEndian(r[10..], (ushort)offset);
            offset += s.Length;
            strings.Add(s);
        }

        byte[] t = new byte[stringsStart + offset];
        header.CopyTo(t, 0);
        int pos = stringsStart;
        foreach (byte[] s in strings)
        {
            s.CopyTo(t, pos);
            pos += s.Length;
        }

        return t;
    }
}
