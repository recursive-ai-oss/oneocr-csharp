# Text normalization v1

The library normalizes text in three places: the extraction model's character ids (NFC), the grounding fold (NFKC) and the
training agreement check (NFC). The runtime's `string.Normalize` cannot be used for this: under invariant globalization it leaves
non-ASCII text unchanged, under ICU it composes, and ICU's Unicode version follows the OS. Normalization is therefore our own and
reads no data at run time. A port must produce the same strings.

## Forms

NFC and NFKC exactly as UAX #15 defines them (https://www.unicode.org/reports/tr15/), with the default (not stream-safe-limited)
behavior:

1. **Decompose** each code point: Hangul syllables arithmetically (Unicode Standard 3.12), everything else through the tables,
   recursively and to the end. NFC uses canonical decompositions; NFKC uses compatibility decompositions, which include the
   canonical ones.
2. **Reorder** each run of code points with a non-zero canonical combining class by that class, stably.
3. **Compose**: walk the result; a code point combines with the last starter before it when no code point between them has a class
   greater than or equal to its own (adjacent always qualifies) and the pair has a primary composite. The composite replaces the
   starter. Primary composites are the canonical two-code-point decompositions minus `CompositionExclusions.txt`, singletons and
   non-starter decompositions; Hangul L+V and LV+T compose arithmetically.

Both forms are idempotent. Text already in the form is returned unchanged (the same string instance in C#).

## Coverage and pass-through

The tables are generated from the full `UnicodeData.txt` of **Unicode 17.0.0** (`UnicodeData.txt` SHA-256
`2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c`, `CompositionExclusions.txt` SHA-256
`2f239196ef3b5b61db5cc476e9bd80f534d15aa1b74e1be1dea5d042a344c85f`): 403 combining-class ranges, 2,081 canonical and 3,849
compatibility decompositions, 961 compositions. That covers every character of charset-v1 (Latin, Greek incl. polytonic, Cyrillic),
every combining mark that composes into them, and the compatibility characters the NFKC fold depends on (ligatures, full-width
forms, superscripts, no-break and other spaces, fractions, roman numerals).

A code point the tables do not list (unassigned, or added after 17.0.0) has combining class 0, no decomposition and no
composition, and **passes through unchanged**. An unpaired surrogate is such a code point: it is copied as it is and never causes
an error. Noncharacters pass through too.

The pinned version changes only with a spec revision; a newer Unicode never alters the result of text that already normalizes
under 17.0.0 except for newly assigned characters.

## Regenerating

`dotnet run --project tools/NormalizationTables -- <ucd-dir> src/OneOcr/Unicode/NormalizationTables.g.cs` with the two files in
`<ucd-dir>`; the tool checks their SHA-256 against the pinned values above.

## Quick paths

NFC returns its input when every UTF-16 unit is below U+0300; NFKC when every unit is below U+00A0. These are optimizations; the
result is the same as running the three steps.

## Checks

The test suite compares both forms against the runtime's ICU-backed `Normalize` over every Unicode scalar value, every base in
Latin, Greek, Cyrillic and Latin Extended Additional / Greek Extended with every combining mark (and with every ordered pair of
the marks Vietnamese and polytonic Greek stack), whenever it runs with ICU; the same inputs are asserted in both modes by the
explicit cases.
