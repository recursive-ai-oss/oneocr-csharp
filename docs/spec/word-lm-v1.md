# Word frequency file, version 1

A file holds how often words occurred in running text. The recognizer's word segmentation (`oneocr-page-v1.md`, "Word segmentation")
uses it to take back a space the network put inside a word. It is produced by `training/wordlm/build_wordlm.py` and read by
`WordFrequencyModel`. All integers are little-endian. Another implementation must reproduce the lookup and the comparison below
exactly: they use integers only.

## What the table holds

Words of Latin letters only, each lowercased by the simple one-to-one lowercase mapping of its code points (a character without one,
and U+0130, stay as they are), as in `char-lm-v1.md`. A word with a digit, an apostrophe, a hyphen, a code symbol or a letter of
another script is not in the table. A word is 1 to 40 UTF-8 bytes. The builder reads the Latin-script books marked `train` in
`training/data/texts.yaml` (the books the character language model is built from; Alice and Pride and Prejudice, the evaluation
page sets' texts, and the held-out books are not among them) and keeps words seen at least `min_count` times (default 3).

A count is stored as `q = clamp(round(4 * log2(count)), 1, 255)`, a quarter of a bit: the count is about `2^(q/4)`. `q` is never 0.

## Layout

```
magic        4 bytes  "OCWF"
version      u16      1
flags        u16      0 (a reader rejects any other value)
total        u32      the number of word tokens the counts come from
words        u32      the number of words W in the table
q_total      u8       clamp(round(4 * log2(total)), 1, 255)
q_missing    u8       the q taken for a word that is not in the table
blocks       u32      ceil(W / 16)
offsets      u32 x blocks   byte offset of each block in the word data
word data:
  entries, W of them, in ascending byte order of the word (strictly increasing):
    shared   u8       bytes the word shares with the previous entry's word (0 for the first entry of a block)
    suffix   u8       bytes that follow; shared + suffix is 1 to 40
    bytes    suffix bytes of UTF-8
    q        u8       1 to 255
```

Blocks hold 16 entries (the last holds the rest); block `i` starts at `offsets[i]`, which is where block `i - 1` ended (the first
at 0). A reader checks all of this while it parses: lengths stay inside the word and the file, words strictly rise, no bytes
remain after the last entry. The builder sets `q_missing = max(1, q(min_count) - 4)`, a word seen about half as often as the
rarest word in the table.

## Lookup

Binary search over the blocks on the first word of each (the last block whose first word is not above the target), then a scan of
that block, rebuilding each word from the previous one's prefix. Words compare as unsigned bytes, a prefix before its extensions.

## Use

The page reader's word segmentation (`oneocr-page-v1.md`, "Word segmentation", rule 3) takes two neighbouring pieces `a` and `b`
(lowercased, letters only) with joined word `ab`, and compares, in quarter-bits,

    gain = q(ab) + q_total - q(a) - q(b)        (q of a word that is not in the table is q_missing)

which is four times the base-2 log-odds that the letters are one word and not two independent words. All comparisons are on
integers, so every platform decides alike. The table holds 217,000 words from the Latin-script books of 15 languages and is 1.3 MB.
It knows nothing of numbers, codes, names or words with an apostrophe or hyphen, and such words are never joined with it.
