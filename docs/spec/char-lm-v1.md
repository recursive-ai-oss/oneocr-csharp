# Character language model file, version 1

A file holds one n-gram model of words per script. The decoder (`decoding.md`) uses it to rank alternative readings of a text
line. It is produced by `training/charlm/build_charlm.py` and read by `CharLanguageModel`. All integers are little-endian.

## What the model scores

A word is a run of non-space characters, case-folded: each character is looked up by its simple lowercase form (Unicode
one-to-one lowercase mapping; a character without one, and U+0130, stay as they are), so `NASI`, `Nasi` and `nasi` are one word and
the alphabet holds lowercase letters. The model scores one word at a time, one symbol after another, starting from a
word-start symbol and ending with a word-end symbol. There is no context across spaces. Symbols are small integers:

| id | meaning |
|---|---|
| 0 | word start (only ever the first symbol of a context) |
| 1 | word end (only ever the last symbol of an n-gram) |
| 2 … 251 | the script's alphabet, in the file's order (`id = index + 2`) |

The alphabet holds at most 250 characters. A character whose folded form is not in a script's alphabet makes its word unscorable for
that script (the decoder gives such words no language-model score).

## Layout

```
magic       4 bytes   "OCLM"
version     u16       1
scripts     u16       number of script models, 1 or more
script model, repeated `scripts` times:
  name_len  u8        1 or more
  name      ASCII     printable, no spaces ("latin", "greek", "cyrillic")
  order     u8        highest n-gram order N, 1 to 8
  alphabet  u8        number of characters A, 1 to 250
  characters, A times:  len u8 (1 to 8), then len bytes of UTF-8 (one charset class)
  mean      f32       average log probability per symbol on the training words, in nats, between -50 and 0
  tables, for n = 1 .. N:
    count   u32
    rows, `count` of them:  key n bytes (symbol ids), prob u8, and for n < N also bow i8
```

Rows of a table are sorted by key, comparing bytes from the first to the last (strictly increasing, no duplicates). Every
byte of a key is below A + 2; byte 0 may only be the first byte of a key, byte 1 only the last.

* `prob` q means ln P = -0.1 q (q = 0 … 254).
* `bow` q (signed) means a back-off weight ln b = 0.1 q.

## Probabilities

The builder estimates interpolated Kneser–Ney probabilities (Kneser and Ney 1995; Chen and Goodman 1998): absolute
discounting with D = n1 / (n1 + 2 n2) per order from the counts of counts, lower orders on continuation counts (the number of
distinct symbols that precede the n-gram; n-grams that begin a word use their plain counts). It then prunes n-grams by
a minimum count per order, keeps every context of a kept n-gram, and writes the model in back-off form, as ARPA files do:
each kept n-gram stores its full interpolated probability, and each context stores the weight that makes the pruned model sum
to 1 over the words that follow it, `(1 - Σ P(w|h)) / (1 - Σ P_lower(w|h'))` over the kept extensions `w` of the context `h`.
The first table holds every symbol of the alphabet and the word end, and the word start as a context (its row only carries a
back-off weight).

## Scoring

`LogProb(context, s)`: let `ctx` be the last `min(len, N-1)` symbols of the context (the word start counts as the first
symbol of a word). Starting with `back = 0` and `L = len(ctx)`, down to 0:

1. If the table of order `L + 1` has the row `ctx[len-L..] ++ s`, return `back - 0.1 * prob`.
2. Otherwise, if `L > 0` and the table of order `L` has the row `ctx[len-L..]`, add `0.1 * bow` of that row to `back`.
3. Continue with `L - 1`.

If no row is found (a file that lacks a unigram), the result is `back - 14`. The sum is accumulated in `float` in that order.

## Limits

Reading checks, before allocating, the file size (`OcrLimits.MaxLanguageModelFileBytes`), scripts, order, the n-gram total
over the file, and that the bytes of a table are present. Violations are `OcrLimitException` (a bound) or `OcrModelException`
(structure).

## Sizes

The model shipped with the recognizer (order 4, minimum counts 1, 3, 4, 6; the 187 training books, case-folded) is 1.42 MB: latin 1,015,999 bytes
(130 characters), greek 301,157 bytes (57), cyrillic 104,747 bytes (57).
