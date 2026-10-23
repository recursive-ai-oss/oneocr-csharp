# CTC decoding

Input: per-step class scores (`logits`, `T` steps of `C` classes, class 0 the CTC blank) from a recognizer, and the charset.
Output: text and a confidence. Every mode gives the same bits on every CPU path, OS and degree of parallelism.

## Greedy (default)

Arg-max per step (lowest index on ties), merge repeats, drop blanks (Graves et al. 2006, sec. 3.1). Confidence: mean over
emitted characters of the largest softmax probability seen while that character was the winning class (0 for empty text).

## Beam search

Prefix beam search over CTC alignments (Graves et al. 2006; Hannun et al. 2014, "First-pass large vocabulary continuous speech
recognition using bi-directional recurrent DNNs").

### Arithmetic

All beam arithmetic is IEEE double precision with `+ - * /` and comparisons only, evaluated in the order written here, no fused
multiply-add. The exponential and logarithm are our own:

* `exp(x)`: `x <= -700` (or NaN) gives 0, `x > 700` is clamped to 700; `k = floor(x * 1.4426950408889634 + 0.5)`;
  `r = (x - k * 6.93147180369123816490e-01) - k * 1.90821492927058770002e-10`; the degree-13 Taylor polynomial of exp in Horner form
  with coefficients `1/n!` (`n = 13 … 0`), each step `p = p * r + c`; the result is `p * 2^k` built from the exponent bits.
* `ln(x)`: `x = m * 2^e` with `m` in `[1, sqrt 2)` (halve `m` and add 1 to `e` when it is above `sqrt 2`);
  `s = (m - 1) / (m + 1)`, `z = s * s`, `q` the Horner form of `1 + z/3 + z²/5 + … + z⁹/19` (coefficients `1/(2j+1)`), result
  `e * 0.6931471805599453 + 2 * s * q`.
* `logadd(a, b) = a + ln(1 + exp(b - a))` for `a >= b`; `a` when `b` is minus infinity or `b - a < -40`.

Log probabilities of classes: `lp(j) = (z_j - max) - ln(S)`, with `z` the float logits, `max` their largest, `S` the float sum
of `exp(z_j - max)` accumulated as in `numerics.md` (the same 16-lane order as the softmax elsewhere). The `float` subtraction
is exact; the rest is double.

### Steps

State: a set of prefixes (at most `BeamWidth`), each with `pb` and `pnb`, the log probabilities of the steps so far with the
prefix emitted and the last step blank / non-blank. Start: the empty prefix with `pb = 0`, `pnb = -inf`.

For each step `t`:

1. Candidates: the classes `j >= 1` whose logit is within 9.2 of `max`, at most 8, best first (equal logits keep the lower class).
2. For each prefix `p` with `total = logadd(pb, pnb)` and last class `c` (none for the empty prefix), in beam order:
   * `p` gets `pb += total + lp(0)`;
   * if `c` exists and `pnb` is not `-inf`: `p` gets `pnb += pnb + lp(c)` (the run of `c` continues);
   * for each candidate `j`, the extended prefix `p·j` gets `pnb += (pb if j == c else total) + lp(j)`.
   Contributions to the same prefix are combined with `logadd`; new prefixes are listed in creation order.
3. Rank the resulting prefixes by `logadd(pb, pnb) + R(prefix)` (below), best first, ties to the prefix created earlier, and keep
   the first `BeamWidth`.

After the last step each kept prefix gets `final = logadd(pb, pnb) + R_end(prefix)`; the best (ties: created earlier) is the
text. If a language model is set (whether or not a script fits the line), the best prefix whose code words (below) match the greedy
reading's is taken, and the greedy reading if none does.

Confidence: `exp(logadd(pb, pnb) / n)` of the chosen text with `n` emitted characters, clamped to `[0, 1]` (0 for empty text).
It is the per-character geometric mean of the probability of the text over all alignments, and is not calibrated.

### Prefix score `R`

A *word* is a run of non-space characters in the prefix (the space is the charset entry `" "`). `R` is the sum of a part for the
finished words, `InsertionBonus` per character, and a part for the open word.

* **Language model** (when set): for a script chosen once per line (the model whose alphabet contains the most letters of the
  greedy text's characters; the first on ties; none if no letter fits), each character `s` of a word that is not *protected*
  adds `LanguageModelWeight * max(LogProb(context, fold(s)) - mean, -1)`, where `fold` is the simple lowercase mapping (see
  `char-lm-v1.md`), `mean` is the file's mean and the context is the word's earlier symbols led by the word start. A finished word additionally adds the same for the word end. The open word counts
  without the word end in `R` and with it in `R_end`.
* **Protected words**: a word is protected, and gets no language-model score, if it contains a decimal digit, one of
  `@ # _ / \ = % + * ~ ^ | < > [ ] { }`, or a character whose folded form is outside the script's alphabet.
* **Lexicon** (when set): the open or finished word adds `LexiconBoost` when it spells a lexicon word exactly (same code points)
  and `LexiconBoost / 2` when it is a proper prefix of one and at least two characters long; nothing otherwise. Words with
  characters not in the charset are dropped from the lexicon.

### Code words

The code words of a text are its space-delimited words that contain a decimal digit or one of the symbols above, in order.
Beam search with a language model returns a reading only if its code words are those of the greedy reading, position by position,
except that a word listed in the lexicon may stand in for the one it replaces. Numbers, identifiers, amounts and dates are
therefore read exactly as greedy decoding reads them; the language model only changes the other words.

## Parameters

| name | default | meaning |
|---|---|---|
| mode | greedy | greedy or beam |
| BeamWidth | 8 | 1 to `OcrLimits.MaxBeamWidth` (64) |
| LanguageModelWeight | 0.8 | at least 0 |
| LexiconBoost | 2 | at least 0 |
| InsertionBonus | 0 | per emitted character |

The candidate count (8), the 9.2 window and the −1 floor were chosen on held-out synthetic lines (see the decoding evaluation in
`bench/README.md`) and are part of this version.

## Limits

Before the search starts, `steps * BeamWidth * 9 + 1` (the most prefixes it can create) is compared with
`OcrLimits.MaxBeamNodes` (default 2^22), and `BeamWidth` with `OcrLimits.MaxBeamWidth`.
