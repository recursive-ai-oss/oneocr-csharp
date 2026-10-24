# Document extraction model, `docextract-v0`

A small transformer over the words of a page. It labels words; deterministic code joins the labelled words into fields, so a value is always the text of words that are on the page. The container section is in [model-container-v1.md](model-container-v1.md), the numeric rules of the kernels in [numerics.md](numerics.md). The model's graph and training are described in `training/export/docextract-architecture.md`; this document repeats what a reader of the weights file needs and specifies what the library does around it: featurization, windows, assembly and normalization.

## Input

The words of an `OcrPage` in reading order, with empty texts dropped; box = the word's box in page units (origin top left, y down); page size W x H (each at least 1). `unit` = the median word height (the mean of the two middle values for an even count), at least 1; word height and width are at least 1e-3. All of this is in double precision until the values are stored as fp32 (`feats`, `geo`). Non-finite coordinates count as 0 and every coordinate is limited to ±1e9.

| input | definition |
|---|---|
| `chars` `[N, max_chars]` | the word in NFC; each code point (not UTF-16 unit) is its index in the container `charset`, or `unk_char` when absent; the first `max_chars` kept; the rest 0. A text that cannot be normalized (unpaired surrogates) is read as it is, an invalid unit as U+FFFD. |
| `bins` `[N, 4]` | `clamp(floor(bins * v), 0, bins - 1)` for `v` = x0/W, y0/H, x1/W, y1/H |
| `feats` `[N, 8]` | x0/W, y0/H, x1/W, y1/H, w/W, h/H, `min(h / unit, 8) - 1`, `a / (1 + a)` with `a = w / h` |
| `geo` `[N, 4]` | the box divided by `unit` |

## Graph

Width `d` (192), `layers` (5) pre-layer-norm blocks with `heads` (6) heads and an `ffn` (512) feed-forward layer with tanh GELU; `char_dim` 48, `char_conv` 128, `pair_hidden` 32, `rel_dim` 64.

1. Word vector: character embedding, convolution (kernel 3, zero padding), ReLU, maximum over the word's valid positions (`char_pool`), linear `char_proj`. Add the rows `pos_x0[bins0]`, `pos_x1[bins2]`, `pos_y0[bins1]`, `pos_y1[bins3]` (in that order) and `feat_proj(feats)`; layer norm `emb_ln` gives `emb`.
2. Pair bias `[layers * heads + 2, N, N]`: 13 features of word *j* relative to word *i* (numerics.md) through `pair1`, ReLU, `pair2`. Channels `6l .. 6l + 5` are the head biases of block *l*; the last two are the biases of the two pointer relations.
3. Block *l*: `u = ln1(x)`; `q, k, v = qkv(u)` (each `[N, heads, head size]`, in that order along the 3d axis); per head `softmax(q kᵀ / sqrt(head size) + bias) v`; `x += proj(heads)`; `x += fc2(gelu(fc1(ln2(x))))`.
4. `final = final_ln(x)`. Heads: `tags = tag_head(final)` (`[N, 1 + 2 * classes]`: 0 = O, `1 + 2c` = B-class *c*, `2 + 2c` = I-class *c*), `roles = role_head(final)` (O, B-KEY, I-KEY, B-VAL, I-VAL), `doc = doc_head(mean of final)`, and the pointers `rel[r, i, j] = q_i,r · k_j,r / sqrt(rel_dim) + pair_bias[layers * heads + r, i, j]` with `q = rel_q(final)`, `k = rel_k(final)` viewed `[N, 2, rel_dim]`. Relation 0 (item): a word of a line item or tax row points to the first word of that item or row (the anchor), the anchor to itself. Relation 1 (key/value): a VAL word points to the first word of its key, a KEY word to itself.

## Windows

A page of at most `max_words` words is one window. A longer page is run in windows of `max_words` words starting every `max_words * 3 / 4` words (512 and 384), the last window ending at the last word: starts 0, s, 2s, … while `start + max_words < N`. A word belongs to the window whose centre is nearest the word's centre (`(start + end) / 2` against `index + 0.5`; the earlier window on a tie) and takes its tag logits, role logits and pointer row from it. Pointers are scored inside a window only; the pointer target is the window-local argmax (the lowest index on a tie) shifted by the window start, so a pointer never leaves its window. The document-type logits of a multi-window page are the mean of the windows' logits weighted by the number of words each window owns, in double precision, rounded to fp32.

## Assembly

1. Document type = argmax of the document logits (lowest index on a tie); its probability is the softmax. A caller that asks for a schema (`ExtractInvoice`, `ExtractReceipt`) reads with that schema's classes whatever the model predicts and gets a note when the two differ; the predicted type is still reported.
2. Tag classes of the other schemas are masked: for `invoice` the classes without the prefix `receipt.`, for `receipt` those without `BT-`, for `other` those with neither prefix. A class name with neither prefix belongs to every type (a model with its own classes).
3. Per word the tag is the argmax over the allowed tags (the O tag included; the lowest index on a tie); its probability is `exp(l - max) / Σ exp(l' - max)` over the allowed tags, summed in index order.
4. A span is a B tag plus the following I tags of the same class on consecutive words; an I tag whose previous word is not in a span of its class starts a span. Span confidence = the mean probability of its words' tags (double precision).
5. Header classes (not item or tax classes) keep their highest-confidence span (the first on a tie). Item classes (`BT-126 BT-129 BT-130 BT-131 BT-146 BT-152 BT-153`, `receipt.itemName receipt.quantity receipt.unitPrice receipt.amount`) and tax classes (`BT-116 BT-117 BT-119`): the span's first word follows relation 0 to an anchor (the pointer of the pointer, at most 4 steps; stop at a word that points to itself; on a cycle, the smallest index on it; after 4 steps, wherever it is). Spans with the same family and anchor form one item or tax row, each class once (highest confidence), ordered by anchor; fields in an item are ordered by their first word.
6. Roles: the argmax per word, spans as above for KEY and VAL; a VAL span pairs with the KEY span that contains the word its first word points to by relation 1; otherwise it is dropped. The pair's confidence is the smaller of the two spans'.
7. The text of a span is its words joined by one space.

Differences from taking the argmax of float32 probabilities: the library takes the argmax of the logits (softmax is monotone; it differs only where two probabilities round to the same float), and its probabilities come from the library's own exp, so confidences agree with a float32 NumPy softmax to about 1e-6, not bit for bit.

## Normalization

Each field carries the text as read and a normalized `Value` derived from it by the same code the grounding of LLM answers uses ([grounding.md](grounding.md)): dates (`BT-2 BT-9 BT-72 receipt.date`) to `yyyy-MM-dd` (`--MM-dd` without a year; a numeric date that reads two ways under the page's date order stays as read), amounts (the money classes) to a canonical decimal with `.` as the decimal mark, quantities and rates (`BT-129 receipt.quantity`, `BT-152 BT-119`, the sign `%` dropped) likewise, `BT-5` to an ISO 4217 code when the text names exactly one. The number style of the page is the one its unambiguous numbers use (point decimal when none is).

## Limits

`OcrLimits.MaxWordsPerPage` (words read from a page), `MaxWordsPerWindow`, `MaxCharsPerWord`, `MaxPositionBins`, `MaxTransformerLayers`, `MaxExtractionClasses` and the existing model-file, tensor and activation limits bound the model file and the work before anything sized by them is allocated. Memory per window is dominated by the pair bias, `(layers * heads + 2) * max_words² * 4` bytes (33 MB at 512 words, 5 layers, 6 heads), so a page of any length costs the same memory and time linear in the number of windows.
