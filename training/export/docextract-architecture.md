# docextract-v0: architecture of the document extraction model

Draft of the model-container architecture `docextract-v0`: a small layout-aware transformer over the words of a page. It reads words
(text and box) and labels them; deterministic code assembles the labels into JSON, so the model cannot produce a value that is not on
the page. Reference implementation: `models/docextract.py` (training graph), `export/docextract.py` (`Reference`, the exported graph on
plain tensors), `models/docextract_assemble.py` (assembly). Parity fixtures: `parity/docextract_fixtures.py`.

Papers: LayoutLM (Xu et al. 2020) and LayoutLMv2 (Xu et al. 2021) for words with 2D positions and a learned spatial term in attention;
Kim et al. 2016 for the character CNN word embedding; Xiong et al. 2020 for pre-layer-norm transformers; Hendrycks and Gimpel 2016
for GELU; Dozat and Manning 2017 for scoring head pointers with a bilinear form. Only the papers were used.

## What it predicts

For a page of N words the model runs in windows (next section); the table gives the shapes inside one window of N <= `max_words` = 512 words.

| output | shape | meaning |
|---|---|---|
| `tags` | [N, 95] | field tag per word: 0 = O, 1 + 2c = B-class c, 2 + 2c = I-class c; 47 classes (`config.classes_list`: 35 invoice classes by EN 16931 business term, 12 receipt classes) |
| `roles` | [N, 5] | O, B-KEY, I-KEY, B-VAL, I-VAL: generic key/value spans on any document |
| `doc` | [3] | document type: invoice, receipt, other |
| `rel` | [2, N, N] | pointer logits, row i over candidate words j. Relation 0 (group): a word of a line item or tax-breakdown row points to the first word of that item or row (the anchor), the anchor to itself. Relation 1 (kv): a VAL word points to the first word of its key, a KEY word to itself |

Class order (never reordered once shipped, append only): `BT-1 BT-2 BT-5 BT-9 BT-13 BT-20 BT-27 BT-31 BT-35 BT-37 BT-38 BT-40 BT-44
BT-48 BT-50 BT-52 BT-53 BT-55 BT-72 BT-84 BT-86 BT-106 BT-110 BT-112 BT-115` (header), `BT-126 BT-129 BT-130 BT-131 BT-146 BT-152 BT-153`
(line), `BT-116 BT-117 BT-119` (tax row), then `receipt.merchant date time subtotal tax tip total paymentMethod` (header) and
`receipt.itemName quantity unitPrice amount` (line). The same list is in the manifest as `config.classes_list`.

## Windows (pages longer than `max_words`)

The rule is the library's (`docs/spec/docextract-v0.md`); `models.docextract.window_plan` and `page_outputs` implement it, and evaluation uses it.

* A page of at most 512 words is one window. A longer page runs in windows of 512 words starting every 384 (`max_words * 3 / 4`), the last
  window ending at the last word: starts 0, 384, 768, ... while `start + 512 < N`. Featurization (`unit`, page size, bins) is computed once for the
  whole page, not per window.
* A word belongs to the window whose centre `(start + end) / 2` is nearest the word's centre `index + 0.5`; on a tie, the earlier window. It takes
  its tag logits, role logits and pointer row from that window.
* Pointers are scored inside a window only: the target is the window-local argmax (lowest index on a tie) shifted by the window start.
* The document-type logits are the mean of the windows' logits weighted by the number of words each window owns, in double precision, rounded to fp32.

## Input featurization (per page; float64 until stored as fp32)

Words are the page model's words in reading order (empty text dropped), box = `bounds` in page pixels (top-left origin), page size W x H.
`unit` = median word height (mean of the two middle values when N is even), at least 1.

* chars: NFC, each code point -> its index in the container `charset` (charset-v1, 0 = blank is never a text character; space does
  not occur in a word); a code point not in the charset -> 622 (`config.unk_char`); first 32 code points kept; right padded with 0.
* bins: `clamp(floor(128 * v), 0, 127)` for v = x0/W, y0/H, x1/W, y1/H.
* feats[8]: x0/W, y0/H, x1/W, y1/H, w/W, h/H, min(h/unit, 8) - 1, a/(1 + a) with a = w/h. (w = max(x1 - x0, 1e-3), h likewise.)
* geo[4]: box / unit.

No logarithm anywhere: every nonlinearity is `exp` (softmax), `tanh` (GELU) or `sqrt` (layer norm).

## Graph (fp32 compute, fp16 tensors in the file)

d = 192, 5 blocks, 6 heads (head size 32), ffn 512, char dim 48, char conv 128, pair hidden 32, rel dim 64. Notation: `lin(name, x)` = x Wᵀ + b
with `W` stored `[out, in]`; `ln(name, x)` = layer norm over the last axis, eps 1e-5, `weight` and `bias`.

| step | op | tensors |
|---|---|---|
| char embedding | gather rows of `char_emb.weight` [623, 48] (row 0 is zero) | `char_emb.weight` |
| char CNN | conv1d, kernel 3, zero padding 1, 48 -> 128, ReLU, then positions past the word's last character set to 0, then max over the 32 positions: `char_pool` [N, 128] | `char_conv.weight` [128, 48, 3], `char_conv.bias` |
| word | `char_proj` 128 -> 192 | `char_proj.*` |
| position | add rows `pos_x0[bins0]`, `pos_y0[bins1]`, `pos_x1[bins2]`, `pos_y1[bins3]` (each [128, 192]) and `feat_proj(feats)` (8 -> 192); then `emb_ln`. Output `emb` [N, 192] | `pos_x0/y0/x1/y1.weight`, `feat_proj.*`, `emb_ln.*` |
| pair bias | for every ordered pair (i, j): 13 features of j relative to i (below) -> `pair1` 13 -> 32, ReLU -> `pair2` 32 -> 32. Output `pair_bias` [32, N, N]: channels 6l..6l+5 are the head biases of block l, channels 30 and 31 are the biases of the two pointer relations | `pair1.*`, `pair2.*` |
| block l (pre-LN) | `u = ln1(x)`; `qkv = lin(u)` 192 -> 576, split into q, k, v (each [N, 6, 32], in that order along the 576 axis, head-major within each); per head `s = q kᵀ / sqrt(32) + pair_bias[6l + h]`; `a = softmax_j(s) v`; `x += proj(concat heads)`; `x += fc2(gelu(fc1(ln2(x))))`, fc1 192 -> 512, fc2 512 -> 192 | `blocks.l.ln1/qkv/proj/ln2/fc1/fc2.*` |
| final | `final_ln`; output `final` [N, 192] | `final_ln.*` |
| heads | `tags = tag_head(final)` [N, 95]; `roles = role_head(final)` [N, 5]; `doc = doc_head(mean over words of final)` [3] | `tag_head.*`, `role_head.*`, `doc_head.*` |
| pointers | `q = rel_q(final)`, `k = rel_k(final)`, each 192 -> 128 viewed [N, 2, 64]; `rel[r, i, j] = q[i, r] · k[j, r] / 8 + pair_bias[30 + r, i, j]` | `rel_q.*`, `rel_k.*` |

GELU is the tanh form, `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`, so the kernels need no `erf`. Pair features for words i, j with
`d(a) = a_j - a_i` on `geo` (cx, cy = box centres; h = y1 - y0), and `squash(x, s) = x / (s + |x|)`:
`squash(d(cx), 1), squash(d(cx), 16), squash(d(cy), 1), squash(d(cy), 16), squash(d(x0), 1), (…, 16), x1, y0, y1 likewise, squash(d(h), 1)`
(13 values, in this order).

Parameters 1.976 M, 3.95 MB fp16 (fixtures and the export report give the exact count). Cost per page of N words, MACs:
`N * 590k` (char CNN) + `N * 1.72M` (blocks) + `N^2 * 2.9k` (attention scores and mix, five blocks) + `N^2 * 1.4k` (pair MLP) + `N^2 * 0.13k` (pointers):
about 0.35 G at N = 150 and 1.0 G at N = 300.

## Layer types the C# side needs (forward / backward)

Embedding gather (scatter-add), conv1d k = 3 with ReLU, max over positions (ties: first position), linear, layer norm, tanh-GELU,
softmax attention with an additive per-head bias, the pair-feature squash, mean over words, and the dot-product pointer scores.
New against the existing kernels: gather/scatter, layer norm, softmax rows, GELU, conv1d (the 1-D case of the existing conv), and the
pair MLP (two linears over N^2 rows; backward flows into `pair1`/`pair2` only, the features are inputs). Nothing else: no LSTM, no
batch norm, no dropout, no `erf`, `log` or `pow`.

## Assembly (deterministic; `models/docextract_assemble.py`)

1. Document type = argmax of `doc`. Tags of classes outside that type's schema are masked before the argmax (invoice: BT-*; receipt:
   receipt.*; other: none).
2. Span = a B tag plus the following I tags of the same class on consecutive words (an I with no matching predecessor starts a span).
   Confidence = mean softmax probability of the chosen tags over the span's words.
3. Header classes keep their highest-confidence span. Line and tax classes: the span's first word is mapped through relation 0 (argmax over
   j, followed to a fixed point in at most 4 steps; a cycle ends at its smallest index) to an anchor; spans with the same anchor form one line
   item (or tax row), one span per class, ordered by anchor.
4. Key/value: spans of KEY and VAL roles; a VAL span's first word is mapped through relation 1; if the target lies in a KEY span the pair is
   emitted.
5. Output text of a span = its words joined by one space; JSON carries word indices, union box and confidence. Value normalization (dates, amounts,
   currency) is a separate step of the schema layer, not part of this model.

## Fine-tuning a custom schema

The backbone is everything up to `final`. A user schema replaces `tag_head` (95 -> 1 + 2 * K outputs) and, if it has repeated groups or key/value
links, keeps `rel_q`/`rel_k`. Backbone layers are frozen or trained at a lower rate; with all layer types above, backward passes are plain.

## Container section (proposal; v1 layout, the library repo's spec is not edited here)

`architecture` = `docextract-v0`, `name` = `document-extraction`. `input` = `{"kind": "words", "max_words": 512, "max_chars": 32, "bins": 128, "features": 8}`
(no image size). `charset` = charset-v1 (the alphabet of the character embedding; entry 0 `<blank>` is padding). `config` = the numbers above plus
`classes_list`, `roles_list`, `doc_types_list`, `unk_char`. `calibration` = `{"raw": "word-mean-max-run", "scripts": {}}` (empty; confidences are
raw softmax means, see Assembly). Tensor names and shapes: the table above and the fixtures' `model.onm`.
