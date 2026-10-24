# Page detector `dbnet-lite-v0`

(`dbnet-lite-v1` adds an optional stride-16 layout branch and a two-scale orientation head: [detector-v1.md](detector-v1.md).)

One page image in; text lines, layout regions and the page's orientation out. The container section is in [model-container-v1.md](model-container-v1.md), the operation order in [numerics.md](numerics.md). The text branch is the probability-map branch of Liao et al., "Real-time Scene Text Detection with Differentiable Binarization" (AAAI 2020): the threshold map and the differentiable binarization exist only for training and are not in the file. The layout head applies the same shrink-and-recover idea to region boxes.

## Outputs

For a padded input of height *H* and width *W* (multiples of 32):

| output | shape | meaning |
|---|---|---|
| `prob` | [1, H/2, W/2] | text-line core logit |
| `layout` | [classes, H/8, W/8] | per-class region-core logits |
| `orient` | [4] | logits for the page content being rotated 0, 90, 180, 270 degrees counter-clockwise from upright |

Layout classes, in channel order, are the manifest's `layout_classes` (`text, title, list, table, figure, header_footer, form_field, checkbox` in the first model). Detection is line level: one box per text line segment, which is what the recognizer reads.

## Pre-processing

1. Convert to 8-bit gray.
2. Scale so the long side is 1024 pixels: `s = 1024 / max(w, h)`, each side `max(1, round_half_even(side * s))`, resampled with the imaging library's automatic filter in gamma-encoded values (no linear-light conversion; the model was trained on gamma-space bilinear resampling). Pages smaller than 1024 are enlarged the same way; a page already 1024 on the long side is not resampled.
3. Pad right and bottom with white (255) to a multiple of 32 in both directions.
4. Each pixel `p` becomes `(p / 255 - mean) / std` in fp32 with the manifest's `input.mean` and `input.std` (0.5 and 0.5, so `p / 127.5 - 1`); `invert` flips `p / 255` first.

An A4 page gives 1024 x 736 and about 1.86 GMACs. The network is fully convolutional and runs on any multiple-of-32 size up to the reader's `MaxDetectorSide` limit (default 4096).

## Layer graph

`conv(k, s, g)` is a *k* x *k* convolution with stride *s*, *g* groups, zero padding `k / 2`, bias, batch norm folded in. ReLU follows unless noted. `DW(cin, cout, s)` is `dw` = `conv(3, s, g = cin)` then `pw` = `conv(1, 1, 1)`, `cin -> cout`, each with ReLU. Tensor names are `<path>.weight`, `<path>.bias`; weights are `[cout, cin / g, kh, kw]`.

| step | operation | tensors |
|---|---|---|
| `stem` | `conv(3, 2)`, 1 to `stem` channels | `stem` |
| `stage<i>.<j>` (i = 0..3, j < depths[i]) | `DW(cin, widths[i], 2 if j = 0 else 1)`; stage outputs are at strides 4, 8, 16, 32 | `stage<i>.<j>.dw`, `.pw` |
| `lat<i>` | `conv(1, 1)`, `widths[i]` to `fpn`, no ReLU, on the stage-*i* output | `lat<i>` |
| top-down | `p16 = lat2 + up2(lat3)`; `p8 = DW(fpn, fpn)(lat1 + up2(p16))`; `p4 = DW(fpn, fpn)(lat0 + up2(p8))`. `up2` repeats each value in a 2 x 2 block; `p16` is not smoothed | `smooth8`, `smooth4` |
| layout head | `layout_blocks` blocks `DW(·, layout)` on `p8`, then `conv(1, 1)` to `classes`, no ReLU | `layout.<j>`, `layout_out` |
| text head | `t4 = DW(fpn, text)(p4)`; `t2 = DW(text, text)(up2(t4) + conv(1, 1)(stem output; stem to text, no ReLU))`; `prob = conv(1, 1)` `text` to 1, no ReLU | `text4`, `stem_proj`, `text2`, `prob_out` |
| `orient` | global average of the stage-3 output, then linear `widths[3]` to 4 | `orient.weight`, `orient.bias` |

The reference for this table is a float64 evaluation on plain loops (`tests/OneOcr.Tests/TestDetectorModels.cs`) and PyTorch fixtures from `training/parity/dbnet_fixtures.py`.

## Post-processing

Both heads use one rule: a cell is on when its logit exceeds a threshold (compared on logits, in fp32: `ln(0.3 / 0.7) = -0.847297860387203` for text, `ln(0.6 / 0.4) = 0.405465108108164` for layout; a threshold `p` the manifest gives is converted as `float32(ln(p / (1 - p)))`); 4-connected components of on cells are labelled in the order of their first cell in row-major order; each component's bounding cell range is the shrunk core, which is grown back. A component's score is the mean sigmoid of its cells' logits (informational).

**Text lines** (`prob`, stride 2): components of fewer than 2 cells are dropped. The core in network pixels is `x0 = 2 * minCol`, `x1 = 2 * (maxCol + 1)`, likewise `y`. With `c = min(x1 - x0, y1 - y0)` the box grows by `d = 0.3 / (1 - 2 * 0.3) * c * expand = 0.75 c * expand` on each side (the training shrank each line by 0.3 of its short side per side).

**Layout regions** (`layout`, stride 8), for every class the manifest calls trained and no other: components of any size. Core `x0 = 8 * minCol` etc., `core = min(x1 - x0, y1 - y0)`; grown by `core / 3` when `core <= 60`, else by 20, times the layout expansion (default 0.85). Then, for every class except `table` and `figure`: the box becomes the union of the text lines (network pixels, before scaling back) whose centres lie inside it, each side limited to 8 pixels beyond the region's own box; with no line inside, or an empty union, the box stays. There is no non-maximum suppression: regions of different classes may overlap.

Boxes are divided by `s` to page pixels and clipped to the page; a box that is empty after clipping is dropped. (The training-side reference does not clip.) More than `MaxLinesPerPage` lines or `MaxPageElements` regions is a limit error.

**Orientation:** softmax over `orient` with the library's exp; the winner (lowest index on a tie) is the number of counter-clockwise quarter turns the content is rotated by. A page whose winner probability is below 0.9 is read as it is. Otherwise it is turned clockwise by that many quarter turns and detected again on the upright copy; the page model then describes the upright page, and a diagnostic says so. In the engine a turn the head asks for is confirmed by the classical orientation decision (sample lines read both ways up; see page-orientation-v1.md) and that decision is the one acted on: the head has only seen flat pages, and on tilted phone photos it called upright pages upside down with probability 0.99.

**Values.** Defaults (used when the manifest has no `config.postprocess`): text threshold 0.3, text expansion 1.0, layout threshold 0.6, layout expansion 0.85. A model overrides any of them with `config.postprocess` (`text_threshold`, `text_expand`, `layout_threshold`, `layout_expand`; see the container spec).

## Use in the page pipeline

When an engine has a detector, `Ocr` uses it instead of the built-in classical line finding:

- Each detected line box is narrowed to the ink inside it (the page's Otsu threshold, as the classical finder uses), so the recognizer's crop rules, written for ink-tight lines, apply unchanged; with no ink inside, the detected box is used.
- Lines inside a `text`, `title`, `list` or `header_footer` region (the smallest one holding the line's centre) form one reading-order leaf; other lines are leaves of their own, so the cells of a table row on one baseline join into a row.
- Regions fill `OcrPage.Regions` with the role of their class (`header_footer` is a header in the upper half of the page and a footer in the lower half), the score as confidence and the blocks whose centres lie inside. Blocks inside a `header_footer` region get the block role Header or Footer.
- Without a detector, page output is exactly what it was before.

## Evaluation order and memory

Every output value of the network is an independent sum, so a port may evaluate the layers whole or in bands of rows with any band height and any number of threads and get the same bits. The C# implementation keeps its working set small by running the stem inside the first stage, the stride-4 merge with the text block, and the stride-2 text head in row bands, each band recomputing the few input rows its neighbours share (the stem rows and the stage-0 lateral rows); no stride-2 map and no stride-4 lateral exists whole. With the shipped size, an A4 page needs about 23 MB of scratch at degree 1 (the three stride-2 maps and the stride-4 lateral and smoothed maps of a whole-map evaluation take about 100 MB). Tests hold the banded and whole-map results bit-equal for band heights from one row up.
