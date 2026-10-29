# `dbnet-lite-v0` and `dbnet-lite-v1` container architectures

The training-side description of the `architecture` section of the model container for the page detector; the runtime's normative specs are `docs/spec/detector-v0.md` and `docs/spec/detector-v1.md`. The trainer (`train.detect`),
the exporter (`export.dbnet`) and the forward parity fixtures (`parity.dbnet_fixtures`) already produce files in this shape using
the container's existing envelope (version 1: magic, manifest, 64-byte-aligned fp16 tensors); only the architecture id, the layer
graph, the pre- and post-processing below are new. Reference implementation of the graph on plain tensors:
`export.dbnet.FoldedReference`; of the post-processing: `eval.detect`.

Model: `models/dbnet.py`. Text detection is the probability-map branch of Liao et al., "Real-time Scene Text Detection with
Differentiable Binarization" (2020); the threshold branch and the differentiable binarization exist only for training and are
**not in the file**. The layout head applies the same shrink-and-recover idea to region boxes. Everything is convolution, depthwise
convolution, ReLU, nearest 2x upsampling and addition, plus one linear layer on a global average.

## Task

One page image in, three outputs:

| output | shape (H, W the padded input) | meaning |
|---|---|---|
| `prob` | [1, H/2, W/2] | text-line core probability (logit; apply sigmoid) |
| `layout` | [8, H/8, W/8] | per class region-core probability (logit; sigmoid) |
| `orient` | [4] | logits for page rotation 0, 90, 180, 270 degrees counter-clockwise from upright |

Layout classes, index order = `config.layout_classes`: `text, title, list, table, figure, header_footer, form_field, checkbox`.
Detection is **line level** (one box per text line segment): the recognizer reads line crops, and the training truth (DocLayNet's
cells, merged along a baseline) is line-shaped. `form_field` and `checkbox` are labelled only in `dbnet-lite-v1` models, from exact-truth synthetic pages (no real-page labels
exist); `config.layout_trained` lists the classes a model's training labelled, and a reader must ignore the other channels.

## Pre-processing

1. Convert to gray (the runtime's own gray conversion; the model was trained on PIL `convert("L")`, so expect small differences).
2. Scale so the **long side is 1024 px** (round to nearest, at least 1), bilinear with area pre-reduction when shrinking
   (`Image.resize(..., BILINEAR, reducing_gap=2.0)` in training; the runtime's Auto filter is the intended substitute).
   Pages smaller than 1024 are scaled up the same way. Scale factor `s = 1024 / max(w, h)`.
3. Pad right and bottom with white (255) to a multiple of 32 in both directions: `W = ceil(round(w s) / 32) * 32`, same for H.
4. Pixel `p` -> `(p / 255 - 0.5) / 0.5`, float32.

An A4 page gives 1024 x 736 (about 1.86 GMACs). The model is fully convolutional and also runs on any other multiple-of-32 size.

## Manifest `config`

```json
{"arch": "dbnet-lite", "stem": 16, "widths": [32, 72, 144, 224], "depths": [2, 3, 4, 3], "fpn": 56,
 "text": 32, "layout": 112, "layout_blocks": 3, "classes": 8, "layout_classes": ["text", "title", "list", "table", "figure", "header_footer", "form_field", "checkbox"]}
```

`input`: `{"height": 0, "channels": 1, "mean": 0.5, "std": 0.5, "invert": false}` (height 0: no fixed height, a multiple of 32).
`charset`: `["<blank>", ...layout class names]` only to satisfy the envelope's charset rules; it is not an output alphabet.
`calibration`: empty `scripts` (no confidence calibration for detection).

## Layer graph (tensors fp16 in the file, fp32 compute)

Notation: `conv(k, s, g)` is a k x k convolution with stride s, groups g, zero padding k/2 (floor), bias, and batch norm already
folded in; `relu` follows unless noted. `DW(cin, cout, s)` is the pair `dw` = conv(3, s, g = cin), cin -> cin, then
`pw` = conv(1, 1, 1), cin -> cout, each followed by ReLU. Tensor names are `<path>.weight` and `<path>.bias`; weights are
PyTorch layout `[cout, cin/g, kh, kw]`.

| step | op | tensor path(s) |
|---|---|---|
| `stem` | conv(3, 2), 1 -> 16, ReLU. Output [16, H/2, W/2] | `stem` |
| `stageI.J` (I = 0..3, J < depths[I]) | `DW(cin, widths[I], stride 2 if J = 0 else 1)`; cin = previous output. Stage outputs are at strides 4, 8, 16, 32 with 32, 72, 144, 224 channels | `stageI.J.dw`, `stageI.J.pw` |
| `lat0..lat3` | conv(1, 1), widths[I] -> 56, **no ReLU**, applied to the stage I output | `latI` |
| top-down | `p16 = lat2 + up2(lat3)`; `p8 = DW(56, 56)(lat1 + up2(p16))` (`smooth8`); `p4 = DW(56, 56)(lat0 + up2(p8))` (`smooth4`). `up2` is nearest-neighbour 2x (each value repeated in a 2x2 block). Note `p16` is not smoothed | `smooth8.*`, `smooth4.*` |
| layout head | `h = p8`; three `DW(·, 112)` blocks (`layout.0` 56 -> 112, `layout.1`, `layout.2` 112 -> 112); `layout = conv(1)` 112 -> 8, no ReLU. Output [8, H/8, W/8] | `layout.J.dw/pw`, `layout_out` |
| text head | `t4 = DW(56, 32)(p4)` (`text4`, stride 4); `t2 = DW(32, 32)(up2(t4) + conv(1)(stem output, 16 -> 32, no ReLU))` (`text2`, stride 2); `prob = conv(1)` 32 -> 1, no ReLU. Output [1, H/2, W/2] | `text4.*`, `stem_proj`, `text2.*`, `prob_out` |
| `orient` | global average over H x W of the stage 3 output (224 values, sum in a fixed order then divide by the count), linear 224 -> 4 | `orient.weight` [4,224], `orient.bias` |

Parameters stored: 309,077 (0.62 MB fp16; container file about 0.63 MB). 94 tensors. No batch-norm tensors are stored.

Where `H/2` etc. are exact because the padded input is a multiple of 32; every stride-2 convolution sees an even-sized input, so
`out = in / 2` with padding 1 and no edge cases.

## `dbnet-lite-v1`: what differs from `v0`

Same layer types (convolution, depthwise 3x3, ReLU, nearest 2x upsample, add, channel mean, linear) and the same tensor naming.
No new layer type, no dilation, no other kernel size. Three changes; each is a config switch, and `v0` files stay valid
(`layout16_blocks` absent or 0 and `orient16` absent or false mean the `v0` graph).

1. **Stride-16 context branch for the layout head** (`layout16_blocks` = n > 0, width `layout16`):
   `g = DW(56 -> layout16)(p16)` then n - 1 further `DW(layout16, layout16)` blocks at stride 16 (tensors `layout16.J.dw/pw`);
   `m = conv(1) layout16 -> layout` (`layout_mix`, bias, **no ReLU**) applied to `g` **at stride 16**, then nearest-upsampled 2x
   (a 1x1 convolution commutes with nearest upsampling, so the multiply is done on the 4x smaller map);
   the stride-8 layout head's first block output is **added** to it: `h = DW(fpn -> layout)(p8) + up2(m)` (`layout.0`), then the remaining
   `layout.J` blocks as before. Here `fpn` is the config's FPN width; widths in `v0` text (56) are the `v0` config's values.
2. **Orientation from two scales** (`orient16` true): the 4 logits are one linear layer over the concatenation
   `[mean(stage2 output), mean(stage3 output)]` (channels of stage 2 first), `orient.weight` [4, widths[2] + widths[3]].
3. **Widths and depths** are config only (the table below gives the `v1` values).

`config` additions (`v1`): `layout16_blocks`, `layout16`, `orient16`, and the two brokered members every detector file carries from
this version on:

```json
"postprocess": {"text_threshold": 0.3, "text_expand": 0.5, "layout_threshold": 0.6, "layout_expand": 0.85},
"layout_trained": ["text", "title", "list", "table", "figure", "header_footer", "form_field", "checkbox"]
```

`postprocess` values are probabilities (thresholds on `sigmoid(logit)`) and growth factors for the rules in *Post-processing*;
a reader uses the file's values (a `v0` file without them uses the defaults written there: 0.3, 1.0, 0.6, 0.85). For 0.2.0 the text growth factor is 0.5, not the geometric inverse (1.0) of the training shrink: its synthetic-page training boxes are line-box height, about 1.7x the ink height, so the predicted boxes come out taller than the ink and a factor of 0.5 gives crops the recognizer reads best (picked on the end-to-end gate pages, two numbers over a coarse grid; see the weights' report).
`layout_trained` names the classes whose channels were trained; the others are untrained noise.

## Cost

MACs for the inference graph (convolutions and the linear layer; no BN, ReLU, add, upsample):

| input | GMACs |
|---|---|
| 1024 x 736 (A4 portrait) | 1.863 |
| 1024 x 1024 | 2.592 |

Largest layers: `text2.pw` (193 M at 1024 x 736), `smooth4.pw`, `layout.1.pw`, `layout.2.pw` (about 135 M each). Activation peak
is the stride-2 stage: 16 + 32 channels at 512 x 368 (about 36 MB fp32 at once if the stem output is kept for `stem_proj`).
The stem output must stay alive until `text2`; free stage outputs after their lateral.

## Post-processing

Both heads use the same rule: threshold the sigmoid map, label 4-connected components, take each component's bounding cell range as
the shrunk core, grow it back. Scores are the mean probability over the component's cells (informational; not used to filter).

**Text lines** (`prob`, stride 2):
1. `P = sigmoid(prob)`; mask = `P > 0.3`.
2. 4-connected components; drop components with fewer than 2 cells.
3. Core box in input pixels: `x0 = 2 * minCol`, `x1 = 2 * (maxCol + 1)`, same for y. Let `c = min(x1 - x0, y1 - y0)`.
4. Grow by `d = (0.3 / (1 - 2 * 0.3)) * c * expand`, on each side (the training shrink was 0.3 of the short side per side, so the core's
   short side is 0.4 m and the offset is 0.75 of it); `expand` = 1.0. (Validation on DocLayNet's PDF cells preferred 0.7, which shrinks boxes; the benchmark pages' ink-aligned boxes score best at 1.0 (clean F1 0.997 vs 0.979), so 1.0, the geometric inverse of the training shrink, stays the default; the recognizer's crop padding absorbs the difference.)
5. Divide by the pre-processing scale `s` to page pixels. Clip to the page.

**Layout regions** (`layout`, stride 8), per class:
1. `P = sigmoid(layout[c])`; mask = `P > 0.6`; 4-connected components (no minimum size).
2. Core box `x0 = 8 * minCol`, `x1 = 8 * (maxCol + 1)` etc.; `core = min(x1 - x0, y1 - y0)`.
3. Grow by `d = core / 3` when `core <= 60` else `20`, times `expand_layout` = 0.85. (Training shrank by `min(0.2 m, 20)`; 0.85 and the 0.6 threshold were picked on DocLayNet validation pages.)
4. Snap (only for classes other than `table` and `figure`): replace the box by the union of the text lines (step above, in the same
   input pixels, before dividing by `s`) whose centres fall inside it, limited to the box grown by 8 px; no line inside: keep the
   box. This restores ink-accurate edges that the 8 px stride cannot give.
5. Divide by `s`.
No non-maximum suppression: a component is a region, and adjacent regions of one class are separated by the gap in the map. A
region of another class may overlap (the maps are independent sigmoids); the C# reading-order code picks by role.

**Orientation:** softmax over `orient`; the arg-max is the page rotation. Trained on upright pages plus 25 % random 90/180/270
rotations of the training pages. A confidence below 0.9 should leave the page unrotated (the head is a cheap guard, not a classifier
to trust blindly on pages with little text).

The layout threshold and expansion were picked on DocLayNet validation pages; the text values are the training-geometry defaults.
Both text expansions were reported on the benchmark pages, which were not used to select anything.

## Per-line angle and rotated line boxes

Each text line's angle is read from its component in the text map (principal axis of the cell coordinates, from exact integer moments);
the runtime's rule, filters and quad construction are specified in the library's `docs/spec/detector-v1.md`, "Line axis". The training
side only supplies the oriented targets (`train.detect_data.text_targets`). `eval.quad_diag` measured the angle on the bench pages with
the 0.2.0 detector: median error 0.07 deg on phone (95 % of 931 lines within 0.5 deg) against 0.23 deg (73 %) for one page skew.

## Numeric notes for the C# kernels

- Depthwise 3x3 (stride 1 and 2) is new relative to the recognizer's kernels: per channel, 9 taps in (ky, kx) order, FMA, same
  accumulation convention as the dense convolution's tap order; no im2col needed. Pointwise 1x1 convolutions are plain GEMMs over
  the channels-last activation (K = cin; zero-pad K to a multiple of 16 as the existing convolution does).
- The ReLU after each convolution is the only non-linearity; sigmoid is applied only to the three output maps and only in the
  post-processing (use the runtime's own `exp`; thresholds can be applied on logits: `P > 0.3` is `logit > ln(0.3 / 0.7) = -0.8473`,
  `P > 0.6` is `logit > 0.4055`; compare logits to these in float32 to avoid the sigmoid for the mask).
- Nearest 2x upsample + add can be fused into the lateral addition (read the coarse map at `(y/2, x/2)`).
- Average pool for `orient`: sum over 7,168 + positions in row-major order in float32 then divide.
- fp32-vs-fp64 differences of the reference per layer are in the fixtures' `fp32_vs_fp64_floor` (tens of nanounits at most, about
  2e-7 absolute for activations of order 1).

## Parity fixture layers

`input` [1,H,W]; `stem`, `stage0` .. `stage3`, `p16`, `p8`, `p4`, (v1: `layout16`, `layout_mix`) `layout_feat`, `layout`, `text4`, `text2`, `prob` ([C,H',W']); `orient`
([4]). Five tiles plus one "wide" tile with the head outputs only; see `parity/dbnet_fixtures.py`. Each file under 1 MB.

## Container v1 additions proposed

- `architecture`: `dbnet-lite-v0`; `name`: `text-detector-layout`.
- `config`: as above; a reader rejects an unknown `arch`, a `widths`/`depths` length other than 4, or any width above 1024.
- Tensors: the names in the table (94). All fp16.
- `input.height = 0` means "any multiple of 32"; `input` otherwise as for the recognizers. The recognizer's charset rule is satisfied
  by the placeholder charset; a reader may ignore it for this architecture.
- Limits: input side up to 4096; activation buffer at the stride-2 stage is `(16 + 32) * H/2 * W/2` floats.
