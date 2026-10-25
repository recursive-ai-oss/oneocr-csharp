# Page detector `dbnet-lite-v1`

`dbnet-lite-v1` is [`dbnet-lite-v0`](detector-v0.md) with three optional changes. Everything else is as in v0: inputs, outputs, pre- and post-processing, layer types (convolution, 3x3 depthwise convolution, ReLU, nearest 2x upsampling, addition, channel mean, linear), tensor naming, the evaluation-order freedom and the numeric rules in [numerics.md](numerics.md). There is no new layer type, kernel size or dilation.

A v1 file with `layout16_blocks` absent or 0 and `orient16` absent or false is exactly the v0 graph. The v1 members are an error in a file whose architecture is `dbnet-lite-v0`.

## 1. Stride-16 context branch for the layout head

With `layout16_blocks` = *n* > 0 and width `layout16` = *w16*:

```
p16 = lat2 + up2(lat3)                       (as in v0; p16 is not smoothed)
g   = DW(fpn -> w16)(p16)                    layout16.0
g   = DW(w16 -> w16)(g)                      layout16.1 .. layout16.(n-1)
m   = conv1x1(g), w16 -> layout, bias, no ReLU      layout_mix
h   = DW(fpn -> layout)(p8) + up2(m)         layout.0 output plus the upsampled mix
h   = layout.1 ... layout.(layout_blocks-1)  as in v0
layout = layout_out(h)                       as in v0
```

`layout` is the config's layout width, `fpn` its FPN width. The 1x1 convolution commutes with nearest upsampling, so the multiply runs on the map that is four times smaller. All branch maps are at stride 16: height `H/16`, width `W/16`. Because the padded input is a multiple of 32, `H/8` and `W/8` are always exactly twice `H/16` and `W/16`, so the upsample never meets a partial pixel; the stride-32 map may be odd (`W = 160` gives 5 columns), which the tests cover. The addition is one fp32 addition per element and commutes, so its order does not matter.

Tensors: `layout16.<j>.dw.weight/.bias` (`[c,1,3,3]`, `[c]`) and `layout16.<j>.pw.weight/.bias` for `j < n` (the first block has `fpn` inputs, the others `layout16`), and `layout_mix.weight` `[layout, layout16, 1, 1]`, `layout_mix.bias` `[layout]`.

## 2. Orientation from two scales

With `orient16` true the four logits are one linear layer over the concatenation `[mean(stage2 output), mean(stage3 output)]`, stage 2's channels first; `orient.weight` is `[4, widths[2] + widths[3]]`. Each mean is the row-major fp32 sum over the map divided by the position count ([numerics.md](numerics.md)); each logit is one dot product over the `widths[2] + widths[3]` joined values in the 16-lane order (the lanes continue across the join: the concatenation is one vector), plus the bias.

## 3. Widths and depths

They come from `config` only. The 0.2.0 configuration is `widths [32, 72, 176, 288]`, `depths [2, 3, 6, 4]`, `fpn 48`, `text 32`, `layout 96`, `layout16 160`, `layout16_blocks 4`, `orient16 true`.

## Line axis (post-processing, any detector version)

A text line's component of the text map also gives the line's own axis, used to read lines that lean (a photographed page keeps some
perspective lean after the page-wide deskew). The text map and the component are those of [detector-v0](detector-v0.md#post-processing);
nothing about the boxes changes.

For a component of `n >= 12` cells (cell `(x, y)` has its centre at `(x + 0.5, y + 0.5)` in cell units, one cell = 2 network pixels), with
mean `(mx, my)` over the cell indices and `a = Var x`, `b = Cov(x, y)`, `c = Var y` (divisors `n`, sums in the order the flood fill visits the cells):

- `h = (a - c) / 2`, `d = sqrt(h*h + b*b)`; no axis when `d <= 1e-9`.
- `cos 2t = h / d`; `cos t = sqrt(max(0, (1 + cos 2t) / 2))`, `sin t = sqrt(max(0, (1 - cos 2t) / 2))`, negated when `b < 0`. The axis points right (`cos t >= 0`); `sin t > 0` means the line descends to the right.
- Eigenvalues `major = (a + c)/2 + d`, `minor = max(0, (a + c)/2 - d)`; `elongation = major / (minor + 1/12)` (one cell's own variance).
- The line frame has its origin at the centroid `(mx + 0.5, my + 0.5)`, `u` along the axis and `v` across it, pointing down: `u = dx*cos t + dy*sin t`, `v = dy*cos t - dx*sin t` for a cell centre offset `(dx, dy)`. Its extent is `[min u - 0.5, max u + 0.5] x [min v - 0.5, max v + 0.5]`, then grown on every side by `0.75 * text_expand * min(length, thickness)` like the axis-aligned box.
- Everything is scaled to page pixels by `2 / scale` (the text map stride over the page scale). The page point of `(u, v)` is `(cx + u*cos t - v*sin t, cy + u*sin t + v*cos t)`.

Only `+ - * /` and square roots are used, so the axis is bit-identical on every platform and degree of parallelism.

A line is **read along its axis** when its axis passes all of these; otherwise it takes the axis-aligned path described in the engine's
line-crop rules, unchanged:

| test | value |
|---|---|
| cells | at least 20 |
| elongation | at least 100 (the component is about ten times longer than thick) |
| `|sin t|` | at least `sin 0.3 degrees`, and at least `0.15 * thickness / length` (the angle of a short, thick component is not believed), and at most `sin 15 degrees` |

The values were chosen on held-out camera-simulated pages and on straight pages (so that straight text stays on the axis-aligned path).

Reading a line along its axis:

1. **Ink band.** The detected rectangle is sampled into an upright strip of one-pixel cells (below), and the usual ink-band rule (the main band and any band holding at least 40% as much ink) finds the line's ink rectangle in the line frame; each band is one line.
2. **Crop.** The crop rectangle has the margins of the axis-aligned rule (a quarter of the line height above and below, half of it left and right, the height raised to 0.85 of the page's median) but is taken in the line frame and is not clamped to the page. It is sampled (below) and scaled to the model height exactly as an axis-aligned crop.
3. **Words.** A word's extent along the line comes from the alignment; its rectangle is the ink inside that extent and the line's rows, found on the line-frame pixel grid with the page pixel under each cell centre (nearest pixel). Its `quad` is that rectangle's corners (top-left, top-right, bottom-right, bottom-left of the text's own frame) mapped to the page, and its box is the box around the quad. Characters use the same rows and their alignment edges. A line's `quad` is the smallest rectangle in the first word's direction that holds all its words' corners.
4. **Reading order** sees the straightened ink rectangle, unrotated, placed at the centre of the tilted one.

**Sampling.** The value at a page point `(px, py)` (pixel `(x, y)` covers `[x, x+1)`) is bilinear between the four pixel centres around it: `fx = px - 0.5`, `x0 = floor(fx)`, `wx = fx - x0`, likewise `y`; neighbours outside the page repeat the edge pixel; `top = p00*(1-wx) + p10*wx`, `bottom = p01*(1-wx) + p11*wx`, `value = top*(1-wy) + bottom*wy` in double precision, rounded half up to a byte. No fused multiply-add is used, so the bytes are the same on every platform.

## Evaluation order and memory

As for v0, every output value is an independent sum, so a port may evaluate whole or in bands. The C# reader keeps the v0 banded stages; the stride-16 branch runs on whole stride-16 maps (a quarter of the stride-8 map's size), `p16` is kept until the branch has consumed it, and stage 2's channel means are taken before stage 2 is dropped. On an A4 page at the 0.2.0 widths the scratch in use at once peaks at 24.3 MB at degree 1 (22.8 MB for v0 at its own widths).

## Reader checks

Before a tensor is read or a buffer allocated: `layout16_blocks` is in `[0, MaxConvBlocks]` and counts toward the total block limit with the backbone and the layout head; `layout16` is positive and within `MaxLayerWidth` when the branch exists; the stride-16 buffers are checked against `MaxActivationElements` with the other activation sizes; every tensor shape is checked against `config`. `orient16` must be a boolean.

## Tolerances

As in [numerics.md](numerics.md), with these additional layers against the float64 evaluation: `p16` Atol 4e-6, `layout16` 5e-6, `layout_mix` 5e-6 (Rtol 1e-5). `layout_feat`, `layout` and `orient` keep their v0 values. Worst error measured on random-weight models at the 0.2.0 widths and a narrow model, on every path, as a fraction of the tolerance: `p16` 0.24, `layout16` 0.38, `layout_mix` 0.29, `layout_feat` 0.55, `layout` 0.39, `orient` 0.14. Against PyTorch fp32 fixtures at the 0.2.0 widths (and a second model with the branch only) every layer differs by about PyTorch's own fp32 error.

## Page-conditional text threshold (post-processing, any detector version)

Camera pages carry sensor noise that the detector sometimes turns into faint speckle; on them a higher text threshold reads better, while
on scans, screenshots and clean pages the manifest's `text_threshold` is right. A model may carry a rule that picks the text threshold
per page. It needs all three members of `config.postprocess`; with none of them the rule is off, with some of them the file is rejected.

| member | meaning |
|---|---|
| `rule_sigma_min` | the page's noise estimate must be above this |
| `rule_junk_max` | the page's speck density must be below this |
| `rule_text_threshold` | text probability above which a cell is on, on pages the rule selects (0.5 is logit 0) |

**Noise estimate `sigma`.** Take the gray page at detector scale: the page resized so that its long side is 1024 pixels (the resize of
the network input, before padding), unpadded area only. Apply the 3x3 kernel `[[1,-2,1],[-2,4,-2],[1,-2,1]]` to every interior pixel
(`(h-2) x (w-2)` responses; a page under 3 pixels in either direction has `sigma = 0`). Build the exact integer histogram of the absolute
responses (0 to 2040), take the lower median (the value at rank `(n+1) div 2`, counting from 1, n the number of responses) and divide by
4.047. This is Immerkaer's noise estimate (Immerkaer 1996, "Fast noise variance estimation") with the median of a half-normal
distribution, 0.6745 x 6, in place of the mean. The median is an integer, so `sigma` is exact.

**Speck density `junk`.** In the text map (probabilities, one per cell, padding included), count the 4-connected components of
cells with probability above 0.3 that have 2 to 12 cells and a mean cell probability below 0.5. This count always uses 0.3, whatever
`text_threshold` is. Divide by the map's cell count times 4, over 10^6 (the map has stride 2, so a cell stands for 4 input pixels).

**Rule.** If `sigma > rule_sigma_min` and `junk < rule_junk_max`, label the text map with `rule_text_threshold`; otherwise with
`text_threshold`. Nothing else changes: boxes, axes, layout regions and orientation are computed as before from the chosen labelling.
`sigma` is computed only for models that carry the rule. The integer histogram makes it identical on every platform; the speck density
depends on the text map, so it is as reproducible as the map.

Constants of the first model with the rule, chosen on held-out pages (camera-simulated pages gain; clean, layout and screenshot pages
have `sigma` under the minimum): `rule_sigma_min` 0.1235, `rule_junk_max` 0.6358, `rule_text_threshold` 0.5. The script
`training/export/add_postprocess_rule.py` adds the three members to an existing container without touching the tensors.
