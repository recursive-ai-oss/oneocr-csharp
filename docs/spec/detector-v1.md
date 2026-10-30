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

## Ink on camera photos (page reader, any detector version)

The ink inside a detected line (the ink band, the line's ends, the word rectangles) is found with the page's Otsu threshold: ink is the
smaller class. On a photo whose light falls off across the page one threshold cannot fit every line, and a line the threshold does not
fit loses its ends or its words. The page reader therefore binarizes lines of **camera photos** by itself. A page is a camera photo when
the detector's noise estimate `sigma` (see the page-conditional threshold below) is above `0.1`; screenshots and clean renders have flat colours,
and pages without a noise estimate (a model without the rule) are never treated as photos. On every other page nothing here applies and the
output is unchanged.

On a photo, for each detected line (axis-aligned rectangle, or the straightened strip of a line read along its axis):

1. **Line threshold.** Let `H` be the histogram of the line's gray values and `(t, bright)` Otsu's threshold over `H` with ink as the smaller class.
   The line uses `(t, dark ink)` instead of the page's when
   - the page reads ink as bright, at least half of the page's detected lines read ink as dark by their own Otsu threshold, and this line does too
     (the page threshold was fixed by the dark majority of a shadowed photo; a dark-mode screen has bright ink in most lines and keeps the page's reading); or
   - the page reads ink as dark, more than half of the line's pixels are at or below the page's threshold (the threshold lies inside the line's
     paper), and the line's own reading is dark ink with `t` different from the page's.

   The line's own reading is used only when it is credible: the pixels at or below `t` are at least 1.5% of the line, and the mean gray of the
   rest exceeds the mean of those by at least 40. Otherwise the page's threshold stands. The threshold then serves the ink band, the line's ends and
   the word rectangles of that line.
2. **Line ends.** After the ink band is found, an end of the line's ink that is more than 0.8 of the band's height short of the detected
   rectangle's end (left or right, along the line) is moved to the detector's end. Print at the end of a line is where blur and uneven light leave
   it faintest, and a crop that stops there loses the words.
3. **Word rectangles.** On every page: when the ink found inside a word's aligned extent is narrower than a quarter of the extent (and the extent is
   at least half the line's height), the word's rectangle is the aligned extent over the line's rows, as when no ink is found. A rectangle of a pixel or
   two otherwise tears the word out of its line in reading order.

## Line ends on flat pages (page reader, any detector version)

The detector's box is a few pixels uncertain at the ends of a line, and a box that stops inside the last letter (or just before it) leaves
the ink band, and with it the crop the recognizer reads, a letter short: `Nairobi` read as `Nairob`. Measured on exact-truth pages, the box
ends about 1 px inside the ink on typical lines and a letter or more inside on a few percent of them. After the ink band of a line is
found, each end is therefore followed outwards on **pages that are not camera photos** (photos keep the rule above). Let `B` be the band's height in pixels.

1. **Follow.** From the band's last ink column, scan columns outwards over the band's rows. A column has ink when any of its pixels is on the
   ink side of the threshold the band was found with. Scanning stops at the first run of more than `max(1, round(0.25 * B))` columns without
   ink (a letter gap is below this, a word gap above it), at `1.0 * B` columns from the old end, at the page edge, or at the box of the
   nearest other detected line on the same row (two lines overlapping vertically by at least half the smaller height; the line on the side of
   this one's centre that the end is on). The candidate end is the last column with ink.
2. **Letters only.** The candidate is taken only when the stretch gained is at least `ceil(0.25 * B)` columns and its ink pixels are at most 0.65
   of its area (stretch columns times `B`). A shorter stretch is the letters' anti-aliased edge, and moving the crop for it only shifts what the
   recognizer sees by a pixel. A stretch that is mostly ink is a button, a rule or a border, not letters. Otherwise the end stays.
3. The two ends are decided independently. Nothing else about the strip changes; the crop rule (margins of half and a quarter of the line's
   height, scale) applies to the new extent.

Without step 2, about one strip in ten on screenshot pages grew by a whole line height onto the fill of buttons and bars. The constants
(0.25 gap, 1.0 reach, 0.25 minimum, 0.65 fill) come from a sweep of seven variants over generated layout and screenshot pages.

## Joined lines on flat pages (page reader, any detector version)

Lines set solid (leading barely above the type size, as in statute and report text) touch row by row: a descender reaches the next line's
ascenders, so no pixel row between two lines is empty. The detector can then box two or three lines as one, and the ink-band rule above
(a gap of more than an eighth of the box height ends a band) cannot cut it. A recognizer line crop that holds two lines reads as noise. On
**pages that are not camera photos**, with at least 8 detected lines:

1. **Typical box.** `T` is the median height of the page's detected boxes (faint-text recoveries left out).
2. **Tall band.** A band of ink rows (found as above, rows counted over the box's width) at least `ceil(1.5 * T)` rows tall is cut.
3. **Valley.** Let `count[r]` be the ink pixels in row `r` over the box's width, `above(r)` and `below(r)` the densest rows on either side
   within the band. A row is a valley when `count[r] <= 0.2 * min(above(r), below(r))` (so it lies between two dense bands, not in one
   line's ascender or descender rows) and at least `max(2, floor(R / 2))` rows remain on each side, `R` being the reference height
   (`T` at first). The valley with the smallest ratio is the cut (ties: nearest the band's middle); the cut row belongs to neither part and
   blank rows at the cut ends are trimmed.
4. **Again.** Each part is cut again by steps 2 and 3 while it is at least `ceil(1.5 * R')` tall, with `R'` the smaller of `R` and its
   sibling's height. A box of three lines of small type is cut twice; two lines of normal type once.
5. **No duplicates.** A box is often drawn over a line that has a box of its own (the lines of a shaded note beside running text). A part
   is dropped when another detected line that is shorter than this box and shares at least half of the narrower width covers at least half
   of the part's rows. When every part would be dropped, all are kept.
6. The parts go through the ink-band mass rule with a lower bar than ordinary bands: a part is kept with a tenth of the heaviest part's
   ink (ordinary bands need 40%), because a short last line has little ink beside a full one. Each part's horizontal extent is taken from
   its own rows.

7. A box this tall is cut upright first (the principal axis of a component that joins lines leans with them, and straightening along it
   smears the rows between the lines). If it was cut, the parts are used; otherwise the box goes on as any other, along its axis if it
   leans (see "Line axis"), where the same cut is tried in the straightened strip with `T` unchanged and without step 5.

Without step 5 a line already read through its own box was read twice. The constants (1.5, 0.2, the cover share 0.5) were chosen on held-out
real-PDF pages (IRS layout, scans) and on generated solid-set pages; on those, a tall-band factor anywhere from 1.2 to 1.7 gives identical
output, and a valley share of 0.35 made generated layout pages worse by splitting single lines.

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

`dbnet-lite-v1` 0.2.2 is 0.2.1 with these three members added and the manifest version changed: the tensor bytes are identical to 0.2.1's,
so the forward-pass parity fixtures of 0.2.1 apply unchanged. Only the page-level behaviour differs.

`dbnet-lite-v1` 0.3.0 has the 0.2.x architecture and tensor layout with new weights (a fine-tune of 0.2.1 on windows of real web pages and
light-on-dark synthetic pages, blended 1:3 with its starting weights). Its page rule is `rule_sigma_min` 0.1235, `rule_junk_max`
1,000,000 (the junk-density condition never holds back the rule) and `rule_text_threshold` 0.5. It has its own forward-pass fixtures.

## Faint text on dark backgrounds (post-processing, any detector version)

Light text a few percent brighter than a dark background (small menu and footer text on dark bands) reaches the text map with
probabilities of only a few percent, below `text_threshold`, so the ordinary labelling misses it. A model may carry a second,
much lower threshold that is used only where the text sits on a dark background. All five members of `config.postprocess` are given
together, or none (some of them: the file is rejected). Without them nothing below applies and the output is as before.

| member | meaning |
|---|---|
| `faint_text_threshold` | text probability above which a cell is on in the second labelling (0.02 in the first model) |
| `faint_bg_max` | largest median gray value (0 to 255) of a candidate's cell box that counts as a dark background (110) |
| `faint_min_cells` | fewest cells of a candidate (6) |
| `faint_min_aspect` | smallest width over height of a candidate's cell box (1.5) |
| `faint_min_confidence` | smallest calibrated word confidence kept from a recovered line (0.9) |

**Candidates.** Skipped on pages where the page-conditional rule above chose its threshold (camera pages). Otherwise: the cells of every
ordinary component and their 8-neighbours are removed from the map (so a line's own fringe is never a candidate), the rest is labelled
with `faint_text_threshold` (4-connected, as the ordinary labelling) and a component is a candidate when it has at least
`faint_min_cells` cells (and at least the ordinary minimum), its cell box is at least `faint_min_aspect` times as wide as tall, it does not
overlap an ordinary component's cell box by more than a third of the smaller box, and the median gray value of the pixels under its cell box
(detector-scale page, one cell is two pixels) is at most `faint_bg_max`.

**Extension.** The map holds only the most visible stretch of a faint line, so the candidate is followed along its rows on the gray page: a column of
the candidate's rows has ink when one of its pixels is at least 5 above the median and the column's mean is not above `faint_bg_max`; the box grows
left and right over ink columns, across gaps of up to its own height (at least 6 pixels), and stops before an ordinary line that shares at least half its
rows. The box is then grown like an ordinary line (`k * text_expand` of its shorter side), and dropped when it covers more than a quarter of the smaller of
itself and an ordinary line's grown box.

**Order.** Recovered lines come after the ordinary lines; they take no part in snapping layout regions to lines.

**Reading.** A recovered line is cropped and recognized like any other, but (a) it takes no part in the page's typical line height, so the other
lines are cropped exactly as without it, (b) words under `faint_min_confidence` and single-character words are dropped from it, (c) a line that has no
word left, or whose ink box covers more than a quarter of the smaller of itself and an ordinary line's ink box, is dropped. The page therefore
changes only where a recovered line survives.

The script `training/export/add_faint_rule.py` writes the five members into a container's manifest and leaves the tensors untouched.
Constants were chosen on held-out tune pages only (screenshots of real pages, synthetic screens, scans, layouts, phone pages and receipts): on every
tune set except the real-screenshot one the output is byte-identical to the container without the members.
