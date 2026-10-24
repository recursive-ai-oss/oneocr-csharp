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

## Evaluation order and memory

As for v0, every output value is an independent sum, so a port may evaluate whole or in bands. The C# reader keeps the v0 banded stages; the stride-16 branch runs on whole stride-16 maps (a quarter of the stride-8 map's size), `p16` is kept until the branch has consumed it, and stage 2's channel means are taken before stage 2 is dropped. On an A4 page at the 0.2.0 widths the scratch in use at once peaks at 24.3 MB at degree 1 (22.8 MB for v0 at its own widths).

## Reader checks

Before a tensor is read or a buffer allocated: `layout16_blocks` is in `[0, MaxConvBlocks]` and counts toward the total block limit with the backbone and the layout head; `layout16` is positive and within `MaxLayerWidth` when the branch exists; the stride-16 buffers are checked against `MaxActivationElements` with the other activation sizes; every tensor shape is checked against `config`. `orient16` must be a boolean.

## Tolerances

As in [numerics.md](numerics.md), with these additional layers against the float64 evaluation: `p16` Atol 4e-6, `layout16` 5e-6, `layout_mix` 5e-6 (Rtol 1e-5). `layout_feat`, `layout` and `orient` keep their v0 values. Worst error measured on random-weight models at the 0.2.0 widths and a narrow model, on every path, as a fraction of the tolerance: `p16` 0.24, `layout16` 0.38, `layout_mix` 0.29, `layout_feat` 0.55, `layout` 0.39, `orient` 0.14. Against PyTorch fp32 fixtures at the 0.2.0 widths (and a second model with the branch only) every layer differs by about PyTorch's own fp32 error.
