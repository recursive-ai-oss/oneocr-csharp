# Draft: `svtr-lite-v1` container architecture

A proposed `architecture` section for the model container, for a runtime that reads `svtr-lite` models. No runtime does: it was a candidate in the recognizer comparison, and the shipped recognizer is `crnn-v0`. The trainer, the
exporter (`export.export`) and the parity fixtures (`parity.fixtures`) already produce files in this shape using the container's
existing envelope (magic, manifest, 64-byte-aligned fp16 tensors); only the architecture id and the layer graph below are new.

Model: `models/svtr.py` (a lightweight variant of the recognizer in Du et al., "SVTR: Scene Text Recognition with a Single Visual
Model", 2022). Input: 1 x 32 x W, `(p / 255 - 0.5) / 0.5`, as for the other recognizers. Output: T = ceil(ceil(W / 2) / 2) steps of
`classes` logits, greedy CTC with blank = 0.

## Manifest `config`

```json
{"arch": "svtr-lite", "stem": [32, 64], "mix_blocks": 2, "merge": 128, "dim": 192, "blocks": 6, "heads": 4, "mlp": 2, "classes": 622}
```

## Layer graph (all tensors fp16 in the file, fp32 compute)

Let `gelu` be the tanh form `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`.

| step | op | tensors (shape) |
|---|---|---|
| `stem.0` | conv 3x3, stride (2,2), pad 1, 1 -> 32, gelu | `stem.0.weight` [32,1,3,3], `stem.0.bias` [32] |
| `stem.1` | conv 3x3, stride (2,2), pad 1, 32 -> 64, gelu | `stem.1.weight` [64,32,3,3], `stem.1.bias` |
| `mix.i` (i < mix_blocks) | `x = x + gelu(conv 3x3, stride 1, pad 1, 64 -> 64)` | `mix.i.weight` [64,64,3,3], `mix.i.bias` |
| `merge` | conv 3x3, stride (2,1), pad 1, 64 -> 128, gelu; output [128, 4, T] | `merge.weight` [128,64,3,3], `merge.bias` |
| `proj` | per column: flatten (channel-major, then height: index `c*4 + h`) to 512, linear 512 -> dim | `proj.weight` [dim,512], `proj.bias` |
| `block.i` (i < blocks) | see below | `block.i.{norm0,local,norm1,qkv,out,norm2,fc1,fc2}.{weight,bias}` |
| `fc` | layer norm (`norm.weight/bias`, eps 1e-5), then linear dim -> classes | `norm.*`, `fc.weight` [classes,dim], `fc.bias` |

Batch norm is folded into the convolutions at export (fp64, rounded once), so the file holds only conv weights and biases.

Block `i`, on a sequence `x` of T vectors of size `dim`:

1. `x = x + gelu(conv1d_k3_pad1(layer_norm_0(x)))` over the width, dense `dim -> dim`; `block.i.local.weight` is [dim, dim, 3].
2. `x = x + out(attention(layer_norm_1(x)))`: `qkv` is one linear `dim -> 3 dim` whose output splits as `[q | k | v]`, each
   split into `heads` consecutive slices of `dim / heads`; `softmax(q k^T / sqrt(dim / heads)) v` per head over all T positions
   (lines are batch-of-one at inference, so no masking), heads concatenated, then `out` linear `dim -> dim`.
3. `x = x + fc2(gelu(fc1(layer_norm_2(x))))`, `fc1` is `dim -> mlp * dim`.

Layer norm: biased variance, eps 1e-5, over the last axis, `y = (x - mean) / sqrt(var + eps) * weight + bias`.

## Layer types the runtime needs beyond `crnn-v0`

Strided convolution (including stride (2,1)), 1-D convolution (a width-3 convolution is a 3x3 convolution with height 1), GELU
(tanh form, composed from the runtime's own `tanh`), layer norm, linear, softmax, multi-head attention (a batched matrix product
pair), residual add. No recurrent layer. Attention memory and time grow with T^2: a 1000 px line has T = 250.

## Parity fixture layers

`input`, `stem.0`, `stem.1`, `mix.0`, `mix.1`, `merge` ([C,H,W]); `proj` ([T,dim]); `block.0` .. `block.5` ([T,dim], after
each block); `fc` ([T,classes]).
