# Numeric rules

The operation order that makes the library's output bit-identical across CPU paths (scalar, 128-bit vectors, AVX2 and AVX-512), operating systems and degrees of parallelism. A port that follows these rules reproduces the C# results exactly. All arithmetic is IEEE 754 binary32 unless stated; `fma(a, b, c)` is the fused multiply-add with a single rounding. Weights are stored as binary16 and widened exactly to binary32 on load.

## Dot products

A dot product of two length-*n* rows is computed with 16 partial sums, initially +0. Element *k* is fused into partial *k mod 16*: `p[k mod 16] = fma(a[k], b[k], p[k mod 16])`, for all *k* in order, including a final group shorter than 16. The partials are then added in a fixed tree:

1. `p[i] = p[i] + p[i+8]` for i = 0..7
2. `p[i] = p[i] + p[i+4]` for i = 0..3
3. `p[i] = p[i] + p[i+2]` for i = 0..1
4. result `p[0] + p[1]`

A bias, when present, is added to the finished sum. Vector paths keep the 16 partials in registers (one 512-bit, two 256-bit or four 128-bit); padding a row with zeros to a multiple of 16 does not change the result. Computing several outputs at once (a tile of A rows against B rows), reading one row in several contiguous segments whose lengths are multiples of 16, reducing four outputs' partials together with the same pairwise additions, or, for a dot product of at most 16 taps, keeping each tap's product in its own vector across output channels (lane k holds `fma(w[k], x[k], 0)`, the missing lanes are +0) are all the same computation: every output sees the same fused multiply-adds in the same order and the same tree. Parallel work is split across independent outputs; no reduction crosses threads.

## Layers

- **Convolution** output value = dot product of the weight row with the zero-padded input patch, plus the bias, with both flattened in the order (kernel row, kernel column, input channel): tap index `k = (ky * kw + kx) * c + channel`, so the lane of a tap is `k mod 16`. (The container stores weights `[out][in][kh][kw]`; readers reorder them at load. The order is chosen so that, with channels-last activations, the taps of one kernel row are adjacent in memory.) Then ReLU `v > 0 ? v : 0`, then max pooling (a value that is not greater than the running maximum, starting at 0, is ignored).
- **LSTM** step, per direction, hidden size H, gate order input, forget, cell, output: `proj[t] = dot(W_ih row, x_t) + (b_ih + b_hh)` (the two biases are added in fp32 once); `z = dot(W_hh row, h_prev) + proj[t]`; `i, f, o = sigmoid(z slice)`, `g = tanh(z slice)`; `c' = fma(f, c, i * g)` with `i * g` rounded first; `h = o * tanh(c')`. State starts at zero. The reverse direction visits steps from last to first and writes its output at the step's own position.
- **Classifier** logits = dot product + bias, per step.
- **Softmax denominator** (for confidence) = sum over classes of `exp(z - max)` accumulated into 16 partials by class index mod 16 with plain additions, combined with the tree above; the winning probability is `1 / sum`.

## Transcendental functions

No platform math library is used. With `Limit = 80`:

- `exp(x)`: clamp `x` to [-Limit, Limit] (`x < Limit ? x : Limit`, then `x > -Limit ? x : -Limit`, so NaN becomes Limit); `n = roundToNearestEven(x * log2e)`; `r = fma(-n, ln2_hi, x)`, then `r = fma(-n, ln2_lo, r)`; `p = c7`, then `p = fma(p, r, c6)`, `c5`, `c4`, `c3`, `c2` in turn; `y = fma(p, r * r, r) + 1`; result `y * 2^n`, with `2^n` built from the exponent bits `(n + 127) << 23`.
- `sigmoid(x) = 1 / (1 + exp(-x))`; `tanh(x) = 1 - 2 / (exp(x + x) + 1)`.
- Constants (binary32): `log2e = 1.44269504088896341`, `ln2_hi = 0.693145751953125`, `ln2_lo = 1.42860677e-6`, `c2 = 0.5`, `c3 = 1/6`, `c4 = 1/24`, `c5 = 1/120`, `c6 = 1/720`, `c7 = 1/5040`, each rounded to binary32.

Measured accuracy: exp and sigmoid within 4e-7 relative error, tanh within 2e-7 absolute.

## Detector layers

The page detector (`dbnet-lite-v0`, [detector-v0.md](detector-v0.md)) uses activations laid out channels-last, `[row][column][channel]`, with the channel count padded up to a multiple of 16 by zeros that stay zero. Nothing about this layout is visible in the results; it is how the sums below are kept in the order of the dot-product rule.

- **1x1 convolution** is a dot product over the input channels for every position: the rule above, with the channel count zero-padded to a multiple of 16 (no change to the result), then the bias, then ReLU where the graph has one.
- **3x3 depthwise convolution** (zero padding 1, stride 1 or 2, one filter per channel): the nine products of a position and channel are the elements of one dot product, tap *k* = `ky * 3 + kx` in lane *k*: `p[k] = fma(w[k], x[k], +0)`, lanes 9 to 15 stay +0, then the tree above and the bias. A tap that falls outside the image multiplies a zero padding value and contributes `fma(w, 0, +0) = +0`, so it is simply +0. Vector paths hold the nine products for 4, 8 or 16 channels at a time; a channel is never split across lanes, so the width cannot change a result.
- **Stem** (3x3, stride 2, one input channel, ReLU) follows the same rule per output channel.
- **Nearest 2x upsampling plus addition**: `dst[y][x][k] = dst[y][x][k] + src[y / 2][x / 2][k]` (one fp32 addition; commutative).
- **Global average** over a stage output of *n* positions, per channel: the positions are added one after the other in row-major order starting from +0 in fp32, then divided by *n* (IEEE division). The orientation logits are dot products of that vector with each row of `orient.weight`, plus the bias.
- **ReLU** `v > 0 ? v : 0` (NaN and -0 give +0). Sigmoid and softmax appear only in post-processing, with the exp above; thresholds are compared on logits.

Rows of an output are split across threads; no reduction crosses a thread boundary.

## Tolerance of the detector against float64 and PyTorch

For each layer the C# fp32 result is compared with a float64 evaluation of the same graph and fp16 weights: `|ours - exact| <= Atol + Rtol * |exact|`, with `Rtol` 1e-5 for every layer and `Atol` as in the table (about three times the worst error seen over the test models on every path). Against PyTorch's fp32 fixtures the allowance is twice that, plus twice PyTorch's own recorded fp32-versus-float64 error for the layer (`fp32_vs_fp64_floor` in `fixtures.json`).

| layer | Atol | | layer | Atol |
|---|---|---|---|---|
| input | 1e-6 (Rtol 1e-6) | | layout_feat | 5e-6 |
| stem | 1e-6 | | layout | 3e-6 |
| stage0 | 1e-6 | | text4 | 5e-6 |
| stage1, stage2, stage3 | 2e-6 | | text2 | 8e-6 |
| p8 | 4e-6 | | prob | 5e-6 |
| p4 | 5e-6 | | orient | 2e-7 |

## Tolerance against PyTorch

Agreement with a PyTorch fp32 computation from the same fp16 weights is a per-layer tolerance, not bit equality. Both sides carry float32 rounding error against the exact answer, so the difference between them is bounded by the sum of the two errors. For layer *L* and every element:

`|ours - pytorch| <= Atol(L) + 2 * Floor(L) + Rtol(L) * |pytorch|`

- `Floor(L)` is PyTorch's own fp32-versus-float64 maximum absolute error for layer *L* over the fixture lines, measured by the fixture generator and stored in `fixtures.json` as `fp32_vs_fp64_floor`. It scales with the activation magnitudes of the model, so a trained model with large, saturating activations gets a proportionally larger allowance than a random-weight one. Our own float32 error is of the same size (both use float32 accumulation over the same number of terms), hence the factor 2.
- `Atol` / `Rtol` are twice the tolerances our kernels must meet against a float64 reference of the same network on random-weight models (error measured on models of the intended size, 3-5x headroom): conv 4e-6 / 2e-5, LSTM 3e-6 / 2e-5, classifier 2e-6 / 2e-5; so 8e-6 / 4e-5, 6e-6 / 4e-5, 4e-6 / 4e-5 here. The input layer must match to 1e-6.
- On top of the layer tolerances, our greedy decoding of our own logits must equal the fixture's `decoded` text for every line.

Observed on the 50 held-out lines of a trained model: the largest difference per layer is 0.9-2.0x the PyTorch floor (conv.5 2.6e-6, lstm.l1 6.6e-5, fc 7.0e-5, within the bound through its relative term).
