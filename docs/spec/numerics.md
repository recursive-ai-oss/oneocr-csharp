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

## Evaluation grouping that leaves every sum alone

These rearrangements change what is computed together or which terms are visited, never the value of any output. A port may use them or not.

- **Taps on the zero border.** A tap whose input value is an exact zero (a kernel row that falls on the zero border above or below the image rows) adds nothing: `fma(0, w, s) = s` for finite `w`, and a partial sum that starts at +0 is never -0, so the +0 or -0 product leaves it unchanged. Such taps may be left out of the dot product while every other tap keeps its lane `k mod 16` and its place in the order. This holds only when every weight of the layer is finite (a zero times an infinity or NaN is NaN); a layer with a non-finite weight visits all taps. Zeros that come from the data are not skipped.
- **Batched recurrence.** The recurrent product `dot(W_hh row, h_prev)` of several lines at the same step may be one matrix product (rows of `h_prev` against the rows of `W_hh`); each output is the same dot product, then `+ proj[t]` as one fp32 addition. A line's step never reads another line's values, so which lines share a product, and in what order, does not matter. The input projections of a run of steps may likewise be computed ahead as one product.
- **Three-by-three single-channel convolution** with 9 taps keeps the 16 partials in registers; lanes 9 to 15 are +0, and a partial `fma(w, x, +0)` is never -0, so adding one of those lanes in the tree returns the other operand unchanged: only lane 0 + lane 8 is a real addition, and the remaining additions are the tree's, in its order.

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
- **Global average** over a stage output of *n* positions, per channel: the positions are added one after the other in row-major order starting from +0 in fp32, then divided by *n* (IEEE division). The orientation logits are dot products of that vector with each row of `orient.weight`, plus the bias; with `dbnet-lite-v1` `orient16` the vector is the stage-2 means followed by the stage-3 means, one dot product over the joined vector ([detector-v1.md](detector-v1.md)).
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
| p16 (v1) | 4e-6 | | layout16, layout_mix (v1) | 5e-6 |

## Tolerance against PyTorch

Agreement with a PyTorch fp32 computation from the same fp16 weights is a per-layer tolerance, not bit equality. Both sides carry float32 rounding error against the exact answer, so the difference between them is bounded by the sum of the two errors. For layer *L* and every element:

`|ours - pytorch| <= Atol(L) + 2 * Floor(L) + Rtol(L) * |pytorch|`

- `Floor(L)` is PyTorch's own fp32-versus-float64 maximum absolute error for layer *L* over the fixture lines, measured by the fixture generator and stored in `fixtures.json` as `fp32_vs_fp64_floor`. It scales with the activation magnitudes of the model, so a trained model with large, saturating activations gets a proportionally larger allowance than a random-weight one. Our own float32 error is of the same size (both use float32 accumulation over the same number of terms), hence the factor 2.
- `Atol` / `Rtol` are twice the tolerances our kernels must meet against a float64 reference of the same network on random-weight models (error measured on models of the intended size, 3-5x headroom): conv 4e-6 / 2e-5, LSTM 3e-6 / 2e-5, classifier 2e-6 / 2e-5; so 8e-6 / 4e-5, 6e-6 / 4e-5, 4e-6 / 4e-5 here. The input layer must match to 1e-6.
- On top of the layer tolerances, our greedy decoding of our own logits must equal the fixture's `decoded` text for every line.

Observed on the 50 held-out lines of a trained model: the largest difference per layer is 0.9-2.0x the PyTorch floor (conv.5 2.6e-6, lstm.l1 6.6e-5, fc 7.0e-5, within the bound through its relative term).

## Document extraction kernels

The `docextract-v0` forward pass (graph in [docextract-v0.md](docextract-v0.md)) uses the rules above and these. Every sum of products is a dot product in the 16-lane order, so a vector path gives the same bits as scalar; everything else is element-wise IEEE arithmetic (add, subtract, multiply, divide, square root, fused multiply-add where stated), which is the same on every width. No sum is split across threads: rows, heads and words are split between threads, each output is computed by one call.

- **Linear layer** `y = x Wᵀ + b`, `W` stored `[out, in]`: one dot product of `in` terms per output, bias added to the finished sum. Inputs whose length is not a multiple of 16 continue the same lanes (the tail is part of the order); `feat_proj` (8 inputs) and `pair1` (13) are zero-padded to 16, which changes nothing because a zero product leaves every partial sum as it was.
- **Embedding rows**: `x[i] = x[i] + table[id]`, one addition per element, applied in the order x0, x1, y0, y1 of the position tables after the character projection; then the projected features are added; then layer norm.
- **Character CNN**, per word: the embeddings of its characters (an id of 0 reads row 0 of the table) with one zero row before and after; output position *p* is the dot product, in the order of taps (k, channel) with k = 0 the previous character, of the weight row `[k][channel]` (the container's `[out, channel, k]` reordered at load) with the three rows from *p* - 1 to *p* + 1, plus the bias. ReLU (`v > 0 ? v : 0`), then the maximum over the word's valid positions (characters with an id other than 0), starting from +0. Positions after the last valid character are not computed. `char_pool` is that maximum.
- **Layer norm** over `d` (epsilon 1e-5 as fp32): `mean = (Σ x) / d` with the sum a dot product with ones; `dx = x - mean`; `variance = (Σ dx²) / d`, a dot product of `dx` with itself; `inv = 1 / sqrt(variance + eps)` (IEEE square root and division); `y = fma(dx * inv, weight, bias)`.
- **GELU** (tanh form): `inner = c0 * fma(c1, (x * x) * x, x)` with `c0 = 0.79788456f`, `c1 = 0.044715f`; `y = (0.5f * x) * (1 + tanh(inner))`, `tanh` being the library's own (see Transcendental functions).
- **Pair features**: with `geo` the box in units of the median word height and the centre `c = (lo + hi) * 0.5f`, `d = a_j - a_i`, `squash(d, s) = d / (s + |d|)` (one division): centre x, centre y, x0, x1, y0, y1 each at s = 1 and s = 16, then the height difference at s = 1; 13 values, in that order. The pair MLP is `pair1` (13 to `pair_hidden`), ReLU, `pair2`; row *i*, column *j* of channel *c* is stored at `c * n * n + i * n + j`.
- **Attention**, per head: `s = q·k` (dot product over the 32 head dimensions), `s = s / sqrt(head size) + bias` (the divisor is the fp32 square root of the head size, 5.656854f for 32; division then addition, two roundings); softmax per row: `m = max s`, `e = exp(s - m)` with the library's exp, `sum = Σ e` as a dot product with ones, `p = e / sum`; `a = Σ_j p_j v_j` as a dot product over `j` with the head's V transposed to contiguous rows, no bias.
- **Pointers**: `rel[r, i, j] = (q_i,r · k_j,r) / sqrt(rel_dim) + pair_bias[layers * heads + r, i, j]`, the division by the fp32 square root (8 for 64).
- **Document logits**: the mean over words of `final` (each column summed in word order in double precision, divided by the word count, rounded to fp32) through `doc_head`.
- **Windows**: see docextract-v0.md; the merge is selection, not arithmetic, except the document logits of a page of several windows, which are a double-precision weighted mean rounded to fp32.

### Tolerance of the document extraction model

Per layer (`char_pool`, `emb`, `pair_bias`, `block.<l>`, `final`, `tags`, `roles`, `doc`, `rel`), against PyTorch's float64 run of the same fp16 weights: `max |ours - exact| <= 2 * Floor(L) + 8 ulp * max |exact|`, where `Floor(L)` is PyTorch's own fp32-versus-float64 maximum error for the layer (`fp32_vs_fp64_floor` in the fixtures) and the second term (8 times 2^-23 of the layer's largest value) covers layers where PyTorch happens to be nearly exact. Against PyTorch's fp32 result: `max |ours - pytorch| <= 3 * Floor(L) + 8 ulp * max |exact|`. Observed on two random-weight models over pages of 24 to 300 words on all four paths: our error is 0.4 to 1.13 times PyTorch's for every layer except the document logits (2.05 times a very small floor, inside the ulp term); the difference from PyTorch's own result is at most 1.44 times the floor. Results are bit-identical across paths and degrees.

## Line sampling

Reading a leaning line (see [detector-v1](detector-v1.md#line-axis-post-processing-any-detector-version)) samples the page at points off the pixel grid.
The sampler is bilinear in double precision with a fixed order of operations (`top`, `bottom`, then their blend), edge pixels repeated and the
result rounded half up to a byte. It uses no fused multiply-add and no library trigonometry: the axis comes from `sqrt`, `+ - * /` only, so
every platform and every degree of parallelism produces the same bytes. The sampled crop then goes through the same resize as any line crop.

## Libraries outside the kernels

Every crop is scaled with `RecursiveAI.OneImage` (linear light, 8-bit gray), which is outside these rules. Its sRGB tables are built with the
platform's `powf`; macOS's differs from glibc's by one binary32 ulp in 2 of the 256 decode entries, and that is enough to change one pixel of a
crop, a logit, and the last bit of a raw word confidence (text and boxes were unchanged on the sampled pages). Taking the power in double and
rounding once, `(float)Math.Pow(x, 2.4)`, gives tables equal to glibc's on x64 and arm64 and to macOS's, and leaves every x64 output unchanged.
Until OneImage ships that, raw confidence can differ in the last bit between macOS and Linux; `OneOcr.Bench mathhash` prints the table hashes.
The platform identity check is described in `bench/README.md`.
Two library `Math.Log` calls remain outside the kernels (the detector's logit thresholds and a column-width comparison in the page reader); they are
double precision, rounded to binary32 or compared, and agreed on every platform tried.
