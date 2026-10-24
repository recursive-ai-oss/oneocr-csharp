# Backward numerics

How gradients, the CTC loss and the optimizer step are computed, in an order that makes them bit-identical across CPU paths (scalar, 128-bit vectors, AVX2, AVX-512), operating systems and degrees of parallelism. A port that follows these rules reproduces the C# results exactly. It extends `numerics.md`: all arithmetic is IEEE 754 binary32 unless stated, `fma` is the fused multiply-add, and a "dot product" is the 16-lane dot product defined there (element *k* into partial *k mod 16*, then the fixed pairwise tree). Gradients are for the `crnn-v0` recognizer; weights are the fp32 values widened from the container's fp16, and fine-tuning updates fp32 master copies of them.

## Principle

Every sum over many terms is a dot product in the library's order; every other operation is element-wise. A gradient is therefore a set of independent outputs, each with one fixed order, so tiling and threading cannot change a bit. Sums that would otherwise be scatter-adds are written as gathers (each output reads its terms in a stated order), and the per-line gradients of a batch are added in line order. No reduction crosses threads.

## Layers

Notation: `dY` is the gradient of a layer's output, `T` the number of positions or time steps, activations are channels-last as in the forward pass.

- **Linear** (`y = x Wᵀ + b`, `T` rows): `dW[o][i]` = dot over t of `dY[t][o]` and `x[t][i]`; `db[o]` = dot over t of `dY[t][o]` and 1; `dx[t][i]` = dot over o of `dY[t][o]` and `W[o][i]`. The dot products read the operands as rows of the transposed matrices, so the lane of term *t* (or *o*) is *t* mod 16 (or *o* mod 16).
- **Sum of a row** (bias gradients) is always the dot product with a row of ones: `fma(x, 1, p)` adds exactly, so this is the 16-lane sum.
- **ReLU and max pooling.** Forward: the pooled value is the maximum over the window starting from 0, scanning rows then columns, replacing the running maximum only when strictly greater (so the first of equal maxima wins and a NaN never wins). The winner's flat index in the convolution output is recorded, or "none" when no value was positive. Backward: the pooled element's gradient is assigned to the winner; "none" passes nothing (this is the ReLU's zero gradient at and below 0). Trailing partial windows are dropped, as in the forward pass.
- **Convolution** (taps in (ky, kx, channel) order, rows padded with zeros to a multiple of 16, as in `numerics.md`). With `P` output positions, `dC[p][o]` the gradient of the convolution output (after the pool backward) and `col[p][k]` the input windows used by the forward pass:
  - `db[o]` = dot over p of `dC[p][o]` and 1;
  - `dW[o][k]` = dot over p of `dC[p][o]` and `col[p][k]` (zero in the padding columns, whose window values are zero);
  - `dcol[p][k]` = dot over o of `dC[p][o]` and `W[o][k]`;
  - `dx[iy][ix][c]` = the sum of `dcol[p][(ky·kw + kx)·C + c]` over the taps (ky, kx) whose window covers the input element, p being the output position `(iy + padH − ky, ix + padW − kx)`; started from +0 and added in ascending ky, then ascending kx, with plain additions. The first block's input gradient is not computed.
- **LSTM** (per direction, hidden size H, gate order i, f, g, o; `b = b_ih + b_hh` is one parameter, and its gradient is the gradient of either bias). The forward pass stores the activated gates, the cell state, `tanh` of the cell state and the hidden state each step started from (zeros for the first step visited). The backward pass visits the steps in the reverse of the forward visiting order. For each element j, with `dh = dOut[t][column + j] + dhRec[j]` (`dhRec` from the step visited next, zero for the last):
  ```
  dtc = dh * o            do = dh * tc
  dc  = dcNext + dtc * (1 - tc * tc)
  di = dc * g   dg = dc * i   df = dc * cPrev   dcNext = dc * f
  dz_i = di * (i * (1 - i))      dz_f = df * (f * (1 - f))
  dz_g = dg * (1 - g * g)        dz_o = do * (o * (1 - o))
  ```
  every operation rounded separately (no fused multiply-add), `cPrev` = 0 for the first step visited. `dhRec` for the previous step is the dot product over the 4H gates of `dz` and `W_hh[·][j]`. After the recurrence, with `dz` stored by time step: `dW_ih[g][i]` = dot over t of `dz[t][g]` and `x[t][i]`; `dW_hh[g][j]` = dot over t of `dz[t][g]` and `hPrev[t][j]`; `db[g]` = dot over t of `dz[t][g]` and 1 (t ascending by time index in all three). The input gradient `dx[t][i]` is the dot product over the 8H gates, forward direction's 4H first and then the reverse direction's, of `dz[t][·]` and `W_ih[·][i]`.
- **Classifier** is a linear layer on the top LSTM output.

## CTC loss

Definition: Graves et al. 2006 (sections 3.1–3.2, 4.1). Blank is class 0; targets use classes 1 and up. Double precision, scalar, using the double `exp` and `log` of `decoding.md` (`DetMath`) so the result is the same everywhere:

1. Log softmax per step: `m` = maximum logit; `s` = sum over classes in ascending order of `exp(z − m)`; `lp[k] = (z[k] − m) − log(s)`.
2. Forward and backward variables over the blank-extended label sequence (2L + 1 entries), log domain, `logadd(a, b) = a + log(1 + exp(b − a))` for `a ≥ b` (returning `a` when `b − a < −40` or `b` is −∞); both include the emission of their own step. Skips of two entries are allowed only between different non-blank labels.
3. `logP` = `logadd` of the last two forward entries at the last step; loss = `−logP`. When `logP` is −∞ (no alignment) the loss is +∞ and the gradient is zero; the training step then counts the line as 0 (`zero_infinity`).
4. Gradient with respect to the logits: `scale · (exp(lp[k]) − exp(acc[k] − lp[k] − logP))`, where `acc[k]` is the `logadd`, in ascending entry order, of forward + backward over the entries labelled `k`. Rounded to binary32 once.

**Reduction.** The batch loss is the mean over the lines of each line's loss, **not** divided by the target length (PyTorch's `reduction='mean'` also divides by it; the parity fixtures use the plain mean). A line's logit gradient is scaled by `1/B`.

## Batch gradient and clipping

Per line: forward, loss, backward into that line's own gradient buffers. Then for every parameter element, `g = g₀`, `g += g₁`, … in line order (plain additions). The global gradient norm is the double-precision sum over tensors, in parameter order, of each tensor's dot product with itself, then the square root; when clipping is on and the norm exceeds the limit every gradient is multiplied by `float(limit / (norm + 1e-6))`.

Parameter order: for each convolution block `weight` (`[Out][KPadded]`, the forward layout) and `bias`; for each LSTM layer, forward direction then reverse: `W_ih`, `W_hh`, `b`; classifier `weight`, `bias`.

## AdamW

Loshchilov and Hutter 2019, algorithm 2, fp32 master weights, per element, each operation rounded separately, in PyTorch's single-tensor order. For step *t* (from 1), with the scalars computed in double and rounded once to binary32:

```
c1 = 1 − β1ᵗ,  c2 = 1 − β2ᵗ   (running double products β1ᵗ, β2ᵗ)
stepSize = float(lr / c1)      sqrtC2 = float(sqrt(c2))
decay = float(1 − lr · wd)     (only for tensors flagged for decay)
p = p * decay
m = m + (1 − β1) * (g − m)
v = v * β2 + (1 − β2) * (g * g)
p = p − stepSize * (m / (sqrt(v) / sqrtC2 + eps))
```

Decay is applied to weight matrices and, by default, not to biases (a setting). Square root and division are correctly rounded IEEE operations, so the step is identical everywhere.

## Tolerances

Against float64 (a straightforward double-precision implementation written from the definitions, on random-weight models of the intended size, widths with and without tails): `|ours − reference| ≤ Atol + Rtol·|reference|`.

| Layer | Atol | Rtol | Measured max (abs; rel where \|ref\| > 0.05) |
|---|---|---|---|
| conv (dW, db, dx) | 7e-6 | 3.5e-5 | 5.1e-6; 2.5e-5 |
| LSTM (dW_ih, dW_hh, db, dx) | 1e-6 | 5e-6 | 5.9e-7; 5.2e-6 |
| classifier (dW, db, dx) | 1e-6 | 5e-6 | 7.2e-7; 4.5e-6 |
| CTC loss and logit gradient vs brute-force sum over all alignments | 1e-6 | 1e-6 | (loss within 1e-10) |
| ReLU/pool routing, Im2Col/Col2Im adjoint | exact / 1e-6 relative | | |

The LSTM gradient is of the network as computed, with the approximate sigmoid and tanh of `numerics.md`; the reference uses exact functions, which is most of the LSTM difference.

Against PyTorch fixtures (a trained model, fp32 references with their float64 floor): `|ours − pytorch| ≤ 2·Atol + 2·Floor + 2·Rtol·|pytorch|` with `Floor` the fixture's recorded `max_abs` fp32-versus-float64 error of that tensor. Observed worst use of that allowance: conv 0.21, LSTM 0.34, classifier 0.30 (per-layer fixtures); CTC loss and logit gradient 0.42; optimizer step on stored gradients 0.001 of the allowance. Whole-model parameter gradients on a batch of four lines are within about 0.45 of the allowance for every tensor but one: the first convolution's weights differ by up to 10 floors at the taps of one output channel, which is a single ReLU/pool decision lying within rounding of a tie going the other way in the two implementations; the test allows one such flip. Weights after three real steps are compared only where the gradient is at least 50 floors in every step (below that the sign of the first Adam step is decided by noise in either implementation).
