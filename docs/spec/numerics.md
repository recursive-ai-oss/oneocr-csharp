# Numeric rules

The operation order that makes the library's output bit-identical across CPU paths (scalar, AVX2, and later AVX-512 and 128-bit vectors), operating systems and degrees of parallelism. A port that follows these rules reproduces the C# results exactly. All arithmetic is IEEE 754 binary32 unless stated; `fma(a, b, c)` is the fused multiply-add with a single rounding. Weights are stored as binary16 and widened exactly to binary32 on load.

## Dot products

A dot product of two length-*n* rows is computed with 16 partial sums, initially +0. Element *k* is fused into partial *k mod 16*: `p[k mod 16] = fma(a[k], b[k], p[k mod 16])`, for all *k* in order, including a final group shorter than 16. The partials are then added in a fixed tree:

1. `p[i] = p[i] + p[i+8]` for i = 0..7
2. `p[i] = p[i] + p[i+4]` for i = 0..3
3. `p[i] = p[i] + p[i+2]` for i = 0..1
4. result `p[0] + p[1]`

A bias, when present, is added to the finished sum. Vector paths keep the 16 partials in registers (one 512-bit, two 256-bit or four 128-bit); padding a row with zeros to a multiple of 16 does not change the result. Parallel work is split across independent outputs; no reduction crosses threads.

## Layers

- **Convolution** output value = dot product of the weight row (input channel, kernel row, kernel column order) with the zero-padded input patch in the same order, plus the bias. Then ReLU `v > 0 ? v : 0`, then max pooling (a value that is not greater than the running maximum, starting at 0, is ignored).
- **LSTM** step, per direction, hidden size H, gate order input, forget, cell, output: `proj[t] = dot(W_ih row, x_t) + (b_ih + b_hh)` (the two biases are added in fp32 once); `z = dot(W_hh row, h_prev) + proj[t]`; `i, f, o = sigmoid(z slice)`, `g = tanh(z slice)`; `c' = fma(f, c, i * g)` with `i * g` rounded first; `h = o * tanh(c')`. State starts at zero. The reverse direction visits steps from last to first and writes its output at the step's own position.
- **Classifier** logits = dot product + bias, per step.
- **Softmax denominator** (for confidence) = sum over classes of `exp(z - max)` accumulated into 16 partials by class index mod 16 with plain additions, combined with the tree above; the winning probability is `1 / sum`.

## Transcendental functions

No platform math library is used. With `Limit = 80`:

- `exp(x)`: clamp `x` to [-Limit, Limit] (`x < Limit ? x : Limit`, then `x > -Limit ? x : -Limit`, so NaN becomes Limit); `n = roundToNearestEven(x * log2e)`; `r = fma(-n, ln2_hi, x)`, then `r = fma(-n, ln2_lo, r)`; `p = c7`, then `p = fma(p, r, c6)`, `c5`, `c4`, `c3`, `c2` in turn; `y = fma(p, r * r, r) + 1`; result `y * 2^n`, with `2^n` built from the exponent bits `(n + 127) << 23`.
- `sigmoid(x) = 1 / (1 + exp(-x))`; `tanh(x) = 1 - 2 / (exp(x + x) + 1)`.
- Constants (binary32): `log2e = 1.44269504088896341`, `ln2_hi = 0.693145751953125`, `ln2_lo = 1.42860677e-6`, `c2 = 0.5`, `c3 = 1/6`, `c4 = 1/24`, `c5 = 1/120`, `c6 = 1/720`, `c7 = 1/5040`, each rounded to binary32.

Measured accuracy: exp and sigmoid within 4e-7 relative error, tanh within 2e-7 absolute.

## Tolerance against PyTorch

Agreement with a PyTorch fp32 computation from the same fp16 weights is a per-layer tolerance, not bit equality; the tolerances live beside the parity tests.
