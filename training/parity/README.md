# Parity fixtures

Reference data the C# kernels are checked against. Both generators read a trained model container
and write plain files; nothing here is committed output.

## Forward: `parity/fixtures.py`

See its docstring. Layer outputs for real lines, fp32 from the container's fp16-rounded, batch-norm-folded
weights, with the fp32-vs-fp64 floor in `fixtures.json`.

## Backward: `parity/backward.py`

```
uv run python -m parity.backward --fixtures FORWARD_FIXTURE_DIR --out OUT_DIR [--seed 21]
```

Reads `model.onm`, `lines/` and the truth text from a forward fixture directory. Single-threaded CPU,
seeded, deterministic: regenerating gives identical `SHA256SUMS`. Needs about 90 MB for crnn-v0.

### Files

```
model.onm                         the container the references come from
fixtures.json                     index of everything below, conventions, fp32-vs-fp64 floors
gradcheck.json                    finite-difference check of every analytic gradient (fp64)
SHA256SUMS                        sha256 of every other file
lines/<id>.png                    8-bit gray crops (same as the forward fixtures)
layers/<id>/<layer>.x.f32         layer input
layers/<id>/<layer>.gy.f32        upstream gradient (seeded standard normal, shape of the layer output)
layers/<id>/<layer>.d[<t>].f32    expected gradient w.r.t. tensor <t>: "x" (the input) or a parameter name
model/<batch>/<id>.logits.f32     fp32 logits [T, classes] of a batch line
model/<batch>/<id>.dlogits.f32    d(batch loss)/d(logits)
optimizer/grads_step{1,2,3}/<param>.f32
optimizer/weights_after_step{1,3}/<param>.f32
```

All tensors are little-endian float32, row-major. A tensor over 200,000 floats is split into
`<name>.part00.f32`, `.part01.f32`, ... (concatenate in order); every entry in `fixtures.json` lists its
`files` with their sha256 and its `shape`. Every file is under 1 MB except `model.onm`.

### Layers (`layer_lines[].layers[]`)

Two lines, each with these layers in network order: for each conv block `conv.<i>.conv`, `conv.<i>.relu`
and (where the block pools) `conv.<i>.pool`; then `lstm.l0`, `lstm.l1`, `fc`. Each entry has `kind`, `attrs`,
`input`, `grad_output`, `expected` (gradient per tensor) and `params` (which `expected` keys are parameters).
Shapes: conv/relu/pool tensors are `[C,H,W]`; LSTM and fc tensors are `[T, features]`.

- **conv**: stride 1, zero padding from `attrs`, cross-correlation, weight `[Cout,Cin,kh,kw]` with the batch norm
  already folded in; parameters `conv.<i>.weight`, `conv.<i>.bias`. Bias gradient is the sum of the
  upstream gradient over H and W.
- **relu**: derivative 0 at exactly 0 (gradient passes where input > 0).
- **maxpool**: kernel = stride, trailing partial windows dropped. The gradient goes to the first maximum in
  row-major window order. Post-ReLU activations tie at 0 constantly, so this rule decides where the gradient
  lands in an isolated pool test (in the full chain the ReLU mask removes it again).
- **lstm.l<k>**: one bidirectional layer, zero initial state, gate order i,f,g,o (PyTorch's). The backward
  direction reads the sequence reversed and its outputs are stored at the original time steps; output is
  `[forward h | backward h]`. Parameters per direction (suffix `_reverse` for backward): `lstm.weight_ih_l<k>`
  `[4h,in]`, `lstm.weight_hh_l<k>` `[4h,h]`, `lstm.bias_l<k>` `[4h]` = **b_ih + b_hh as one tensor**. Its
  gradient is therefore the gradient of either bias, and the optimizer treats it as one parameter.
- **fc**: `y = x Wᵀ + b`, parameters `fc.weight` `[classes, 2h]`, `fc.bias`.

### Whole model (`model_level`)

Four lines run one at a time (no padding), under two sets of targets (`model_level.batches`): `truth`, each
line's own text (the model fits it, loss about 0.01), and `shifted`, each image paired with the next line's
text (loss about 255, strong gradients; the optimizer steps and parameter gradients use this one). Per sample:
`target` (class ids, charset index; blank is 0), `logits` (the same for both batches), `dlogits`, per-sample
`nll` (and `nll_fp64`).

- **CTC**: negative log-likelihood summed over all alignments (Graves et al. 2006), blank 0, over softmax of
  the logits, in natural log.
- **Reduction**: batch loss = mean over the B samples of the per-sample NLL (no division by target or
  input length). `dlogits` is the gradient of that mean, so a sample's block is its own gradient / B. For
  a "sum" reduction multiply by B.
- `builtin_ctc_loss_check` records how far `torch.nn.functional.ctc_loss` (fp64) is from the reference here.

### Optimizer (`optimizer`)

AdamW with decoupled weight decay, applied per parameter tensor, step counter t from 1, m = v = 0 at start:

```
p <- p * (1 - lr * weight_decay)
m <- b1*m + (1-b1)*g ;  v <- b2*v + (1-b2)*g*g
p <- p - lr/(1-b1^t) * m / (sqrt(v/(1-b2^t)) + eps)
```

lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight decay 0.01, **applied to biases too**, no clipping. Start
weights are the container values widened to fp32 (LSTM biases summed). Parameter names are `parameters.names`
in `fixtures.json`. Three real steps on the shifted batch: gradients are recomputed at each step's weights in fp32
(`grads_step1..3`), results are `weights_after_step1` and `weights_after_step3`; feeding the stored
gradients to an optimizer alone must give the same weights. Where |g| is far below eps the update is
sensitive to gradient noise, so the first step is close to `lr * sign(g)` for most weights; compare with a
tolerance, not bit-for-bit.

### Floors and checks

- `fp32_vs_fp64_floor`: the fp32 PyTorch result against the same function evaluated in fp64 on the same
  fp32 inputs, per layer and tensor: `max_abs`, `ref_max_abs`, `max_abs_over_ref_max` (scale-relative, the
  useful tolerance basis for gradients), and `max_rel_above_1e-3_of_max` (relative error over entries
  above 1e-3 of the largest). Taken as the maximum over lines. Model level: CTC loss, `dlogits`, and step-1
  parameter gradients end to end. The optimizer entry has fp32-vs-fp64 replays of the update.
- `gradcheck.json`: each analytic fp64 gradient against central finite differences along random directions
  (tilted toward the gradient), per layer and for the whole-model CTC loss (shifted batch) against the parameters
  and against the logits; plus `torch.autograd.gradcheck` on a small CTC case. Max-pool is checked on its input plus a
  jitter, because ties are not differentiable.
