# docextract-v0: backward pass and fine-tuning

The training counterpart of [docextract-v0.md](docextract-v0.md), written so that another implementation can reproduce the gradients and the weights. It follows [backward-numerics.md](backward-numerics.md) (every long sum a 16-lane dot product, everything else separately rounded IEEE operations, no fused multiply-add outside a dot) and [numerics.md](numerics.md) (the library's `exp`, `tanh`).

## Training forward pass

One page is one example and one window (at most `max_words` words). The forward pass is the inference pass of docextract-v0 with every activation kept; its outputs are bit-identical to inference (tested). Dropout does not exist. Layer-normalization inputs, the softmax probabilities of every head, the GELU `tanh` terms and the pair-MLP hidden activations are saved for the backward pass.

## Backward kernels

- **Layer normalization** (`y = xhat * w + b`, `xhat = (x - mean) * inv`): per row with `g = dY * w`, `dX = inv * ((g - mean(g)) - xhat * mean(g * xhat))`, both means 16-lane dot products divided by `d`. `dw`, `db` are the column sums of `dY * xhat` and `dY` (dot products with ones along the row axis).
- **GELU** (tanh form, constants of the forward pass): `dX = dY * (0.5 (1 + t) + 0.5 x (1 - t²) c0 (1 + 3 c1 x²))` with `t = tanh(c0 (x + c1 x³))` saved from the forward pass. It differentiates the function as computed, with the library's `tanh`.
- **Softmax with bias** (`s = raw / scale + bias`): per row `ds = p * (dP - dot(p, dP))`; the pair-bias gradient is `ds`, the query-key gradient `ds / scale`.
- **Scores and weighted sum**: `dQ[i][c] = Σj dS[i][j] K[j][c]`, `dK[j][c] = Σi dS[i][j] Q[i][c]`, `dP[i][j] = dot(dA[i], V[j])`, `dV[j][c] = Σi P[i][j] dA[i][c]`, all 16-lane dots (over `j` or `i` ascending).
- **Pair-bias MLP** `pair2(relu(pair1(features)))` over the `n²` ordered pairs in row-major `(i, j)` order: weight gradients are 16-lane dots over pairs in that order; the features are data and get no gradient.
- **Pointer layers** (`rel_q`, `rel_k`, items and key–value): the same score adjoint, with the pointer scores' own loss.
- **Embeddings**: the gradient of a table row is the sum of the row gradients with that id, added in word order with plain additions on one thread. **Row 0 of `char_emb` (the padding id) receives no gradient** (PyTorch's `padding_idx`). Position tables likewise add in word order.
- **Character convolution**: taps ordered (tap, channel) as in the runtime layout; its weight gradient is mapped back to the container order `(out, channel, tap)` only when a model is written.

## Losses

Cross-entropy in double precision with the library's own exponential and logarithm: `Σ w[t] · nll / Σ w[t]` over words whose target is not negative (PyTorch `reduction='mean'`, `ignore_index`, `weight`). A negative target contributes neither loss nor gradient; no counted word gives loss 0 and zero gradients. The gradient `scale · w[t] / Σ w · (softmax - onehot)` is rounded once to binary32. Terms: tag (BIO over `1 + 2 · classes` tags), role, document type (one per page), item pointer and key–value pointer (a word points at its anchor word; targets are word indices). The loss of an example is the weighted sum of its terms; the batch gradient is the sum over examples in example order, each example scaled by `1 / batch size`. Examples may run in parallel, each into its own gradient set; the sets are added in example order, so no reduction crosses threads and the result does not depend on the degree.

## Parameters and optimizer

Trainable tensors have a fixed order: `char_emb`, `char_conv` (weight, bias), `char_proj`, the four position tables, `feat_proj`, `emb_ln`, per block `ln1 qkv proj ln2 fc1 fc2` (weight, bias each), `pair1`, `pair2`, `final_ln`, `tag_head`, `role_head`, `doc_head`, `rel_q`, `rel_k`. AdamW follows backward-numerics.md with one addition: a per-tensor learning-rate scale (the tensor's step uses `scale · lr`). Weight decay applies to weight matrices and tables, not to biases or layer-norm parameters. Optional global-norm clipping uses the 16-lane dot of each gradient with itself.

A head-only run (`BackboneScale` 0) trains the tag head and its bias only; every other tensor and every other loss term is off. With a positive scale every tensor the active losses reach trains at `scale ×` the head's rate. The role, document and pointer heads train only when their own loss is on.

## Tolerances

Against PyTorch fp32 reference gradients (fixtures from `training/parity/docextract_backward.py`, seed 31, the random-7 model), per element `|err| ≤ atol + rtol · |ref|` with the tolerance chosen against a float64 reference as in numerics.md: PyTorch's own fp32 error is measured, and the C# error must stay within 2× of it plus the absolute floor.

| layer | atol | rtol |
|---|---|---|
| layer normalization | 2e-7 | 1e-6 |
| GELU | 2e-6 | 1e-5 |
| attention (scores, softmax, weighted sum) | 3e-7 | 2e-6 |
| pair MLP | 1e-6 | 5e-6 |
| pointer scores | 3e-7 | 2e-6 |
| cross-entropy | 2e-7 | 1e-6 |
| whole model, every parameter | 1e-6 | 1e-5 |

Measured worst use of the allowance is at most 0.14 for every layer except GELU (0.46, from the approximate `tanh`). C# to C# the forward values, loss, gradients and updated weights are bit-identical on every kernel path and at degrees 1 and 4.

## Fine-tuning a custom schema

See "Fine-tuning a custom schema" in [docextract-v0.md](docextract-v0.md) for the container, and the label rules below.

- **Labels.** The labelling model is asked for the schema's fields (each nullable). Only values the grounding step reports as grounded (exactly or after normalization) become labels; a value computed rather than printed is not labelled; a page with a value that is not on the page is excluded from training and evaluation. The words a value was read from, if they form one run of consecutive words, are tagged `B` for the first and `I` for the rest of the field's class; every other word is `O`. A value whose words are not one run, or a word claimed by two fields, is ignored by the loss (target −1) and not scored.
- **Split.** Usable pages are ordered by `SHA-256("<seed>:<id>")` (first 8 bytes, little endian), then id; the first `max(1, round(n · fraction))` (at most `n − 1`) are held out.
- **Batches.** Page order per epoch is a SplitMix64 permutation seeded with `(seed << 32) xor epoch`; step `s` takes `batch` consecutive positions of the stream. Learning rate: linear warm-up to the peak over `warmup` steps, then linear decay to 10% of the peak over the remaining steps.
- **Class weights.** Tag `O` weighs 1, every `B-`/`I-` tag the field weight (default 3), against the dominance of `O`.
- **Resume.** The checkpoint holds the step, the trainable weights and the AdamW moments, tied by a SHA-256 fingerprint to the starting model, the schema, the settings and every training label file; a continued run ends with the model bytes of an uninterrupted one.
- **Scores.** Per field and held-out page: correct when the model's field has exactly the label's words (no value on either side also counts as correct); recall counts labelled pages only; agreement counts pages where either side has a value. The suggested `LocalFirst` threshold is the lowest field confidence from which at least the stated share (default 0.99) of the tuned model's values were exactly right, with at least 5 values.
