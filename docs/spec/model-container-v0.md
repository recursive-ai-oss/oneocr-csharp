# Model container, version 0

Status: **provisional**. Version 0 exists to prove the pipeline end to end: one recognizer, written by the PyTorch exporter in `training/` and read by the C# loader. It will be replaced by version 1 before the first release; files of version 0 are not promised to load after that.

The container is a cross-language contract: one weights file serves every implementation. Everything below is little-endian.

## File layout

| Offset | Size | Content |
|---|---|---|
| 0 | 8 | Magic: ASCII `ONEOCRM` followed by one zero byte |
| 8 | 4 | `uint32` format version, `0` |
| 12 | 4 | `uint32` manifest length *M* in bytes |
| 16 | *M* | Manifest: UTF-8 JSON, no BOM |
| 16 + *M* | pad | Zero bytes up to the next multiple of 64 — the start of the data region *D* |
| *D* | … | Tensor data, each tensor starting at a multiple of 64 from *D*, zero padding between tensors |

A reader rejects the file — before allocating anything sized by it — when the magic or version is wrong, the manifest is not valid UTF-8 JSON, or any length, offset, shape or count exceeds the reader's limits or the file's size. Tensors may not overlap. All padding bytes (after the manifest and between tensors) must be zero. The file ends exactly where the last tensor ends: no trailing padding, and bytes after the last tensor are an error. Tensors need not be listed in offset order.

The manifest is strict JSON: no comments, no trailing commas, no duplicate member names. Members this version does not name are ignored (`provenance` among them). Sizes, offsets and shapes are JSON integers.

## Manifest

```json
{
  "architecture": "crnn-v0",
  "name": "line-recognizer",
  "version": "0.1.0",
  "license": "MIT OR Apache-2.0",
  "input": { "height": 32, "channels": 1, "mean": 0.5, "std": 0.5, "invert": false },
  "charset": ["<blank>", " ", "!", "..."],
  "config": { },
  "tensors": [
    { "name": "conv.0.weight", "dtype": "f16", "shape": [64, 1, 3, 3], "offset": 0, "length": 1152 }
  ],
  "provenance": { }
}
```

- `architecture` names the layer graph (below); a reader rejects any it doesn't know.
- `input`: line crops are grayscale, scaled to `height` keeping aspect ratio; a pixel `p` in 0–255 becomes `x = p / 255` (`1 - p / 255` when `invert`), then `(x - mean) / std`, all in float32 with a correctly rounded division. `channels` is 1 for `crnn-v0`; `std` is positive.
- `charset`: the output classes in order. Index 0 is the CTC blank; its text is a placeholder (the exporter writes `<blank>`) and is never output. Each other entry is a non-empty NFC string, usually one code point; entries, the blank placeholder included, are unique. (Readers do not check NFC; they check emptiness, uniqueness and length.)
- `tensors`: `dtype` is `f16` (IEEE 754 binary16) in version 0; values are widened to `float32` for computation. `offset` is relative to *D*; `length` equals the element count times 2. Tensors are stored row-major in PyTorch's own dimension order.
- `provenance`: free-form record of the training run (commit, config, seed, data manifest hashes). Readers ignore it.

## Architecture `crnn-v0`

A convolutional stack, then a bidirectional LSTM, then a per-timestep classifier, decoded by greedy CTC. `config`:

```json
{
  "conv": [
    { "out": 64, "kernel": [3, 3], "padding": [1, 1], "pool": [2, 2] },
    { "out": 128, "kernel": [3, 3], "padding": [1, 1], "pool": null }
  ],
  "lstm": { "hidden": 256, "layers": 2 },
  "classes": 700
}
```

- `config.conv[i]`: `out` channels, `kernel`, `padding` and `pool` (an array of two integers `[h, w]`, or `null`, or absent for no pooling). Kernel and pool sizes are at least 1, padding at least 0. `config.classes` equals the charset length.
- **Conv block *i*** (`conv.<i>`): 2-D convolution, stride 1, the given kernel `[h, w]` and zero padding `[h, w]`, with bias; batch normalization is folded into the weight and bias at export. Then ReLU. Then, when `pool` is `[ph, pw]`, max pooling with kernel and stride `[ph, pw]` and no padding (a trailing partial window is dropped; the block's input has `in` channels, the previous block's `out`, or `input.channels` for the first). A line image too narrow to leave at least one column after every block reads as empty text. Tensors: `conv.<i>.weight` `[out, in, kh, kw]`, `conv.<i>.bias` `[out]`.
- **To sequence:** the stack's output height must be exactly 1 for the configured input height (a reader checks this). Shape `[C, 1, W]` becomes a sequence of `W` vectors of size `C`, left to right.
- **BiLSTM** (`lstm`): `layers` stacked bidirectional layers; each layer's input is the previous layer's forward and reverse outputs concatenated (forward first). Gate order and equations are PyTorch's `nn.LSTM` (input, forget, cell, output; `c' = f*c + i*g`, `h' = o*tanh(c')`), zero initial state. Tensors use PyTorch's names under `lstm.`: `lstm.weight_ih_l<k>`, `lstm.weight_hh_l<k>`, `lstm.bias_ih_l<k>`, `lstm.bias_hh_l<k>`, and the same with suffix `_reverse`.
- **Classifier** (`fc`): linear layer `2 * hidden → classes` with bias: `fc.weight` `[classes, 2 * hidden]`, `fc.bias` `[classes]`. `classes` equals the charset length. A reader ignores tensors it does not need.
- **Decoding:** greedy CTC: take the arg-max class per timestep (lowest index on ties), merge repeats, drop blanks, concatenate charset entries. The confidence of a line is computed from the per-timestep softmax probabilities; its exact definition is the implementation's until version 1. The C# reader's: for each emitted character (a run of equal non-blank arg-max steps), the largest softmax probability of its class over the run; the line confidence is the mean of those, and 0 for an empty line.

## Numerics

Weights are stored in fp16 and computed in fp32. Implementations agree with the PyTorch reference within a documented per-layer tolerance; bit-exact agreement is only required between the C# library's own code paths. The order of floating-point operations that makes those paths bit-identical is written down in [numerics.md](numerics.md).
