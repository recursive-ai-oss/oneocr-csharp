# Model container, version 1

Status: current. Version 1 is version 0 (`model-container-v0.md`) with format version `1` and one added manifest member, `calibration`. Writers produce version 1; readers read both. A version 0 file loads and has no calibration: its calibrated word confidence is its raw confidence, and the reader reports a diagnostic saying so. Everything else in version 0 stands and is repeated here so this document is complete.

The container is a cross-language contract: one weights file serves every implementation. Everything below is little-endian.

## File layout

| Offset | Size | Content |
|---|---|---|
| 0 | 8 | Magic: ASCII `ONEOCRM` followed by one zero byte |
| 8 | 4 | `uint32` format version, `1` (a reader also accepts `0`; any other value is an error) |
| 12 | 4 | `uint32` manifest length *M* in bytes |
| 16 | *M* | Manifest: UTF-8 JSON, no BOM |
| 16 + *M* | pad | Zero bytes up to the next multiple of 64 — the start of the data region *D* |
| *D* | … | Tensor data, each tensor starting at a multiple of 64 from *D*, zero padding between tensors |

A reader rejects the file — before allocating anything sized by it — when the magic or version is wrong, the manifest is not valid UTF-8 JSON, or any length, offset, shape or count exceeds the reader's limits or the file's size. Tensors may not overlap. All padding bytes (after the manifest and between tensors) must be zero. The file ends exactly where the last tensor ends: no trailing padding, and bytes after the last tensor are an error. Tensors need not be listed in offset order.

The manifest is strict JSON: no comments, no trailing commas, no duplicate member names. Members this version does not name are ignored (`provenance` among them). In a version 0 file `calibration` is such a member and is ignored, whatever it holds. Sizes, offsets and shapes are JSON integers.

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
  "calibration": {
    "raw": "word-mean-max-run",
    "scripts": {
      "Latin": { "x": [0.0, 0.5, 1.0], "y": [0.0, 0.4, 0.95] },
      "Greek": { "x": [0.0, 1.0], "y": [0.0, 1.0] }
    }
  },
  "provenance": { }
}
```

- `architecture` names the layer graph (below); a reader rejects any it doesn't know.
- `input`: line crops are grayscale, scaled to `height` keeping aspect ratio; a pixel `p` in 0–255 becomes `x = p / 255` (`1 - p / 255` when `invert`), then `(x - mean) / std`, all in float32 with a correctly rounded division. `channels` is 1 for `crnn-v0`; `std` is positive.
- `charset`: the output classes in order. Index 0 is the CTC blank; its text is a placeholder (the exporter writes `<blank>`) and is never output. Each other entry is a non-empty NFC string, usually one code point; entries, the blank placeholder included, are unique. (Readers do not check NFC; they check emptiness, uniqueness and length.)
- `tensors`: `dtype` is `f16` (IEEE 754 binary16) in versions 0 and 1; values are widened to `float32` for computation. `offset` is relative to *D*; `length` equals the element count times 2. Tensors are stored row-major in PyTorch's own dimension order.
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
- **Decoding:** greedy CTC: take the arg-max class per timestep (lowest index on ties), merge repeats, drop blanks, concatenate charset entries. Confidences are defined under *Confidence* below.

## Calibration (version 1)

A recognizer says how sure it is per character; users want the probability that a **word** is exactly right. The model file carries the map from one to the other, fitted by the trainer on a held-out calibration split.

**Character probability.** The network's output at a step is softmax over the classes. A character is a *run*: a maximal sequence of consecutive steps whose arg-max is the same non-blank class (a blank step ends a run, so "ll" is two runs). Its probability is the largest softmax probability of that class over the run's steps.

**Raw word confidence** (`"raw": "word-mean-max-run"`, the only value defined; any other is an error): the words of a line are the maximal sequences of runs whose characters are not whitespace. A character is whitespace when its charset entry is non-empty and consists only of whitespace code units (`char.IsWhiteSpace`-style: U+0020, U+00A0, U+2003 and the other Unicode White_Space characters). Spaces are not part of any word; a line with no non-space run has no words. The raw confidence of a word is the **mean** of its runs' probabilities, summed left to right in float32 and divided by the number of runs. A whole line's raw confidence (`RecognizedLine.Confidence`) uses the same rule over every run of the line, spaces included; it is 0 for an empty line.

**Script of a word.** Count the word's letters (Unicode categories L*, by code point) in three groups: Greek (U+0370–03FF, U+1F00–1FFF), Cyrillic (U+0400–052F, U+1C80–1C8F, U+2DE0–2DFF, U+A640–A69F), Latin (every other letter). The script with the most letters wins; ties go to the first of Latin, Greek, Cyrillic. A word with no letters (digits, punctuation, symbols) is Latin.

**Calibrated word confidence.** The manifest member:

```json
"calibration": { "raw": "word-mean-max-run", "scripts": { "Latin": { "x": [...], "y": [...] }, "Greek": {...}, "Cyrillic": {...} } }
```

* `scripts` is an object whose names are exactly `Latin`, `Greek` or `Cyrillic` (case-sensitive, no duplicates). Any other name is an error. A script may be missing; its calibrated confidence is then its raw confidence. `scripts` may be empty.
* Each script has `x` and `y`, JSON arrays of numbers of equal length, between 1 and 64 knots (more is a limit error, `MaxCalibrationKnots`).
* `x` is strictly increasing and every value is in [0, 1]; `y` is non-decreasing and every value is in [0, 1]. Numbers are IEEE doubles.
* A version 1 file must have the `calibration` object; the reader rejects a file without it.
* The map is piecewise linear through the knots `(x[i], y[i])` and constant outside them: for `r <= x[0]` the value is `y[0]`, for `r >= x[last]` it is `y[last]`. For `x[i] < r < x[i+1]`, in IEEE double precision with each operation rounded on its own (no fused multiply-add):

  `t = (r - x[i]) / (x[i+1] - x[i])`, `p = (y[i+1] - y[i]) * t`, `value = y[i] + p`

  and the result is converted to float32. A one-knot curve is the constant `y[0]`. `r` is the float32 raw confidence converted to double.

**Line confidence** in the result type is the calibrated confidences of the line's words averaged with each word weighted by its number of runs (double precision, converted to float32); 0 for a line with no words. This is a reader convention for reporting, not part of the file.

## Numerics

Weights are stored in fp16 and computed in fp32. Implementations agree with the PyTorch reference within a documented per-layer tolerance; bit-exact agreement is only required between the C# library's own code paths. The order of floating-point operations that makes those paths bit-identical is written down in [numerics.md](numerics.md).
