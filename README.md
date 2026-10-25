# oneocr

Free, fast, local OCR and document understanding for .NET, written from first principles with no runtime dependencies beyond the .NET base class library. Small detection, layout and recognition models, trained by us and run by our own deterministic CPU kernels.

Work in progress; not yet released.

- `src/` — the libraries (`RecursiveAI.OneOcr` and companions).
- `training/` — the public training pipeline (data generation, PyTorch training, export), so the shipped models can be reproduced.
- `docs/spec/` — the versioned formats other implementations can rely on.

## Publishing with NativeAOT

A NativeAOT binary is compiled for one instruction-set level, and .NET's default for x64 (`x86-64-v2`) has no AVX2 or FMA. At that level
the OCR kernels fall back to their exact scalar path: the text is the same, but a page takes about a hundred times longer. The
`RecursiveAI.OneOcr` package therefore sets `IlcInstructionSet` to `x86-64-v3` (AVX2 + FMA, any x86-64 CPU since about 2015) for x64
NativeAOT publishes whose project has not chosen a level; a value in your project or on the command line always wins. Use
`x86-64-v4` to get the AVX-512 kernels on machines that have them. A binary built for v3 refuses to start on an older CPU. Arm needs nothing.

To see what a process runs, read `OcrEngine.KernelPath`. `OcrEngine.PerformanceNotice` is non-null when the scalar path is running on a CPU
that could run AVX2, and says how to fix it (the usual causes are an app published below v3, or `DOTNET_EnableAVX2=0` /
`DOTNET_EnableHWIntrinsic=0` in the environment). It never throws and is not part of the page, so page JSON is the same on every path.

## License

Code and model weights are licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option. © Recursive AI, LLC.
