# oneocr

Free, fast, local OCR and document understanding for .NET, written from first principles with no runtime dependencies beyond the .NET base class library. Small detection, layout and recognition models, trained by us and run by our own deterministic CPU kernels.

Work in progress; not yet released.

- `src/` — the libraries (`RecursiveAI.OneOcr` and companions).
- `training/` — the public training pipeline (data generation, PyTorch training, export), so the shipped models can be reproduced.
- `docs/spec/` — the versioned formats other implementations can rely on.

## License

Code and model weights are licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option. © Recursive AI, LLC.
