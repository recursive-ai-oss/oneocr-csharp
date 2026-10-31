# Fuzzing

`fuzz/OneOcr.Fuzz` is one program with a target per input surface. A target feeds arbitrary bytes to the code that reads them; only the
typed exceptions the API promises may come out (`OcrFormatException`, `OcrLimitException`, `OcrModelException`, `OcrProviderException`,
`ArgumentException` for schemas, OneImage's `ImageException` for decoding), and whatever parses must satisfy the invariants named below.
Anything else, a timeout (20 s) or an out-of-memory (2 GB heap) is a finding.

| target | input | also checked |
|---|---|---|
| `container` | model container v0/v1 (recognizer, and the `dbnet-lite-v0/v1` detector seeds) | parser, builder, one forward pass, word splitting, detector decode |
| `docextract` | document-extraction container | parser, builder, featurization, forward pass, assembly |
| `charlm`, `lexicon`, `wordfreq` | character model, lexicon, word frequency files | beam search with the model; lookups |
| `pagejson` | `oneocr.page` JSON | parsed pages write back to the same bytes |
| `review` | review-import JSON | limits, no control characters |
| `llm-anthropic`, `llm-openai`, `llm-gemini` | provider response bytes | answer is a bounded JSON object |
| `llm-schema` | JSON Schema text | per-provider rewrites |
| `llm-answer` | a model's answer text through the whole read path (first byte picks the outputs) | alignment, grounding, re-ask; returned pages round-trip |
| `training-files` | label, reviewed-lines, extraction-label and checkpoint files (first byte picks the file) | the one `InvalidOperationException` allowed is "checkpoint of another run" |
| `pdf` | PDF bytes through the PDF path | page JSON round trip |
| `image` | image bytes through OneImage and the page path with random small models | page JSON round trip |
| `image-real` | the same with the shipped models (needs the model files at build time) | page JSON round trip |
| `density` | image file headers (PNG, JPEG, TIFF) | never throws; an answer is finite and in 10..10000 dpi; deterministic |
| `normalization` | UTF-16 code units, lone surrogates included | NFC/NFKC idempotence, amount/date/currency parsing, bounded edit distance |

Seeds live in `fuzz/corpus/<target>/`; `ONEOCR_FUZZ_TARGET=<target> OneOcr.Fuzz seeds <dir>` writes them. Minimized reproducers of
findings are in `testdata/fuzz/` with a test in `FuzzRegressionTests`.

    OneOcr.Fuzz                      run under libFuzzer (SharpFuzz); the target comes from ONEOCR_FUZZ_TARGET
    OneOcr.Fuzz replay <file>...     run saved inputs once each
    OneOcr.Fuzz mutate <s> <dir>     in-process mutation loop without coverage feedback, for a quick local run
    OneOcr.Fuzz sweep --degree N --timeout S [--out F] PATH...
                                     no-crash/no-hang sweep: the bundled engine over every image, PDF and JSON file under PATH

## Coverage-guided campaign

Needs Linux x64, bash, curl, tar, xz and a .NET 10 SDK; installs nothing system-wide.

    export FUZZ_TOOLS=$HOME/oneocr-fuzz-tools   # clang 20.1.8 (for libFuzzer's runtime), libfuzzer-dotnet, sharpfuzz; about 11 GB
    export FUZZ_OUT=$HOME/oneocr-fuzz-out       # builds, grown corpora, findings, logs; keep it outside the checkout
    export FUZZ_BUILD_ARGS="-p:OneOcrModelFile=... -p:OneOcrDetectorFile=... -p:OneOcrLanguageModelFile=... -p:OneOcrWordFrequencyFile=..."
    fuzz/setup-tools.sh                          # once; downloads are pinned and SHA-256 checked
    FUZZ_SECONDS=600 fuzz/run-campaign.sh run    # every target, or name some

`build` instruments `OneOcr`, `OneOcr.Llm`, `OneOcr.Training`, `OneOcr.Pdf` and the OnePDF and OneImage assemblies they call. With
`FUZZ_NO_BUILD=1`, several `run <target>` processes can share one build. Per target `$FUZZ_OUT/<target>/` gets `corpus/`, `min/`
(same coverage, fewest inputs), `artifacts/` (`crash-*`, `timeout-*`, `oom-*`) and `run.log`; `$FUZZ_OUT/report.txt` has a row with
executions, libFuzzer features (the coverage measure with SharpFuzz), corpus sizes and findings. A finding does not stop the run.
Run it away from builds and benchmarks; each target uses one core.

    fuzz/run-campaign.sh repro <target> <artifact>   replay a finding through the instrumented build
    fuzz/run-campaign.sh promote [targets...]        add the minimized inputs to fuzz/corpus (smallest first, 256 KB per target)

For every finding: minimize it, fix it in safe code with a typed exception, add the reproducer to `testdata/fuzz/` with a regression
test, `promote`, and re-run the target.
