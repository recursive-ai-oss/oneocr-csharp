# Fuzzing

`fuzz/OneOcr.Fuzz` holds one target per input surface. `container` (model container bytes, version 0 and 1 with calibration, through the parser, the recognizer builder, one forward pass and word splitting, under tight `OcrLimits`; only `OcrModelException` and `OcrLimitException` may come out). For the page detector architectures (`dbnet-lite-v0` and `dbnet-lite-v1`) the target also builds the detector, runs it on a 32 by 64 tile and decodes the maps; its seeds are the `dbnet-*.seed` files (the v1 ones: stride-16 branch with two-scale orientation, orientation head only, branch only). `fuzz/corpus/container/` holds the seed corpus (small valid containers written by `OneOcr.Fuzz seeds <dir>`).

Three more targets, chosen with `ONEOCR_FUZZ_TARGET`: `charlm` (character language model bytes through the parser and, when they parse, a beam search with the model attached) `lexicon` (lexicon file bytes) and `pagejson` (`oneocr.page` JSON through the reader under tight limits; whatever parses must write and read back to the same bytes). Their seeds are in `fuzz/corpus/charlm/`, `fuzz/corpus/lexicon/` and `fuzz/corpus/pagejson/`, written by `ONEOCR_FUZZ_TARGET=charlm OneOcr.Fuzz seeds <dir>`.

    OneOcr.Fuzz                         run under libFuzzer (SharpFuzz); the target comes from ONEOCR_FUZZ_TARGET
    OneOcr.Fuzz replay <file>...        run saved inputs once each
    OneOcr.Fuzz mutate <seconds> <dir> [seed]
                                        in-process mutation loop without coverage feedback, for a quick local run

For a coverage-guided run, instrument `OneOcr.dll` with the `sharpfuzz` tool and start the program through SharpFuzz's `libfuzzer-dotnet` host (built with `clang++ -fsanitize=fuzzer`), with this corpus as the seed directory.

`llm-anthropic`, `llm-openai` and `llm-gemini` run provider response bytes through that provider's parser (Azure OpenAI and OpenAI-compatible servers share the OpenAI parser); `llm-schema` runs JSON Schema text through the schema checks and the per-provider rewrites. Only `OcrProviderException`, `OcrLimitException` (and `ArgumentException` for schemas) may come out. Seeds are in `fuzz/corpus/llm-*/`, written by `ONEOCR_FUZZ_TARGET=llm-openai OneOcr.Fuzz seeds <dir>`. The first mutation run found that a string read out of invalid UTF-8 threw `InvalidOperationException`; response bodies are now checked for valid UTF-8 first.

`review` runs a review-import file through the parser (`docs/spec/review-queue.md`) under tight limits; only `OcrFormatException` and `OcrLimitException` may come out. Seeds are in `fuzz/corpus/review/`. The first mutation run found that invalid UTF-8 inside a string threw `InvalidOperationException`; the whole input is now checked for valid UTF-8 first.

`pdf` runs PDF bytes through the whole PDF path (`OneOcr.Pdf`: parsing, text-layer conversion, image location, a low-resolution render and recognition of the regions) under tight limits; only the typed exceptions may come out, and whatever is read must write to page JSON that reads back to the same bytes. Seeds are in `fuzz/corpus/pdf/`, written by `ONEOCR_FUZZ_TARGET=pdf OneOcr.Fuzz seeds <dir>`.

`wordfreq` runs word frequency file bytes (`docs/spec/word-lm-v1.md`) through the parser and, when they parse, lookups of arbitrary byte strings under tight limits; only `OcrModelException` and `OcrLimitException` may come out. Seeds are in `fuzz/corpus/wordfreq/`, written by `ONEOCR_FUZZ_TARGET=wordfreq OneOcr.Fuzz seeds <dir>`.

A minimized reproducer of a finding gets a regression test asserting a typed error.

`docextract` runs document-extraction model container bytes (`docextract-v0`) through the parser, the model builder and, when they build, featurization, a forward pass and assembly on a fixed seven-word page under tight limits; only `OcrModelException` and `OcrLimitException` may come out. Seeds are in `fuzz/corpus/docextract/`, written by `ONEOCR_FUZZ_TARGET=docextract OneOcr.Fuzz seeds <dir>`. A 600 s mutation run executed 31.5 million inputs, 0.8 million of them parsed and built, with no findings.
