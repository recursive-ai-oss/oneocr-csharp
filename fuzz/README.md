# Fuzzing

`fuzz/OneOcr.Fuzz` holds one target per input surface. Today: `container` (model container bytes through the parser, the recognizer builder and one forward pass, under tight `OcrLimits`; only `OcrModelException` and `OcrLimitException` may come out). `fuzz/corpus/container/` holds the seed corpus (small valid containers written by `OneOcr.Fuzz seeds <dir>`).

    OneOcr.Fuzz                         run under libFuzzer (SharpFuzz); the target comes from ONEOCR_FUZZ_TARGET
    OneOcr.Fuzz replay <file>...        run saved inputs once each
    OneOcr.Fuzz mutate <seconds> <dir> [seed]
                                        in-process mutation loop without coverage feedback, for a quick local run

For a coverage-guided run, instrument `OneOcr.dll` with the `sharpfuzz` tool and start the program through SharpFuzz's `libfuzzer-dotnet` host (built with `clang++ -fsanitize=fuzzer`), with this corpus as the seed directory.

A minimized reproducer of a finding goes to `testdata/` with a regression test asserting a typed error.
