# bench — clean-page OCR baseline

A small evaluation set of clean A4 pages at 300 DPI with exact truth, an engine-agnostic scorer, and a runner
that measures Tesseract (`tessdata_best` and `tessdata_fast`, stock English models) as a black box.

Everything is generated or fetched by scripts; no images, models or outputs are committed — only the manifests
in `manifests/` (SHA-256 of every page image and truth file, and of every font and text source).

## The page set `clean300-v1`

30 pages, 2480 x 3508 px, 8-bit grayscale PNG, one column, Latin script. HTML is rendered by headless Chromium
(Playwright); the truth is the browser's own layout: one rectangle per word from a DOM Range, in document order,
so no OCR is involved. Page text = blocks in reading order, words separated by spaces, blocks by newlines.
A word that wraps at a line-end hyphen (`pre-possessed`) has one box spanning both lines.

Content per page: a heading or two, paragraphs of public-domain prose, a paragraph of generated numbers, dates and
amounts, and a short numbered list. Fonts rotate over seven held-out families:
Gelasio, Carlito, Caladea, Source Serif 4, Source Sans 3, IBM Plex Serif, IBM Plex Sans (all SIL OFL 1.1; licence
files and hashes in `manifests/assets.json`). Prose is sampled from Project Gutenberg #1342 and #11 (public domain
in the US; only the book text between the START/END markers is used, so the Gutenberg trademark licence does not apply).
Size, line height, margins, justification and page fill vary per page from the seed.

Outputs per page: `pages/pNNN.png`, `truth/pNNN.json` (words with boxes, font, layout parameters),
`truth/pNNN.txt` (what the scorer reads).

## Commands

All of them run from `bench/` with `uv`. `ONEOCR_BENCH_DIR` (default `bench/.data`) is where data and results go;
`PLAYWRIGHT_BROWSERS_PATH` can point the Chromium download somewhere persistent.

```
uv sync --frozen
uv run playwright install chromium
uv run python -m oneocr_bench fetch            # fonts + text, verified against manifests/assets.json
uv run python -m oneocr_bench gen              # render the set from seed 20261023
uv run python -m oneocr_bench gen --check      # same, then compare every hash with manifests/clean300-v1.json
uv run python -m oneocr_bench draw p001        # truth boxes over a page, to look at
uv run python -m oneocr_bench score --hyp DIR  # score DIR/<page-id>.txt for any engine
```

The page hashes are tied to the pinned Chromium build (recorded in the manifest) and the fonts; a different
build can shift anti-aliasing and so the PNG hash, never the truth text.

### Tesseract baselines

```
bash scripts/install_tesseract.sh              # distro packages via apt-get download + dpkg -x, no root
uv run python -m oneocr_bench tess --psms 3 4 6
```

`install_tesseract.sh` puts the binary under `~/tools/tesseract` and downloads `eng.traineddata` from the official
`tessdata_best` and `tessdata_fast` repositories, recording versions and SHA-256 in `versions.txt`.
`tess` runs every page as its own process, one at a time, for each model and page-segmentation mode, once with
`OMP_THREAD_LIMIT=1` and once with default threading; it reports wall seconds per page (process start and model
load included), peak RSS, and CER / WER / reading-order error, and marks the best mode per model.
Timing runs belong on an otherwise idle machine.

## Scoring

Text is normalised the same way on both sides: Unicode NFC, typographic quotes and dashes folded to ASCII,
`ﬁ`/`ﬂ` split, whitespace collapsed; case and punctuation are kept. Metrics are micro-averaged over the pages.

- **CER / WER** are order-free: words are a bag; identical words cancel, each leftover truth word is paired with the
  closest leftover hypothesis word, the rest are deletions or insertions. A word split or merged by an engine costs
  a little more than in an ordered alignment. Details in `oneocr_bench/score.py`.
- **Reading-order error**: word-sequence edit distance divided by truth words. It also contains recognition
  errors, so it is read together with WER: order error close to WER means the order is right.

A missing hypothesis file scores as empty output.
