# bench — OCR evaluation sets and baselines

Six small evaluation sets with exact truth, one per quality category, an engine-agnostic scorer, and a runner that
measures Tesseract (`tessdata_best` and `tessdata_fast`, stock English models) as a black box.

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

## The sets

| set | category | pages | what |
|---|---|---|---|
| `clean300-v1` | clean 300 DPI print | 30 | the page set described above |
| `degraded-v1` | degraded scans | 40 | clean-style pages (new seeds and texts), then scan degradation |
| `screenshot-v1` | screenshots and UI | 40 | UI and web screens at 1x and 2x device scale |
| `layout-v1` | complex layout | 40 | columns, tables, forms |
| `phone-v1` | phone photos | 40 | clean-style pages, then simulated camera capture |
| `receipts-v1` | receipts | 100 | CORD v2 test split |

Every generated set is reproducible from its seed (in `oneocr_bench/sets.py`) and the pinned fonts, text and Chromium
build; `manifests/<set>.json` holds the SHA-256 of every page image and truth file. Truth is never produced by an OCR
engine: it is browser layout geometry (DOM ranges), the geometry of our own transforms, or a dataset's shipped
annotation. Pages that are JPEG on purpose (`degraded`, `phone`) are written by Pillow at the recorded quality, so their
hashes also depend on the pinned Pillow and NumPy versions (recorded in each manifest).

Truth files per page: `truth/<id>.json` (words with `box` = [x0, y0, x1, y1] in image pixels, a `block` index, a
`quad` where the page was transformed, the generation parameters), `truth/<id>.txt` (words in reading order, a new line
when the block changes; this is what the scorer reads). A word that wraps at a hyphen or em dash has its box on the first
line and `"split": true`.

- **degraded-v1** is rendered like `clean300-v1`, then, per page and from the page's own seed: skew up to 2 degrees,
  faint ink (ink factor 0.3 to 1), greyish paper, downsampling to 150 or 200 DPI, blur up to 0.8 px, Gaussian noise and
  speckle, JPEG quality 30 to 85. Parameters are in `truth/<id>.json` under `degradation`; truth quads and boxes are moved
  with the image by the same homography.
- **phone-v1** places a page on a desk-coloured background in a 2016 x 2688 canvas with perspective (corner jitter up to
  4.5 %, rotation up to 5 degrees), tints the paper, then applies vignette, a lighting gradient and sometimes a soft
  shadow band, Gaussian blur 0.6 to 2.2 px, sensor noise and JPEG 55 to 88. Quads are the transformed word corners; `box` is
  their axis-aligned box.
- **screenshot-v1** composes an app bar, an optional sidebar and one content template (form, table, settings list, dialog,
  dashboard, web article) in a light, dark, sepia or blue theme at 12 to 16 px, in the seven held-out families, at 1x for
  even pages and 2x for odd pages. Reading order is document order: app bar left to right, sidebar top to bottom, then
  the content top to bottom and left to right (forms label, value; tables row by row). A screen with text outside the
  viewport is redrawn. No system UI font is used.
- **layout-v1** pages are two-column articles, three-column newsletters, table pages, forms and mixed pages. Reading
  order is the document order: title, byline and abstract; then column by column, each top to bottom; tables row by row,
  cell by cell, header first; forms by section, row by row, label before value, a checkbox's label after its box.
- **receipts-v1** is the CORD v2 test split (100 receipt photos) with CORD's own word texts and quads as truth.
  Reading order: words sharing a `row_id` form a row; rows are sorted top to bottom by the mean y of their word
  centres; words inside a row go left to right. Empty words are dropped; `box` is the axis-aligned box around the quad.
  Each page is checked: box inside the image, non-empty text, annotation size equal to image size.

**CORD attribution (CC BY 4.0):** Park et al., "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing", Document
Intelligence Workshop at NeurIPS 2019, https://github.com/clovaai/cord, licensed under
https://creativecommons.org/licenses/by/4.0/. The images are fetched by `training/datasets/fetch_cord.py` and copied
into the set directory; they are not redistributed here.

**CORD truth is partial.** CORD publishes only part of each receipt (store and payment header blocks were removed, and
some regions are marked as not text). An engine that reads the whole receipt produces words that are not in the truth.
For this set read the recall scores (errors on truth words only) next to CER/WER, and treat detection precision as a
lower bound.

Fonts for every generated set are the seven held-out families only.

## Commands

All of them run from `bench/` with `uv`. `ONEOCR_BENCH_DIR` (default `bench/.data`) is where data and results go;
`PLAYWRIGHT_BROWSERS_PATH` can point the Chromium download somewhere persistent; `ONEOCR_DATASETS` (default
`~/oneocr/datasets`) is where the CORD fetch put its data.

```
uv sync --frozen
uv run playwright install chromium
uv run python -m oneocr_bench fetch                       # fonts + text, verified against manifests/assets.json
uv run python -m oneocr_bench gen --set degraded-v1        # render one set (default clean300-v1); --set all for every set
uv run python -m oneocr_bench gen --set all --check        # render, then compare every hash with the manifests
uv run python -m oneocr_bench draw --set phone-v1 p001     # truth boxes (blue) and quads (red) over a page, to look at
uv run python -m oneocr_bench score --set degraded-v1 --hyp DIR      # one set; DIR/<page-id>.txt
uv run python -m oneocr_bench score --hyp-root ROOT                  # every set found as ROOT/<set>/, one table
```

The page hashes are tied to the pinned Chromium build, Pillow and NumPy (recorded in the manifests) and the fonts; a
different build can shift anti-aliasing and so an image hash, never the truth text.

## Scoring

Text is normalised the same way on both sides: Unicode NFC, typographic quotes and dashes folded to ASCII,
`ﬁ`/`ﬂ` split, whitespace collapsed; case and punctuation are kept. Metrics are micro-averaged over the pages.

- **CER / WER** are order-free: words are a bag; identical words cancel, each leftover truth word is paired with the
  closest leftover hypothesis word, the rest are deletions or insertions. A word split or merged by an engine costs
  a little more than in an ordered alignment. Details in `oneocr_bench/score.py`.
- **Reading-order error**: word-sequence edit distance divided by truth words. It also contains recognition
  errors, so it is read together with WER: order error close to WER means the order is right.

Besides CER/WER/order, `score` reports **recall-CER and recall-WER**: errors on the truth side only (a truth word that
is missing or misread; extra hypothesis words cost nothing). They matter where truth is partial (receipts).

**Word detection F1 at IoU 0.5.** If every page has a sidecar `<page-id>.json` next to its `<page-id>.txt`, the scorer also
reports detection precision, recall and F1. Sidecar format, coordinates in pixels of the page image as given in `pages/`:

```json
{"words": [{"box": [x0, y0, x1, y1]}, {"quad": [x1, y1, x2, y2, x3, y3, x4, y4], "text": "optional, ignored"}]}
```

Boxes are matched one to one, highest IoU first, text not compared; truth quads count as their axis-aligned boxes.
A missing sidecar on any page means no F1 for that run. A missing hypothesis file scores as empty output.

## Running Tesseract

```
bash scripts/install_tesseract.sh                          # distro packages via apt-get download + dpkg -x, no root
uv run python -m oneocr_bench tess --set degraded-v1 --psms 3 4 6 11
uv run python -m oneocr_bench tess-report                  # one table over every set that has results
bash scripts/tess_all.sh                                   # all sets in turn (optionally under a lock file, see the script)
```

`install_tesseract.sh` puts the binary under `~/tools/tesseract` and downloads `eng.traineddata` from the official
`tessdata_best` and `tessdata_fast` repositories, recording versions and SHA-256 in `versions.txt`.
`tess` runs every page as its own process, one at a time, with one thread, for each model and page-segmentation mode,
asking for text and TSV output with `-c tessedit_create_txt=1 -c tessedit_create_tsv=1` (the TSV word rows become the box sidecar). Wall seconds per page include
process start and model load; peak RSS is the child's. The best mode per model (lowest CER, then WER; lowest recall-CER for
`receipts-v1`) is run again with default threading for its timing, and the two texts are compared for equality. Results land in
`$ONEOCR_BENCH_DIR/results/<set>/` with `best-<model>/<set>` links that `score --hyp-root results/best-best` reads. Timing
runs belong on an otherwise idle machine; CPU, cores and load average before, during and after are in each `tesseract.json`.

## Tesseract 5.3.4 baselines (2026-10-23)

Stock `eng` models, one thread unless noted, AMD Ryzen 9 7950X3D (16 cores, 32 threads), Linux; load average 1.0 to 2.6
during the runs (that includes the benchmark's own busy thread; nothing else heavy ran, but this is not an isolated
benchmark host). "Default" is Tesseract's own threading. PSM per model is the one with the lowest CER (lowest recall-CER for
receipts); the second table gives the best PSM per metric separately. Pages: 30 / 40 / 40 / 40 / 40 / 100.

| category (set) | model | PSM | CER % | WER % | order % | recall-CER % | F1@0.5 | s/page 1 thread | s/page default | peak RSS MB (1t / default) | load1 before / max during / after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean 300 DPI print (clean300-v1) | best | 11 | 0.24 | 0.73 | 0.74 | 0.19 | 0.753 | 1.58 | 0.94 | 91 / 91 | 2.62 / 1.7 / 1.62 |
| clean 300 DPI print (clean300-v1) | fast | 6 | 0.26 | 0.83 | 0.85 | 0.22 | 0.753 | 0.76 | 0.64 | 72 / 72 | 2.62 / 1.7 / 1.62 |
| degraded scans (degraded-v1) | best | 3 | 0.91 | 2.55 | 2.75 | 0.84 | 0.722 | 1.11 | 0.68 | 66 / 66 | 1.62 / 1.92 / 1.92 |
| degraded scans (degraded-v1) | fast | 3 | 1.19 | 3.36 | 3.58 | 1.06 | 0.721 | 0.57 | 0.49 | 66 / 66 | 1.62 / 1.92 / 1.92 |
| screenshots and UI (screenshot-v1) | best | 11 | 9.94 | 14.38 | 32.18 | 9.74 | 0.681 | 0.42 | 0.29 | 95 / 95 | 1.92 / 1.96 / 1.96 |
| screenshots and UI (screenshot-v1) | fast | 11 | 8.39 | 12.52 | 30.42 | 8.27 | 0.687 | 0.24 | 0.23 | 86 / 86 | 1.92 / 1.96 / 1.96 |
| complex layout (layout-v1) | best | 3 | 12.64 | 11.10 | 12.50 | 12.42 | 0.684 | 1.78 | 1.09 | 92 / 92 | 1.96 / 1.93 / 1.84 |
| complex layout (layout-v1) | fast | 3 | 12.55 | 11.25 | 12.72 | 12.31 | 0.685 | 0.95 | 0.83 | 79 / 79 | 1.96 / 1.93 / 1.84 |
| phone photos (phone-v1) | best | 6 | 6.31 | 12.60 | 14.14 | 5.76 | 0.529 | 1.36 | 0.87 | 108 / 108 | 1.84 / 2.39 / 2.39 |
| phone photos (phone-v1) | fast | 6 | 6.22 | 12.57 | 14.13 | 5.68 | 0.529 | 0.78 | 0.68 | 94 / 93 | 1.84 / 2.39 / 2.39 |
| receipts (receipts-v1) | best | 6 | 137.29 | 298.75 | 302.13 | 35.31 | 0.231 | 0.63 | 0.43 | 196 / 194 | 2.39 / 2.33 / 2.33 |
| receipts (receipts-v1) | fast | 6 | 163.60 | 324.19 | 327.02 | 35.97 | 0.216 | 0.37 | 0.34 | 177 / 176 | 2.39 / 2.33 / 2.33 |

Best PSM chosen separately for each metric (the hardest baseline for a per-metric gate); same runs as above.

| category (set) | model | CER % (PSM) | WER % (PSM) | order % (PSM) |
|---|---|---|---|---|
| clean 300 DPI print (clean300-v1) | best | 0.24 (11) | 0.69 (6) | 0.70 (6) |
| clean 300 DPI print (clean300-v1) | fast | 0.26 (6) | 0.83 (6) | 0.85 (6) |
| degraded scans (degraded-v1) | best | 0.91 (3) | 2.55 (3) | 2.75 (3) |
| degraded scans (degraded-v1) | fast | 1.19 (3) | 3.36 (3) | 3.58 (3) |
| screenshots and UI (screenshot-v1) | best | 9.94 (11) | 14.38 (11) | 30.89 (4) |
| screenshots and UI (screenshot-v1) | fast | 8.39 (11) | 12.52 (11) | 29.59 (4) |
| complex layout (layout-v1) | best | 12.64 (3) | 11.10 (3) | 12.50 (3) |
| complex layout (layout-v1) | fast | 12.55 (3) | 11.25 (3) | 12.72 (3) |
| phone photos (phone-v1) | best | 6.31 (6) | 12.60 (6) | 14.14 (6) |
| phone photos (phone-v1) | fast | 6.22 (6) | 12.57 (6) | 14.13 (6) |
| receipts (receipts-v1) | best | 64.99 (4) | 88.07 (3) | 93.54 (4) |
| receipts (receipts-v1) | fast | 70.20 (4) | 92.45 (3) | 99.12 (4) |

Reading the table:

- **Receipts**: CORD's truth is partial, so CER/WER and order include every printed word that is not in the truth (header
  and payment blocks, blurred store names): use recall-CER and recall-WER there. The PSM-by-metric table for receipts is
  on plain CER and is not meaningful for the same reason.
- **Detection F1** of Tesseract's TSV words against our truth boxes is a lower bound: truth boxes are layout boxes from
  the browser (full line height, word advance width), Tesseract reports tight ink boxes, so short lowercase words often fall
  below IoU 0.5 even when found (clean pages: 0.75). Compare engines against the same convention, not against 1.0.
- Page timings include process start and model load; screenshot and receipt pages are small, phone and layout pages large.

## Running the C# engine

`bench/OneOcr.Bench` runs `OcrEngine.ToText` over a page directory and writes `<page-id>.txt` for the scorer, plus
`timing.json` (one-time model load, per-page time, peak working set). One process per thread count, so peak memory is per
configuration:

```
dotnet run -c Release --project OneOcr.Bench -- run --model model.onm --pages PAGES --out OUT --degree 1
dotnet run -c Release --project OneOcr.Bench -- run --model model.onm --pages PAGES --out OUT --degree 0   # library default
uv run python -m oneocr_bench score --hyp OUT
```

`OneOcr.Bench lines --pages PAGES --truth TRUTH` checks line finding against lines derived from the truth word boxes
(precision and recall at IoU 0.5) and times grayscale conversion, line finding and line cropping. `OneOcr.Bench kernels --model F --width 1000 --degree 1` prints the time of each stage (convolution blocks, LSTM layers,
classifier) for one line. `OneOcr.Bench randommodel --out F`
writes a random-weight model for pipeline checks.

## Decoding evaluation

`OneOcr.Bench decode --model M --lines DIR --lm charlm-v1.bin [--modes greedy,beam:8,lm:8:0.8:0] [--max N]` reads truth-line crops
(`DIR/labels.jsonl` with `id` and `text`, `DIR/images/<id>.png`, 32 px high, cut from the pages by their truth word boxes), runs the
network once per line and decodes the same output in every mode. Modes: `greedy`; `beam:W`; `lm:W:ALPHA:BETA` (beam search with the
character model, weight ALPHA, per-character bonus BETA); `lexlm:W:ALPHA:BETA:BOOST` (adds `--lexicon FILE`). Columns: CER and WER
over all lines; `numCER`, `numTokErr`: character errors and the share of misread words among truth words that contain a digit
("numbers and IDs"), with their count; `otherTokErr` for the other words; mean decode time per line (the network is not included).
`--max N` keeps every k-th line so that at most N are read.

`recognizer crnn-v0 0.2.0`, `charlm-v1` (order 4, 1.42 MB), width 8, weight 0.8, bonus 0. Per set up to 2,500 lines; one image of
`clean300-v1` is unreadable (a truncated file in the line set) and is skipped.

| set (lines) | mode | CER % | WER % | digit words wrong % (of n) | other words wrong % |
|---|---|---|---|---|---|
| clean300-v1 (796) | greedy | 0.11 | 0.43 | 0.00 (590) | 0.46 |
|  | beam 8 | 0.11 | 0.43 | 0.00 (590) | 0.46 |
|  | beam 8 + model | 0.10 | 0.34 | 0.00 (590) | 0.37 |
| degraded-v1 (1126) | greedy | 40.57 | 60.51 | 34.37 (803) | 62.36 |
|  | beam 8 | 40.54 | 60.42 | 34.25 (803) | 62.28 |
|  | beam 8 + model | 40.18 | 58.31 | 34.37 (803) | 60.00 |
| screenshot-v1 (875) | greedy | 1.82 | 6.06 | 3.54 (311) | 6.36 |
|  | beam 8 | 1.82 | 5.99 | 3.54 (311) | 6.28 |
|  | beam 8 + model | 1.72 | 5.40 | 3.54 (311) | 5.62 |
| layout-v1 (1551) | greedy | 4.01 | 7.81 | 9.15 (481) | 7.73 |
|  | beam 8 | 4.01 | 7.81 | 9.15 (481) | 7.73 |
|  | beam 8 + model | 3.94 | 7.47 | 9.15 (481) | 7.37 |
| phone-v1 (2264) | greedy | 14.60 | 21.38 | 16.54 (133) | 21.68 |
|  | beam 8 | 14.48 | 21.29 | 15.79 (133) | 21.63 |
|  | beam 8 + model | 13.38 | 18.55 | 16.54 (133) | 18.68 |
| receipts-v1 (2297) | greedy | 35.98 | 56.80 | 57.58 (1082) | 56.13 |
|  | beam 8 | 35.96 | 56.93 | 57.76 (1082) | 56.21 |
|  | beam 8 + model | 36.15 | 56.76 | 57.58 (1082) | 56.05 |

The language model never changes a word that contains a digit or a code symbol (`docs/spec/decoding.md`), so those columns equal
greedy's by construction. Decode time per line, one thread (Ryzen 9 7950X3D, .NET 10, a GPU training job running beside, so information only):
greedy 0.01 to 0.08 ms, beam search 0.07 to 0.62 ms, beam search with the model 0.09 to 0.75 ms; the network pass takes 2 to 20 ms
per line, so beam search with the model adds 3 to 7 % to a line. On the 30 `clean300-v1` pages the whole run takes 611 ms per page
(greedy) and 620 ms (model) at one thread, 192 and 198 ms at the default degree.

Page level on `clean300-v1` (`OneOcr.Bench run ... --beam 8 --lm charlm-v1.bin`, scored with the page scorer): CER 0.171 %, WER 0.337 %
greedy; CER 0.157 %, WER 0.271 % with the model.
