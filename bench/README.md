# bench — OCR evaluation sets and baselines

Six small evaluation sets with exact truth, one per quality category, an engine-agnostic scorer, and a runner that
measures Tesseract (`tessdata_best` and `tessdata_fast`, stock English models) as a black box.

Everything is generated or fetched by scripts; no images, models or outputs are committed — only the manifests
in `manifests/` (SHA-256 of every page image and truth file, and of every font and text source).

## The page set `clean300-v1`

30 pages, 2480 x 3508 px, 8-bit grayscale PNG, one column, Latin script. HTML is rendered by headless Chromium
(Playwright); the truth is the browser's own layout: one rectangle per word from a DOM Range, in document order,
so no OCR is involved. Page text = blocks in reading order, words separated by spaces, blocks by newlines.
A word that wraps at a line-end hyphen (`pre-possessed`) has one box per printed fragment (`pre-`, `possessed`) and
stays one word in the text file (bench-v1 gave it one box spanning both lines).

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

Truth files per page: `truth/<id>.json` (words with `box` = [x0, y0, x1, y1] in image pixels, the layout rectangle; `ink`, the
box tight to the printed ink; a `block` index; `quad` and `ink_quad` where the page was transformed; the generation
parameters), `truth/<id>.txt` (words in reading order, a new line when the block changes; this is what the scorer reads).
A word that wraps at a hyphen or em dash is one entry per printed fragment, each with `"split": true` and `"part": 0, 1`;
the text file joins the fragments without a space.

**Ink boxes** (`ink`) are computed from the rendered pixels, never by hand: inside the word's layout box (one pixel of margin
sideways) the background is the most common grey level; every pixel that differs from it by at least 30 % of the word's
peak contrast is ink; `ink` is the extent of those pixels. They are computed on the clean render, before any degradation, and
moved by the same homography as `box` (`ink_quad` is the transformed rectangle, `ink` its axis-aligned box). A word with no
visible ink (peak contrast under 12 grey levels) keeps its layout box. CORD's word quads already enclose the printed word, so
for `receipts-v1` `ink` equals `box`.

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

**Scorer versions** (`--scorer v2|v3|v3.1`, recorded as `scorer` in every result and report). The metrics above are the same in all of them;
they differ only in where the truth word sequence comes from (and v3.1 in one extra normalisation).

- **v2** (the default of `score`, unchanged since bench-v1): the words of the truth text `<id>.txt`. That text joins what the page
  shows as separate ink pieces (a word wrapped at a hyphen or dash, `breakfast-room` for `breakfast-` at a line end and `room` on
  the next) and splits what is one piece of ink (a web page's punctuation node next to the word it touches: `Intelligence` `,`).
- **v3** (the default of `gate`): wherever the truth has word boxes (`<id>.json`), the truth word sequence is the box words, in
  their order, joined exactly where the ink is continuous:
  1. a box whose text is only punctuation or symbols (Unicode categories P* and S*) is joined to each list neighbour (the box before,
     the box after) whose layout box touches it: the horizontal gap is at most 0.5 px plus 2 % of the box height and the boxes overlap
     vertically by at least half of the shorter one. So `Intelligence` `,` is `Intelligence,`, `1` `/` `2` is `1/2`, `(` `see` `)` is `(see)`;
  2. nothing else is joined. Two boxes with letters or digits stay two words however close they are, so each fragment of a
     wrapped word stays a word (`breakfast-`, `room`), and a mark with a space before it stays alone;
  3. boxes with no text left after normalisation are dropped; a truth without a sidecar, or with no words in it, uses the truth text.

  Normalisation, the bag pairing, the order distance, ECE matching and the F1 boxes are exactly as in v2 (F1 compares the boxes as
  given, not the joined ones). Every engine's output is scored against the same sequence. A set without word boxes scores the same
  under both versions. `scripts/test_scorer_v3.py` pins the rule; `scripts/scorer_diff.py` lists every hypothesis word whose
  outcome differs between the versions, checks that the two truth sequences differ only by cutting the same characters differently,
  and samples the differences for a manual check. Known limit: PDF truth cut into small-caps or letter-spaced pieces stays in pieces
  (a mark touching only one of them joins that one).
- **v3.1** (`--scorer v3.1`): v3 plus one normalisation rule, applied to truth and hypothesis of every engine alike: two apostrophes in a row
  (after typographic single quotes are folded to `'`) are one double quote `"`. A closing quote typeset as `''` and read as `"` (or the reverse)
  is then no error. `scripts/quote_fold_diff.py` counts the pages and words the rule changes. `score` and `gate` accept it; the gate
  needs competitor numbers under it (`competitors-freeze --rescore` adds them); v2 and v3 behave exactly as before.

Besides CER/WER/order, `score` reports **recall-CER and recall-WER**: errors on the truth side only (a truth word that
is missing or misread; extra hypothesis words cost nothing). They matter where truth is partial (receipts).

**Word detection F1 at IoU 0.5.** If every page has a sidecar `<page-id>.json` next to its `<page-id>.txt`, the scorer also
reports detection precision, recall and F1. Sidecar format, coordinates in pixels of the page image as given in `pages/`:

```json
{"words": [{"box": [x0, y0, x1, y1]}, {"quad": [x1, y1, x2, y2, x3, y3, x4, y4], "text": "optional, ignored"}]}
```

Boxes are matched one to one, highest IoU first, text not compared; truth quads count as their axis-aligned boxes.
A missing sidecar on any page means no F1 for that run. A missing hypothesis file scores as empty output.

Boxes are scored against the truth's **ink** box by default (`--boxes ink`), which is what an engine that reports tight word
boxes produces; `--boxes layout` scores against the layout rectangle (bench-v1's convention, except that a hyphen-wrapped word
now has one box per line). Where a truth has no `ink` member (a bench-v1 directory) the layout box is used and the output
says `[layout boxes]`.

**Expected calibration error (ECE).** If every sidecar word also has `text` and a confidence (`confidence`, or `conf` as the C#
bench writes it), `score` adds ECE for the set:

- Each hypothesis word is matched to the page's truth words the way the order-free WER pairs them: words equal after
  normalisation cancel one for one (of equal hypothesis words the earlier ones match); the leftover truth and hypothesis words
  are then paired closest edit distance first. A word is **correct** only if it matched exactly. A hypothesis word paired
  with a different truth word is wrong; one left without a truth word is wrong too (`ECE`).
- 15 equal-width bins on [0, 1]; ECE = sum over bins of (words in bin / all words) * |accuracy in bin - mean confidence in bin|,
  micro-averaged over the pages of the set.
- `ECE-m` (`ece_matched` in the JSON) leaves out the hypothesis words that found no truth word at all. Use it for `receipts-v1`,
  whose truth is partial: words the engine read correctly but CORD does not annotate count as wrong in `ECE`, so `ECE` is an
  upper bound there and `ECE-m` is the figure to quote, labelled as such.
- This is the library's calibration metric (15 bins, word accuracy) measured per page set instead of per script on line crops;
  matching is the page's bag of words, not an ordered line alignment. The result JSON has `ece.{words, accuracy, mean_confidence,
  ece, matched_words, matched_accuracy, ece_matched}`.

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
- **Detection F1** in the tables above is against bench-v1's layout boxes and is a lower bound: truth boxes are layout boxes
  from the browser (full line height, word advance width), Tesseract reports tight ink boxes, so short lowercase words often
  fall below IoU 0.5 even when found (clean pages: 0.75). Against the ink boxes of bench-v2 the same outputs score
  0.998 / 0.983 / 0.817 / 0.942 / 0.885 / 0.231 (clean, degraded, screenshot, layout, phone, receipts); see below.
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

## The gate table in one command

```
# quality + calibration (runs OneOcr.Bench on all six sets, scores, compares with the Tesseract results)
uv run python -m oneocr_bench gate --model MODEL.onm --detector DETECTOR.onm \
    --eval EVAL_DIR --tess-results RESULTS_ROOT --out OUT

# speed + memory: separate mode, needs the machine lock claimed by the same job (below)
uv run python -m oneocr_bench gate --model MODEL.onm --detector DETECTOR.onm --timing --tess-results RESULTS_ROOT --out OUT
```

`--detector` is optional (without it the classical line finder runs). `--eval` is the frozen set directory (`<set>/pages`,
`<set>/truth`; env `ONEOCR_EVAL`). `--tess-results` is a root holding `<set>/tesseract.json` and `<set>/hyp/` from `tess`
(default `$ONEOCR_BENCH_DIR/results`); the table reuses them and runs Tesseract again only with `--rerun-tess` (into a
writable directory under `--work`, since the frozen sets are read-only). `--sets` restricts the run, `--reuse` re-scores engine
outputs already under `--work/out`.

Accuracy mode runs `OneOcr.Bench run --words 1 --pagejson 1`, scores with the bench scorer (ink boxes) and writes
`OUT/accuracy.json`; `--timing` writes `OUT/timing.json`. Both then rewrite `OUT/gate.md` and `OUT/gate.json`; the speed section
appears once a timing run exists. Accuracy is not timing-sensitive and can run under `nice` when the machine is quiet.

- **Quality verdict**, per category and metric: ours against the **best value any measured engine reaches**. With
  `--competitors DIR` (a frozen competitor-results directory, below) the candidates are Tesseract best-of (the best PSM for that metric of
  `tessdata_best`, the second table under *Tesseract 5.3.4 baselines*) and every gated competitor configuration; the table names the
  engine that holds the best value, keeps Tesseract best and fast in their own columns, and a second table lists every engine. Without
  `--competitors` the baseline is Tesseract best-of alone. Win means strictly lower error; clean print (clean300-* and clean-real) may tie, an error within 0.2
  percentage points (scorer v2) or 0.05 percentage points (scorer v3) above the baseline passes as a tie. Receipts are judged on recall-CER and recall-WER (CORD truth is partial); their
  order is shown, not judged.
- **Calibration verdict**: ECE (15 bins) ≤ 0.05. On receipts ECE is an upper bound; ECE-m is shown next to it.
- **Detection**: ink-box F1@0.5 of ours, and of the selected Tesseract runs rescored with the same scorer.
- **Speed and memory** (`--timing` only), two parts so each fits a timing window:
  - `--timing --model M [--detector D]` times our engine per set at degree 1 and at the default degree min(4, cores): one process per run,
    three runs, mean of the runs (whole-process ms/page with start and model load included, page-only ms as information, peak RSS).
    Writes `timing.json`. The timed process is the shipped command line, `oneocr ocr` (src/OneOcr.Tool), published NativeAOT for the host
    (`dotnet publish -r <rid>`, no JIT warm-up). x64 is compiled for AVX-512 (`x86-64-v4`) when the host has it, else AVX2 + FMA (`x86-64-v3`);
    `--isa` overrides, and `timing.json` and `gate.md` record the level and the binary's hash. Accuracy mode still runs `OneOcr.Bench`, which
    writes the word-box sidecars; its text and page JSON are byte-identical to the tool's.
  - `--timing --tess-timing` (no model needed) times Tesseract the same way: stock `tessdata_best` and `tessdata_fast` at the PSM `tess` selected
    for that set (read from `--tess-results`), one process per page, `OMP_THREAD_LIMIT` 1 and the default degree, three runs, mean ms/page and
    peak RSS from `wait4`. Writes `tess-timing.json`, merged per set (`--sets` runs a few sets per window; `--pages N` takes the first N pages).
  - The report pairs them: degree 1 against one Tesseract thread, the default degree against the same number of threads. Speed passes when
    ours is faster than `tessdata_fast`; slower than fast but faster than best with the quality row won is an *exception*; memory passes when
    peak RSS at the default degree is at most 300 MB (degree 1 is shown for information). Each file records CPU, cores, and load before, after every run and after.
- **The lock**: Either `--timing` mode exits with status 75 and does nothing else unless the lock file (`~/WORKER-BENCH.lock`, env `ONEOCR_LOCK`)
  names one of its own ancestor processes as `pid=` (the one exception: `--tess-timing --pages 2` or fewer runs as a smoke test while no lock file exists). Claim it in the job's shell, so it is released however the job ends:

```
set -o noclobber; echo "gate pid=$$ $(date -Is)" > ~/WORKER-BENCH.lock || exit 75
trap 'rm -f ~/WORKER-BENCH.lock' EXIT
```

The gate judges on `--scorer v3` (default) and lists the other version beside it in a second table (ours, best engine and verdict
under each); `gate.json` carries both. Tesseract's best-of values are scored again under v3 from its hypothesis directories. The
competitor directory must hold v3 numbers: `competitors-freeze` writes both versions, and `competitors-freeze --rescore FROZEN --out
OUT --eval EVAL` adds v3 to a directory made before, from its hypotheses alone (names and baseline flags kept; no engine runs). The
CER/WER/order/F1/ECE numbers are those of `score` with the same `--scorer`. `scripts/full_run.sh` passes `SCORER` (default v3).

## Where the page error comes from

```
bench/scripts/page_causes.sh EVAL_ROOT OUT_ROOT MODEL DETECTOR CHARLM WORDFREQ SET [SET ...]
PAGE_CAUSES_ARGS="--aggregate --ref OTHER_OUTPUT_ROOT ..." bench/scripts/page_causes.sh ...
```

Reads the sets with the given model files and runs `training/eval/page_causes.py`, which puts every truth word in one class
(missed, merged with a neighbour, split, misread in a correct box) and counts extra words, in truth characters and as CER/WER
points. It also reports two truth artifacts (the same output scored against the truth boxes' words instead of the truth text,
which isolates the cost of words the page wraps at a hyphen; truth words whose boxes touch), the visibility of missed words
(contrast inside the ink box), and, for outputs of other engines given with `--ref`, how many of them read each missed word or
each low-contrast word (information only). `--aggregate` prints no single word or page, for sets whose pages must stay unseen.

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

## Competitor baselines (information rows)

Released competitors run as black boxes on the same pages as Tesseract and scored with the same scorer. Nothing of theirs is
read, loaded or reused by our code; their output is only compared with our exact truth. They are information, not gates.
The page files used on the Mac were checked against the manifests first (`scripts/verify_pages.py PAGES`).

### Apple Vision

`vision/main.swift` is a small Swift program over the public Vision API (`VNRecognizeTextRequest`, English, automatic language
detection off). It writes `<id>.txt` with the lines in the order Vision returns them and `<id>.json` with one box per word
(`VNRecognizedText.boundingBox(for:)` on each whitespace-separated word, converted to pixels, y down).

```
bash vision/run.sh PAGES OUT [cpu]     # builds the CLI with swiftc, then runs accurate/fast x language correction off/on over PAGES/<set>/
uv run python -m oneocr_bench competitors-report OUT --eval EVAL_DIR     # where the truth directories are
```

Mac mini (Mac14,3, Apple M2, 8 cores, 8 GB), macOS 27.0.1 (26A434), Swift 6.4. Vision gives no control over threads, so there is
one timing column: mean wall seconds per page inside one process (request and results included; the first page of a set also pays
the one-time model load, so the mean is a little high for small sets). `usesCPUOnly` gave identical text and the same speed, so only
the default rows are shown. The Mac's load average was 1.2 before the runs and about 4 during them. `fast` is a different, much weaker
recognizer, shown for completeness. Configuration names: `accurate`/`fast` is the recognition level, `corroff`/`corron` language correction.

| category (set) | engine / configuration | CER % | WER % | order % | recall-CER % | recall-WER % | F1@0.5 | s/page | peak RSS MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean 300 DPI print (clean300-v1) | accurate-corroff | 0.29 | 0.70 | 0.70 | 0.24 | 0.63 | 0.992 | 0.18 (n=30) | 121 |
| clean 300 DPI print (clean300-v1) | accurate-corron | 0.33 | 0.67 | 0.67 | 0.25 | 0.56 | 0.987 | 0.21 (n=30) | 138 |
| clean 300 DPI print (clean300-v1) | fast-corroff | 1.18 | 2.56 | 2.69 | 1.08 | 2.33 | 0.737 | 0.07 (n=30) | 119 |
| clean 300 DPI print (clean300-v1) | fast-corron | 1.29 | 2.56 | 2.69 | 1.19 | 2.33 | 0.751 | 0.15 (n=30) | 166 |
| degraded scans (degraded-v1) | accurate-corroff | 0.42 | 1.22 | 1.25 | 0.36 | 1.13 | 0.984 | 0.15 (n=40) | 105 |
| degraded scans (degraded-v1) | accurate-corron | 0.38 | 1.05 | 1.06 | 0.30 | 0.90 | 0.975 | 0.17 (n=40) | 124 |
| degraded scans (degraded-v1) | fast-corroff | 10.34 | 16.48 | 16.91 | 8.83 | 15.38 | 0.678 | 0.05 (n=40) | 106 |
| degraded scans (degraded-v1) | fast-corron | 10.40 | 16.13 | 16.56 | 8.92 | 15.03 | 0.712 | 0.13 (n=40) | 182 |
| screenshots and UI (screenshot-v1) | accurate-corroff | 0.45 | 1.86 | 33.01 | 0.35 | 1.30 | 0.983 | 0.09 (n=40) | 113 |
| screenshots and UI (screenshot-v1) | accurate-corron | 0.28 | 1.46 | 32.78 | 0.17 | 0.86 | 0.977 | 0.11 (n=40) | 126 |
| screenshots and UI (screenshot-v1) | fast-corroff | 31.29 | 37.89 | 44.07 | 25.81 | 37.76 | 0.642 | 0.03 (n=40) | 104 |
| screenshots and UI (screenshot-v1) | fast-corron | 31.55 | 37.26 | 43.57 | 25.91 | 37.13 | 0.642 | 0.05 (n=40) | 130 |
| complex layout (layout-v1) | accurate-corroff | 0.48 | 1.14 | 15.25 | 0.39 | 1.00 | 0.986 | 0.20 (n=40) | 124 |
| complex layout (layout-v1) | accurate-corron | 0.48 | 1.08 | 15.27 | 0.37 | 0.91 | 0.980 | 0.24 (n=40) | 151 |
| complex layout (layout-v1) | fast-corroff | 2.48 | 3.86 | 50.12 | 2.40 | 3.74 | 0.730 | 0.08 (n=40) | 141 |
| complex layout (layout-v1) | fast-corron | 2.51 | 3.86 | 50.14 | 2.43 | 3.74 | 0.731 | 0.18 (n=40) | 203 |
| phone photos (phone-v1) | accurate-corroff | 0.67 | 1.17 | 1.57 | 0.60 | 1.09 | 0.990 | 0.16 (n=40) | 97 |
| phone photos (phone-v1) | accurate-corron | 0.65 | 1.13 | 1.54 | 0.56 | 1.03 | 0.985 | 0.19 (n=40) | 120 |
| phone photos (phone-v1) | fast-corroff | 37.46 | 47.04 | 47.47 | 37.17 | 46.97 | 0.528 | 0.06 (n=40) | 95 |
| phone photos (phone-v1) | fast-corron | 37.42 | 46.04 | 46.47 | 37.13 | 45.97 | 0.577 | 0.13 (n=40) | 148 |
| receipts (receipts-v1) | accurate-corroff | 12.55 | 23.48 | 61.38 | 7.06 | 18.22 | 0.869 | 0.09 (n=100) | 175 |
| receipts (receipts-v1) | accurate-corron | 13.75 | 25.15 | 63.22 | 7.45 | 18.31 | 0.854 | 0.09 (n=100) | 197 |
| receipts (receipts-v1) | fast-corroff | 52.98 | 70.64 | 80.98 | 46.88 | 66.26 | 0.612 | 0.04 (n=100) | 240 |
| receipts (receipts-v1) | fast-corron | 52.81 | 69.56 | 80.40 | 46.64 | 65.18 | 0.615 | 0.05 (n=100) | 253 |

### PaddleOCR and docTR

Installed with `competitors/setup.sh` into two virtual environments of their own (apart from this project's),
CPU only, stock models downloaded by the packages themselves. Versions (`versions.txt`): paddleocr 3.7.0, paddlepaddle 3.3.1,
paddlex 3.7.2, python-doctr 1.1.0, torch 2.14.1 (CPU), CPython 3.13.

```
bash competitors/setup.sh                          # once
bash competitors/run_all.sh                        # accuracy rows (every page) and a first timing pass; claims the machine lock
bash competitors/timing_chunks.sh                  # timing/RSS rows in short chunks that each claim and release the lock
uv run python -m oneocr_bench competitors-report COMPETITOR_OUT --eval EVAL_DIR [--timing .../out-timing]
```

`competitors/run.py` is the runner (one process per configuration, thread setting and set; see its docstring):

- **paddle-v5-mobile** = PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec; **paddle-v5-server** = PP-OCRv5_server_det + PP-OCRv5_server_rec.
  Document orientation, unwarping and text-line orientation steps are off (the sets are upright). Detection input is capped at 1920 px
  on the longer side (`text_det_limit_type=max`): with the stock setting the server model took 21 GB and over a minute on a screenshot page.
  `enable_mkldnn=False`, because the default oneDNN path raised an error in the pinned paddlepaddle on the benchmark machine. Paddle returns text
  lines; its word boxes are the line boxes cut in proportion to character counts, so F1 is approximate, and low on `phone-v1`, where
  lines are rotated. The server model drops spaces on degraded and phone pages (`acceptedthechallenge`), which is most of its CER/WER there.
- **doctr-default** = `ocr_predictor(pretrained=True)` (db_resnet50 + crnn_vgg16_bn at this version); **doctr-mobile** = db_mobilenet_v3_large +
  crnn_mobilenet_v3_small. Word boxes are docTR's. Its detector resizes pages to a fixed square input, which costs it on large pages.
- Reading order is each engine's own line order. Receipts: read the recall columns, as for Tesseract.

Ryzen 9 7950X3D (16 cores, 32 threads), Linux. `s/page 1 thread` is the first n=8 pages with one CPU thread (`cpu_threads=1`,
`torch.set_num_threads(1)`); `s/page default` is every page with the engine's own threading (Paddle 10 threads, torch all). Times include
reading the file and exclude model load (1 to 2 s, in each `timing.json`); peak RSS includes model load.

**These timings are not publishable.** The first pass ran for 5 hours on a shared machine; the load average before and after each run is in
the last column (up to 11), and other jobs on the machine overlapped part of it. The accuracy columns do not depend on load. The
timing-only rerun in short lock-claiming chunks (`timing_chunks.sh`) did not get the lock: other long jobs kept re-claiming it.
Rerun `bash competitors/timing_chunks.sh` on a free machine and pass `--timing` to the report.

| category (set) | engine / configuration | CER % | WER % | order % | recall-CER % | recall-WER % | F1@0.5 | s/page 1 thread | s/page default | peak RSS MB (1t / default) | load1 before / after (default run) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean 300 DPI print (clean300-v1) | doctr-default | 0.89 | 2.49 | 2.59 | 0.82 | 2.42 | 0.979 | 10.10 (n=8) | 1.98 (n=30) | 1589 / 1963 | 1.2 / 8.8 |
| clean 300 DPI print (clean300-v1) | doctr-mobile | 0.97 | 2.55 | 2.63 | 0.89 | 2.49 | 0.975 | 1.59 (n=8) | 0.53 (n=30) | 1379 / 1702 | 5.5 / 7.8 |
| clean 300 DPI print (clean300-v1) | paddle-v5-mobile | 0.41 | 0.82 | 0.83 | 0.24 | 0.55 | 0.764 | 13.22 (n=8) | 12.08 (n=30) | 1439 / 1472 | 1.8 / 9.8 |
| clean 300 DPI print (clean300-v1) | paddle-v5-server | 2.07 | 2.85 | 2.98 | 1.67 | 2.36 | 0.721 | 76.63 (n=8) | 36.90 (n=30) | 13932 / 13930 | 7.9 / 7.2 |
| degraded scans (degraded-v1) | doctr-default | 2.62 | 6.07 | 14.20 | 2.55 | 6.06 | 0.968 | 9.62 (n=8) | 1.68 (n=40) | 1554 / 1946 | 2.9 / 9.9 |
| degraded scans (degraded-v1) | doctr-mobile | 1.79 | 5.13 | 14.28 | 1.71 | 5.07 | 0.977 | 1.44 (n=8) | 0.47 (n=40) | 1363 / 1742 | 6.3 / 5.9 |
| degraded scans (degraded-v1) | paddle-v5-mobile | 2.46 | 4.12 | 4.62 | 2.36 | 4.06 | 0.517 | 10.29 (n=8) | 8.32 (n=40) | 1420 / 1432 | 2.5 / 10.0 |
| degraded scans (degraded-v1) | paddle-v5-server | 63.75 | 53.87 | 54.29 | 47.69 | 53.87 | 0.374 | 68.96 (n=8) | 30.70 (n=40) | 13883 / 13971 | 1.1 / 6.4 |
| screenshots and UI (screenshot-v1) | doctr-default | 2.92 | 7.57 | 20.89 | 2.85 | 7.57 | 0.959 | 3.98 (n=8) | 0.63 (n=40) | 1407 / 1529 | 3.4 / 7.7 |
| screenshots and UI (screenshot-v1) | doctr-mobile | 1.67 | 5.51 | 19.89 | 1.55 | 5.38 | 0.970 | 1.19 (n=8) | 0.38 (n=40) | 1294 / 1378 | 4.8 / 7.1 |
| screenshots and UI (screenshot-v1) | paddle-v5-mobile | 1.53 | 3.79 | 17.87 | 1.43 | 3.49 | 0.820 | 5.39 (n=8) | 3.98 (n=40) | 1463 / 1543 | 3.2 / 9.6 |
| screenshots and UI (screenshot-v1) | paddle-v5-server | 9.72 | 11.23 | 24.18 | 7.35 | 11.06 | 0.826 | 43.95 (n=8) | 18.79 (n=40) | 14823 / 14887 | 1.1 / 6.6 |
| complex layout (layout-v1) | doctr-default | 1.49 | 4.16 | 63.74 | 1.37 | 4.08 | 0.970 | 14.88 (n=8) | 2.25 (n=40) | 1616 / 2135 | 4.7 / 11.6 |
| complex layout (layout-v1) | doctr-mobile | 1.54 | 4.05 | 63.66 | 1.35 | 3.95 | 0.972 | 1.72 (n=8) | 0.55 (n=40) | 1391 / 1843 | 6.1 / 9.5 |
| complex layout (layout-v1) | paddle-v5-mobile | 0.52 | 0.96 | 62.93 | 0.36 | 0.73 | 0.837 | 18.69 (n=8) | 13.99 (n=40) | 1452 / 1486 | 5.1 / 10.0 |
| complex layout (layout-v1) | paddle-v5-server | 1.91 | 2.92 | 63.93 | 1.66 | 2.65 | 0.817 | 94.22 (n=8) | 39.35 (n=40) | 13927 / 13931 | 1.8 / 7.4 |
| phone photos (phone-v1) | doctr-default | 3.83 | 9.60 | 41.52 | 3.64 | 9.57 | 0.951 | 10.84 (n=8) | 1.87 (n=40) | 1515 / 1965 | 2.5 / 9.9 |
| phone photos (phone-v1) | doctr-mobile | 3.10 | 8.71 | 43.14 | 2.87 | 8.57 | 0.975 | 1.56 (n=8) | 0.49 (n=40) | 1358 / 1734 | 7.6 / 10.0 |
| phone photos (phone-v1) | paddle-v5-mobile | 3.08 | 5.62 | 5.71 | 2.96 | 5.60 | 0.213 | 10.55 (n=8) | 8.95 (n=40) | 1476 / 1489 | 1.9 / 9.8 |
| phone photos (phone-v1) | paddle-v5-server | 90.59 | 71.25 | 71.29 | 64.54 | 71.25 | 0.182 | 73.87 (n=8) | 34.68 (n=40) | 14823 / 14877 | 1.0 / 7.1 |
| receipts (receipts-v1) | doctr-default | 11.60 | 24.06 | 27.94 | 3.97 | 9.55 | 0.895 | 2.33 (n=8) | 0.44 (n=100) | 1403 / 1466 | 3.0 / 7.8 |
| receipts (receipts-v1) | doctr-mobile | 24.10 | 60.01 | 64.64 | 5.37 | 13.89 | 0.758 | 1.16 (n=8) | 0.37 (n=100) | 1263 / 1367 | 8.0 / 10.5 |
| receipts (receipts-v1) | paddle-v5-mobile | 15.26 | 29.11 | 38.16 | 10.34 | 25.56 | 0.826 | 2.13 (n=8) | 1.94 (n=100) | 1154 / 1741 | 3.2 / 9.8 |
| receipts (receipts-v1) | paddle-v5-server | 14.44 | 24.27 | 34.45 | 8.98 | 20.18 | 0.841 | 23.11 (n=8) | 11.90 (n=100) | 11295 / 19505 | 1.1 / 6.1 |

Reading the rows: Apple Vision accurate is the strongest engine on the five page sets (CER 0.3 to 0.7 %); on receipts docTR default has
the best recall-CER (3.97 %, Vision 7.06 %). PP-OCRv5 mobile is next on clean print (0.41 %) and layout (0.52 %); docTR is 1 to 4 % CER.
PaddleOCR's word-box F1 is low partly because of the line-box splitting described above.

## Using the eval sets from another project

- **Tag:** git tag `bench-v1` is the scorer and the generators; the frozen eval directory `bench-v1/<set>/` (for example
  `bench-v1/clean300-v1/`) holds the sets: frozen copies of the generated sets with a `SHA256SUMS` file, chmod read-only.
- **Rules:** never modify either. Any change to a set or to the scorer becomes `bench-v2`.
- **Pinning:** other projects may vendor `bench/` pinned to the tag.

## bench-v2

Same pages and same text truth as bench-v1 (every page image and every `<id>.txt` is byte-identical; `gen --set all --check`
reproduces the page hashes), with three additions:

1. **ECE** in `score`, defined under Scoring.
2. **One box per printed fragment** of a hyphen- or em-dash-wrapped word (about 70 entries over 5 sets); text truth unchanged.
3. **Ink-tight boxes** (`ink`) next to the layout boxes, and `--boxes ink|layout` (default `ink`).

CER, WER, recall and reading-order numbers are identical to bench-v1's on any input: `scripts/compare_scorers.py OLD_SCORE_PY
OLD_ROOT NEW_ROOT OUTPUT_ROOT` runs the bench-v1 scorer (saved from the tag) and this one on the same outputs and requires
equal totals and per-page values. The frozen copy is `bench-v2/<set>/` (`pages/`, `truth/`) with
`SHA256SUMS` at its root, made by `scripts/freeze_sets.sh`, read-only. Never modify the copy or the scorer; a change is a
new version.

Tesseract `tessdata_best` (best PSM per set, as in the baseline table), F1@0.5 on the same outputs:

| set | bench-v1 boxes (one box for a wrapped word) | bench-v2 layout boxes | bench-v2 ink boxes |
|---|---|---|---|
| clean300-v1 | 0.7529 | 0.753 | 0.998 |
| degraded-v1 | 0.7220 | 0.722 | 0.983 |
| screenshot-v1 | 0.6807 | 0.681 | 0.817 |
| layout-v1 | 0.6844 | 0.685 | 0.942 |
| phone-v1 | 0.5289 | 0.530 | 0.885 |
| receipts-v1 | 0.2310 | 0.231 | 0.231 |
- **Scoring:** `score` takes a truth directory and an output directory, needs no environment variable and writes nothing to either:

```
uv run python -m oneocr_bench score EVAL_DIR/clean300-v1 OUTPUT_DIR [--boxes ink|layout] [--json result.json]
```

  The first argument is a set directory (its `truth/` is used) or a `truth/` directory itself; `truth/<id>.txt` is the reading-order
  truth and `truth/<id>.json` the word boxes. `OUTPUT_DIR` holds `<id>.txt` and, optionally, `<id>.json` box sidecars (format above); a
  missing `<id>.txt` scores as empty output. The `--set`/`--hyp` and `--hyp-root` forms still work and read the data under
  `ONEOCR_BENCH_DIR`. Checked with nothing written to the frozen copy: the Tesseract outputs for `clean300-v1` give
  CER 0.2427 %, WER 0.7300 %, order 0.7394 % (best, PSM 11) and 0.2586 % / 0.8329 % / 0.8517 % (fast, PSM 6), the numbers in the
  baseline table above.

## A sealed set

The bench-v2 pages have been looked at while the engine was developed, so they are a development set. For quality claims a
second set is cut once, with the same recipes, scorer and category sizes (30 / 40 / 40 / 40 / 40 / 100), and is not looked at,
scored against or tuned on until a release check.

- **Recipes.** Each category is rendered like its `-v1` set from a seed that is not in this repository (`gen --seed N --out DIR` is
  the public seed override; `seal build ROOT --seeds FILE` reads six seeds from a private file and renders all categories).
  Fonts are the same seven held-out families. `--avoid-text DIR` drops every Gutenberg passage that shares an eight-word run
  with a truth under an earlier eval root, so the sealed prose is new text; `seal verify` reports any such run or any identical
  page image as a failure.
- **What is real.** Nothing in it is a photograph or a scan. Prose is public-domain text, truth is browser geometry or our own
  transform of it, and `degraded`, `phone` and `receipts` are simulated captures. `receipts-v1` is synthetic text from
  `receipts_synth.py` (stores, items, tax, totals, card lines; faint thermal ink; the phone simulation on a taller canvas),
  because every CORD v2 split has been used for training or tuning, so no unseen CORD receipts exist. Synthetic receipts
  have full truth, so recall-CER equals CER on them; they do not test the dataset's printer, paper or photo variety.
- **Layout.** `<set>/pages`, `<set>/truth`, `<set>/MANIFEST.json`, `SEALED.json` (what each category is made of) and `SHA256SUMS`,
  all read-only.
- **Check and run.** `seal verify ROOT --avoid-text <earlier eval roots>` checks hashes, truth structure, image sizes, ink boxes
  against pixels and overlap with earlier sets, printing counts only. `gate --sealed ROOT [--dry-run]` is the usual gate on that
  root; its `gate.md` is headed as sealed. `--dry-run` lists the files and runs nothing.
