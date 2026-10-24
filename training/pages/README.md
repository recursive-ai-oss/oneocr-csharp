# pages: synthetic full pages with exact truth

Generates document pages (prose, tables, forms, invoices, receipts) as HTML, renders them in headless Chromium, takes every
box from the browser's own layout, then degrades the image (resolution, rotation, perspective, light, blur, noise, JPEG) and
transforms the truth with the same homography. Nothing is read from an image and no model labels anything.

```
uv run python -m pages --n 1000 --seed 7 --split train --out $OUT --workers 8
uv run python -m pages.check $OUT --split train --n 1000 --workers 8 --report $OUT/check-train.json
uv run python -m pages.overlay $OUT/train/images/train-0000001.jpg overlay.png      # look at a page with its truth drawn
```

Needs `ONEOCR_FONT_ROOT` and `ONEOCR_TEXT_ROOT` (see `data/fonts.py`, `data/texts.py`) and a Chromium for Playwright
(`PLAYWRIGHT_BROWSERS_PATH`, `uv run playwright install chromium`). Settings are in `configs/pages-v0.yaml`.

Page `i` depends only on `(seed, split, i)`: the same seed gives byte-identical truth and images, whatever the worker count.
Output under `OUT/<split>/`: `images/<id>.png|jpg`, `truth/<id>.json`, `manifest.jsonl` (id, kind, words, profile, SHA-256 of image
and truth), `run.json` (config, seed, commit, library versions, throughput).

## Page kinds

| kind | content |
|---|---|
| prose | 1, 2 or 3 columns, title, headings, paragraphs, bulleted/numbered lists, figures with captions, small tables, footnote, running header, page number |
| table | one to three tables per page: ruled, row-ruled, header-ruled, boxed or unruled; zebra, coloured header band, spanning header cells, row spans |
| form | key-value fields (line, box, inline and cell styles, 1-3 grid columns), empty fields, checkbox groups, sections; en de fr es it nl |
| invoice | letter-head variants, three party arrangements, three meta-block styles, item tables (columns, rules, band, zebra), totals and tax breakdown, bank block, footer; en de fr es it nl |
| receipt | narrow till slip in mono or sans, item lines with quantity and unit price, tax, tip, total, payment; alone or on a page-sized surface |

Prose comes from Project Gutenberg books (`data/texts.yaml`), Latin, Greek and Cyrillic; invoices, forms and receipts use
invented companies, people, addresses, phone numbers, e-mail addresses on `example.*` domains, and IBANs with the correct
length and mod-97 check digits but random bank and account numbers.

## Splits

* `train` uses the `train` fonts of `data/fonts.yaml`, the `train` books of `data/texts.yaml`, the `train` degradation ranges,
  and layouts whose template hash is not in the validation share. `val` uses the held-out fonts (falling back to training
  fonts for scripts they do not cover; the page records its fonts), the held-out books, the held-out degradation ranges and the
  validation share of layouts (one template hash in eight; the template is the set of discrete layout choices recorded in
  `synthetic.template`).
* The fonts the page evaluation sets are set in (`never` in `data/font_families.yaml`) are not in `data/fonts.yaml`, and the
  checker verifies no page uses them. The two books the evaluation text comes from are not in `data/texts.yaml`; the checker verifies
  no page cites them.

## Truth format

`truth/<id>.json` is a `oneocr.page` version 1 document (docs/spec/oneocr-page-v1.md) with one page, readable by `OcrPageJson.Read`
and with the same member names, order and joining rules. Coordinates are pixels of the stored image, top-left origin, y down.
Words are numbered in reading order (the DOM order of the generated markup); that numbering is what `cells`, `keyValues`, `fields`
and `checkboxes` refer to. Every object has `source: "textLayer"` and confidence 1, which is what exact truth means here.

Boxes are the browser's rectangles: a word box is the Range rectangle of the word (the font's line box height, not an ink-tight
box); a line is the union of its words; a block is the union of its lines (a table block is the table's rectangle). With rotation or
perspective every box gets a `quad` (top-left, top-right, bottom-right, bottom-left of the transformed box) and `bounds` is the
quad's axis-aligned extent.

Members beyond the spec (a reader ignores them):

| where | member | meaning |
|---|---|---|
| region | `docLayNet` | the DocLayNet class, or null for the roles DocLayNet has no class for |
| page | `documentType` | `prose`, `table`, `form`, `invoice`, `receipt` |
| page | `fields` | `{id, name, value, lineItem, words}`: `id` is the EN 16931 business term for invoices (`BT-1`) or `receipt.<name>`; `value` is the normalized value (ISO date, plain decimal, country code); `lineItem` is the zero-based item index for line fields, else null; `words` are word indices |
| page | `checkboxes` | `{bounds, checked, label}`: the box, whether it is marked, and the word indices of its label |
| document | `synthetic` | id, seed, split, kind, language, template and its split, fonts, books, render dpi, the full degradation parameters, image file and size |

Layout roles map onto DocLayNet like this (`role` is the `oneocr.page` region role):

| generated element | `role` | `docLayNet` |
|---|---|---|
| paragraph, address, meta and totals blocks | text | Text |
| document title | title | Title |
| section heading | title | Section-header |
| list item (also receipt item lines) | list | List-item |
| figure, logo, barcode | figure | Picture |
| figure and table captions | text | Caption |
| footnote | text | Footnote |
| table, item table | table | Table |
| running head / foot, page number | header / footer | Page-header / Page-footer |
| form field (label and value, or a question with its options) | formField | none (our own role) |
| checkbox | checkbox | none (our own role) |

Table `cells` carry row, column and spans with the word indices of each cell. `keyValues` link a label to its value in forms,
invoice meta blocks, totals and receipts; a checkbox question links to the label of the marked option (nothing is linked when none is
marked).

### Invoice field ids

BT-1 number, BT-2 issue date, BT-5 currency, BT-9 due date, BT-13 order reference, BT-20 payment terms, BT-27 seller name, BT-31
seller VAT id, BT-35/37/38/40 seller street, city, postcode, country, BT-44 buyer name, BT-48 buyer VAT id, BT-50/52/53/55 buyer
street, city, postcode, country, BT-72 delivery date, BT-84 IBAN, BT-86 BIC, BT-106 sum of line net amounts, BT-110 total VAT,
BT-112 total with VAT, BT-115 amount due; per line BT-126 line id, BT-129 quantity, BT-130 unit code, BT-131 net amount, BT-146
unit price, BT-152 VAT rate, BT-153 item name; per tax rate BT-116 taxable amount, BT-117 tax amount, BT-119 rate. A field is
listed once per place it is printed as a field; copies of the same text elsewhere on the page (a sender line, the footer) are
plain text.

Receipt fields: `receipt.merchant`, `date`, `time`, `itemName`, `quantity`, `unitPrice`, `amount` (per item, with `lineItem`),
`subtotal`, `tax`, `tip`, `total`, `paymentMethod`.

## Degradation

Profiles `clean`, `scan`, `photo`, `screen` (weights in the config). The page is rendered at 96, 150, 200 or 300 dpi, resampled to
the drawn final resolution, then (scan) rotated slightly on a bed or (photo) put in perspective on a desk, lit unevenly, thinned
(faint print), blurred, noised, reduced to grey or black and white, and saved as PNG or JPEG. Every drawn parameter is stored in
the truth (`synthetic.degradation`), so evaluation can slice by it. `train` and `heldout` ranges are disjoint.

## Checks

`pages.check` regenerates a sample of pages from their seeds and verifies: the truth and image are byte-identical to the stored
ones; the page's words in order equal the whitespace-separated tokens of the rendered text (`innerText`), so no word is missing,
duplicated, reordered or differently cased; every word box holds ink in the clean render; every word box still differs from its
surroundings in the stored degraded image; the structure rules of the `oneocr.page` reader hold; the template, fonts and books
belong to the split. `jsoncheck` style reading by the C# `OcrPageJson.Read` was run over the sample set too.

## Not covered

Ink-tight word boxes (the boxes are line-box height), rotated or vertical text, handwriting, curved pages, scripts beyond Latin,
Greek and Cyrillic (prose only; invoices, forms and receipts are Latin), real photographs.
