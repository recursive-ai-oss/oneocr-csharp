# Web line set: data card

Pages listed 1192; rendered ok 1098; results: {'ok': 1098, 'http 404': 56, 'robots.txt': 8, 'http 403': 28, 'copyright or permission notice': 1, 'Error: Page.goto: net::ERR_TOO_MANY_REDI': 1}

Candidate visual lines 120679; kept after every check and the per-site text cap: 59159

| dropped | lines | share of candidates |
|---|---|---|
| pixels_faint | 9429 | 7.8% |
| length_or_symbols | 4237 | 3.5% |
| covered_neighbour | 2933 | 2.4% |
| pixels_inkheight | 2111 | 1.7% |
| heldout_font | 941 | 0.8% |
| pixels_margin_ink | 514 | 0.4% |
| gap | 422 | 0.3% |
| superscript | 326 | 0.3% |
| pixels_ends | 299 | 0.2% |
| pixels_wordblank | 152 | 0.1% |
| charset | 124 | 0.1% |
| pixels_huge | 82 | 0.1% |
| mixed_sizes | 21 | 0.0% |

## train: 42314 lines, 866 pages, 8 sites

| site | pages | lines |
|---|---|---|
| fda | 130 | 8368 |
| justice | 130 | 7608 |
| cfpb | 121 | 6286 |
| eia | 73 | 4804 |
| gsa | 83 | 4792 |
| doi | 99 | 4034 |
| energy | 100 | 3856 |
| ntsb | 130 | 2566 |

em size (device px): p10 15.0, median 22.5, p90 36.0; characters per line: median 39, p90 94; crop width at 32 px: median 396
device-pixel ratio: {2.0: 14108, 1.5: 5023, 1.25: 5293, 1.0: 17890}; anti-aliasing: {'grayscale': 29509, 'subpixel': 12805}
lines set in the stand-in font for a held-out family: 11520
lines with a viewport narrower than 500 px: 3359

Font families the browser used (lines containing each):

Public Sans (Fontations) 9397, Open Sans (Fontations) 7201, Helvetica 6481, Open Sans (stand-in) 4319, Figtree 3461, Liberation Sans (Fontations) 2952, Jost 2331, Roboto 2095, Public Sans Web 1239, Merriweather (Fontations) 935, Jost (Fontations) 926, Figtree Light (Fontations) 343, Public Sans SemiBold (Fontations) 227, DejaVu Sans Mono (Fontations) 158, Public Sans Light (Fontations) 82, Public Sans ExtraBold (Fontations) 48, Figtree Light ExtraBold (Fontations) 47, Verdana 23, Segoe UI Webfont 21, Liberation Serif (Fontations) 20, Merriweather 10, Noto Sans (Fontations) 6, Figtree Light Light (Fontations) 5, Roboto (Fontations) 3, Font Awesome 6 Pro Solid (Fontations) 1

lines with a digit: 8560; all-capital lines: 1076

## heldout: 16291 lines, 227 pages, 2 sites

| site | pages | lines |
|---|---|---|
| federalreserve | 129 | 12959 |
| nrc | 98 | 3332 |

em size (device px): p10 16.0, median 20.0, p90 32.0; characters per line: median 69, p90 104; crop width at 32 px: median 691
device-pixel ratio: {1.25: 2874, 1.0: 6410, 2.0: 5182, 1.5: 1825}; anti-aliasing: {'grayscale': 11456, 'subpixel': 4835}
lines set in the stand-in font for a held-out family: 3338
lines with a viewport narrower than 500 px: 1635

Font families the browser used (lines containing each):

Liberation Sans (Fontations) 12065, Open Sans (Fontations) 3338, Liberation Serif (Fontations) 882, Arial 6

lines with a digit: 4849; all-capital lines: 159


## Notes

- Rendered with Chromium 1187 (Playwright 1.55) on Linux, one run of `gen.web_lines render` over `data/web_urls.txt` (1,192 pinned pages of 10 sites; one site, cfpb, was re-rendered after a content-security-policy fix and merged). Pages change over time; `pages.jsonl` in the run output keeps each response's SHA-256.
- "Helvetica", "Arial", "Verdana" and similar names are the CSS names where the browser did not report a platform font; on the render host they resolve to Liberation Sans / Nimbus Sans. No proprietary system fonts are installed there.
- Held-out font families (Source Sans, Source Serif, IBM Plex, Gelasio, Carlito, Caladea and every family the font manifests mark as held out) are never trained on: where the browser used one, the page's text is set in Open Sans (a train-pool font, pinned in `data/fonts.yaml`) instead, and the line carries `substituted: true`. Lines that still show a held-out family after that are dropped (`heldout_font`). The layout, size, colour and anti-aliasing are the real page's; the typeface is not.
- `repeated_text` is not in the table above: navigation and footer lines repeat on every page, so each site keeps at most 2 copies of a text (26k + 2.6k lines were removed that way; the counts are in `run.json`).
- Every line passed two pixel checks: on the screenshot before the crop (`gen/web_lines.py`: ink present across every word's box, ink ends within 0.7 em of the box ends, ink height 0.3-1.2 of the box height, at most one unexplained ink column in the crop's side margins) and again on the saved 32 px crop (`gen/web_lines_check.py`, which pruned 430 train and 124 held-out lines and now reports 0 failures).
- Split: whole sites. Held-out = federalreserve + nrc (227 pages, 16,291 lines; mostly Liberation Sans, so easier than the training sites). Training = 8 sites.

## Baseline: the shipped recognizer (0.6.2, container SHA-256 73ed99cb...53f0, fp16-rounded folded weights, fp32 compute, CPU)

| split | lines | CER % | WER % | exact lines % |
|---|---|---|---|---|
| held-out sites | 16,291 | 0.276 | 1.684 | 89.99 |
| training sites (for reference) | 42,314 | 0.920 | 4.624 | 84.84 |

Held-out by site: federalreserve 0.115 / 0.685 (12,959 lines), nrc 1.626 / 10.484 (3,332 lines, set in the stand-in font). Training sites, CER %: eia 4.02, cfpb 1.69, energy 0.67, fda 0.64, gsa 0.61, justice 0.31, doi 0.22, ntsb 0.11. By size (device px): under 14 px 2.55 (1,458 lines), 14-17 px 1.01. Of the erroneous characters on the training sites, 62% are substitutions, 26% deletions and 12% insertions, mostly on small grey text at device-pixel ratio 1 and on a few web fonts (Jost 5.2, DejaVu Sans Mono 4.2). Slices by publisher, ratio, anti-aliasing, size and font are written by `eval/web_lines_eval.py`.
