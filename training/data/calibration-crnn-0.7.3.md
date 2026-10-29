# Reporting curve of crnn-v0 0.7.3: data manifest

0.7.3 has the weights and the decision `calibration` of 0.7.1 unchanged; only the `report_calibration` member (the curve a page
reports its word confidence with) is new. Method: `eval/split_half_fit.py`, which fits with `eval.calibrate.fit_map` (isotonic,
at most 64 knots per script).

## Fit samples

The synthetic and CORD sets of `calibration-crnn-0.7.2.md` (`calib`, `calib_receipt`, `calib6` weight 1, `calib_cord` weight 8, line
crops of the 0.7.1 recognizer), the line crops of the generated receipt photos of the calibration and evaluation splits
(`gen_calib`, `gen_dev`, weight 1), and the page samples of those 250 photos (100 calibration + 150 evaluation; 13,664 words from
the full pipeline, `eval.page_samples`), counted K times. The sealed receipt-photo set was not used.

## Weight

Split-half across the two splits: the curve is fitted with the page samples of one split and its ECE is measured on the page samples
of the other, then the splits are swapped. A weight is eligible while every guard set (line sets: `eval`, `val`, the five page-style
benchmark line sets, `receipts-v1`, `cord_dev`) stays at ECE 0.03 or lower, the calibration gate being 0.05. Among the eligible
weights the lowest mean held-out ECE wins.

| K | fit calib, test dev | fit dev, test calib | mean | worst guard set |
|---|---|---|---|---|
| 0 | 0.0361 | 0.0414 | 0.0387 | 0.0361 (cord_dev, not eligible) |
| 5 | 0.0326 | 0.0369 | 0.0348 | 0.0278 |
| 10 | 0.0299 | 0.0332 | 0.0315 | 0.0261 |
| **25** | **0.0234** | **0.0265** | **0.0249** | 0.0259 |
| 50 | 0.0190 | 0.0204 | 0.0197 | 0.0323 (not eligible) |
| 100 | 0.0128 | 0.0149 | 0.0139 | 0.0406 (not eligible) |

K = 25 was used for the final fit on both splits. SHA-256 of the fitted map: `b6a3bc7f1ed377e60713202c1b07453e629c91421d12cb6323ed851778f29b8a`.
The photo split is the one in `calibration-crnn-0.7.2.md`.
