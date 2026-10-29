# Confidence calibration of crnn-v0 0.7.2: data manifest

0.7.2 has the weights of 0.7.1 unchanged (the tensor region of the container is byte-identical) and a refitted `calibration` member.
Only the calibration changed. This variant was not shipped: the bundled recognizer keeps 0.7.1's decision calibration and adds only a reporting curve (`calibration-crnn-0.7.3.md`). Method: `eval/calibrate.py fit` (isotonic map per script, at most 64 knots), page-level samples from
`eval/page_samples.py`.

## Fit samples (calibration splits only)

| Sample set | Words / lines | Weight | Label source |
|---|---|---|---|
| `calib` (held-out synthetic lines, `configs/lines-v4.yaml`) | 30,000 lines | 1 | exact-truth synthetic |
| `calib_receipt` (receipt-style synthetic lines, `configs/lines-v5.yaml`) | 15,000 lines | 1 | exact-truth synthetic |
| `calib6` (targeted-error synthetic lines, `configs/lines-v6.yaml`) | 24,000 lines | 1 | exact-truth synthetic |
| `gen_calib` (engine line crops of the calibration photos) | 1,840 lines | 1 | generated receipt photos with transcriptions (see below) |
| `calib_cord` (CORD v2 calibration crops) | 1,967 lines | 8 | CORD v2 annotations, CC BY 4.0 |
| page samples of the calibration photos | 100 pages, 5,705 words | 25 | same photos and transcriptions |

The page samples are the words the full pipeline (detector 0.3.0, character model, word frequencies) returns for the 100
calibration photos, each marked correct when it equals a transcribed word under the benchmark scorer's pairing. Words the
transcription does not cover are left out, as in the scorer's matched ECE.

## Weight choice

Chosen on the calibration photos only, by split-half: the map is fitted with the page samples of half the pages (odd and even
page order) and scored on the other half. ECE of the 5,705-word set on the held-out half (15 equal bins):

| Page-sample weight | 0 | 10 | 25 | 50 |
|---|---|---|---|---|
| ECE, held-out half | 0.0355 | 0.0338 | 0.0287 | 0.0255 |

The error falls as the weight rises. 25 was used; larger weights move the maps of other kinds of text further from their own
samples.

## Receipt photos and the split

The photos were generated with Google Gemini 3.1 Flash Image and transcribed by Anthropic Claude Sonnet, as recorded in
`datasets/LICENCES.md`; the photos and transcriptions are not in the repository. The split (`gen/receipt_photos.py split`,
seed `receipts-gen-1`, assignment by image SHA-256) has 1,166 training, 100 calibration and 150 evaluation photos. The three parts
share no image id and no image hash (checked). SHA-256 of `split.json`:
`09cc853d99119e1e15229eb167f7f8e70260a4df07d8027ec414d95dccab5654`. This fit uses the calibration part only; the evaluation part
is for reporting.

## Output

Fitted map (all three scripts) SHA-256 `1e6792e40d30094535518eada7e83d2ec98b14e67710cd15147240aee7b2840c`.

## Using the map as a reporting curve

The map above also ships as a `report_calibration` member next to the unchanged `calibration` of 0.7.1
(`export/add_report_calibration.py IN.onm OUT.onm --from-container <0.7.2 file>`). The runtime then decides with `calibration`
(filters, word joiner, angle search: the page text is byte-identical to 0.7.1) and reports the refitted confidence
(`docs/spec/model-container-v1.md`, "Reporting curve"). Only the calibration photos and the sets listed above were used to fit it.
Resulting file: SHA-256 `1b74db86ac047aad6307bc35a3aa16ab05b904d300de322909242bb87b3c1e55` (0.7.1 tensors, 0.7.2 reporting curve).
Page-level ECE-m on the 150 evaluation photos 0.0362 -> 0.0235; every bench set stays at or below 0.012.
