# Dataset fetch scripts

Pinned, reproducible fetches of the standard document datasets used for training and evaluation. Only scripts, manifests, split digests and notes are tracked; images and annotations are never committed.

| Dataset | Version | Terms | Command |
|---|---|---|---|
| CORD (receipts) | v2, revision pinned in `manifests/cord.json` | CC BY 4.0 | `python3 fetch_cord.py` |
| DocLayNet (layout) | 1.0.0 core | CDLA-Permissive-1.0 | `python3 fetch_doclaynet.py` |
| PubLayNet (layout) | 1.0.0: val + train shard 0 | annotations CDLA-Permissive-1.0, images PMC Open Access terms | `python3 fetch_publaynet.py` |
| UNLV-ISRI | not fetched | unclear | see `LICENCES.md` |

`LICENCES.md` has the verdict for each, with the quoted clauses and sources. Read it before using PubLayNet images.

## Running

Standard library only; run from this directory. Data goes to `$ONEOCR_DATASETS` (default `~/oneocr/datasets`) as `<name>/<version>/`:

```
images/<split>/…        annotations/…        index/<split>.jsonl (id, file, sha256, bytes)
dedup.json              splits.json          DONE.json
```

- Downloads resume (`.part` files), are size-checked and SHA-256-checked against `manifests/*.json`, and retry on network errors. Archives are deleted after extraction unless `--keep-raw`.
- A second run finds `DONE.json` and does nothing. `--force` re-extracts.
- CORD is parquet, so `fetch_cord.py` re-runs itself under `uv run --no-project --with pyarrow` when pyarrow is missing; nothing is installed into a project.
- Options: `--root DIR`, `--keep-raw`, `--force`.

## Pins and splits

- Each manifest pins URL, size and SHA-256 per archive. CORD's hashes are the Hub's LFS object ids at a fixed revision. DocLayNet and PubLayNet publish no checksums, so their SHA-256 values are those observed at the first fetch (trust on first use), and the fetch fails if a later download differs.
- Official splits are kept: CORD train/dev/test (800/100/100); DocLayNet train/val/test from its COCO files (69,375/6,489/4,999); PubLayNet val plus one of the seven official train shards (its test split has no labels in the 1.0.0 package).
- `splits/<name>-<version>.json` is each fetch's `splits.json`: counts, bytes, and a digest per split over its sorted `(id, image sha256)` pairs, which pins the split exactly without listing 80,000 ids. The full id lists are `index/*.jsonl` on the worker.
- Real images are de-duplicated across splits by SHA-256 of the file bytes. `dedup.json` records within-split duplicates and cross-split collisions; for a collision the copy in the higher-priority split (test, dev, val, train) is kept and `drop` lists the ids to exclude from the other. Byte hashing does not catch re-encoded or re-photographed copies.

Datasets whose labels ship with them (CORD, DocLayNet, PubLayNet) are label sources in their own right; see `LICENCES.md` for attribution.

## Additions for the page detector

- `fetch_doclaynet_extra.py` fetches DocLayNet's "extra" archive (8.0 GB, SHA-256 pinned in `manifests/doclaynet_extra.json`, same CDLA-Permissive-1.0 licence) and keeps only its per-page JSON: the PDF's text cells, used as text-line truth. The PDFs are not extracted.
- `filter_publaynet_licence.py` looks up each PubLayNet article's licence code in PMC's Open Access cloud bucket metadata and writes `licence/allowed_{train,val}.txt` for articles coded exactly `CC0` or `CC BY`. Result on the fetched shards: train shard 0, 38,309 of 39,841 articles kept (CC BY 38,217; CC0 92; excluded: TDM 1,035, no code 467, CC BY-NC 26, CC BY-NC-ND 4); val, 7,755 of 7,797 kept.
