# Terms verdicts

Checked 2026-10-23 from each dataset's own licence text and the source collection's terms. Quotes are verbatim.

## CORD v2 — usable (CC BY 4.0)

- Source: https://github.com/clovaai/cord (README): "This work is licensed under a Creative Commons Attribution 4.0 International License", linking http://creativecommons.org/licenses/by/4.0/. The dataset card of the official release, https://huggingface.co/datasets/naver-clova-ix/cord-v2, carries `license: cc-by-4.0`.
- Use for training, evaluation and publishing numbers is allowed with attribution (Park et al., "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing", Document Intelligence Workshop at NeurIPS 2019); images are not redistributed by us.
- Official splits: 800 train / 100 dev / 100 test. The README notes that some labels of the paper's full class set were removed for Indonesian legal reasons; only the released classes exist.

## DocLayNet 1.0.0 — usable (CDLA-Permissive-1.0)

- Source: https://github.com/DS4SD/DocLayNet (LICENSE is the Community Data License Agreement – Permissive – Version 1.0; the Hugging Face card says "License: CDLA-Permissive-1.0").
- §2.1: "Data Provider(s) hereby grant(s) to You a worldwide, non-exclusive, irrevocable (except as provided in Section 5) right to: (a) Use Data; and (b) Publish Data."
- §3.1 conditions apply only if we publish the data: give recipients the licence text and keep credit and attribution. §3.4: "This Agreement imposes no obligations or restrictions on Your Use or Publication of Results." Models trained on the data and metrics computed on it are Results.
- The pages are rendered from public-source documents (annual reports, papers, laws, tenders, manuals, patents); the licence is the provider's grant for the dataset as published. We do not republish pages.

## PubLayNet 1.0.0 — annotations usable; images usable for training and aggregate numbers, not for redistribution

- Source: `LICENSE.txt` inside `labels.tar.gz` (same text as LICENSE.md in https://github.com/ibm-aur-nlp/PubLayNet): "The annotations in this dataset belong to IBM and are licensed under a Community Data License Agreement – Permissive – Version 1.0 License. ## Images IBM does not own the copyright of the images. Use of the images must abide by the PMC Open Access Subset Terms of Use."
- So the CDLA covers the annotations only. The images are PubMed Central pages. The README says they come from the "PubMed Central Open Access Subset (commercial use collection)".
- PMC, https://pmc.ncbi.nlm.nih.gov/tools/textmining/: "Commercial use allowed: CC0, CC BY, CC BY-SA, CC BY-ND". https://pmc.ncbi.nlm.nih.gov/tools/openftlist/: "License terms vary. Please refer to the license statement in each article for specific terms of use. Users of this dataset are directly and solely responsible for compliance with copyright restrictions".
- Consequences: (1) every image is under a Creative Commons licence that allows commercial use, so training a layout model on it and publishing metrics is allowed; (2) the images stay on the worker and are never redistributed, which also avoids BY-SA and BY-ND conditions on redistribution; (3) CC BY attribution applies to redistributed copies, not to model weights, and `licence/pmcids_*.txt` (written by the fetch) lists the articles used so a notice or per-article lookup can be produced; (4) the licence per article is not in the package, so the BY-SA / BY-ND share is unknown without a lookup in PMC's file list. Residual risk: a court could read BY-ND/BY-SA differently for model training; this is a legal judgement, not a licence-text question.
- PMC's rule against systematic downloading applies to the PMC website, not to this dataset, which IBM redistributed.

## UNLV-ISRI OCR test data — no-go (not fetched)

- Source: Google Code Archive project `isri-ocr-evaluation-tools` (https://code.google.com/archive/p/isri-ocr-evaluation-tools, project metadata at https://storage.googleapis.com/google-code-archive/v2/code.google.com/isri-ocr-evaluation-tools/project.json): `"license":"asf20","contentLicense":""`. The Apache 2.0 licence is the project's, which holds the evaluation source code; no content licence is stated for the page images and ground truth, and the downloaded archive (checked for one 7.8 MB set, then deleted) holds only `.tif`, `.txt` and `.uzn` files with no readme, notice or licence.
- The images are scans of business letters, DOE reports, magazines, newspapers and legal documents from 1992–1996, each with its own third-party copyright. The tool sources carry "Copyright 1996 The Board of Regents of the Nevada System of Higher Education, on behalf, of the University of Nevada, Las Vegas, Information Science Research Institute".
- Verdict: unclear; not clearly usable for evaluation and publication of numbers. Not fetched. Options: leave it out (recommended; the held-out synthetic, rendered and public-domain sets cover clean print); or ask UNLV's ISRI in writing for terms; or use only the DOE sets, which may be US-government works, after a per-document check.
