# Financial evaluation sets: sources and licence notes

Evaluation pages only; no page of these sets may enter a training set. The generated sets (`fin-scan`, `fin-fax`, `fin-phone`,
`fin-tables`) contain invented names, accounts and amounts; nothing in them comes from a real person or company. The `-real` sets
are pages of federal financial reports, with the text layer as truth (`python -m oneocr_bench finreal`). The PDFs are not
redistributed; `fin-dev-real-urls.json` lists every source URL with its size, page count and SHA-256.

## Real documents

Pages that carry a ©, "(c) year", "used with permission" or "all rights reserved", and pages with an embedded image, are dropped
by the cut. The verdicts below were read from the publishers' own pages on 2026-11-25.

| Publisher | PDFs | Notice (primary source) | Verdict |
|---|---|---|---|
| Bureau of the Fiscal Service: Monthly Treasury Statement (fiscaldata.treasury.gov) | 6 | fiscaldata.treasury.gov/about-us, Licensing: "The data on this site is available to copy, adapt, redistribute, or otherwise use for non-commercial and commercial purposes." treasurydirect.gov/terms.htm (covers fiscal.treasury.gov): "Government works generally cannot be copyrighted". | usable |
| U.S. Department of the Treasury: Agency Financial Report (home.treasury.gov) | 1 | The site's own copyright page could not be fetched. The basis is 17 U.S.C. 105(a) (law.cornell.edu/uscode/text/17/105): "Copyright protection under this title is not available for any work of the United States Government"; the Treasury-run MyMoney.gov says: "No copyright may be claimed for any work on this web site that was created or maintained by Federal employee in the course of their duties." | usable, resting on 17 U.S.C. 105 |

Considered and not used: GAO and the State Department (the servers refuse requests that identify themselves honestly, so nothing
was fetched), FDIC (fdic.gov/about/website-policies has no affirmative public-domain statement: unclear), the Department of Labor
and HUD (policy pages not retrievable: unclear), SEC EDGAR (terms not checked).

## Fonts of the generated sets

Held out of every training pool: Source Sans 3, Source Serif 4, IBM Plex Sans, IBM Plex Serif, Gelasio, Carlito, Caladea (the held-out
families of the line and page sets) and IBM Plex Mono, DM Mono, Barlow. All are SIL Open Font License 1.1, each family's own
`OFL.txt` from the Google Fonts repository at the commit pinned in `manifests/assets.json` and `manifests/fin-assets.json` (file URLs and
SHA-256 of every font file and licence file). Used unmodified, so Reserved Font Names do not apply: "Permission is hereby granted,
free of charge, to any person obtaining a copy of the Font Software, to use, study, copy, merge, embed, modify, redistribute, and sell
modified and unmodified copies of the Font Software". Fonts are rendered into page images; the font files are not distributed.
