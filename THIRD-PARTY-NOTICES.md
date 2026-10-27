# Third-party notices

## Unicode Character Database 17.0.0

`src/OneOcr/Unicode/NormalizationTables.g.cs` is generated from `UnicodeData.txt` and `CompositionExclusions.txt` of the Unicode
Character Database 17.0.0 (https://www.unicode.org/Public/17.0.0/ucd/). Those data files are used under the Unicode License v3,
reproduced here.

```
UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2026 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.
```


## Training and evaluation data behind the shipped models

The weights in `RecursiveAI.OneOcr.Models` and `RecursiveAI.OneOcr.Models.Documents` are trained from random initialization
on the data below. No third-party pretrained weights are used. None of the data or fonts is redistributed in this repository
or in the packages; the packages carry only our own weights. The notices are here because attribution and licence terms
were followed in making them, and so that you can check where everything came from. The scripts that fetch each source, with
pinned hashes, are in `training/datasets/` and `training/data/`; `training/datasets/LICENCES.md` has the licence reading for
each dataset with the operative clauses.

### CORD v2

Receipt images and annotations for training the detector, the recognizer's receipt fine-tune and the extraction model's
receipt examples, and for evaluation numbers.

- Park, S., Shin, S., Lee, B., Lee, J., Surh, J., Seo, M., Lee, H.: "CORD: A Consolidated Receipt Dataset for Post-OCR
  Parsing", Document Intelligence Workshop at NeurIPS 2019. https://github.com/clovaai/cord
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), http://creativecommons.org/licenses/by/4.0/. The
  dataset's README states: "This work is licensed under a Creative Commons Attribution 4.0 International License".
- Changes: word crops and boxes were cut from the images and used with their labels; the weights are a derived model.
- The images are not redistributed.

### DocLayNet 1.0.0

Page images, layout annotations and (from the "extra" archive) text cells, for training the page detector.

- Pfitzmann, B., Auer, C., Dolfi, M., Nassar, A. S., Staar, P.: "DocLayNet: A Large Human-Annotated Dataset for
  Document-Layout Analysis", KDD 2022. https://github.com/DS4SD/DocLayNet
- Licence: Community Data License Agreement – Permissive, Version 1.0 (CDLA-Permissive-1.0),
  https://cdla.dev/permissive-1-0/. Section 3.4: "This Agreement imposes no obligations or restrictions on Your Use or
  Publication of Results." Models trained on the data are Results.
- The pages are not redistributed.

### PubLayNet 1.0.0

Layout annotations and page images, for training the page detector.

- Zhong, X., Tang, J., Jimeno Yepes, A.: "PubLayNet: largest dataset ever for document layout analysis", ICDAR 2019.
  https://github.com/ibm-aur-nlp/PubLayNet
- The annotations belong to IBM and are licensed under CDLA-Permissive-1.0 (https://cdla.dev/permissive-1-0/).
- The images come from the PubMed Central Open Access Subset. IBM does not own their copyright, and use of the images must
  abide by the PMC Open Access Subset Terms of Use. We trained only on images of articles whose PMC licence code is exactly
  CC0 or CC BY (training shard 0: 38,309 of 39,841 articles; validation: 7,755 of 7,797). Attribution for CC BY articles is
  owed to each article's authors; the article identifiers used are listed by `training/datasets/fetch_publaynet.py` in
  `licence/pmcids_*.txt` when the data is fetched. The images are not redistributed.

### Project Gutenberg texts

Public-domain books in 19 languages (217 titles; the identifiers are in `training/data/texts.yaml`) supply the text that is
rendered into synthetic line images, and the text the character language model (`charlm-v1`) and the word frequency table (`wordfreq-v1`) are estimated from. Two further
books (identifiers 1342 and 11) are used only as prose for evaluation pages. Only the body between the START and END markers
is used, with the Project Gutenberg trademark text removed. Terms: https://www.gutenberg.org/policy/permission.html — for
public-domain titles "nobody can grant, or withhold, permission to do with this item as you please", "including any commercial
use".

### Fonts used to render synthetic training data

Synthetic lines and pages are rendered from open fonts only, taken from the Google Fonts repository
(https://github.com/google/fonts) at commit e7db6ca603fd4862010383ff17ffd6cc4ffac1c0 and checked against the hash of each
family's own licence file. 77 families, 200 font files: 72 under the SIL Open Font License 1.1
(https://openfontlicense.org), 2 under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0),
3 under the Ubuntu Font Licence 1.0 (https://ubuntu.com/legal/font-licence). The fonts are used to draw
images of text and are not part of any package. Families marked "heldout" were used only to build evaluation sets.
The font files and the full licence texts are in the Google Fonts repository under each family's directory.

| Family | Licence | Files | Use | Copyright |
|---|---|---|---|---|
| abrilfatface | OFL-1.1 | 1 | heldout | (c) 2011, TypeTogether (www.type-together.com), |
| alegreya | OFL-1.1 | 2 | train | 2011 The Alegreya Project Authors (https://github.com/huertatipografica/Alegreya) |
| alegreyasans | OFL-1.1 | 8 | heldout | 2013 The Alegreya Sans Project Authors (https://github.com/huertatipografica/Alegreya-Sans) |
| amaticsc | OFL-1.1 | 2 | train | 2015 The Amatic SC Project Authors (https://github.com/googlefonts/AmaticSC) |
| anonymouspro | OFL-1.1 | 4 | train | (c) 2009, Mark Simonson (http://www.ms-studio.com, mark@marksimonson.com), |
| anton | OFL-1.1 | 1 | train | 2020 The Anton Project Authors (https://github.com/googlefonts/AntonFont.git) |
| arimo | OFL-1.1 | 2 | train | 2026 The Arimo Project Authors (https://github.com/googlefonts/arimo) |
| arvo | OFL-1.1 | 4 | train | (c) 2010-2013, Anton Koovit (anton@korkork.com), with Reserved Font Name 'Arvo' |
| bangers | OFL-1.1 | 1 | train | 2010 The Bangers Project Authors (https://github.com/googlefonts/bangers) |
| bitter | OFL-1.1 | 2 | train | 2011 The Bitter Project Authors (https://github.com/solmatas/BitterPro) |
| cabin | OFL-1.1 | 2 | heldout | 2018 The Cabin Project Authors (https://github.com/impallari/Cabin.git) |
| cardo | OFL-1.1 | 3 | train | (c) 2002-2011, David J. Perry (hospes02@scholarsfonts.net) |
| comfortaa | OFL-1.1 | 1 | train | 2011 The Comfortaa Project Authors (https://github.com/alexeiva/comfortaa), with Reserved Font Name "Comfortaa". |
| cormorantgaramond | OFL-1.1 | 2 | train | 2015 the Cormorant Project Authors (github.com/CatharsisFonts/Cormorant) |
| courierprime | OFL-1.1 | 4 | train | 2015 The Courier Prime Project Authors (https://github.com/quoteunquoteapps/CourierPrime). |
| cousine | OFL-1.1 | 4 | train | 2026 The Cousine Project Authors (https://github.com/googlefonts/cousine) |
| crimsonpro | OFL-1.1 | 2 | train | 2018 The Crimson Pro Project Authors (https://github.com/Fonthausen/CrimsonPro) |
| crimsontext | OFL-1.1 | 6 | train | 2010 The Crimson Text Project Authors (https://github.com/googlefonts/Crimson) |
| cuprum | OFL-1.1 | 2 | train | 2020 The Cuprum Project Authors (https://github.com/alexeiva/cuprum), with Reserved Font Name "Cuprum". |
| cutivemono | OFL-1.1 | 1 | train | 2012 The Cutive Project Authors (https://github.com/googlefonts/cutivemono) |
| didactgothic | OFL-1.1 | 1 | train | (c) 2009, 2010, 2011 Daniel Johnson (<il.basso.buffo@gmail.com>). |
| domine | OFL-1.1 | 1 | heldout | 2020 The Domine Project Authors (https://github.com/googlefonts/domine) |
| ebgaramond | OFL-1.1 | 2 | train | 2017 The EB Garamond Project Authors (https://github.com/octaviopardo/EBGaramond12) |
| exo2 | OFL-1.1 | 2 | train | 2013 The Exo 2 Project Authors (https://github.com/googlefonts/Exo-2.0) |
| firamono | OFL-1.1 | 3 | train | (c) 2012-2013, The Mozilla Corporation and Telefonica S.A. |
| firasans | OFL-1.1 | 8 | train | (c) 2012-2015, The Mozilla Foundation and Telefonica S.A. |
| inconsolata | OFL-1.1 | 1 | train | 2006 The Inconsolata Project Authors |
| inter | OFL-1.1 | 2 | train | 2020 The Inter Project Authors (https://github.com/rsms/inter) |
| istokweb | OFL-1.1 | 4 | train | (c) 2008-2012, 2014, Andrey V. Panov (panov@canopus.iacp.dvo.ru), |
| jetbrainsmono | OFL-1.1 | 2 | heldout | 2020 The JetBrains Mono Project Authors (https://github.com/JetBrains/JetBrainsMono) |
| jost | OFL-1.1 | 2 | train | 2020 The Jost Project Authors (https://github.com/indestructible-type) |
| karla | OFL-1.1 | 2 | heldout | 2019 The Karla Project Authors (https://github.com/googlefonts/karla) |
| lato | OFL-1.1 | 8 | train | (c) 2010-2014 by tyPoland Lukasz Dziedzic (team@latofonts.com) with Reserved Font Name "Lato" |
| ledger | OFL-1.1 | 1 | train | (c) 2012, Denis Masharov <denis.masharov@gmail.com>, |
| librebaskerville | OFL-1.1 | 2 | train | 2012 The Libre Baskerville Project Authors (https://github.com/impallari/Libre-Baskerville) with Reserved Font Name Libre Baskerville. |
| literata | OFL-1.1 | 2 | train | 2017 The Literata Project Authors (https://github.com/googlefonts/literata) |
| lobster | OFL-1.1 | 1 | train | 2010 The Lobster Project Authors (https://github.com/impallari/The-Lobster-Font), with Reserved Font Name "Lobster". |
| lora | OFL-1.1 | 2 | train | 2011 The Lora Project Authors (https://github.com/cyrealtype/Lora-Cyrillic), with Reserved Font Name "Lora". |
| manrope | OFL-1.1 | 1 | train | 2018 The Manrope Project Authors (https://github.com/googlefonts/manrope) |
| merriweather | OFL-1.1 | 2 | train | 2020 The Merriweather Project Authors (https://github.com/EbenSorkin/Merriweather4) with Reserved Font Name "Merriweather". |
| montserrat | OFL-1.1 | 2 | train | 2024 The Montserrat.Git Project Authors (https://github.com/JulietaUla/Montserrat.git) |
| neuton | OFL-1.1 | 6 | train | 2010 The Neuton Project Authors (http://www.21326.info/), with Reserved Font Name "Neuton". |
| notosans | OFL-1.1 | 2 | train | 2022 The Noto Project Authors (https://github.com/notofonts/latin-greek-cyrillic) |
| notosansmono | OFL-1.1 | 1 | train | 2022 The Noto Project Authors (https://github.com/notofonts/latin-greek-cyrillic) |
| notoserif | OFL-1.1 | 2 | train | 2022 The Noto Project Authors (https://github.com/notofonts/latin-greek-cyrillic) |
| oldstandardtt | OFL-1.1 | 3 | train | 2011 The Old Standard Project Authors (amkryukov@gmail.com) |
| opensans | OFL-1.1 | 2 | train | 2020 The Open Sans Project Authors (https://github.com/googlefonts/opensans) |
| oswald | OFL-1.1 | 1 | train | 2016 The Oswald Project Authors (https://github.com/googlefonts/OswaldFont) |
| pacifico | OFL-1.1 | 1 | train | 2018 The Pacifico Project Authors (https://github.com/googlefonts/Pacifico) |
| philosopher | OFL-1.1 | 4 | train | 2011 The Philosopher Project Authors (https://github.com/alexeiva/philosopher) |
| play | OFL-1.1 | 2 | train | (c) 2011, Jonas Hecksher, Playtypes, e-types AS (lasse@e-types.com), with Reserved Font Name 'Play', 'Playtype', 'Playtype Sans'. |
| playfairdisplay | OFL-1.1 | 2 | train | 2017 The Playfair Display Project Authors (https://github.com/clauseggers/Playfair-Display), with Reserved Font Name "Playfair Display" |
| ptmono | OFL-1.1 | 1 | train | Copyright (c) 2011, ParaType Ltd. (http://www.paratype.com/public), |
| ptsans | OFL-1.1 | 4 | train | Copyright (c) 2010, ParaType Ltd. (http://www.paratype.com/public), |
| ptsansnarrow | OFL-1.1 | 2 | heldout | Copyright (c) 2010, ParaType Ltd. (http://www.paratype.com/public), |
| ptserif | OFL-1.1 | 4 | train | (c) 2010, ParaType Ltd. (http://www.paratype.com/public), |
| roboto | OFL-1.1 | 2 | train | 2011 The Roboto Project Authors (https://github.com/googlefonts/roboto-classic) |
| robotocondensed | OFL-1.1 | 2 | train | 2011 The Roboto Project Authors (https://github.com/googlefonts/roboto-classic) |
| robotomono | OFL-1.1 | 2 | heldout | 2015 The Roboto Mono Project Authors (https://github.com/googlefonts/robotomono) |
| robotoslab | Apache-2.0 | 1 | train |  |
| rubik | OFL-1.1 | 2 | train | 2015 The Rubik Project Authors (https://github.com/googlefonts/rubik) |
| ruslandisplay | OFL-1.1 | 1 | train | 2011 Oleg Snarsky, Denis Masharov, Vladimir Rabdu, with Reserved Font Name 'Ruslan', 'Ruslan Display'. |
| russoone | OFL-1.1 | 1 | train | (c) 2011-2012, Jovanny Lemonad (jovanny.ru), with Reserved Font Name "Russo" |
| scada | OFL-1.1 | 4 | train | 2012 The Scada Project Authors (lemonad@jovanny.ru) |
| sharetechmono | OFL-1.1 | 1 | train | (c) 2012, Carrois Type Design, Ralph du Carrois (post@carrois.com www.carrois.com), with Reserved Font Name 'Share' |
| sourcecodepro | OFL-1.1 | 2 | train | 2010, 2012 Adobe Systems Incorporated (http://www.adobe.com/), with Reserved Font Name 'Source'. All Rights Reserved. Source is a trademark of Adobe Systems Incorporated in the United States and/or other countries. |
| spacemono | OFL-1.1 | 4 | train | 2016 The Space Mono Project Authors (https://github.com/googlefonts/spacemono) |
| specialelite | Apache-2.0 | 1 | train |  |
| spectral | OFL-1.1 | 8 | heldout | 2017 The Spectral Project Authors (https://github.com/productiontype/Spectral) |
| tenorsans | OFL-1.1 | 1 | train | (c) 2011, Denis Masharov <denis.masharov@gmail.com>, |
| ubuntu | UFL-1.0 | 8 | train | ------------------------------- |
| ubuntucondensed | UFL-1.0 | 1 | train | ------------------------------- |
| ubuntumono | UFL-1.0 | 4 | heldout | ------------------------------- |
| vollkorn | OFL-1.1 | 2 | heldout | 2017 The Vollkorn Project Authors (https://github.com/FAlthausen/Vollkorn-Typeface) |
| worksans | OFL-1.1 | 2 | train | 2019 The Work Sans Project Authors (https://github.com/weiweihuanghuang/Work-Sans) |
| yanonekaffeesatz | OFL-1.1 | 1 | heldout | 2010 The Yanone Kaffeesatz Project Authors (https://github.com/yanone/kaffeesatz) |
| zillaslab | OFL-1.1 | 8 | train | 2017, The Mozilla Foundation |

Screenshot-style training pages are also rendered with the Noto and Liberation fonts as installed from the Debian/Ubuntu
packages (SIL Open Font License 1.1) and the Ubuntu font family (Ubuntu Font Licence 1.0).

The evaluation pages in `bench/` additionally use these fonts, all under the SIL Open Font License 1.1, from the same Google
Fonts commit (see `bench/manifests/assets.json` for the file hashes). They are kept out of recognizer training.

| Family | Copyright |
|---|---|
| Gelasio | Copyright 2022 The Gelasio Project Authors (https://github.com/SorkinType/Gelasio) |
| Carlito | Copyright 2013 The Carlito Project Authors (https://github.com/googlefonts/carlito), with Reserved Font Name "Carlito" |
| Caladea | Copyright 2012 The Caladea Project Authors (https://github.com/huertatipografica/Caladea) |
| Source Serif 4 | Copyright 2014 The Source Serif 4 Project Authors (https://github.com/adobe-fonts/source-serif) |
| Source Sans 3 | Copyright 2010-2020 Adobe (http://www.adobe.com/), with Reserved Font Name 'Source'. All Rights Reserved. Source is a trademark of Adobe in the United States and/or other countries. |
| IBM Plex Serif | Copyright © 2017 IBM Corp. with Reserved Font Name "Plex" |
| IBM Plex Sans | Copyright © 2017 IBM Corp. with Reserved Font Name "Plex" |

### Evaluation sources a user can fetch

The benchmark harness in `bench/` can build evaluation pages from sources it fetches itself; nothing from them is committed.

- **US federal government works.** Pages and PDFs published by US federal agencies. Works of the United States Government are
  not subject to US copyright: 17 U.S.C. § 105(a), "Copyright protection under this title is not available for any work of
  the United States Government". `realweb.py` skips pages whose text carries a copyright or permission notice. You supply the
  list of PDFs and pages to `realpdf.py` and `realweb.py`, and you are responsible for choosing only federal works.
- **U.S. Web Design System (USWDS) pages.** https://github.com/uswds/uswds — the project is in the public domain, and
  contributions by the General Services Administration are released under CC0 1.0 Universal
  (https://creativecommons.org/publicdomain/zero/1.0/). Fonts, icons and style files bundled in it keep their own licences
  (SIL Open Font License 1.1 for Source Sans, Merriweather and Public Sans; Apache License 2.0 for Roboto Mono and Material
  icons); the harness captures rendered page text, not those files.
- **Wikipedia articles**, when the harness is given a title, are fetched as a permanent revision link and are available under
  CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/); the pages are used for local evaluation only and are not
  redistributed.
- **Our own generated pages** have no third-party terms; the generators and seeds are in `bench/`.
