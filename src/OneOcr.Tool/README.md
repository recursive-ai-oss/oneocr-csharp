# oneocr

The command-line tool: make searchable PDFs (`pdf`), read pages with the local engine (`ocr`), and fine-tune a OneOcr recognizer on your own documents (`train`, `eval`, `review`).

```
oneocr ocr --in page.png                      # text to stdout
oneocr ocr --in scans/ --out text/ --json     # one .txt and one .page.json (the page model) per image
```

`ocr` reads PNG and JPEG with the bundled models and never touches the network. A file that cannot be read (damaged, not an image, a PDF) is reported as `oneocr: <name>: <message>` on stderr, the other files are still read, and the exit code is 3; the tool does not read PDFs, so render them first or use `RecursiveAI.OneOcr.Pdf` from code. `oneocr --version` prints the version. `--degree N` caps the threads (default min(4, cores)); the text does not depend on it.

### Searchable PDFs

```
oneocr pdf scan.jpg --out scan.pdf                      # one image, one page
oneocr pdf pages/ fax.tif --out book.pdf --pdfa         # a folder (by file name), then a multi-frame TIFF; PDF/A-2b
oneocr pdf scanned.pdf --out searchable.pdf             # a PDF: pages that need a text layer get one, every other page is untouched
```

Inputs come first, then the options. Images (PNG, JPEG, TIFF with any number of frames) become the pages of one PDF, each with an invisible text layer so
selection, search and copy match the page; the image is embedded as it is (a JPEG as its own bytes, a 1-bit TIFF page as 1-bit). One PDF in gives a copy
with a layer on the pages that need one (scans, pages with another OCR step's invisible layer, garbled or wrongly mapped text) and the other pages left
alone; images and a PDF, or several PDFs, are not mixed. `--out` is required and is replaced only when the whole file was written (it may be the input
PDF itself). `--pdfa` writes PDF/A-2b, `--title` sets the title of a PDF made from images, `--dpi N` sets the resolution of images that declare none
(default: estimated from the text size), `--keep-ocr-layer` trusts an invisible OCR layer already in the PDF instead of replacing it, and `--model`,
`--detector`, `--wordlm`, `--wordfreq` and `--degree` work as for `ocr`. Stdout gets one line (`book.pdf: 5 pages, 5 given a text layer, 412 words, 3.1 s`);
notes about individual pages (an earlier layer replaced, unreliable text marked) go to stderr. A bad input is one `oneocr: ...` line and exit code 1
(unreadable file) or 2 (wrong usage). Local only, never touches the network; the same input gives the same file.

### Starting fast: publish it NativeAOT

The tool is AOT-clean. `dotnet tool install` gives a framework-dependent app that JIT-compiles on every start, which costs about 300 ms
before the first page is read. For batch jobs and scripts that start the process often, publish a native executable instead:

```
dotnet publish src/OneOcr.Tool -c Release -r linux-x64      # or win-x64, osx-arm64, linux-arm64
dotnet publish src/OneOcr.Tool -c Release -r linux-x64 -p:IlcInstructionSet=x86-64-v4   # AVX-512 machines
```

A native binary is compiled for one instruction-set level. The default for x64 is AVX2 + FMA (`x86-64-v3`, any x86-64 CPU since about
2015); older CPUs refuse to start it. Do not publish x64 below that level: the reader would run its exact scalar path, about a hundred times
slower. `x86-64-v4` uses the AVX-512 kernels on machines that have them. Output is identical at every level.

The same applies to your own apps that reference `RecursiveAI.OneOcr`: set `PublishAot` and an `IlcInstructionSet` of at least `x86-64-v3`.

```
dotnet tool install RecursiveAI.OneOcr.Tool
oneocr train --model recognizer.onm --work run1 --docs scans/ \
    --provider openai --llm-model <model> --key-from-env OPENAI_API_KEY --budget-usd 2
oneocr review export --work run1 --out review/
# edit review/review.json: set "text" on the lines you checked
oneocr review import --work run1 --file review/review.json
oneocr train --model recognizer.onm --work run1 --restart  # trains on the labels already in run1; no network
oneocr eval  --model run1/model.onm --work run1

# a custom document type: schema.json is a JSON Schema whose top-level scalar properties are the fields
oneocr train --extract schema.json --model docextract.onm --work po-run --docs purchase-orders/ \
    --provider openai-compatible --base-url http://localhost:8080/v1 --llm-model <local model>
```

`train --extract` fine-tunes the document-extraction model for your schema from values the labelling model found *on the page* (the report lists held-out accuracy before and after). `train` with `--docs` sends page images to the provider you name, under your own agreement with it. API keys come only from `--key` or from the environment variable you name with `--key-from-env`; nothing is read from the environment otherwise. `eval`, `review export|import` and `train` without `--docs` never touch the network. Using a provider's output to train a model is governed by your own agreement with it; a local open-weight model behind the OpenAI-compatible provider avoids the question. Run `oneocr help` for every option, and see the RecursiveAI.OneOcr.Training package for how lines are chosen for training and for a note on the terms of the providers you label with.
