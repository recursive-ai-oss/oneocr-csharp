# Word frequency table builder

`build_wordlm.py` writes the word frequency file that the page reader uses to take back a space the network put inside a word
(format: `docs/spec/word-lm-v1.md`, use: `docs/spec/oneocr-page-v1.md`, "Word segmentation"). Python standard library only.

    python3 training/wordlm/build_wordlm.py --texts DIR --out wordfreq-v1.bin

`DIR` holds `pg<ID>.txt` for the books of `training/data/texts.yaml` (`uv run python -m data.texts fetch` downloads them); every file
is checked against the SHA-256 in the manifest. Only Latin-script books marked `train` are read, the same books the character
language model is built from, so the held-out books and the two books the page evaluation sets are set in never reach the table.

Options: `--langs all|en` (Latin-script books of every language, or English only; default `all`) and `--min-count` (default 3).
Output is deterministic. The default settings give 1,305,931 bytes and 216,859 words from 10.3 million tokens; English only is
0.17 MB and 27,000 words. Counting words seen twice instead of three times (1.9 MB) changed no page output.
