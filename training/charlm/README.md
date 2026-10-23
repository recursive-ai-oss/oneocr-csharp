# Character language model builder

`build_charlm.py` writes the word-level character n-gram file that beam-search decoding can use (format:
`docs/spec/char-lm-v1.md`, use: `docs/spec/decoding.md`). Python standard library only.

    python3 training/charlm/build_charlm.py --texts DIR --out charlm-v1.bin

`DIR` holds `pg<ID>.txt` for the books of `training/data/texts.yaml` (`uv run python -m data.texts fetch` downloads them); every file
is checked against the SHA-256 in the manifest. Only books marked `train` are read, so the held-out books the recognizer is validated
on and the two books the page evaluation sets are set in never reach the model. Language decides the script (`el` Greek;
`ru`, `bg`, `sr` Cyrillic; every other language Latin).

Options: `--order` (default 4), `--min-counts` (smallest effective count kept per order, default `1,3,4,6`),
`--min-symbol-count` (rarest alphabet character kept, default 5). Output is deterministic. The default settings give 1.42 MB for
the three scripts (Latin 1.02 MB, Greek 0.30 MB, Cyrillic 0.10 MB); a third-order model is 0.4 MB and a few tenths of a point
of word error rate worse on held-out synthetic lines.

Words containing a digit, a code symbol (`@ # _ / \ = % + * ~ ^ | < > [ ] { }`) or a character outside the script are left out of
the training text: the model never sees numbers or identifiers, and the decoder never scores them.
