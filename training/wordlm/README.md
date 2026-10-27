# Word frequency table builder

`build_wordlm.py` writes the word frequency file that the page reader uses to take back a space the network put inside a word
(format: `docs/spec/word-lm-v1.md`, use: `docs/spec/oneocr-page-v1.md`, "Word segmentation"). Python standard library only.

    python3 training/wordlm/build_wordlm.py --texts DIR --web WEB/train/labels.jsonl --out wordfreq-v2.bin

`DIR` holds `pg<ID>.txt` for the books of `training/data/texts.yaml` (`uv run python -m data.texts fetch` downloads them); every file
is checked against the SHA-256 in the manifest. Only Latin-script books marked `train` are read, the same books the character
language model is built from, so the held-out books and the two books the page evaluation sets are set in never reach the table.

Options: `--langs all|en` (Latin-script books of every language, or English only; default `all`) and `--min-count` (default 3).
Output is deterministic. The default settings give 1,305,931 bytes and 216,859 words from 10.3 million tokens; English only is
0.17 MB and 27,000 words. Counting words seen twice instead of three times (1.9 MB) changed no page output.

## Modern vocabulary from the web (`wordfreq-v2`)

Books alone lack words like `stakeholder`, `genealogy` or `validating`. `--web LABELS.jsonl` adds the visible page text of the public-domain US
federal sites rendered by `training/gen/web_lines.py` (its `train/labels.jsonl`; only publishers marked `split: train` in
`training/data/web_sources.yaml` are read, so the held-out sites never reach the table). Each line text counts once per publisher, a word must
occur in at least `--web-min-lines` (2) distinct lines, and the web count is multiplied by `--web-weight` (default 1) before it joins the book count.

The weight is 1: the web counts are plain observed counts, added to the books'. A larger weight (5, 10, 20) taught the table that
`timeframes` is one word, which joined the two words of `time frames` on two held-out real layout pages and cost more there than the extra
modern words won; weight 1 lost nothing on any held-out set and kept most of the gain. The table is 1,316,145 bytes and 218,677 words from
10.54 million tokens (the books' 10.33 million plus 209,000 from 31,000 distinct lines; 1,818 words added).
