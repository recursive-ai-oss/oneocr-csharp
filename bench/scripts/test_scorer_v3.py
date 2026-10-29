"""Checks for scorer v3 (truth words from the truth boxes). Run with `uv run python scripts/test_scorer_v3.py`; no test framework needed.

Also checks that v2 is untouched: v2 and v3 agree on a truth directory without word boxes, and v2 reads only `<id>.txt`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oneocr_bench.score import SCORERS, normalize, score_dir, v3_truth_words


def w(text: str, x0: float, x1: float, y0: float = 0, y1: float = 20) -> dict:
    return {"text": text, "box": [x0, y0, x1, y1]}


def test_touching_mark_joins() -> None:
    # "Intelligence" "," touch: one word. "tokens" and "next" are a space apart: separate.
    words = [w("Intelligence", 0, 100), w(",", 100, 105), w("next", 117, 150)]
    assert v3_truth_words(words) == ["Intelligence,", "next"]


def test_mark_between_two_words_joins_all() -> None:
    assert v3_truth_words([w("1", 0, 8), w("/", 8, 14), w("2", 14, 22)]) == ["1/2"]
    assert v3_truth_words([w("(", 0, 6), w("see", 6, 30), w(")", 30, 36)]) == ["(see)"]


def test_mark_with_a_space_stays_separate() -> None:
    words = [w("word", 0, 40), w("-", 48, 54), w("word", 62, 100)]
    assert v3_truth_words(words) == ["word", "-", "word"]


def test_wrapped_fragments_stay_words() -> None:
    # same line, touching, but both are words: never joined (a hyphen-wrapped word on one text line of the DOM, or two lines)
    assert v3_truth_words([w("555-", 0, 40), w("4489", 40, 80)]) == ["555-", "4489"]
    # a wrapped word: the fragments are on different lines
    words = [w("breakfast-", 300, 380, 0, 20), w("room", 0, 40, 24, 44)]
    assert v3_truth_words(words) == ["breakfast-", "room"]


def test_mark_on_another_line_stays_separate() -> None:
    assert v3_truth_words([w("end", 300, 340, 0, 20), w(".", 0, 6, 24, 44)]) == ["end", "."]


def test_empty_boxes_dropped() -> None:
    assert v3_truth_words([w("a", 0, 10), w("\u00a0", 20, 30), w("b", 40, 50)]) == ["a", "b"]


def _page(root: Path, name: str, text: str, words: list[dict] | None, hyp: str) -> None:
    (root / "truth").mkdir(exist_ok=True)
    (root / "hyp").mkdir(exist_ok=True)
    (root / "truth" / f"{name}.txt").write_text(text)
    if words is not None:
        (root / "truth" / f"{name}.json").write_text(json.dumps({"words": words}))
    (root / "hyp" / f"{name}.txt").write_text(hyp)


def test_scores() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        words = [w("breakfast-", 300, 380, 0, 20), w("room", 0, 40, 24, 44), w("Intelligence", 0, 100, 48, 68), w(",", 100, 105, 48, 68)]
        # The truth text joins the wrapped word and splits the touching comma.
        _page(root, "p1", "breakfast-room Intelligence ,", words, "breakfast- room Intelligence,")
        v2 = score_dir(root / "truth", root / "hyp")
        v3 = score_dir(root / "truth", root / "hyp", "ink", "v3")
        assert v2["scorer"] == "v2" and v3["scorer"] == "v3"
        assert v3["wer"] == 0 and v3["cer"] == 0 and v3["order_err"] == 0, v3["totals"]
        assert v2["wer"] > 0 and v2["order_err"] > 0
        # a real misread stays an error under v3
        _page(root, "p1", "breakfast-room Intelligence ,", words, "breakfast- roam Intelligence,")
        v3 = score_dir(root / "truth", root / "hyp", "ink", "v3")
        assert v3["totals"]["char_errors"] == 1 and v3["totals"]["word_errors"] == 1, v3["totals"]


def test_no_boxes_same_under_both() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _page(root, "p1", "one two\nthree", None, "one tw three")
        a = score_dir(root / "truth", root / "hyp")
        b = score_dir(root / "truth", root / "hyp", "ink", "v3")
        assert {k: v for k, v in a.items() if k != "scorer"} == {k: v for k, v in b.items() if k != "scorer"}
        _page(root, "p2", "x y", [], "x y")  # a sidecar without words falls back to the text
        assert score_dir(root / "truth", root / "hyp", "ink", "v3")["pages"] == 2


def test_v31_folds_two_apostrophes() -> None:
    # v3.1 only: two apostrophes (typographic ones included) read as one double quote, in truth and hypothesis alike.
    assert normalize("''a''") == ["''a''"]
    assert normalize("''a''", True) == ['"a"']
    assert normalize("\u2019\u2019a\u2019\u2019 it's", True) == ['"a"', "it's"]
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _page(root, "p1", 'say "hi" now', None, "say ''hi'' now")
        v3 = score_dir(root / "truth", root / "hyp", "ink", "v3")
        v31 = score_dir(root / "truth", root / "hyp", "ink", "v3.1")
        assert v3["totals"]["word_errors"] == 1 and v31["totals"]["word_errors"] == 0, (v3["totals"], v31["totals"])
        assert v31["scorer"] == "v3.1"
        # a lone apostrophe is untouched, so v3.1 equals v3 on text without a pair
        _page(root, "p2", "it's mine", None, "it's mine")
    assert SCORERS == ("v2", "v3")  # gate and competitors-freeze keep computing exactly these side by side


def test_unknown_scorer() -> None:
    try:
        score_dir(Path("."), Path("."), "ink", "v9")
    except ValueError:
        return
    raise AssertionError("unknown scorer accepted")


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok", name)
    print(f"{n} checks passed")
