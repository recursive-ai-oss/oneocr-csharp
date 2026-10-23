"""Line records for evaluation crops cut from page images by their truth word boxes.

Usage: uv run python -m eval.page_lines PAGES_DIR TRUTH_DIR OUT.jsonl

PAGES_DIR/<id>.png and TRUTH_DIR/<id>.json (words with text and [x0, y0, x1, y1] boxes).
Words are grouped into lines by geometry: words whose boxes start at nearly the same height (within 30% of the box height)
and follow one another left to right form a line. The line text is its words joined by single spaces; the line box is the
union of the word boxes. `LineGen --page-lines` turns the records into crops. Evaluation only: nothing here feeds training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def group_lines(words: list[dict]) -> list[list[dict]]:
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["box"][1], w["box"][0])):
        h = w["box"][3] - w["box"][1]
        for ln in lines:
            ref = ln[0]["box"]
            if abs(w["box"][1] - ref[1]) <= 0.3 * h and w["box"][0] >= ln[-1]["box"][0]:
                ln.append(w)
                break
        else:
            lines.append([w])
    return lines


def main() -> None:
    pages, truth, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows = []
    for tp in sorted(truth.glob("*.json")):
        t = json.loads(tp.read_text(encoding="utf-8"))
        for i, ln in enumerate(group_lines(t["words"])):
            box = [min(w["box"][0] for w in ln), min(w["box"][1] for w in ln), max(w["box"][2] for w in ln), max(w["box"][3] for w in ln)]
            rows.append({"id": f"{t['id']}-l{i:03d}", "text": " ".join(w["text"] for w in ln), "page": str(pages / f"{t['id']}.png"),
                         "box": [round(v, 2) for v in box]})  # fmt: skip
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} lines from {len(list(truth.glob('*.json')))} pages -> {out}")


if __name__ == "__main__":
    main()
