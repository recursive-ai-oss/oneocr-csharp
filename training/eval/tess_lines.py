"""Runs the Tesseract binary (stock models, black box) over a rendered line set and writes hypothesis records.

    uv run python -m eval.tess_lines --set-dir DIR --out OUT --models best,fast --psms 7,13 --scales 1,3 [--by-script]

DIR holds images/<id>.png and labels.jsonl (the same crops our own models read). For every model x page-segmentation mode x
scale one JSONL of {"id", "ref", "hyp"} is written to OUT/<model>-psm<psm>-x<scale>.hyps.jsonl. The scale is an integer
bicubic enlargement of the identical crop (1 = the crop as is): the engine's documentation advises larger text, so the report
takes the best configuration per category, which can only favour the engine. Only documented command-line options are used:
the list-file input (one process reads many images), --psm, -l, --tessdata-dir and OMP_THREAD_LIMIT.

--by-script reads each line with the engine's own model for its script (eng, ell, rus; selected from the reference text, which
is the fair comparison for Greek and Cyrillic lines); without it every line is read with eng.
When the engine process dies on a crop (a division by zero on some tiny blurred crops) that crop counts as an empty reading;
the crops are listed in OUT/crashes.txt.
Tesseract output is only ever scored against the exact truth; it is never a label.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from eval.scoring import line_script, normalize

TESS = os.environ.get("ONEOCR_TESS", str(Path.home() / "tools/tesseract/tess"))
LANG = {"latin": "eng", "greek": "ell", "cyrillic": "rus"}
CHUNK = 250


CRASHES: list[str] = []


def run_chunk(model: str, lang: str, psm: int, paths: list[Path], work: Path, tag: str) -> list[str]:
    """One engine process for the chunk; when it dies (it does on some tiny degenerate crops) the chunk is halved until the
    offending image stands alone, which then counts as an empty reading and is recorded in CRASHES."""
    try:
        return run_once(model, lang, psm, paths, work, tag)
    except subprocess.CalledProcessError:
        if len(paths) == 1:
            CRASHES.append(f"{model} psm{psm} {paths[0].name}")
            return [""]
        h = len(paths) // 2
        return run_chunk(model, lang, psm, paths[:h], work, tag + "a") + run_chunk(model, lang, psm, paths[h:], work, tag + "b")


def run_once(model: str, lang: str, psm: int, paths: list[Path], work: Path, tag: str) -> list[str]:
    lst = work / f"{tag}.lst"
    lst.write_text("".join(f"{p}\n" for p in paths), encoding="utf-8")
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    base = work / tag
    subprocess.run([TESS, model, str(lst), str(base), "-l", lang, "--psm", str(psm), "-c", "tessedit_create_txt=1"],
                   env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    pages = Path(f"{base}.txt").read_text(encoding="utf-8")
    parts = pages.split("\f")
    if len(parts) < len(paths):
        raise RuntimeError(f"{tag}: {len(parts)} pages for {len(paths)} images")
    return [" ".join(p.split()) for p in parts[: len(paths)]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="best,fast")
    ap.add_argument("--psms", default="7,13")
    ap.add_argument("--scales", default="1,3")
    ap.add_argument("--jobs", type=int, default=28)
    ap.add_argument("--by-script", action="store_true")
    ap.add_argument("--tmp", required=True, help="scratch directory for scaled crops and list files")
    a = ap.parse_args()
    root, out, tmp = Path(a.set_dir), Path(a.out), Path(a.tmp)
    out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (root / "labels.jsonl").read_text(encoding="utf-8").splitlines() if x]
    langs = [LANG[line_script(normalize(r["text"]))] if a.by_script else "eng" for r in rows]
    for scale in map(int, a.scales.split(",")):
        sdir = tmp / f"x{scale}"
        sdir.mkdir(exist_ok=True)
        paths = []
        for r in rows:
            src = root / "images" / f"{r['id']}.png"
            if scale == 1:
                paths.append(src)
                continue
            dst = sdir / f"{r['id']}.png"
            if not dst.exists():
                with Image.open(src) as im:
                    im.resize((im.width * scale, im.height * scale), Image.BICUBIC).save(dst)
            paths.append(dst)
        for model in a.models.split(","):
            for psm in map(int, a.psms.split(",")):
                jobs = []
                for lang in sorted(set(langs)):
                    ids = [i for i, lg in enumerate(langs) if lg == lang]
                    for c in range(0, len(ids), CHUNK):
                        jobs.append((lang, ids[c : c + CHUNK]))
                hyp = [""] * len(rows)
                tag0 = f"{model}-p{psm}-x{scale}"
                with ThreadPoolExecutor(a.jobs) as ex:
                    futs = [ex.submit(run_chunk, model, lang, psm, [paths[i] for i in ids], tmp, f"{tag0}-{n}")
                            for n, (lang, ids) in enumerate(jobs)]  # fmt: skip
                    for (_, ids), f in zip(jobs, futs, strict=True):
                        for i, h in zip(ids, f.result(), strict=True):
                            hyp[i] = h
                name = out / f"{model}-psm{psm}-x{scale}.hyps.jsonl"
                name.write_text("".join(json.dumps({"id": r["id"], "ref": r["text"], "hyp": h}, ensure_ascii=False) + "\n"
                                        for r, h in zip(rows, hyp, strict=True)), encoding="utf-8")  # fmt: skip
                print(name.name, f"({len(CRASHES)} engine crashes so far)", flush=True)
        for f in tmp.glob(f"*-x{scale}-*"):
            f.unlink()
        if scale != 1:
            shutil.rmtree(sdir)
    (out / "crashes.txt").write_text("\n".join(sorted(set(CRASHES))) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
