"""The financial-set report: per set and per capture category, our numbers beside every engine's and the best of them.

    python -m oneocr_bench fin-report --eval ROOT --runs RUNS [--ours NAME] [--out report.md] [--json report.json]

ROOT holds `<set>/truth` for the eight sets (fin-scan, fin-fax, fin-phone, fin-tables and their -real twins). RUNS holds the engines'
page outputs (`<id>.txt` and the sidecar `<id>.json` with word boxes), in the layouts the runners write:

    RUNS/ours/<set>                          our engine (named by --ours, default "ours")
    RUNS/comp/<engine-variant>[-default]/<set>   PaddleOCR and docTR variants
    RUNS/tess/<set>/hyp/<model>-psm<N>-1thread   Tesseract, every model and page-segmentation mode as its own row
    RUNS/vision/<configuration>/<set>        Apple Vision

Every number is scorer v3.2 (`score.py`, `amounts.py`): CER, WER and reading order of the content metrics; amount exact match (money,
date and account-like values read exactly in place) by box and, beside it, the order-free text-only match; running-balance
reconciliation on statements. A category pools its synthetic and real set. "Best" is the best value per metric over every other row,
so it can come from different engines. Reading order on the real sets is information only (table pages have no verified order).

For our rows the report also classifies each missed value (undetected, merged with other characters, separator or sign misread, digit misread) and gives the miss rate by capture condition.
"""

from __future__ import annotations

import difflib
import json
from collections import Counter
from pathlib import Path

from . import amounts
from .amounts import CLASSES, _centre, _inside, _key
from .score import score_dir

SETS = ("fin-scan", "fin-fax", "fin-phone", "fin-tables")
CATEGORIES = {"scan": "fin-scan", "fax": "fin-fax", "phone": "fin-phone", "tables": "fin-tables"}


def discover(runs: Path, ours: str) -> dict[str, dict[str, Path]]:
    """configuration -> {set: hypothesis directory}."""
    out: dict[str, dict[str, Path]] = {}

    def add(name: str, set_name: str, path: Path) -> None:
        if path.is_dir() and any(path.glob("*.txt")):
            out.setdefault(name, {})[set_name] = path

    for s in (runs / "ours").glob("fin-*") if (runs / "ours").is_dir() else []:
        add(ours, s.name, s)
    for c in sorted((runs / "comp").glob("*")) if (runs / "comp").is_dir() else []:
        for s in c.glob("fin-*"):
            add(c.name.removesuffix("-default"), s.name, s)
    for s in sorted((runs / "tess").glob("fin-*")) if (runs / "tess").is_dir() else []:
        for h in sorted((s / "hyp").glob("*-1thread")):
            add("tesseract-" + h.name.removesuffix("-1thread"), s.name, h)
    for c in sorted((runs / "vision").glob("*")) if (runs / "vision").is_dir() else []:
        for s in c.glob("fin-*"):
            add("vision-" + c.name, s.name, s)
    return out


def diagnose(truth: dict, hyp: list[dict], tag: dict) -> tuple[str, str, str]:
    """Why a tagged value was not read exactly in place, from the hypothesis words that lie in its region."""
    tw = truth["words"]
    reg = amounts.value_region(tw, tag["words"])
    near = sorted((h for h in hyp if _inside(_centre(h), reg)), key=lambda h: _centre(h)[0])
    if not near:
        return "undetected", "", ""
    want = "".join(amounts._nospace(tw[i]["text"]) for i in tag["words"])
    got = "".join(amounts._nospace(h["text"]) for h in near)
    if want in got:
        return "merged", want, got
    digits = lambda s: "".join(c for c in s if c.isdigit())  # noqa: E731
    if digits(got) == digits(want):
        return "separator or sign misread", want, got
    return "digit misread", want, got


def confusions(want: str, got: str) -> list[str]:
    """Character edits that turn the hypothesis text into the truth, as `truth>read` (`-` for a missing or an extra character)."""
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, want, got, autojunk=False).get_opcodes():
        if op != "equal":
            out.append(f"{want[i1:i2] or '-'}>{got[j1:j2] or '-'}")
    return out


def conditions(truth: dict) -> list[str]:
    d = truth.get("degradation") or {}
    cap = d.get("capture", truth.get("capture"))
    out = [cap]
    if cap == "fax":
        out.append("fax 1-bit " + str(d.get("fax_mode")))
    if cap == "scan" or (cap == "fax" and d.get("fax_mode") == "std"):
        if cap == "fax" or d.get("dpi", 300) <= 200:
            out.append("low resolution (scan <= 200 dpi, fax 100 dpi rows)")
    if abs(d.get("skew_deg", 0) or 0) >= 4:
        out.append("skew >= 4 degrees")
    if truth.get("set", "").startswith("fin-tables"):
        out.append("tables cut")
    out.append(f"doc: {truth.get('doc', '?')}")
    return out


def score_config(eval_root: Path, name: str, dirs: dict[str, Path], analyse: bool) -> dict:
    res: dict = {"sets": {}}
    for s, hyp in sorted(dirs.items()):
        tdir = eval_root / s / "truth"
        r = score_dir(tdir, hyp, "ink", "v3.2")
        row = {"pages": r["pages"], "cer": r["cer"], "wer": r["wer"], "order": r["order_err"], "totals": r["totals"],
               "f1": r["det"]["f1"] if r["det"] else None, "ece": r["ece"]["ece"] if r.get("ece") else None, "amounts": r.get("amounts")}
        per_doc: dict = {}
        causes: Counter = Counter()
        edits: dict[str, Counter] = {}
        qcauses: Counter = Counter()
        slices: dict = {}
        for t in sorted(tdir.glob("*.json")):
            truth = json.loads(t.read_text())
            side = hyp / f"{t.stem}.json"
            hw = json.loads(side.read_text())["words"] if side.exists() else []
            page = amounts.score_page(truth, hw)
            if not page:
                continue
            doc = truth.get("doc", "?")
            amounts.add_page(per_doc.setdefault(doc, {}), page)
            conds = conditions(truth)
            for d in page["decisions"]:
                if d["cls"] == "qty" and analyse and not d["read"]:
                    tag = next(x for x in truth["tags"] if x["words"] == d["words"])
                    qcauses[diagnose(truth, hw, tag)[0]] += 1
                if d["cls"] not in CLASSES:
                    continue
                for c in conds:
                    v = slices.setdefault(c, [0, 0])
                    v[0] += 1
                    v[1] += int(d["read"])
                if analyse and not d["read"]:
                    tag = next(x for x in truth["tags"] if x["words"] == d["words"])
                    cause, want, got = diagnose(truth, hw, tag)
                    causes[cause] += 1
                    if cause in ("separator or sign misread", "digit misread"):
                        edits.setdefault(cause, Counter()).update(confusions(want, got))
        row["by_doc"] = {k: amounts.summarize(v) for k, v in per_doc.items()}
        if analyse:
            row["causes"] = dict(causes)
            row["qty_causes"] = dict(qcauses)
            row["edits"] = {k: dict(v) for k, v in edits.items()}
            row["slices"] = {k: {"values": v[0], "missed": v[0] - v[1]} for k, v in slices.items()}
        res["sets"][s] = row
    return res


def pooled(sets: dict, names: list[str]) -> dict | None:
    rows = [sets[n] for n in names if n in sets]
    if len(rows) != len(names):  # an engine with only some of the sets is not pooled
        return None
    tot: Counter = Counter()
    cnt: Counter = Counter()
    txt: Counter = Counter()
    cls: dict[str, list[int]] = {}
    for r in rows:
        for k in ("char_errors", "ref_chars", "word_errors", "ref_words", "order_errors"):
            tot[k] += r["totals"][k]
        a = r["amounts"]
        if not a:
            continue
        for k in ("rows", "reconciled"):
            cnt[k] += a["counts"].get(k, 0)
        for c, v in a["counts"].get("values", {}).items():
            cls.setdefault(c, [0, 0, 0])[0] += v
            cls[c][1] += a["counts"]["read"].get(c, 0)
            cls[c][2] += a["text_only"]["counts"]["read"].get(c, 0)
    n = sum(cls.get(c, [0])[0] for c in CLASSES)
    return {"cer": tot["char_errors"] / tot["ref_chars"], "wer": tot["word_errors"] / tot["ref_words"],
            "order": tot["order_errors"] / tot["ref_words"], "values": n,
            "amount": sum(cls.get(c, [0, 0])[1] for c in CLASSES) / n if n else None,
            "text_only": sum(cls.get(c, [0, 0, 0])[2] for c in CLASSES) / n if n else None,
            "classes": {c: {"values": v[0], "box": v[1] / v[0], "text": v[2] / v[0]} for c, v in cls.items() if v[0]},
            "reconciliation": cnt["reconciled"] / cnt["rows"] if cnt["rows"] else None, "rows": cnt["rows"]}


def pct(x: float | None, d: int = 2) -> str:
    return "-" if x is None else f"{100 * x:.{d}f}"


def build(eval_root: Path, runs: Path, ours: str) -> dict:
    configs = discover(runs, ours)
    out = {"ours": ours, "configs": {}}
    for name, dirs in configs.items():
        # an engine run that has not finished a set is left out of the report, not scored as if it had read nothing
        done = {s: d for s, d in dirs.items() if len(list(d.glob("*.txt"))) >= len(list((eval_root / s / "truth").glob("*.txt")))}
        if done:
            out["configs"][name] = score_config(eval_root, name, done, name == ours)
    return out


def markdown(rep: dict) -> str:
    ours = rep["ours"]
    cfg = rep["configs"]
    L: list[str] = []
    for cat, base in CATEGORIES.items():
        for kind, names in (("synthetic", [base]), ("real", [base + "-real"]), ("pooled", [base, base + "-real"])):
            rows = {n: pooled(c["sets"], names) for n, c in cfg.items()}
            rows = {n: r for n, r in rows.items() if r}
            if not rows:
                continue
            note = " (order is information: table pages have no verified order)" if kind != "synthetic" else ""
            L.append(f"### {cat}, {kind} ({', '.join(names)}){note}\n")
            L.append("| engine | CER % | WER % | order % | amount exact % (box) | amount exact % (text-only) | money | date | account | qty (single digits mostly), in place | reconciliation % |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|")

            def line(n: str, r: dict) -> str:
                cl = r["classes"]
                f = lambda c: pct(cl[c]["box"], 1) + " / " + pct(cl[c]["text"], 1) if c in cl else "-"  # noqa: E731
                return (f"| {n} | {pct(r['cer'])} | {pct(r['wer'])} | {pct(r['order'])} | {pct(r['amount'], 1)} | {pct(r['text_only'], 1)} | "
                        f"{f('money')} | {f('date')} | {f('account')} | {pct(cl['qty']['box'], 1) if 'qty' in cl else '-'} | {pct(r['reconciliation'], 1)} |")

            order = [ours] + sorted(n for n in rows if n != ours)
            for n in order:
                if n in rows:
                    L.append(line(n, rows[n]))
            others = {n: r for n, r in rows.items() if n != ours}
            if others:
                best = {"cer": min(r["cer"] for r in others.values()), "wer": min(r["wer"] for r in others.values()),
                        "order": min(r["order"] for r in others.values()),
                        "amount": max((r["amount"] or 0) for r in others.values()), "text_only": max((r["text_only"] or 0) for r in others.values()),
                        "reconciliation": max((r["reconciliation"] or 0) for r in others.values()) or None,
                        "qty": max((r["classes"].get("qty", {}).get("box") or 0) for r in others.values()) or None}
                L.append(f"| **best other** | {pct(best['cer'])} | {pct(best['wer'])} | {pct(best['order'])} | {pct(best['amount'], 1)} | "
                         f"{pct(best['text_only'], 1)} |  |  |  | {pct(best['qty'], 1)} | {pct(best['reconciliation'], 1)} |")
            L.append("")
    if ours in cfg:
        L.append("### Our amount misses by cause (money, date, account values not read exactly in place)\n")
        L.append("| set | values | missed | " + " | ".join(["undetected", "merged", "separator or sign misread", "digit misread"]) + " |")
        L.append("|---|---|---|---|---|---|---|")
        for s, r in sorted(cfg[ours]["sets"].items()):
            if "causes" not in r:
                continue
            a = r["amounts"]
            n = a["amount_values"]
            miss = n - sum(a["counts"]["read"].get(c, 0) for c in CLASSES)
            cs = r["causes"]
            L.append(f"| {s} | {n} | {miss} | " + " | ".join(str(cs.get(k, 0)) for k in
                     ["undetected", "merged", "separator or sign misread", "digit misread"]) + " |")
        L.append("")
        L.append("### Our quantity-cell misses (mostly single digits) by cause\n")
        L.append("| set | values | missed | undetected | merged | separator or sign misread | digit misread |")
        L.append("|---|---|---|---|---|---|---|")
        for s, r in sorted(cfg[ours]["sets"].items()):
            q = ((r["amounts"] or {}).get("classes") or {}).get("qty")
            if not q:
                continue
            cs = r.get("qty_causes", {})
            L.append(f"| {s} | {q['values']} | {sum(cs.values())} | " + " | ".join(str(cs.get(k, 0)) for k in
                     ["undetected", "merged", "separator or sign misread", "digit misread"]) + " |")
        L.append("")
        for cause in ("separator or sign misread", "digit misread"):
            tot: Counter = Counter()
            for r in cfg[ours]["sets"].values():
                tot.update(r.get("edits", {}).get(cause, {}))
            L.append(f"### Our most frequent character edits in {cause}s (truth>read, all sets)\n")
            L.append(", ".join(f"`{k}` {v}" for k, v in tot.most_common(14)) + "\n")
        L.append("### Our miss rate by condition (values missed / values, conditions overlap)\n")
        L.append("| set | " + " | ".join(["fax 1-bit fine", "fax 1-bit std", "low resolution", "skew >= 4 degrees", "phone", "tables cut"]) + " |")
        L.append("|---|---|---|---|---|---|")
        keys = ["fax 1-bit fine", "fax 1-bit std", "low resolution (scan <= 200 dpi, fax 100 dpi rows)", "skew >= 4 degrees", "phone", "tables cut"]
        for s, r in sorted(cfg[ours]["sets"].items()):
            sl = r.get("slices", {})
            L.append(f"| {s} | " + " | ".join((f"{sl[k]['missed']}/{sl[k]['values']}" if k in sl else "-") for k in keys) + " |")
        L.append("")
        L.append("### Our exact match per class, in place (%; qty = quantity cells, mostly single digits, reported on its own)\n")
        L.append("| set | money | date | account | qty |")
        L.append("|---|---|---|---|---|")
        for s, r in sorted(cfg[ours]["sets"].items()):
            cl = (r["amounts"] or {}).get("classes", {})
            L.append(f"| {s} | " + " | ".join((pct(cl[c]["exact"], 1) + f" ({cl[c]['values']})") if c in cl else "-" for c in ["money", "date", "account", "qty"]) + " |")
        L.append("")
        L.append("### Our amount exact match by document type (box, %)\n")
        L.append("| set | statement | invoice | receipt | real |")
        L.append("|---|---|---|---|---|")
        for s, r in sorted(cfg[ours]["sets"].items()):
            bd = r["by_doc"]
            L.append(f"| {s} | " + " | ".join(pct(bd[k]["amount_exact"], 1) if k in bd else "-" for k in ["statement", "invoice", "receipt", "real"]) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def main(a) -> None:
    rep = build(Path(a.eval).expanduser(), Path(a.runs).expanduser(), a.ours)
    md = markdown(rep)
    if a.out:
        Path(a.out).expanduser().write_text(md)
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(md)
