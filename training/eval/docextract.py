"""Metrics for the document extraction model: field exact match and F1, line item and key/value link F1, document type accuracy.

Inputs are examples from train/docextract_data.py (truth words or OCR words) and a model (checkpoint, or the exported container
read back with its fp16-rounded weights). Predictions come from models/docextract_assemble.py, so what is scored is the JSON the
library would produce.

Comparison: a predicted value is right when its text equals the truth text after NFC and removal of all white space (a word split or
joined differently by the reader is not an extraction error). On OCR input the truth text is the printed text, so a misread
character is an error.

Definitions (per class, micro over pages):
  * header field (everything but line and tax fields): TP when the predicted text for the class equals a truth text of that class;
    FP when a prediction has no equal truth value; FN when a truth value has no equal prediction. accuracy = TP / truth count.
  * line items: item-exact F1 over whole items (all fields of an item equal, multiset match), and item-field F1 over tuples
    (class, text, item name text), which also tests the grouping.
  * tax rows: exact F1 over (rate, taxable amount, tax amount) rows.
  * key/value links: F1 over (key text, value text) pairs.
CORD annotates part of a receipt (no merchant, date or payment, and unmapped categories): there a prediction counts as a false
positive only when its words overlap an annotated word, item-exact F1 is not reported and the amount/unit price split is the
mapping of `menu.price` to amount, `menu.unitprice` to unit price.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from models.docextract import CLASSES, DOC_TYPES, GROUP_CLASSES, LINE_CLASSES, DocExtract, collate
from models.docextract_assemble import assemble


def norm(s: str) -> str:
    return "".join(unicodedata.normalize("NFC", s).split())


@torch.no_grad()
def predict(model: DocExtract, examples: list[dict], dev: str = "cpu", batch: int = 16) -> list[dict]:
    model.eval()
    outs = []
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["texts"]))
    res: dict[int, dict] = {}
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        b = collate([examples[i] for i in idx])
        o = model(b["chars"].to(dev), b["feats"].to(dev), b["bins"].to(dev), b["geo"].to(dev), b["mask"].to(dev))
        for k, i in enumerate(idx):
            n = len(examples[i]["texts"])
            res[i] = {
                "tags": o["tags"][k, :n].float().cpu().numpy(),
                "roles": o["roles"][k, :n].float().cpu().numpy(),
                "doc": o["doc"][k].float().cpu().numpy(),
                "rel": o["rel"][k, :, :n, :n].float().cpu().numpy(),
            }
    for i in range(len(examples)):
        outs.append(res[i])
    return outs


def multiset_prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def match(pred: Counter, truth: Counter) -> tuple[int, int, int]:
    tp = sum((pred & truth).values())
    return tp, sum(pred.values()) - tp, sum(truth.values()) - tp


def truth_items(truth: dict) -> dict:
    items: dict = defaultdict(dict)
    for f in truth["fields"]:
        if f["group"] is not None:
            items[f["group"]].setdefault(f["cls"], []).append(norm(f["text"]))
    return items


def item_name(d: dict) -> str:
    for c in ("BT-153", "receipt.itemName"):
        if c in d:
            return d[c][0] if isinstance(d[c], list) else d[c]
    return ""


def score_page(ex: dict, res: dict, acc: dict) -> None:
    truth = ex["truth"]
    partial = bool(ex["meta"].get("partial"))
    known = ex.get("known")
    role_known = ex["roles"] != -100
    a = assemble(res, ex["texts"], ex["boxes"])
    acc["doc"].append(DOC_TYPES.index(a["documentType"]) == ex["doc"])

    def counted(words) -> bool:  # CORD: only predictions touching annotated words can be false positives
        return (not partial) or any(known[w] for w in words)

    # header fields
    t_h = defaultdict(Counter)
    for f in truth["fields"]:
        if f["group"] is None:
            t_h[f["cls"]][norm(f["text"])] += 1
    p_h = defaultdict(Counter)
    for f in a["fields"]:
        if f["id"] not in GROUP_CLASSES and counted(f["words"]):
            p_h[f["id"]][norm(f["text"])] += 1
    for c in set(t_h) | set(p_h):
        tp, fp, fn = match(p_h[c], t_h[c])
        s = acc["header"][c]
        s[0] += tp
        s[1] += fp
        s[2] += fn
    # item fields
    t_it = truth_items(truth)
    t_tuples, p_tuples = Counter(), Counter()
    for g, d in t_it.items():
        if g[0] == "line":
            nm = item_name(d)
            for c, vs in d.items():
                for v in vs:
                    t_tuples[(c, v, nm)] += 1
    p_items = []
    for it in a["lineItems"]:
        d = {}
        for f in it["fields"]:
            d.setdefault(f["id"], []).append(norm(f["text"]))
        p_items.append((d, it))
    for d, it in p_items:
        if not counted([w for f in it["fields"] for w in f["words"]]):
            continue
        nm = item_name(d)
        for c, vs in d.items():
            for v in vs:
                p_tuples[(c, v, nm)] += 1
    tp, fp, fn = match(p_tuples, t_tuples)
    for k, v in zip(("tp", "fp", "fn"), (tp, fp, fn), strict=True):
        acc["item_field"][k] += v
    for c in LINE_CLASSES:
        tp, fp, fn = match(
            Counter({k: v for k, v in p_tuples.items() if k[0] == c}), Counter({k: v for k, v in t_tuples.items() if k[0] == c})
        )
        s = acc["item_cls"][c]
        s[0] += tp
        s[1] += fp
        s[2] += fn
    if not partial:
        te = Counter(frozenset((c, tuple(vs)) for c, vs in d.items()) for g, d in t_it.items() if g[0] == "line")
        pe = Counter(frozenset((c, tuple(vs)) for c, vs in d.items()) for d, _ in p_items)
        tp, fp, fn = match(pe, te)
        for k, v in zip(("tp", "fp", "fn"), (tp, fp, fn), strict=True):
            acc["item_exact"][k] += v
        tt = Counter(frozenset((c, tuple(vs)) for c, vs in d.items()) for g, d in t_it.items() if g[0] == "tax")
        pt = Counter()
        for it in a["taxBreakdown"]:
            d = {}
            for f in it["fields"]:
                d.setdefault(f["id"], []).append(norm(f["text"]))
            pt[frozenset((c, tuple(vs)) for c, vs in d.items())] += 1
        tp, fp, fn = match(pt, tt)
        for k, v in zip(("tp", "fp", "fn"), (tp, fp, fn), strict=True):
            acc["tax_row"][k] += v
    # key/value links
    tk = Counter((norm(k["key"]), norm(k["value"])) for k in truth["kv"])
    pk = Counter()
    for kv in a["keyValues"]:
        if (not partial) or any(role_known[w] for w in kv["value"]["words"]):
            pk[(norm(kv["key"]["text"]), norm(kv["value"]["text"]))] += 1
    tp, fp, fn = match(pk, tk)
    for k, v in zip(("tp", "fp", "fn"), (tp, fp, fn), strict=True):
        acc["kv"][k] += v


def new_acc() -> dict:
    return {
        "doc": [],
        "header": defaultdict(lambda: [0, 0, 0]),
        "item_cls": defaultdict(lambda: [0, 0, 0]),
        "item_field": Counter(),
        "item_exact": Counter(),
        "tax_row": Counter(),
        "kv": Counter(),
    }


def summarize(acc: dict) -> dict:
    def prf(v):
        d = multiset_prf(*v)
        d["accuracy"] = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else None
        return d

    header = {c: prf(v) for c, v in sorted(acc["header"].items(), key=lambda kv: CLASSES.index(kv[0])) if v[0] + v[2] > 0}
    tot = [sum(v[i] for v in acc["header"].values()) for i in range(3)]
    item_cls = {c: prf(v) for c, v in acc["item_cls"].items() if v[0] + v[2] > 0}
    return {
        "doc_type_accuracy": float(np.mean(acc["doc"])) if acc["doc"] else None,
        "pages": len(acc["doc"]),
        "header_micro": prf(tot),
        "header": header,
        "item_field": prf([acc["item_field"][k] for k in ("tp", "fp", "fn")]),
        "item_field_by_class": item_cls,
        "item_exact": prf([acc["item_exact"][k] for k in ("tp", "fp", "fn")]) if acc["item_exact"] else None,
        "tax_row": prf([acc["tax_row"][k] for k in ("tp", "fp", "fn")]) if acc["tax_row"] else None,
        "kv": prf([acc["kv"][k] for k in ("tp", "fp", "fn")]),
    }


def evaluate(model: DocExtract, examples: list[dict], dev: str = "cpu", by_kind: bool = True) -> dict:
    preds = predict(model, examples, dev)
    groups = defaultdict(new_acc)
    allacc = new_acc()
    for ex, res in zip(examples, preds, strict=True):
        score_page(ex, res, allacc)
        if by_kind:
            score_page(ex, res, groups[ex["meta"]["kind"]])
    out = {"all": summarize(allacc)}
    if by_kind:
        out["by_kind"] = {k: summarize(v) for k, v in sorted(groups.items())}
    return out


def load_model(path: str, dev: str = "cpu") -> DocExtract:
    """A training checkpoint (.pt) or an exported container (.onm; fp16-rounded weights widened to fp32)."""
    if path.endswith(".onm"):
        from export.container import read_container

        manifest, tensors = read_container(path)
        model = DocExtract(manifest["config"])
        model.load_state_dict({k: torch.from_numpy(v) for k, v in tensors.items()})
    else:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = DocExtract(ck["config"])
        model.load_state_dict(ck["model"])
    return model.to(dev).eval()


def latency(model: DocExtract, examples: list[dict], n: int = 40) -> dict:
    """Per page forward + assembly on one torch CPU thread (a proxy for the C# kernels, which are not the same code)."""
    torch.set_num_threads(1)
    model = copy.deepcopy(model).cpu().eval()
    sel = examples[:: max(1, len(examples) // n)][:n]
    ms, words = [], []
    for ex in sel:
        t0 = time.perf_counter()
        r = predict(model, [ex], "cpu", 1)[0]
        assemble(r, ex["texts"], ex["boxes"])
        ms.append((time.perf_counter() - t0) * 1000)
        words.append(len(ex["texts"]))
    return {
        "pages": len(sel),
        "mean_ms": float(np.mean(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "mean_words": float(np.mean(words)),
        "max_ms": float(np.max(ms)),
        "threads": 1,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="pickles from train.docextract_data")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--latency", action="store_true")
    a = ap.parse_args()
    model = load_model(a.model, a.device)
    report = {}
    for path in a.data:
        d = pickle.load(open(path, "rb"))
        for src in ("generated", "cord"):
            for kind in ("truth", "ocr"):
                ex = d.get(src, {}).get(kind) or []
                if ex:
                    key = f"{Path(path).stem}/{src}/{kind}-words"
                    report[key] = evaluate(model, ex, a.device)
                    if a.latency and kind == "ocr":
                        report[key + "/latency"] = latency(model, ex)
    text = json.dumps(report, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
