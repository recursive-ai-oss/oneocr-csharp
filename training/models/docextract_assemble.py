"""Deterministic assembly of the extraction model's word labels into JSON (the reference the C# side reproduces).

The model labels words; this code only groups, orders and joins them, so every value in the output is the text of words that are on
the page (`words` lists their indices). Rules, in order:

1. Document type = argmax of the document head. Field classes of the other schemas are masked out of the tag argmax
   (invoice: BT-* classes; receipt: receipt.* classes; other: no fields, only key/value pairs).
2. Tags: argmax over the allowed tags. A span starts at a B tag (or at an I tag whose previous word is not in a span of the same
   class) and continues over following I tags of that class. Span confidence = mean softmax probability of the chosen tags.
3. Header fields (not line or tax classes): one value per class, the span with the highest confidence.
4. Line fields and tax fields: the span's group = the anchor word its first word points to, followed through the item pointer until
   it points to itself (at most 4 steps; a cycle ends at the smallest word index on it). Groups are ordered by anchor index. Within a
   group a class appears once (highest confidence). Tax fields (BT-116/117/119) and line fields never share a group.
5. Key/value: role tags give KEY and VAL spans. A VAL span pairs with the KEY span that contains the word its first word points to
   in the key/value pointer relation; if that word is in no KEY span the value is left unpaired.
Text of a span = its words joined by one space.
"""

from __future__ import annotations

import numpy as np

from models.docextract import (
    CLASSES,
    DOC_TYPES,
    GROUP_CLASSES,
    INVOICE_HEADER,
    INVOICE_LINE,
    INVOICE_TAX,
    RECEIPT_HEADER,
    RECEIPT_LINE,
    REL_GROUP,
    REL_KV,
)

ALLOWED = {
    "invoice": set(INVOICE_HEADER + INVOICE_LINE + INVOICE_TAX),
    "receipt": set(RECEIPT_HEADER + RECEIPT_LINE),
    "other": set(),
}


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def spans_from_tags(labels: list, probs: list[float]) -> list[dict]:
    """labels[i] is None or ("B"|"I", key); returns spans of consecutive words with one key."""
    spans, cur = [], None
    for i, (lab, p) in enumerate(zip(labels, probs, strict=True)):
        if lab is None:
            cur = None
            continue
        kind, key = lab
        if cur is not None and kind == "I" and cur["key"] == key and cur["words"][-1] == i - 1:
            cur["words"].append(i)
            cur["p"].append(p)
        else:
            cur = {"key": key, "words": [i], "p": [p]}
            spans.append(cur)
    for s in spans:
        s["confidence"] = float(np.mean(s.pop("p")))
    return spans


def decode_tag(t: int) -> tuple[str, int] | None:
    if t == 0:
        return None
    c = (t - 1) // 2
    return ("B" if (t - 1) % 2 == 0 else "I", c)


def assemble(out: dict, texts: list[str], boxes: np.ndarray) -> dict:
    """`out`: numpy arrays from the model for one page: tags [N, T], roles [N, 5], doc [3], rel [2, N, N] (logits)."""
    n = len(texts)
    doc_probs = softmax(out["doc"])
    doc = DOC_TYPES[int(doc_probs.argmax())]
    allowed = ALLOWED[doc]
    allowed_tags = np.zeros(out["tags"].shape[1], dtype=bool)
    allowed_tags[0] = True
    for c, name in enumerate(CLASSES):
        if name in allowed:
            allowed_tags[1 + 2 * c : 3 + 2 * c] = True
    tl = np.where(allowed_tags[None, :], out["tags"], -1e30)
    tp = softmax(tl)
    tag = tp.argmax(1)
    labels = [decode_tag(int(t)) for t in tag]
    spans = spans_from_tags(labels, [float(tp[i, tag[i]]) for i in range(n)])

    def box_of(words):
        b = boxes[words]
        return [float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())]

    def field(s):
        name = CLASSES[s["key"]]
        return {
            "id": name,
            "text": " ".join(texts[i] for i in s["words"]),
            "words": s["words"],
            "bounds": box_of(s["words"]),
            "confidence": s["confidence"],
        }

    header: dict[str, dict] = {}
    grouped: dict[tuple[str, int], dict[str, dict]] = {}
    ptr = out["rel"][REL_GROUP].argmax(1) if n else np.zeros(0, dtype=int)

    def anchor(i: int) -> int:
        seen = [i]
        for _ in range(4):
            j = int(ptr[seen[-1]])
            if j == seen[-1]:
                return j
            if j in seen:
                return min(seen[seen.index(j) :])
            seen.append(j)
        return seen[-1]

    for s in spans:
        f = field(s)
        name = f["id"]
        if name in GROUP_CLASSES:
            a = anchor(s["words"][0])
            fam = "tax" if name in INVOICE_TAX else "line"
            g = grouped.setdefault((fam, a), {})
            if name not in g or f["confidence"] > g[name]["confidence"]:
                g[name] = f
        elif name not in header or f["confidence"] > header[name]["confidence"]:
            header[name] = f

    items = [
        {"anchor": a, "fields": [g[k] for k in sorted(g, key=lambda k: g[k]["words"][0])]}
        for (fam, a), g in sorted(grouped.items())
        if fam == "line"
    ]
    tax = [
        {"anchor": a, "fields": [g[k] for k in sorted(g, key=lambda k: g[k]["words"][0])]}
        for (fam, a), g in sorted(grouped.items())
        if fam == "tax"
    ]

    rp = softmax(out["roles"])
    rt = rp.argmax(1)
    rlabels = []
    for t in rt:
        rlabels.append(None if t == 0 else (("B" if t in (1, 3) else "I"), "KEY" if t in (1, 2) else "VAL"))
    rspans = spans_from_tags(rlabels, [float(rp[i, rt[i]]) for i in range(n)])
    key_of_word = {}
    keys = [s for s in rspans if s["key"] == "KEY"]
    for k in keys:
        for w in k["words"]:
            key_of_word[w] = k
    kvptr = out["rel"][REL_KV].argmax(1) if n else np.zeros(0, dtype=int)
    pairs = []
    for v in (s for s in rspans if s["key"] == "VAL"):
        k = key_of_word.get(int(kvptr[v["words"][0]]))
        if k is not None:
            pairs.append(
                {
                    "key": {"text": " ".join(texts[i] for i in k["words"]), "words": k["words"], "bounds": box_of(k["words"])},
                    "value": {"text": " ".join(texts[i] for i in v["words"]), "words": v["words"], "bounds": box_of(v["words"])},
                    "confidence": float(min(k["confidence"], v["confidence"])),
                }
            )
    return {
        "documentType": doc,
        "documentTypeConfidence": float(doc_probs.max()),
        "fields": [header[k] for k in sorted(header, key=CLASSES.index)],
        "lineItems": items,
        "taxBreakdown": tax,
        "keyValues": pairs,
    }
