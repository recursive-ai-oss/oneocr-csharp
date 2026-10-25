"""Word-level document extraction model: a small layout-aware transformer over the words of a page (docextract-v0).

Input is the page's words in reading order: their text and boxes. Output is, per word, a BIO field tag, a key/value role tag
and two pointers (word -> anchor word of its line item / tax row, value word -> first word of its key), plus a page-level
document type. Deterministic code (models/docextract_assemble.py) turns the tags into JSON, so the model can only label words
that are on the page.

Ingredients, all with published precedents:
  * words as sequence elements with 2D position embeddings, a transformer encoder over them: Xu et al., "LayoutLM" (2020) and
    "LayoutLMv2" (2021, learned relative spatial bias in self-attention). Only the papers were used; this is our own code.
  * character CNN word embedding: Kim et al., "Character-Aware Neural Language Models" (2016).
  * pre-layer-norm transformer: Xiong et al. (2020); GELU with the tanh approximation: Hendrycks and Gimpel (2016).
  * pointer scoring with a bilinear form between word pairs: Dozat and Manning, "Deep Biaffine Attention for Neural Dependency
    Parsing" (2017), here a plain dot product plus a geometric pair bias.

Layer types: embedding lookup, conv1d (k = 3), max over positions, linear, layer norm, GELU, ReLU, softmax attention with an additive
bias, a two-layer ReLU MLP over word pairs. See export/docextract-architecture.md for the exact graph the C# side reproduces.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = "docextract-v0"

# Field classes. The order is part of the model contract: never reorder once a model ships; append only.
INVOICE_HEADER = [
    "BT-1",
    "BT-2",
    "BT-5",
    "BT-9",
    "BT-13",
    "BT-20",
    "BT-27",
    "BT-31",
    "BT-35",
    "BT-37",
    "BT-38",
    "BT-40",
    "BT-44",
    "BT-48",
    "BT-50",
    "BT-52",
    "BT-53",
    "BT-55",
    "BT-72",
    "BT-84",
    "BT-86",
    "BT-106",
    "BT-110",
    "BT-112",
    "BT-115",
]
INVOICE_LINE = ["BT-126", "BT-129", "BT-130", "BT-131", "BT-146", "BT-152", "BT-153"]
INVOICE_TAX = ["BT-116", "BT-117", "BT-119"]
RECEIPT_HEADER = [
    "receipt.merchant",
    "receipt.date",
    "receipt.time",
    "receipt.subtotal",
    "receipt.tax",
    "receipt.tip",
    "receipt.total",
    "receipt.paymentMethod",
]
RECEIPT_LINE = ["receipt.itemName", "receipt.quantity", "receipt.unitPrice", "receipt.amount"]
CLASSES = INVOICE_HEADER + INVOICE_LINE + INVOICE_TAX + RECEIPT_HEADER + RECEIPT_LINE
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
LINE_CLASSES = set(INVOICE_LINE + RECEIPT_LINE)
TAX_CLASSES = set(INVOICE_TAX)
GROUP_CLASSES = LINE_CLASSES | TAX_CLASSES  # fields that belong to a repeated group; the item pointer ties them together
NTAGS = 1 + 2 * len(CLASSES)  # 0 = O, 1 + 2c = B-c, 2 + 2c = I-c
ROLES = ["O", "B-KEY", "I-KEY", "B-VAL", "I-VAL"]
DOC_TYPES = ["invoice", "receipt", "other"]
REL_GROUP, REL_KV = 0, 1  # pointer relations
NREL = 2

MAX_CHARS = 32
UNK_CHAR = 622  # charset-v1 has 622 entries (0 = blank, used here as padding); anything else is UNK
N_CHARS = 623
NBINS = 128
NFEATS = 8
NPAIR = 13


def default_config() -> dict:
    return {
        "d": 192,
        "layers": 5,
        "heads": 6,
        "ffn": 512,
        "char_dim": 48,
        "char_conv": 128,
        "pair_hidden": 32,
        "rel_dim": 64,
        "max_words": 512,
        "classes": len(CLASSES),
        "tags": NTAGS,
        "roles": len(ROLES),
        "doc_types": len(DOC_TYPES),
    }


_CHARSET: dict[str, int] | None = None


def charset_ids() -> dict[str, int]:
    global _CHARSET
    if _CHARSET is None:
        p = Path(__file__).resolve().parents[1] / "data" / "charset-v1.txt"
        lines = p.read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        _CHARSET = {c: i for i, c in enumerate(lines) if i > 0}
    return _CHARSET


def encode_word(text: str) -> list[int]:
    import unicodedata

    cs = charset_ids()
    ids = [cs.get(c, UNK_CHAR) for c in unicodedata.normalize("NFC", text)]
    return ids[:MAX_CHARS]


def featurize(texts: list[str], boxes: list[list[float]], width: float, height: float) -> dict:
    """Model inputs for one page. `boxes` are [x0, y0, x1, y1] in page pixels, top-left origin. Returns numpy-ready lists:
    chars [N, MAX_CHARS] (0 padded), feats [N, NFEATS], bins [N, 4], geo [N, 4] (box in units of the median word height), unit."""
    import numpy as np

    n = len(texts)
    chars = np.zeros((n, MAX_CHARS), dtype=np.int16)
    for i, t in enumerate(texts):
        ids = encode_word(t)
        chars[i, : len(ids)] = ids
    b = np.asarray(boxes, dtype=np.float64).reshape(n, 4)
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    bh = np.maximum(b[:, 3] - b[:, 1], 1e-3)
    bw = np.maximum(b[:, 2] - b[:, 0], 1e-3)
    unit = float(np.median(bh)) if n else 1.0
    unit = max(unit, 1.0)
    nb = np.stack([b[:, 0] / w, b[:, 1] / h, b[:, 2] / w, b[:, 3] / h], 1)
    r = np.minimum(bh / unit, 8.0)
    a = bw / bh
    feats = np.stack([nb[:, 0], nb[:, 1], nb[:, 2], nb[:, 3], bw / w, bh / h, r - 1.0, a / (1.0 + a)], 1).astype(np.float32)
    bins = np.clip(np.floor(nb * NBINS), 0, NBINS - 1).astype(np.int64)
    geo = (b / unit).astype(np.float32)
    return {"chars": chars, "feats": feats, "bins": bins, "geo": geo, "unit": unit}


def squash(x: torch.Tensor, half: float) -> torch.Tensor:
    """x / (half + |x|): a bounded, monotone compression with no transcendental function (half = where it reaches 0.5)."""
    return x / (half + x.abs())


def pair_features(geo: torch.Tensor) -> torch.Tensor:
    """geo [B, N, 4] (x0, y0, x1, y1 in median word heights) -> [B, N, N, NPAIR]; entry (i, j) describes word j relative to word i.
    Six differences (centre x and y, left, right, top, bottom) each at two scales (half-saturation 1 and 16 word heights), and the
    height difference."""
    x0, y0, x1, y1 = geo.unbind(-1)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = y1 - y0

    def d(a):  # a_j - a_i
        return a.unsqueeze(1) - a.unsqueeze(2)

    f = []
    for a in (cx, cy, x0, x1, y0, y1):
        f += [squash(d(a), 1.0), squash(d(a), 16.0)]
    f.append(squash(d(h), 1.0))
    return torch.stack(f, -1)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class DocExtract(nn.Module):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        c = {**default_config(), **(cfg or {})}
        self.cfg = c
        d = c["d"]
        self.char_emb = nn.Embedding(N_CHARS, c["char_dim"], padding_idx=0)
        self.char_conv = nn.Conv1d(c["char_dim"], c["char_conv"], 3, padding=1)
        self.char_proj = nn.Linear(c["char_conv"], d)
        self.pos_x0 = nn.Embedding(NBINS, d)
        self.pos_x1 = nn.Embedding(NBINS, d)
        self.pos_y0 = nn.Embedding(NBINS, d)
        self.pos_y1 = nn.Embedding(NBINS, d)
        self.feat_proj = nn.Linear(NFEATS, d)
        self.emb_ln = nn.LayerNorm(d, eps=1e-5)
        self.blocks = nn.ModuleList(Block(d, c["heads"], c["ffn"]) for _ in range(c["layers"]))
        self.pair1 = nn.Linear(NPAIR, c["pair_hidden"])
        self.pair2 = nn.Linear(c["pair_hidden"], c["layers"] * c["heads"] + NREL)
        self.final_ln = nn.LayerNorm(d, eps=1e-5)
        self.tag_head = nn.Linear(d, c["tags"])
        self.role_head = nn.Linear(d, c["roles"])
        self.doc_head = nn.Linear(d, c["doc_types"])
        self.rel_q = nn.Linear(d, NREL * c["rel_dim"])
        self.rel_k = nn.Linear(d, NREL * c["rel_dim"])
        self.apply(self._init)
        with torch.no_grad():
            self.char_emb.weight[0].zero_()

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear | nn.Conv1d):
            nn.init.normal_(m.weight, std=0.02) if isinstance(m, nn.Linear) else nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.1 if m.num_embeddings == N_CHARS else 0.02)

    def forward(self, chars, feats, bins, geo, mask) -> dict:
        """chars [B,N,L] long (0 pad), feats [B,N,NFEATS], bins [B,N,4] long, geo [B,N,4], mask [B,N] bool (True = real word)."""
        c = self.cfg
        b, n, ln = chars.shape
        e = self.char_emb(chars.view(b * n, ln)).transpose(1, 2)  # [BN, E, L]
        cm = (chars.view(b * n, 1, ln) != 0).to(e.dtype)
        h = F.relu(self.char_conv(e)) * cm  # positions past the word's end are zero, so max over all positions = max over valid ones
        w = self.char_proj(h.amax(-1)).view(b, n, -1)
        x = w + self.pos_x0(bins[..., 0]) + self.pos_x1(bins[..., 2]) + self.pos_y0(bins[..., 1]) + self.pos_y1(bins[..., 3])
        x = self.emb_ln(x + self.feat_proj(feats))
        pb = self.pair2(F.relu(self.pair1(pair_features(geo)))).permute(0, 3, 1, 2)  # [B, L*H+R, N, N]
        neg = torch.zeros(b, 1, 1, n, device=x.device, dtype=pb.dtype).masked_fill(~mask[:, None, None, :], float("-inf"))
        nh = c["heads"]
        for i, blk in enumerate(self.blocks):
            x = blk(x, pb[:, i * nh : (i + 1) * nh] + neg)
        x = self.final_ln(x)
        m = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(1) / m.sum(1).clamp_min(1)
        r = c["rel_dim"]
        q = self.rel_q(x).view(b, n, NREL, r).permute(0, 2, 1, 3)
        k = self.rel_k(x).view(b, n, NREL, r).permute(0, 2, 1, 3)
        rel = q @ k.transpose(-1, -2) / math.sqrt(r) + pb[:, c["layers"] * nh :] + neg
        return {"tags": self.tag_head(x), "roles": self.role_head(x), "doc": self.doc_head(pooled), "rel": rel}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def collate(items: list[dict]) -> dict:
    """items: dicts of numpy arrays from featurize + labels (tags, roles, group_ptr, kv_ptr, doc). Pads to the longest."""
    import numpy as np

    n = max(len(i["feats"]) for i in items)
    bs = len(items)
    out = {
        "chars": torch.zeros(bs, n, MAX_CHARS, dtype=torch.long),
        "feats": torch.zeros(bs, n, NFEATS),
        "bins": torch.zeros(bs, n, 4, dtype=torch.long),
        "geo": torch.zeros(bs, n, 4),
        "mask": torch.zeros(bs, n, dtype=torch.bool),
        "tags": torch.full((bs, n), -100, dtype=torch.long),
        "roles": torch.full((bs, n), -100, dtype=torch.long),
        "group_ptr": torch.full((bs, n), -100, dtype=torch.long),
        "kv_ptr": torch.full((bs, n), -100, dtype=torch.long),
        "doc": torch.zeros(bs, dtype=torch.long),
    }
    for k, it in enumerate(items):
        m = len(it["feats"])
        out["chars"][k, :m] = torch.from_numpy(it["chars"])
        out["feats"][k, :m] = torch.from_numpy(it["feats"])
        out["bins"][k, :m] = torch.from_numpy(it["bins"])
        out["geo"][k, :m] = torch.from_numpy(it["geo"])
        out["mask"][k, :m] = True
        for key in ("tags", "roles", "group_ptr", "kv_ptr"):
            if key in it:
                out[key][k, :m] = torch.from_numpy(np.asarray(it[key], dtype=np.int64))
        out["doc"][k] = int(it.get("doc", 2))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Pages longer than one window


def window_plan(n: int, max_words: int) -> tuple[list[tuple[int, int]], list[int]]:
    """Windows (start, end) over n words and, per word, the window that owns it.

    A page of at most `max_words` words is one window. A longer page runs in windows of `max_words` words starting every
    `max_words * 3 // 4` words (512 and 384), the last window ending at the last word: starts 0, s, 2s, ... while
    `start + max_words < n`. A word belongs to the window whose centre `(start + end) / 2` is nearest the word's centre
    `index + 0.5`; on a tie the earlier window. This is the rule the library's C# reader implements.
    """
    if n <= max_words:
        return [(0, n)], [0] * n
    stride = max_words * 3 // 4
    starts = [0]
    while starts[-1] + max_words < n:
        starts.append(starts[-1] + stride)
    wins = [(s, min(s + max_words, n)) for s in starts]
    owner = []
    for i in range(n):
        best, bd = 0, None
        for k, (s, e) in enumerate(wins):
            d = abs((s + e) / 2 - (i + 0.5))
            if bd is None or d < bd:
                best, bd = k, d
        owner.append(best)
    return wins, owner


@torch.no_grad()
def page_outputs(model: DocExtract, ex: dict, dev: str = "cpu"):
    """Model outputs for one page as numpy arrays, with the window rule applied: tags [N, T], roles [N, 5], doc [3], ptr [2, N].

    Each word takes its tag logits, role logits and pointer from the window that owns it. A pointer is the window-local argmax
    (lowest index on a tie) shifted by the window start, so it never leaves its window. The document logits of a multi-window page
    are the mean of the windows' logits weighted by the number of words each window owns, in double precision.
    """
    import numpy as np

    n = len(ex["texts"])
    wins, owner = window_plan(n, model.cfg["max_words"])
    tags = np.zeros((n, model.cfg["tags"]), dtype=np.float32)
    roles = np.zeros((n, model.cfg["roles"]), dtype=np.float32)
    ptr = np.zeros((NREL, n), dtype=np.int64)
    doc = np.zeros(model.cfg["doc_types"], dtype=np.float64)
    owner_a = np.asarray(owner)
    for k, (s, e) in enumerate(wins):
        sl = slice(s, e)
        m = e - s

        def t(a, sl=sl):
            return torch.as_tensor(np.asarray(a)[None, sl], device=dev)

        o = model(t(ex["chars"]).long(), t(ex["feats"]), t(ex["bins"]).long(), t(ex["geo"]), torch.ones(1, m, dtype=torch.bool, device=dev))
        mine = np.nonzero(owner_a[sl] == k)[0]
        tags[s + mine] = o["tags"][0].float().cpu().numpy()[mine]
        roles[s + mine] = o["roles"][0].float().cpu().numpy()[mine]
        rel = o["rel"][0].float().cpu().numpy()
        ptr[:, s + mine] = rel[:, mine, :].argmax(-1) + s
        doc += len(mine) * o["doc"][0].double().cpu().numpy()
    return {"tags": tags, "roles": roles, "doc": (doc / max(n, 1)).astype(np.float32), "ptr": ptr}
