"""Targeted-error training stream: the batches are drawn toward what the current model gets wrong.

Every `every` steps the model reads a held-out synthetic stream (clean pack of unseen texts, degraded on the GPU with fresh draws).
Its errors are attributed to the character, the font family, the kind of text (digits, amounts, codes, capitals, words, scripts) and
the degradation parameters of each line (size, blur, contrast, noise); each factor's error rate relative to the average becomes a
sampling weight (clipped, damped by `gamma`, averaged with the previous weights in log space):

  * the next stretch of training lines is drawn from the training pack with probability proportional to
    font weight x kind weight x (1 + beta x mean character weight of the line);
  * the size, blur, contrast and noise of those lines are drawn from bins whose probabilities follow the bin error rates.

All text is exact-truth synthetic (the rendered string); only the sampling weights come from the model's behaviour. A version of the
weights is a function of the checkpoint at its step, saved as JSON next to the run, so a resumed run continues with the same weights.
Version v (computed at step v x every) governs steps v x every .. (v + 1) x every - 1.
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from degrade.gpu import degrade, sample_params
from models.crnn import greedy_decode
from train.data import PackSet, to_input

KINDS = ["digits", "amount", "code", "upper", "lower", "greek", "cyrillic", "mixed"]
_AMOUNT = re.compile(r"^[^\w\s]?\d{1,3}([.,' ]\d{3})*([.,]\d{1,2})?\s?[^\w\s]?$")


def text_kind(t: str) -> str:
    """Coarse class of a line's text, used to weight the sampler and to report where the errors are."""
    letters = [c for c in t if c.isalpha()]
    digits = sum(c.isdigit() for c in t)
    if any("\u0370" <= c <= "\u03ff" for c in letters):
        return "greek"
    if any("\u0400" <= c <= "\u04ff" for c in letters):
        return "cyrillic"
    if not letters:
        return "amount" if _AMOUNT.match(t.strip()) else "digits"
    if digits and digits + len(letters) >= 4 and " " not in t.strip():
        return "code"
    if digits >= len(letters):
        return "amount"
    if all(c.isupper() for c in letters):
        return "upper"
    if not digits:
        return "lower"
    return "mixed"


def family(font: str) -> str:
    return font.split("/")[1] if "/" in font else font


class Adaptive:
    """Sampling weights over a clean pack, plus degradation-bin probabilities, one version per measuring step."""

    def __init__(self, pack: PackSet, stream: PackSet, charset: list[str], cfg: dict, side: dict, seed: int, run_dir: Path):
        self.pack, self.stream, self.charset, self.cfg, self.side, self.seed = pack, stream, charset, cfg, side, seed
        self.every = cfg["every"]
        self.dir = run_dir
        index = {c: i for i, c in enumerate(charset)}
        self.fam_names = sorted({family(f) for f in pack.fonts})
        fam_ix = {f: i for i, f in enumerate(self.fam_names)}
        self.fam = np.array([fam_ix[family(f)] for f in pack.fonts], dtype=np.int32)
        self.kind = np.array([KINDS.index(text_kind(t)) for t in pack.texts], dtype=np.int32)
        lens = np.array([max(1, len(t)) for t in pack.texts])
        self.starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        self.lens = lens
        self.flat = np.array([index.get(c, 0) for t in pack.texts for c in (t or " ")], dtype=np.int32)
        self.bin_names = list(cfg["bins"])
        self.base_probs = self._base_bins()
        self.versions: dict[int, dict] = {}
        self.cv = threading.Condition()

    # -- degradation bins -------------------------------------------------------------------------------------------------------
    def _base_bins(self) -> dict[str, np.ndarray]:
        """Bin probabilities of the config's own (unweighted) ranges, from a large draw."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        p = sample_params({**self.side, "_bins": {}}, 200_000, g, "cpu")
        key = {"blur_rel": "blur_rel", "em": "em", "contrast": "contrast", "noise": "noise"}
        out = {}
        for name, edges in self.cfg["bins"].items():
            v = p[key[name]].numpy()
            if name == "blur_rel":
                v = v[v > 0]
            h = np.histogram(np.clip(v, edges[0], np.nextafter(edges[-1], 0)), bins=edges)[0].astype(float)
            out[name] = (h + 1.0) / (h + 1.0).sum()
        return out

    def bins_for(self, v: int) -> dict:
        w = self.versions[v]["bins"]
        return {n: {"edges": self.cfg["bins"][n], "probs": w[n]} for n in w}

    # -- versions ---------------------------------------------------------------------------------------------------------------
    def initial(self) -> dict:
        return {"fonts": {f: 1.0 for f in self.fam_names}, "kinds": {k: 1.0 for k in KINDS}, "chars": {},
                "bins": {n: p.tolist() for n, p in self.base_probs.items()}, "step": 0}  # fmt: skip

    def publish(self, v: int, state: dict) -> None:
        (self.dir / f"adapt-v{v:03d}.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        self._install(v, state)

    def _install(self, v: int, state: dict) -> None:
        char_w = np.ones(len(self.charset))
        for c, w in state["chars"].items():
            char_w[self.charset.index(c)] = w
        font_w = np.array([state["fonts"].get(f, 1.0) for f in self.fam_names])
        kind_w = np.array([state["kinds"].get(k, 1.0) for k in KINDS])
        line_char = np.add.reduceat(char_w[self.flat], self.starts) / self.lens
        w = font_w[self.fam] * kind_w[self.kind] * (1.0 + self.cfg.get("beta", 1.0) * (line_char - 1.0).clip(min=0))
        with self.cv:
            self.versions[v] = {**state, "cum": np.cumsum(w), "bins": state["bins"]}
            self.cv.notify_all()

    def load(self, v: int) -> bool:
        p = self.dir / f"adapt-v{v:03d}.json"
        if p.exists():
            self._install(v, json.loads(p.read_text(encoding="utf-8")))
            return True
        return False

    def wait(self, v: int) -> dict:
        with self.cv:
            self.cv.wait_for(lambda: v in self.versions)
            return self.versions[v]

    def sample(self, step: int, n: int) -> list[int]:
        """Pack indices for the adaptive share of step `step`: a function of the weights version and the step only."""
        st = self.wait(step // self.every)
        rng = np.random.default_rng([self.seed, step])
        return np.searchsorted(st["cum"], rng.random(n) * st["cum"][-1]).clip(max=len(self.pack) - 1).tolist()

    # -- measuring --------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def measure(self, model, v: int, dev: str, prev: dict) -> tuple[dict, dict]:
        """Reads the stream with the current model; returns (next state for version v, report with the confusions)."""
        model.eval()
        g = torch.Generator(device=dev)
        g.manual_seed(self.seed * 7 + v)
        side = {**self.side, "_bins": self.bins_for(v - 1) if (v - 1) in self.versions else {}}
        err_c, tot_c = Counter(), Counter()
        err_f, tot_f, err_k, tot_k = Counter(), Counter(), Counter(), Counter()
        conf: Counter = Counter()
        bin_err = {n: np.zeros(len(e) - 1) for n, e in self.cfg["bins"].items()}
        bin_tot = {n: np.zeros(len(e) - 1) for n, e in self.cfg["bins"].items()}
        ds = self.stream
        order = sorted(range(len(ds)), key=lambda i: ds.widths[i])
        for s in range(0, len(order), 250):
            idx = order[s : s + 250]
            x, w = ds.batch(idx)
            x, w32, params = degrade(x.to(dev), w.to(dev), side, g)
            logits = model(to_input(x, dev), w32).float().cpu()
            for k, i in enumerate(idx):
                hyp = greedy_decode(logits[k, : model.out_len(int(w32[k]))], self.charset)
                ref = ds.texts[i]
                ops = Levenshtein.opcodes(ref, hyp)
                bad = np.zeros(len(ref), dtype=bool)
                for op in ops:
                    if op.tag in ("replace", "delete"):
                        bad[op.src_start : op.src_end] = True
                    if op.tag == "replace" and op.src_end - op.src_start == op.dest_end - op.dest_start:
                        for a, b in zip(ref[op.src_start : op.src_end], hyp[op.dest_start : op.dest_end], strict=True):
                            conf[(a, b)] += 1
                    if op.tag == "insert" and 0 < op.src_start <= len(ref):
                        bad[op.src_start - 1] = True  # an inserted character is charged to its left neighbour
                for c, b in zip(ref, bad, strict=True):
                    tot_c[c] += 1
                    err_c[c] += int(b)
                e = int(bad.sum()) / max(1, len(ref))
                f, kd = family(ds.fonts[i]), text_kind(ref)
                tot_f[f] += 1
                err_f[f] += e
                tot_k[kd] += 1
                err_k[kd] += e
                for n, edges in self.cfg["bins"].items():
                    val = float(params[n][k])
                    if n == "blur_rel" and val <= 0:
                        continue
                    j = int(np.clip(np.searchsorted(edges, val, side="right") - 1, 0, len(edges) - 2))
                    bin_tot[n][j] += 1
                    bin_err[n][j] += e
        model.train()
        gamma, lo, hi = self.cfg.get("gamma", 0.7), self.cfg.get("clip", [0.4, 4.0])[0], self.cfg.get("clip", [0.4, 4.0])[1]
        smooth = self.cfg.get("smooth", 0.5)

        def mix(old: float, ratio: float) -> float:
            target = float(np.clip(ratio**gamma, lo, hi))
            return float(np.exp(smooth * np.log(old) + (1 - smooth) * np.log(target)))

        total = sum(err_c.values()) / max(1, sum(tot_c.values()))
        mean_line = sum(err_f.values()) / max(1, sum(tot_f.values()))
        state = {"step": v * self.every, "stream_char_error": total, "chars": {}, "fonts": {}, "kinds": {}, "bins": {}}
        for c, t in tot_c.items():
            if t >= 20:
                state["chars"][c] = mix(prev["chars"].get(c, 1.0), (err_c[c] / t) / max(total, 1e-9))
        for f, t in tot_f.items():
            state["fonts"][f] = (
                mix(prev["fonts"].get(f, 1.0), (err_f[f] / t) / max(mean_line, 1e-9)) if t >= 10 else prev["fonts"].get(f, 1.0)
            )
        for k, t in tot_k.items():
            state["kinds"][k] = mix(prev["kinds"].get(k, 1.0), (err_k[k] / t) / max(mean_line, 1e-9)) if t >= 10 else 1.0
        for n in self.cfg["bins"]:
            rate = np.where(bin_tot[n] >= 20, bin_err[n] / np.maximum(bin_tot[n], 1), np.nan)
            mean = np.nansum(bin_err[n]) / max(1.0, bin_tot[n].sum())
            ratio = np.nan_to_num(rate / max(mean, 1e-9), nan=1.0)
            p = self.base_probs[n] * np.clip(ratio**gamma, lo, hi)
            old = np.array(prev["bins"][n])
            p = np.exp(smooth * np.log(old) + (1 - smooth) * np.log(p / p.sum()))
            state["bins"][n] = (p / p.sum()).tolist()
        bins_report = {
            n: {
                "edges": self.cfg["bins"][n],
                "error": [round(float(a / b), 4) if b else None for a, b in zip(bin_err[n], bin_tot[n], strict=True)],
            }
            for n in bin_err
        }
        worst = sorted(conf.items(), key=lambda kv: -kv[1])[:25]
        report = {"step": v * self.every, "char_error": total, "top_confusions": [[a, b, c] for (a, b), c in worst],
                  "kind_error": {k: err_k[k] / t for k, t in tot_k.items()},
                  "font_error_worst": sorted(((f, err_f[f] / t) for f, t in tot_f.items() if t >= 10), key=lambda x: -x[1])[:10],
                  "bins": bins_report}  # fmt: skip
        return state, report
