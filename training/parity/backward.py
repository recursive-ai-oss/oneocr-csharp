"""Backward-pass parity fixtures for the C# fine-tuning kernels (crnn-v0).

Usage:
  uv run python -m parity.backward --fixtures FORWARD_FIXTURE_DIR --out OUT_DIR [--seed 21]

FORWARD_FIXTURE_DIR is a directory written by `parity.fixtures` (it supplies model.onm, lines/ and the truth
text). The references use the container's fp16-rounded, batch-norm-folded weights, exactly as the forward
fixtures do. See parity/README.md for the format; this file is the generator and the finite-difference check.

Everything is seeded and runs single-threaded on the CPU, so regenerating gives byte-identical files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from export.container import read_container
from train.data import load_gray, normalize

FORMAT = "oneocr-backward-parity-1"
SHARD_FLOATS = 200_000  # 800 KB: every file stays under 1 MB
NEG = -1.0e30  # log(0) stand-in that keeps autograd free of NaNs in the CTC recursion

# Lines used for per-layer fixtures (two, to keep the weight gradients small) and for the CTC/optimizer batch.
LAYER_LINES = ["val-00002929", "val-00002012"]
BATCH_LINES = ["val-00002929", "val-00002012", "val-00000228", "val-00002657"]
FD_EXTRA = "val-00000965"

ADAMW = {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}


# ---------------------------------------------------------------- parameter view (what the C# loader builds)


def param_names(cfg: dict) -> list[str]:
    names = []
    for i in range(len(cfg["conv"])):
        names += [f"conv.{i}.weight", f"conv.{i}.bias"]
    for k in range(cfg["lstm"]["layers"]):
        for sfx in ("", "_reverse"):
            names += [f"lstm.weight_ih_l{k}{sfx}", f"lstm.weight_hh_l{k}{sfx}", f"lstm.bias_l{k}{sfx}"]
    return names + ["fc.weight", "fc.bias"]


def trainable_params(cfg: dict, t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Container tensors as the trainer sees them: the two LSTM biases are one tensor, b = b_ih + b_hh (fp32 sum)."""
    p = {}
    for n in param_names(cfg):
        if ".bias_l" in n:
            k, sfx = n.split(".bias_l")[1], ""
            if k.endswith("_reverse"):
                k, sfx = k[: -len("_reverse")], "_reverse"
            p[n] = t[f"lstm.bias_ih_l{k}{sfx}"] + t[f"lstm.bias_hh_l{k}{sfx}"]
        else:
            p[n] = t[n]
    return {n: v.float().contiguous() for n, v in p.items()}


# ---------------------------------------------------------------- layer functions (shared by every check)

_lstm_cache: dict[tuple, nn.LSTM] = {}


def lstm_apply(x: torch.Tensor, p: dict[str, torch.Tensor], k: int, hidden: int) -> torch.Tensor:
    """One bidirectional LSTM layer, zero initial state, gates ordered i,f,g,o. x [T,in] -> [T,2*hidden]."""
    key = (x.shape[-1], hidden, x.dtype)
    if key not in _lstm_cache:
        _lstm_cache[key] = nn.LSTM(x.shape[-1], hidden, 1, bidirectional=True, batch_first=True).to(x.dtype)
    m = _lstm_cache[key]
    params = {}
    for sfx, ps in (("", ""), ("_reverse", "_reverse")):
        params[f"weight_ih_l0{ps}"] = p[f"lstm.weight_ih_l{k}{sfx}"]
        params[f"weight_hh_l0{ps}"] = p[f"lstm.weight_hh_l{k}{sfx}"]
        params[f"bias_ih_l0{ps}"] = p[f"lstm.bias_l{k}{sfx}"]
        params[f"bias_hh_l0{ps}"] = torch.zeros_like(p[f"lstm.bias_l{k}{sfx}"])
    return functional_call(m, params, (x[None],))[0][0]


def model_logits(cfg: dict, p: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """x [1,H,W] normalised -> logits [T,classes]."""
    h = x[None]
    for i, c in enumerate(cfg["conv"]):
        h = F.relu(F.conv2d(h, p[f"conv.{i}.weight"], p[f"conv.{i}.bias"], padding=tuple(c["padding"])))
        if c["pool"]:
            h = F.max_pool2d(h, tuple(c["pool"]))
    seq = h[0].squeeze(1).transpose(0, 1)  # [C,1,T] -> [T,C]
    for k in range(cfg["lstm"]["layers"]):
        seq = lstm_apply(seq, p, k, cfg["lstm"]["hidden"])
    return F.linear(seq, p["fc.weight"], p["fc.bias"])


def ctc_nll(logits: torch.Tensor, target: list[int]) -> torch.Tensor:
    """Negative log-likelihood of `target` under CTC (Graves et al. 2006), blank = class 0, one sample.

    Written out in log space so autograd gives the exact gradient w.r.t. the logits (through log-softmax).
    logits [T,C]; target a non-empty list of class ids >= 1.
    """
    logp = F.log_softmax(logits, dim=-1)
    ext = [0]
    for c in target:
        ext += [c, 0]
    S = len(ext)
    idx = torch.tensor(ext)
    skip = torch.tensor([s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2] for s in range(S)])
    neg = torch.full((1,), NEG, dtype=logits.dtype)
    neg2 = torch.full((2,), NEG, dtype=logits.dtype)
    alpha = torch.full((S,), NEG, dtype=logits.dtype)
    alpha = alpha + 0  # fresh tensor
    first = logp[0, idx]
    alpha = torch.cat([first[:2], alpha[2:]])
    for t in range(1, logits.shape[0]):
        a1 = torch.cat([neg, alpha[:-1]])
        a2 = torch.where(skip, torch.cat([neg2, alpha[:-2]]), torch.full_like(alpha, NEG))
        alpha = torch.logsumexp(torch.stack([alpha, a1, a2]), dim=0) + logp[t, idx]
    return -torch.logsumexp(alpha[-2:], dim=0)


def batch_loss(logits: list[torch.Tensor], targets: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean over samples of the per-sample NLL, per-sample NLL [B])."""
    nll = torch.stack([ctc_nll(lg, tg) for lg, tg in zip(logits, targets, strict=True)])
    return nll.mean(), nll


# ---------------------------------------------------------------- vjp harness and floors


def vjp(fn, leaves: dict[str, torch.Tensor], gy: torch.Tensor, dtype: torch.dtype):
    d = {k: v.detach().to(dtype).clone().requires_grad_(True) for k, v in leaves.items()}
    y = fn(d)
    y.backward(gy.to(dtype))
    return {k: v.grad.detach() for k, v in d.items()}


def diff_stats(a32: torch.Tensor, a64: torch.Tensor) -> dict:
    d = (a32.double() - a64.double()).abs()
    ref = a64.double().abs()
    m = float(ref.max())
    big = ref > 1e-3 * m
    return {
        "max_abs": float(d.max()),
        "ref_max_abs": m,
        "max_abs_over_ref_max": float(d.max()) / m if m > 0 else 0.0,
        "max_rel_above_1e-3_of_max": float((d[big] / ref[big]).max()) if big.any() else 0.0,
    }


def merge_stats(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst[k] = max(dst.get(k, 0.0), v)


class Writer:
    def __init__(self, out: Path):
        self.out = out

    def put(self, rel: str, arr: torch.Tensor) -> dict:
        """Writes a tensor as little-endian float32, split into <1 MB parts when large; returns its manifest entry."""
        flat = arr.detach().contiguous().float().reshape(-1).numpy().astype("<f4")
        parts = [flat[i : i + SHARD_FLOATS] for i in range(0, max(flat.size, 1), SHARD_FLOATS)]
        files = []
        for j, part in enumerate(parts):
            name = f"{rel}.f32" if len(parts) == 1 else f"{rel}.part{j:02d}.f32"
            path = self.out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(part.tobytes())
            files.append({"file": name, "sha256": hashlib.sha256(part.tobytes()).hexdigest()})
        return {"shape": list(arr.shape), "files": files}


# ---------------------------------------------------------------- fixtures


def conv_chain(cfg: dict, p: dict, x: torch.Tensor):
    """Yields (block, input, preactivation, relu out, pool out) in fp32 for one line, x [1,H,W]."""
    h = x[None]
    for i, c in enumerate(cfg["conv"]):
        z = F.conv2d(h, p[f"conv.{i}.weight"], p[f"conv.{i}.bias"], padding=tuple(c["padding"]))
        a = F.relu(z)
        q = F.max_pool2d(a, tuple(c["pool"])) if c["pool"] else a
        yield i, h[0], z[0], a[0], q[0]
        h = q


def layer_specs(cfg: dict, p: dict, x: torch.Tensor):
    """(name, kind, attrs, fn, leaves, output_shape) for every backward-relevant layer of one line."""
    specs = []
    h = cfg["lstm"]["hidden"]
    last = None
    for i, hin, z, a, q in conv_chain(cfg, p, x):
        c = cfg["conv"][i]
        pad = tuple(c["padding"])
        w, b = f"conv.{i}.weight", f"conv.{i}.bias"
        specs.append((f"conv.{i}.conv", "conv", {"padding": list(pad), "stride": [1, 1], "dilation": [1, 1]},
                      lambda t, pad=pad, w=w, b=b: F.conv2d(t["x"][None], t[w], t[b], padding=pad)[0],
                      {"x": hin, w: p[w], b: p[b]}, z.shape))  # fmt: skip
        specs.append((f"conv.{i}.relu", "relu", {"derivative_at_zero": 0}, lambda t: F.relu(t["x"]), {"x": z}, a.shape))
        if c["pool"]:
            ks = tuple(c["pool"])
            specs.append((f"conv.{i}.pool", "maxpool", {"kernel": list(ks), "stride": list(ks),
                                                        "tie": "first maximum in row-major window order"},
                          lambda t, ks=ks: F.max_pool2d(t["x"][None], ks)[0], {"x": a}, q.shape))  # fmt: skip
        last = q
    seq = last.squeeze(1).transpose(0, 1).contiguous()  # [T,C]
    for k in range(cfg["lstm"]["layers"]):
        names = [f"lstm.{n}_l{k}{s}" for s in ("", "_reverse") for n in ("weight_ih", "weight_hh", "bias")]
        leaves = {"x": seq, **{n: p[n] for n in names}}
        specs.append((f"lstm.l{k}", "lstm", {"hidden": h, "bidirectional": True, "gate_order": "i,f,g,o",
                                             "initial_state": "zero"},
                      lambda t, k=k: lstm_apply(t["x"], t, k, h), leaves, (seq.shape[0], 2 * h)))  # fmt: skip
        seq = lstm_apply(seq, p, k, h).detach()
    specs.append(("fc", "linear", {}, lambda t: F.linear(t["x"], t["fc.weight"], t["fc.bias"]),
                  {"x": seq, "fc.weight": p["fc.weight"], "fc.bias": p["fc.bias"]}, (seq.shape[0], cfg["classes"])))  # fmt: skip
    return specs


def adamw_manual(p: torch.Tensor, g: torch.Tensor, m: torch.Tensor, v: torch.Tensor, step: int, o: dict):
    """Decoupled-weight-decay Adam (Loshchilov & Hutter 2019), the formulation torch.optim.AdamW implements."""
    b1, b2 = o["betas"]
    p = p * (1 - o["lr"] * o["weight_decay"])
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    denom = (v / (1 - b2**step)).sqrt() + o["eps"]
    return p - o["lr"] / (1 - b1**step) * m / denom, m, v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, help="forward fixture dir (model.onm, lines/, fixtures.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    src, out = Path(a.fixtures), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w = Writer(out)
    shutil.copyfile(src / "model.onm", out / "model.onm")
    manifest, tensors = read_container(src / "model.onm")
    cfg, charset = manifest["config"], manifest["charset"]
    cindex = {c: i for i, c in enumerate(charset)}
    p32 = trainable_params(cfg, {k: torch.from_numpy(v) for k, v in tensors.items()})
    fwd = {r["id"]: r for r in json.loads((src / "fixtures.json").read_text(encoding="utf-8"))["lines"]}

    def line_input(lid: str) -> torch.Tensor:
        (out / "lines").mkdir(exist_ok=True)
        shutil.copyfile(src / "lines" / f"{lid}.png", out / "lines" / f"{lid}.png")
        return torch.from_numpy(normalize(load_gray(src / "lines" / f"{lid}.png")))[None]

    # ---- per-layer
    floors: dict[str, dict] = {}
    layer_lines = []
    for li, lid in enumerate(LAYER_LINES):
        x = line_input(lid)
        entries = []
        for si, (name, kind, attrs, fn, leaves, oshape) in enumerate(layer_specs(cfg, p32, x)):
            rng = np.random.default_rng([a.seed, li, si])
            gy = torch.from_numpy(rng.standard_normal(tuple(oshape)).astype(np.float32))
            g32 = vjp(fn, leaves, gy, torch.float32)
            g64 = vjp(fn, leaves, gy, torch.float64)
            e = {"name": name, "kind": kind, "attrs": attrs, "seed": [a.seed, li, si],
                 "input": w.put(f"layers/{lid}/{name}.x", leaves["x"]),
                 "grad_output": w.put(f"layers/{lid}/{name}.gy", gy), "expected": {}}  # fmt: skip
            for k, v in g32.items():
                e["expected"][k] = w.put(f"layers/{lid}/{name}.d[{k}]", v)
                merge_stats(floors.setdefault(name, {}).setdefault(k, {}), diff_stats(v, g64[k]))
            e["params"] = [k for k in leaves if k != "x"]
            entries.append(e)
        layer_lines.append({"id": lid, "png": f"lines/{lid}.png", "text": fwd[lid]["text"], "width": fwd[lid]["width"],
                            "layers": entries})  # fmt: skip

    # ---- whole model: CTC loss and dlogits for two batches over the same four images:
    # "truth" (the lines' own text: the model fits it, so the loss is small) and "shifted" (each image paired with
    # the next line's text: a large loss and strong gradients, which is what the optimizer steps train on)
    xs = [line_input(lid) for lid in BATCH_LINES]
    truth = [fwd[lid]["text"] for lid in BATCH_LINES]
    batches = {"truth": truth, "shifted": truth[1:] + truth[:1]}
    with torch.no_grad():
        lg32 = [model_logits(cfg, p32, x) for x in xs]
    model_floor: dict[str, dict] = {}
    model_docs: dict[str, dict] = {}
    all_targets: dict[str, list[list[int]]] = {}
    for tag, texts in batches.items():
        targets = all_targets[tag] = [[cindex[c] for c in t] for t in texts]
        for lg, tg, t in zip(lg32, targets, texts, strict=True):
            need = len(tg) + sum(p == q for p, q in zip(tg, tg[1:], strict=False))
            assert lg.shape[0] >= need, f"{t!r} infeasible"

        def ctc_vjp(dtype, targets=targets):
            lg = [v.detach().to(dtype).requires_grad_(True) for v in lg32]
            loss, nll = batch_loss(lg, targets)
            loss.backward()
            return loss.detach(), nll.detach(), [v.grad for v in lg]

        l32, n32, d32 = ctc_vjp(torch.float32)
        l64, n64, d64 = ctc_vjp(torch.float64)
        # cross-check against the framework's own CTC (value and gradient w.r.t. logits)
        lgt = [v.detach().double().requires_grad_(True) for v in lg32]
        ref = sum(
            F.ctc_loss(F.log_softmax(v, -1)[:, None], torch.tensor([tg]), [v.shape[0]], [len(tg)], blank=0, reduction="sum")
            for v, tg in zip(lgt, targets, strict=True)
        ) / len(lgt)
        ref.backward()
        builtin = {"loss_abs_diff": float(abs(ref.detach() - l64)),
                   "dlogits_max_abs_diff": max(float((v.grad - g).abs().max()) for v, g in zip(lgt, d64, strict=True))}  # fmt: skip
        samples = []
        mf = model_floor.setdefault(tag, {})
        for i, lid in enumerate(BATCH_LINES):
            samples.append({"id": lid, "png": f"lines/{lid}.png", "text": texts[i], "width": fwd[lid]["width"],
                            "steps": lg32[i].shape[0], "target": targets[i], "nll": float(n32[i]),
                            "nll_fp64": float(n64[i]),
                            "logits": w.put(f"model/{tag}/{lid}.logits", lg32[i]),
                            "dlogits": w.put(f"model/{tag}/{lid}.dlogits", d32[i])})  # fmt: skip
            merge_stats(mf.setdefault("dlogits", {}), diff_stats(d32[i], d64[i]))
        mf["loss"] = {"max_abs": float(abs(l32.double() - l64))}
        mf["nll"] = {"max_abs": float((n32.double() - n64).abs().max())}
        model_docs[tag] = {"batch": samples, "loss": float(l32), "loss_fp64": float(l64), "builtin_ctc_loss_check": builtin}
    targets = all_targets["shifted"]

    def loss_and_grads(pp: dict[str, torch.Tensor], dtype):
        d = {k: v.detach().to(dtype).clone().requires_grad_(True) for k, v in pp.items()}
        loss, _ = batch_loss([model_logits(cfg, d, x.to(dtype)) for x in xs], targets)
        loss.backward()
        return float(loss.detach()), {k: v.grad.detach() for k, v in d.items()}

    # ---- optimizer: three real steps on the batch (gradients recomputed each step, fp32)
    names = param_names(cfg)
    pcur = {k: v.clone().requires_grad_(True) for k, v in p32.items()}
    opt = torch.optim.AdamW([pcur[n] for n in names], foreach=False, **ADAMW)
    grads, losses, after = {}, {}, {}
    sets = {}
    for step in (1, 2, 3):
        loss, g = loss_and_grads({k: v.detach() for k, v in pcur.items()}, torch.float32)
        losses[step] = loss
        grads[step] = g
        if step == 1:
            _, g64 = loss_and_grads(p32, torch.float64)
            for n in names:
                merge_stats(model_floor.setdefault("param_grads_step1", {}).setdefault(n, {}), diff_stats(g[n], g64[n]))
        for n in names:
            pcur[n].grad = g[n].clone()
        opt.step()
        after[step] = {n: pcur[n].detach().clone() for n in names}
    # fp64 replay of the optimizer on the same (fp32) gradients: the floor for the update arithmetic alone,
    # and a check that torch.optim.AdamW is the textbook formula above
    opt_floor = {}
    for n in names:
        pp, m, v = p32[n].double(), torch.zeros_like(p32[n]).double(), torch.zeros_like(p32[n]).double()
        for step in (1, 2, 3):
            pp, m, v = adamw_manual(pp, grads[step][n].double(), m, v, step, ADAMW)
            if step in (1, 3):
                merge_stats(opt_floor.setdefault(f"step{step}", {}).setdefault(n, {}), diff_stats(after[step][n], pp))
    # step 1 is lr * sign(g) where |g| >> eps, so it is exact; summarise the worst parameter per step
    opt_summary = {s: {"max_abs": max(v["max_abs"] for v in d.values()),
                       "max_abs_over_ref_max": max(v["max_abs_over_ref_max"] for v in d.values())}
                   for s, d in opt_floor.items()}  # fmt: skip
    for step in (1, 2, 3):
        sets[f"grads_step{step}"] = {n: w.put(f"optimizer/grads_step{step}/{n}", grads[step][n]) for n in names}
    for step in (1, 3):
        sets[f"weights_after_step{step}"] = {n: w.put(f"optimizer/weights_after_step{step}/{n}", after[step][n])
                                             for n in names}  # fmt: skip

    model_doc = {
        "batches": model_docs,
        "loss_reduction": "mean over the B samples of each sample's CTC negative log-likelihood (no division by "
        "target or input length); dlogits are gradients of that mean, so each sample's block is "
        "its own gradient divided by B",  # fmt: skip
        "param_gradients": "optimizer/grads_step1/<name> are d(loss)/d(parameter) of the shifted batch at the "
        "container weights",  # fmt: skip
    }
    opt_doc = {
        "algorithm": "AdamW with decoupled weight decay, per tensor, step counter t starting at 1",
        "update": [
            "p <- p * (1 - lr * weight_decay)",
            "m <- b1*m + (1-b1)*g",
            "v <- b2*v + (1-b2)*g*g",
            "p <- p - (lr / (1 - b1^t)) * m / (sqrt(v / (1 - b2^t)) + eps)",
        ],  # fmt: skip
        "lr": ADAMW["lr"],
        "betas": list(ADAMW["betas"]),
        "eps": ADAMW["eps"],
        "weight_decay": ADAMW["weight_decay"],
        "decay_applies_to_biases": True,
        "gradient_clipping": None,
        "m_v_initial": "zero",
        "batch": "shifted",
        "steps": "three real steps on the shifted batch: gradients are recomputed in fp32 at each step's weights; "
        "weights_after_step1/3 are the results; grads_step1..3 let the optimizer be tested alone",  # fmt: skip
        "loss_per_step": losses,
        "sets": sets,
        "fp32_vs_fp64_replay_floor": opt_summary,
    }

    doc = {
        "format": FORMAT,
        "architecture": manifest["architecture"],
        "model": "model.onm",
        "model_sha256": hashlib.sha256((out / "model.onm").read_bytes()).hexdigest(),
        "dtype": "float32 little-endian, row-major; tensors over 200000 floats are split into .partNN.f32 files "
        "(concatenate in order)",  # fmt: skip
        "seed": a.seed,
        "parameters": {
            "names": names,
            "shapes": {n: list(p32[n].shape) for n in names},
            "lstm_bias": "one tensor per direction, b = b_ih + b_hh (fp32 sum of the container values)",
        },  # fmt: skip
        "fp32_vs_fp64_floor": {"layers": floors, "model": model_floor},
        "layer_lines": layer_lines,
        "model_level": model_doc,
        "optimizer": opt_doc,
    }
    (out / "fixtures.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")

    fd = finite_differences(cfg, p32, layer_specs, xs, all_targets["shifted"], a.seed, line_input, fwd)
    (out / "gradcheck.json").write_text(json.dumps(fd, indent=1), encoding="utf-8")

    sums = sorted(f for f in out.rglob("*") if f.is_file() and f.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text(
        "".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out).as_posix()}\n" for f in sums),
        encoding="utf-8",
    )
    big = [f for f in sums if f.stat().st_size >= 1_000_000 and f.name != "model.onm"]
    assert not big, f"files over 1 MB: {big}"
    print(f"fixtures -> {out}: {len(sums)} files; worst finite-difference rel err {fd['worst_rel_err']:.2e}")
    print(json.dumps({"opt": opt_summary, "builtin": builtin}))


# ---------------------------------------------------------------- finite differences (fp64)


def finite_differences(cfg, p32, layer_specs_fn, xs, targets, seed, line_input, fwd, eps: float = 1e-5, ndir: int = 2):
    """Checks each analytic gradient against central differences of the same fp64 function.

    For a random unit direction v: (s(theta + eps v) - s(theta - eps v)) / 2 eps against <grad, v>, where s is the
    scalar <gy, f(theta)> (loss for the whole model). Run on fp64 copies of the fp32 fixture inputs.
    """
    rng = np.random.default_rng([seed, 999])
    rows = []

    def check(label, scalar, leaves64: dict[str, torch.Tensor], grads64: dict[str, torch.Tensor], eps: float = eps):
        for name, leaf in leaves64.items():
            for _ in range(ndir):
                r = torch.from_numpy(rng.standard_normal(tuple(leaf.shape))).double()
                gn = grads64[name].norm()
                # random direction tilted toward the gradient, so <grad, v> is never lost in the noise floor
                v = r / r.norm() + (grads64[name] / gn if gn > 0 else 0)
                v = v / v.norm()
                with torch.no_grad():
                    sp = float(scalar({**leaves64, name: leaf + eps * v}))
                    sm = float(scalar({**leaves64, name: leaf - eps * v}))
                fd = (sp - sm) / (2 * eps)
                an = float((grads64[name] * v).sum())
                rows.append({"check": label, "tensor": name, "fd": fd, "analytic": an,
                             "rel_err": abs(fd - an) / max(abs(fd), abs(an), 1e-8)})  # fmt: skip

    for lid in dict.fromkeys(LAYER_LINES[:1] + [FD_EXTRA]):
        x = line_input(lid)
        for si, (name, kind, _, fn, leaves, oshape) in enumerate(layer_specs_fn(cfg, p32, x)):
            gy = torch.from_numpy(np.random.default_rng([seed, 7, si]).standard_normal(tuple(oshape))).double()
            if kind == "maxpool":
                # post-ReLU activations tie at 0 and max-pool is not differentiable at ties; jitter them apart
                jit = np.random.default_rng([seed, 8, si]).standard_normal(tuple(leaves["x"].shape))
                leaves = {"x": leaves["x"] + 0.1 * torch.from_numpy(jit).float()}
            l64 = {k: v.double() for k, v in leaves.items()}
            g = vjp(fn, leaves, gy, torch.float64)
            check(f"{lid}:{name}", lambda t, fn=fn, gy=gy: (fn(t) * gy).sum(), l64, g)

    # whole model CTC: loss against the container parameters and against the logits
    p64 = {k: v.double() for k, v in p32.items()}
    xs64 = [x.double() for x in xs]

    def loss_fn(t):
        return batch_loss([model_logits(cfg, t, x) for x in xs64], targets)[0]

    leaves = {k: v.clone().requires_grad_(True) for k, v in p64.items()}
    loss_fn(leaves).backward()
    # smaller step: millions of ReLU/max-pool kinks make a whole-network difference less smooth
    check("model:ctc-vs-params", loss_fn, p64, {k: v.grad for k, v in leaves.items()}, eps=1e-7)

    with torch.no_grad():
        lg = {f"logits[{i}]": model_logits(cfg, p64, x) for i, x in enumerate(xs64)}
    lleaves = {k: v.clone().requires_grad_(True) for k, v in lg.items()}
    batch_loss(list(lleaves.values()), targets)[0].backward()
    check("model:ctc-vs-logits", lambda t: batch_loss(list(t.values()), targets)[0], lg,
          {k: v.grad for k, v in lleaves.items()})  # fmt: skip

    # torch.autograd.gradcheck of the CTC reference itself on a tiny random case
    g = torch.Generator().manual_seed(seed)
    tiny = torch.randn(7, 5, dtype=torch.float64, generator=g, requires_grad=True)
    ok = torch.autograd.gradcheck(lambda z: ctc_nll(z, [2, 2, 3]), (tiny,), eps=1e-6, atol=1e-7, rtol=1e-5)
    worst = max(r["rel_err"] for r in rows)
    by_check: dict[str, float] = {}
    for r in rows:
        by_check[r["check"]] = max(by_check.get(r["check"], 0.0), r["rel_err"])
    return {"eps": eps, "directions_per_tensor": ndir, "worst_rel_err": worst, "gradcheck_ctc_tiny": bool(ok),
            "worst_by_check": by_check, "rows": rows}  # fmt: skip


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
