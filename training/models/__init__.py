"""Recognizer architectures. A model config names its architecture in `arch` (absent: the original CRNN).

Every model exposes forward(x[B,1,32,W], widths) -> logits[B,T,classes] and out_len(width) -> T.
"""

from __future__ import annotations

from models.crnn import CRNN
from models.crnn import DEFAULT_CONFIG as CRNN_CONFIG
from models.crnn import macs_per_column as crnn_macs
from models.svtr import DEFAULT_CONFIG as SVTR_CONFIG
from models.svtr import SVTRLite
from models.svtr import macs_per_column as svtr_macs


def build(config: dict):
    arch = config.get("arch", "crnn-v0")
    if arch == "crnn-v0":
        return CRNN(config)
    if arch == "svtr-lite":
        return SVTRLite(config)
    raise ValueError(f"unknown architecture {arch}")


def macs_per_column(config: dict, t: int = 80) -> dict:
    return svtr_macs(config, t) if config.get("arch") == "svtr-lite" else crnn_macs(config)


def default_config(arch: str, classes: int) -> dict:
    if arch == "crnn-v0":
        return {**CRNN_CONFIG, "classes": classes}
    return {**SVTR_CONFIG, "classes": classes}


def fold(model):
    """Container tensors (fp32, batch norm folded) of a trained model."""
    if model.config.get("arch") == "svtr-lite":
        from models.svtr import fold_batch_norm

        return fold_batch_norm(model)
    from models.crnn import fold_batch_norm

    return fold_batch_norm(model)


def load_folded(config: dict, tensors: dict):
    if config.get("arch") == "svtr-lite":
        from models.svtr import load_folded as lf
    else:
        from models.crnn import load_folded as lf
    return lf(config, tensors)
