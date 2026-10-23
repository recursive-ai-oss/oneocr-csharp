"""crnn-v0: convolutional stack -> BiLSTM -> linear -> CTC.

Follows Shi, Bai, Yao, "An End-to-End Trainable Neural Network for Image-based Sequence Recognition
and Its Application to Scene Text Recognition" (2015), with CTC from Graves et al. (2006). The layer
graph is the one in the model container v0 spec; everything here is randomly initialised.

Tensors are named exactly as the container names them: `conv.<i>.weight/bias` (batch norm folded at
export), `lstm.*` (PyTorch's own parameter names) and `fc.*`.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# Accuracy/CPU-cost trade chosen for the first model: six conv blocks reduce 32 px height to 1 (two 2x2 pools make the
# sequence step 4 px wide, three 2x1 pools finish the height), two BiLSTM layers of 192.
DEFAULT_CONFIG = {
    "conv": [
        {"out": 32, "kernel": [3, 3], "padding": [1, 1], "pool": [2, 2]},
        {"out": 64, "kernel": [3, 3], "padding": [1, 1], "pool": [2, 2]},
        {"out": 128, "kernel": [3, 3], "padding": [1, 1], "pool": None},
        {"out": 128, "kernel": [3, 3], "padding": [1, 1], "pool": [2, 1]},
        {"out": 192, "kernel": [3, 3], "padding": [1, 1], "pool": [2, 1]},
        {"out": 192, "kernel": [3, 3], "padding": [1, 1], "pool": [2, 1]},
    ],
    "lstm": {"hidden": 192, "layers": 2},
    "classes": 622,
}

INPUT_HEIGHT = 32
BN_EPS = 1e-5


def output_length(width: int, config: dict) -> int:
    """Sequence length for an input width (pooling drops a trailing partial window)."""
    for c in config["conv"]:
        if c["pool"]:
            width //= c["pool"][1]
    return width


def conv_out_height(config: dict, height: int = INPUT_HEIGHT) -> int:
    for c in config["conv"]:
        kh, ph = c["kernel"][0], c["padding"][0]
        height = height + 2 * ph - kh + 1
        if c["pool"]:
            height //= c["pool"][0]
    return height


class CRNN(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = copy.deepcopy(config or DEFAULT_CONFIG)
        assert conv_out_height(self.config) == 1, "conv stack must reduce the height to 1"
        blocks = []
        cin = 1
        for c in self.config["conv"]:
            layers: list[nn.Module] = [
                nn.Conv2d(cin, c["out"], tuple(c["kernel"]), padding=tuple(c["padding"]), bias=False),
                nn.BatchNorm2d(c["out"], eps=BN_EPS),
                nn.ReLU(inplace=True),
            ]
            if c["pool"]:
                layers.append(nn.MaxPool2d(tuple(c["pool"])))
            blocks.append(nn.Sequential(*layers))
            cin = c["out"]
        self.conv = nn.ModuleList(blocks)
        h = self.config["lstm"]["hidden"]
        self.lstm = nn.LSTM(cin, h, num_layers=self.config["lstm"]["layers"], bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * h, self.config["classes"])

    def features(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.conv:
            x = b(x)
        return x.squeeze(2).transpose(1, 2)  # [B, C, 1, W] -> [B, W, C]

    def forward(self, x: torch.Tensor, widths: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, 1, 32, W] normalised; widths: true input widths when the batch is padded. Returns logits [B, T, classes]."""
        seq = self.features(x)
        if widths is not None:
            lengths = torch.tensor([output_length(int(w), self.config) for w in widths])
            packed = nn.utils.rnn.pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=seq.shape[1])
        else:
            out, _ = self.lstm(seq)
        return self.fc(out)


def fold_batch_norm(model: CRNN) -> dict[str, torch.Tensor]:
    """Folds each conv's batch norm (eval statistics) into weight and bias; returns the container's tensors in fp32."""
    sd: dict[str, torch.Tensor] = {}
    for i, b in enumerate(model.conv):
        conv, bn = b[0], b[1]
        scale = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
        sd[f"conv.{i}.weight"] = (conv.weight.double() * scale.view(-1, 1, 1, 1)).float()
        sd[f"conv.{i}.bias"] = (bn.bias.double() - bn.running_mean.double() * scale).float()
    for k, v in model.lstm.state_dict().items():
        sd[f"lstm.{k}"] = v.detach().clone()
    for k, v in model.fc.state_dict().items():
        sd[f"fc.{k}"] = v.detach().clone()
    return {k: v.detach().cpu().float() for k, v in sd.items()}


class FoldedReference:
    """fp32 forward pass over the container's (folded, fp16-rounded) tensors, with every layer's output exposed.

    This is the reference the C# kernels are checked against, so it reads only the tensors the file holds.
    """

    def __init__(self, config: dict, tensors: dict[str, torch.Tensor], dtype: torch.dtype = torch.float32):
        self.config = config
        self.dtype = dtype
        self.t = {k: v.to(dtype) for k, v in tensors.items()}

    def _lstm_layer(self, k: int, x: torch.Tensor) -> torch.Tensor:
        h = self.config["lstm"]["hidden"]
        m = nn.LSTM(x.shape[-1], h, num_layers=1, bidirectional=True, batch_first=True).to(self.dtype)
        sd = {}
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            sd[f"{name}_l0"] = self.t[f"lstm.{name}_l{k}"]
            sd[f"{name}_l0_reverse"] = self.t[f"lstm.{name}_l{k}_reverse"]
        m.load_state_dict(sd)
        return m(x)[0]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: [1, 1, H, W] normalised. Returns {'conv.i': [C,H,W], 'lstm.lk': [T,2h], 'fc': [T,classes]}."""
        out: dict[str, torch.Tensor] = {}
        x = x.to(self.dtype)
        for i, c in enumerate(self.config["conv"]):
            x = F.conv2d(x, self.t[f"conv.{i}.weight"], self.t[f"conv.{i}.bias"], padding=tuple(c["padding"]))
            x = F.relu(x)
            if c["pool"]:
                x = F.max_pool2d(x, tuple(c["pool"]))
            out[f"conv.{i}"] = x[0].clone()
        seq = x.squeeze(2).transpose(1, 2)
        for k in range(self.config["lstm"]["layers"]):
            seq = self._lstm_layer(k, seq)
            out[f"lstm.l{k}"] = seq[0].clone()
        out["fc"] = F.linear(seq, self.t["fc.weight"], self.t["fc.bias"])[0].clone()
        return out


def greedy_decode(logits: torch.Tensor, charset: list[str]) -> str:
    """Greedy CTC: arg-max per step (lowest index on ties), merge repeats, drop blanks."""
    ids = logits.argmax(-1).tolist()
    out, prev = [], 0
    for i in ids:
        if i != prev and i != 0:
            out.append(charset[i])
        prev = i
    return "".join(out)


def load_folded(config: dict, tensors: dict[str, torch.Tensor]) -> CRNN:
    """A CRNN whose batch norms are exact identities and whose conv weights/biases are the container's tensors.

    Lets the usual batched evaluation code run the fp16-rounded, folded weights in fp32.
    """
    m = CRNN(config)
    sd = {}
    for i in range(len(config["conv"])):
        sd[f"conv.{i}.0.weight"] = tensors[f"conv.{i}.weight"]
        sd[f"conv.{i}.1.weight"] = torch.ones_like(tensors[f"conv.{i}.bias"])
        sd[f"conv.{i}.1.bias"] = tensors[f"conv.{i}.bias"]
        sd[f"conv.{i}.1.running_mean"] = torch.zeros_like(tensors[f"conv.{i}.bias"])
        sd[f"conv.{i}.1.running_var"] = torch.full_like(tensors[f"conv.{i}.bias"], 1.0 - BN_EPS)
        sd[f"conv.{i}.1.num_batches_tracked"] = torch.tensor(0)
    for k in m.lstm.state_dict():
        sd[f"lstm.{k}"] = tensors[f"lstm.{k}"]
    sd["fc.weight"], sd["fc.bias"] = tensors["fc.weight"], tensors["fc.bias"]
    m.load_state_dict(sd)
    return m.eval()
