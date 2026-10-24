"""Page degradation and camera simulation with exact box transforms.

A clean render is resampled to a lower resolution, placed on a canvas through a homography (small rotation for scans, keystone
perspective for phone photos), lit unevenly, thinned, blurred, noised, optionally reduced to grey or to black and white, and
JPEG-compressed on save. The homography is returned so the page truth can be transformed with the image: word, line and
block quads are the images of their boxes; boxes are the axis-aligned bounds of the quads.

Profiles: clean (resolution only), scan (rotation, light, faint print, noise), photo (perspective on a desk, strong light,
blur), screen (resolution and JPEG only). Every parameter is drawn from a seeded generator and returned for logging. The `train`
and `heldout` range sets in the config are disjoint, so evaluation can use degradations training never saw.

Runs on CPU with torch (one thread per worker process) or on a GPU when a device is passed.
"""

from __future__ import annotations

import io
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def draw(rng: random.Random, ranges) -> float:
    """A value from a [lo, hi] range or from a list of such ranges (one picked uniformly, repeats weight it)."""
    if isinstance(ranges[0], (int, float)):
        ranges = [ranges]
    lo, hi = rng.choice(ranges)
    return rng.uniform(lo, hi)


def sample(rng: random.Random, cfg: dict) -> dict:
    """All degradation parameters for one page. `cfg` is a range set (`ranges`, `p`, `profiles`)."""
    r, p = cfg["ranges"], cfg["p"]
    profile = rng.choices(list(cfg["profiles"]), list(cfg["profiles"].values()))[0]
    d: dict = {"profile": profile, "dpi": round(draw(rng, r["dpi"]), 1)}
    geometric = profile in ("scan", "photo")
    d["rot"] = round(draw(rng, r["rot"]) * rng.choice([-1, 1]), 3) if profile == "scan" else 0.0
    d["persp"] = round(draw(rng, r["persp"]), 4) if profile == "photo" else 0.0
    d["margin"] = round(draw(rng, r["margin"]), 4) if geometric else 0.0
    d["light"] = round(draw(rng, r["light"]), 3) if profile in ("scan", "photo") and rng.random() < p["light"] else 0.0
    d["shadow"] = round(rng.uniform(0.15, 0.5), 3) if profile == "photo" and rng.random() < p["shadow"] else 0.0
    d["faint"] = round(draw(rng, r["faint"]), 3) if profile in ("scan", "photo", "clean") and rng.random() < p["faint"] else 1.0
    d["blur"] = round(draw(rng, r["blur"]) * d["dpi"] / 150, 3) if profile != "screen" and rng.random() < p["blur"] else 0.0
    d["motion"] = int(round(draw(rng, r["motion"]) * d["dpi"] / 150)) if profile == "photo" and rng.random() < p["motion"] else 0
    d["motion_axis"] = rng.choice([0, 1])
    d["read_noise"] = round(draw(rng, r["read"]), 4) if profile != "screen" and rng.random() < p["noise"] else 0.0
    d["shot_noise"] = round(draw(rng, r["shot"]), 4) if profile != "screen" and d["read_noise"] > 0 else 0.0
    d["gray"] = profile in ("scan", "clean") and rng.random() < p["gray"]
    d["binarize"] = profile == "scan" and rng.random() < p["binarize"]
    d["jpeg"] = int(round(draw(rng, r["jpeg_q"]))) if rng.random() < p["jpeg"][profile] else 0
    d["bg"] = [round(c, 3) for c in _bg_color(rng, profile)]
    d["seed"] = rng.randrange(2**31)
    return d


def _bg_color(rng: random.Random, profile: str) -> tuple[float, float, float]:
    if profile == "photo":
        base = rng.choice(
            [(0.15, 0.1, 0.07), (0.45, 0.33, 0.2), (0.3, 0.3, 0.32), (0.75, 0.75, 0.72), (0.08, 0.08, 0.08), (0.55, 0.5, 0.42)]
        )
    else:
        base = rng.choice([(0.0, 0.0, 0.0), (0.35, 0.35, 0.35), (0.85, 0.85, 0.85), (1.0, 1.0, 1.0), (0.2, 0.2, 0.25)])
    j = rng.uniform(0.9, 1.1)
    return tuple(min(1.0, c * j) for c in base)


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 matrix taking the four src points to the dst points (direct linear transform)."""
    A = []
    for (x, y), (u, v) in zip(src, dst, strict=False):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(A, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def _smooth(g: torch.Generator, n: int, h: int, w: int, cells: int) -> torch.Tensor:
    """Smooth random field in [-1, 1], (h, w): a cells x cells grid of values, bicubically upsampled."""
    base = torch.rand(1, 1, cells, cells, generator=g) * 2 - 1
    return F.interpolate(base, size=(h, w), mode="bicubic", align_corners=False)[0, 0].clamp(-1, 1)


def _gauss_kernel(sigma: float) -> torch.Tensor:
    r = max(1, int(math.ceil(sigma * 3)))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2 * sigma * sigma))
    return k / k.sum()


def _sep(x: torch.Tensor, kx: torch.Tensor | None, ky: torch.Tensor | None) -> torch.Tensor:
    c = x.shape[1]
    if kx is not None:
        x = F.conv2d(
            F.pad(x, (kx.numel() // 2, kx.numel() // 2, 0, 0), mode="replicate"), kx.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c
        )
    if ky is not None:
        x = F.conv2d(
            F.pad(x, (0, 0, ky.numel() // 2, ky.numel() // 2), mode="replicate"), ky.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c
        )
    return x


def apply(img: Image.Image, d: dict, render_dpi: float, device: str = "cpu") -> tuple[Image.Image, np.ndarray, int]:
    """Degrade `img` (RGB, rendered at `render_dpi`) as parameters `d` say.

    Returns the final image, the 3x3 homography taking clean-image pixel coordinates (continuous, pixel i spans [i, i+1)) to final
    image coordinates, and the JPEG quality to save with (0 = save as PNG)."""
    g = torch.Generator().manual_seed(d["seed"])
    rng = random.Random(d["seed"])
    w0, h0 = img.size
    f = min(1.0, d["dpi"] / render_dpi)
    w1, h1 = max(8, round(w0 * f)), max(8, round(h0 * f))
    if (w1, h1) != (w0, h0):
        img = img.resize((w1, h1), Image.Resampling.BOX if f < 0.6 else Image.Resampling.LANCZOS)
    S = np.diag([w1 / w0, h1 / h0, 1.0])
    m = d["margin"]
    wc, hc = round(w1 * (1 + 2 * m)), round(h1 * (1 + 2 * m))
    ox, oy = (wc - w1) / 2, (hc - h1) / 2
    corners = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float64)
    dst = corners + [ox, oy]
    if d["rot"]:
        a = math.radians(d["rot"])
        c, s = math.cos(a), math.sin(a)
        ctr = np.array([wc / 2, hc / 2])
        dst = (dst - ctr) @ np.array([[c, -s], [s, c]]).T + ctr
    if d["persp"]:
        jit = np.array([[rng.uniform(-1, 1) * d["persp"] * w1, rng.uniform(-1, 1) * d["persp"] * h1] for _ in range(4)])
        # pull one edge in to get a keystone, as a tilted camera does
        k = rng.uniform(0, d["persp"]) * w1 * rng.choice([-1, 1])
        jit[0, 0] += k
        jit[1, 0] -= k
        dst = dst + jit
        # rotation about the page centre, a few degrees
        a = math.radians(rng.uniform(-4, 4))
        ctr = np.array([wc / 2, hc / 2])
        dst = (dst - ctr) @ np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]).T + ctr
        # keep the page inside the canvas
        lo, hi = dst.min(0), dst.max(0)
        shift = np.array([max(0, 2 - lo[0]) - max(0, hi[0] - (wc - 2)), max(0, 2 - lo[1]) - max(0, hi[1] - (hc - 2))])
        dst = dst + shift
    H1 = _homography(corners, dst)
    H = H1 @ S
    x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    if np.allclose(H1, np.eye(3), atol=1e-9):
        page, alpha = x, torch.ones(1, 1, h1, w1, device=device)
    else:
        Hinv = torch.from_numpy(np.linalg.inv(H1).astype(np.float32)).to(device)
        ys, xs = torch.meshgrid(torch.arange(hc, device=device) + 0.5, torch.arange(wc, device=device) + 0.5, indexing="ij")
        pts = torch.stack([xs, ys, torch.ones_like(xs)], -1) @ Hinv.T
        sx, sy = pts[..., 0] / pts[..., 2], pts[..., 1] / pts[..., 2]
        grid = torch.stack([2 * sx / w1 - 1, 2 * sy / h1 - 1], -1)[None]
        page = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        alpha = F.grid_sample(torch.ones(1, 1, h1, w1, device=device), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    hh, ww = page.shape[-2:]
    if d["faint"] < 1.0:
        page = 1 - (1 - page) * d["faint"]
    bg = torch.tensor(d["bg"], device=device).view(1, 3, 1, 1).expand(1, 3, hh, ww)
    if m > 0:
        bg = bg * (1 + 0.15 * _smooth(g, 1, hh, ww, 6).to(device))[None, None]
    out = page * alpha + bg * (1 - alpha)
    if d["light"] or d["shadow"]:
        fld = torch.ones(hh, ww)
        if d["light"]:
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, hh), torch.linspace(-1, 1, ww), indexing="ij")
            a = rng.uniform(0, 2 * math.pi)
            fld = fld + d["light"] * (0.6 * (math.cos(a) * xx + math.sin(a) * yy) + 0.5 * _smooth(g, 1, hh, ww, 5))
        if d["shadow"]:
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, hh), torch.linspace(-1, 1, ww), indexing="ij")
            a = rng.uniform(0, 2 * math.pi)
            c0 = rng.uniform(-0.6, 0.6)
            soft = rng.uniform(0.05, 0.4)
            fld = fld * (1 - d["shadow"] * torch.sigmoid((math.cos(a) * xx + math.sin(a) * yy - c0) / soft))
        out = out * fld.clamp(0.3, 1.3).to(device)[None, None]
    if d["blur"] > 0.05:
        k = _gauss_kernel(d["blur"]).to(device)
        out = _sep(out, k, k)
    if d["motion"] >= 2:
        k = torch.ones(d["motion"], device=device) / d["motion"]
        out = _sep(out, k if d["motion_axis"] == 0 else None, k if d["motion_axis"] == 1 else None)
    if d["read_noise"] > 0:
        n1 = torch.randn(out.shape, generator=g).to(device)
        n2 = torch.randn(out.shape, generator=g).to(device)
        out = out + d["read_noise"] * n1 + d["shot_noise"] * out.clamp(min=0).sqrt() * n2
    out = out.clamp(0, 1)
    if d["gray"] or d["binarize"]:
        lum = 0.299 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.114 * out[:, 2:3]
        if d["binarize"]:
            local = F.avg_pool2d(F.pad(lum, (15, 15, 15, 15), mode="replicate"), 31, stride=1)
            lum = (lum > local * 0.88).float()
        out = lum.expand(-1, 3, -1, -1)
    arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    res = Image.fromarray(arr)
    if d["gray"] or d["binarize"]:
        res = res.convert("L")
    return res, H, d["jpeg"]


def encode(img: Image.Image, jpeg_q: int) -> tuple[bytes, str]:
    buf = io.BytesIO()
    if jpeg_q:
        img.convert("L" if img.mode == "L" else "RGB").save(buf, "JPEG", quality=jpeg_q)
        return buf.getvalue(), "jpg"
    img.save(buf, "PNG", compress_level=3)
    return buf.getvalue(), "png"


def transform_box(H: np.ndarray, box) -> tuple[list[float], list[float] | None]:
    """Axis-aligned bounds and the quad (TL, TR, BR, BL; 8 numbers) of a box under homography H. The quad is None for a pure scale."""
    l, t, r, b = box
    pts = np.array([[l, t, 1], [r, t, 1], [r, b, 1], [l, b, 1]], dtype=np.float64) @ H.T
    xy = pts[:, :2] / pts[:, 2:3]
    bounds = [float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())]
    plain = abs(H[0, 1]) < 1e-12 and abs(H[1, 0]) < 1e-12 and abs(H[2, 0]) < 1e-12 and abs(H[2, 1]) < 1e-12
    return bounds, None if plain else [float(v) for v in xy.reshape(-1)]
