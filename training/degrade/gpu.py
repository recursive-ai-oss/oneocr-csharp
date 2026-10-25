"""Batched GPU degradation and camera simulation for line crops.

Input: clean lines as rendered by `LineGen --clean` (black text on white, 48 px high, 255 = paper), padded to a common width.
Output: the 8-bit, 32 px high crops the recognizers read, plus every sampled parameter, so evaluation can slice by them.

Order, mirroring what a camera/scanner/screen path does to a page and what the library then does to a crop:
  1. low resolution: the crop is resampled to the pixel height the type would have in the source image (em 12 px is a crop
     about 22 px high; everything below is then applied at that resolution, where blur and noise have their true scale)
  2. perspective: a random homography (corner jitter) - keystone, tilt, shear
  3. composition: text coverage x ink colour over a paper colour (grey, tinted, or light-on-dark), optional sub-pixel
     (LCD) anti-aliasing as a per-channel horizontal shift of the coverage
  4. uneven light: smooth multiplicative field (gradient + low-frequency noise)
  5. blur: Gaussian (defocus) and, sometimes, horizontal motion blur
  6. sensor noise: read noise + signal-dependent shot noise
  7. luma, then a JPEG round trip (8x8 DCT, standard luminance quantisation table scaled by quality)
  8. 8-bit rounding and the scale to 32 px height (bicubic with antialiasing), which is what the library does to a crop

Everything is seeded: the sampling uses one torch.Generator; the same seed and batches give the same pixels on the same
device. JPEG follows the baseline luminance path of ITU-T T.81 without chroma subsampling or entropy coding (lossless).
Ranges and probabilities come from the line-set config (`gpu_degrade`); training and held-out ranges are disjoint.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

IN_H = 48
OUT_H = 32
# pixel heights of the intermediate "source" resolution; a sample is put on the nearest level above its simulated crop height
LEVELS = (16, 20, 24, 28, 32, 36, 40, 44, 48)
# a crop (ink box plus margins) is about this many ems high; used to turn a font size into a source resolution
CROP_PER_EM = 1.8

_QLUM = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100,
    103, 99,
]  # fmt: skip


def out_width(w: int) -> int:
    """Width of the 32 px crop made from a clean line w pixels wide (aspect kept, rounded half up, at least 1)."""
    return max(1, int(w * OUT_H / IN_H + 0.5))


def _uniform(ranges, n: int, g: torch.Generator, dev) -> torch.Tensor:
    """Draw from a list of [lo, hi] ranges: pick an entry uniformly (repeats weight it), then uniform inside it."""
    r = ranges if isinstance(ranges[0], (list, tuple)) else [ranges]
    t = torch.tensor(r, dtype=torch.float32, device=dev)
    i = torch.randint(0, len(r), (n,), generator=g, device=dev)
    return t[i, 0] + (t[i, 1] - t[i, 0]) * torch.rand(n, generator=g, device=dev)


def _binned(spec: dict, n: int, g: torch.Generator, dev) -> torch.Tensor:
    """Draw from bins [edges[i], edges[i+1]) chosen with probabilities spec["probs"], uniform inside a bin (the targeted-error loop sets
    these from the model's measured errors; see train/adaptive.py)."""
    e = torch.tensor(spec["edges"], dtype=torch.float32, device=dev)
    pr = torch.tensor(spec["probs"], dtype=torch.float32, device=dev)
    i = torch.multinomial(pr, n, replacement=True, generator=g)
    return e[i] + (e[i + 1] - e[i]) * torch.rand(n, generator=g, device=dev)


def sample_params(side: dict, n: int, g: torch.Generator, dev) -> dict[str, torch.Tensor]:
    """Per-sample parameters from one side ('train' or 'heldout') of the config's gpu_degrade section."""
    R, P = side["ranges"], side["p"]

    def flag(name: str) -> torch.Tensor:
        return torch.rand(n, generator=g, device=dev) < P[name]

    bins = side.get("_bins", {})

    def draw(name: str) -> torch.Tensor:
        return _binned(bins[name], n, g, dev) if name in bins else _uniform(R[name], n, g, dev)

    em = draw("em")
    src_h = em * CROP_PER_EM
    wh = torch.tensor(LEVELS, device=dev)[(src_h.unsqueeze(1) <= torch.tensor(LEVELS, device=dev).unsqueeze(0)).int().argmax(1)]
    wh = torch.where(src_h > LEVELS[-1], torch.full_like(wh, LEVELS[-1]), wh)
    delta = draw("contrast")
    paper = delta + (255 - delta) * torch.rand(n, generator=g, device=dev)
    paper = torch.maximum(paper, torch.full_like(paper, 140.0)).clamp(max=255.0)
    ink = (paper - delta).clamp(min=0.0)
    dark = flag("dark")
    tint = flag("tint")
    chroma = (torch.rand(n, 2, 3, generator=g, device=dev) - 0.5) * 2 * 45 * tint.view(-1, 1, 1)
    bg = (torch.where(dark, ink, paper).view(-1, 1) + chroma[:, 0]).clamp(0, 255)
    fg = (torch.where(dark, paper, ink).view(-1, 1) + chroma[:, 1]).clamp(0, 255)
    q = _uniform(R["jpeg_q"], n, g, dev).round().clamp(1, 100)
    extra = {}
    if "resample" in P:  # the reader's own resampling: a second bilinear pass at a random phase and a residual tilt (0-2 passes)
        tilt = R.get("tilt_deg", [0.0, 0.3])
        extra["resample"] = (torch.rand(n, generator=g, device=dev) < P["resample"]).int() + (
            torch.rand(n, generator=g, device=dev) < P.get("resample2", 0.0)).int()  # fmt: skip
        extra["rs_angle"] = (torch.rand(n, 2, generator=g, device=dev) * 2 - 1) * _uniform(tilt, n, g, dev).view(-1, 1) * math.pi / 180
        extra["rs_shift"] = torch.rand(n, 2, 2, generator=g, device=dev)
    if "thermal" in P:  # thermal print: ink dropout and row-wise gain streaks
        on = flag("thermal")
        extra["dropout"] = _uniform(R["dropout"], n, g, dev) * on
        extra["streak"] = _uniform(R["streak"], n, g, dev) * on
    return extra | {
        "em": em,
        "src_h": wh.float(),
        "blur_rel": draw("blur_rel") * (torch.rand(n, generator=g, device=dev) < P["blur"]),
        "motion": _uniform(R["motion"], n, g, dev) * flag("motion"),
        "noise": draw("noise"),
        "shot": _uniform(R["shot"], n, g, dev),
        "jpeg_q": torch.where(flag("jpeg"), q, torch.zeros_like(q)),
        "contrast": delta,
        "persp": _uniform(R["persp"], n, g, dev) * flag("persp"),
        "persp_offsets": torch.rand(n, 4, 2, generator=g, device=dev) * 2 - 1,
        "light": _uniform(R["light"], n, g, dev) * flag("light"),
        "light_dir": torch.rand(n, 2, generator=g, device=dev) * 2 - 1,
        "subpixel": flag("subpixel"),
        "dark": dark,
        "tint": tint,
        "bg": bg,
        "fg": fg,
    }


def _dct() -> torch.Tensor:
    k = torch.arange(8).view(-1, 1).double()
    x = torch.arange(8).view(1, -1).double()
    d = torch.cos((2 * x + 1) * k * math.pi / 16) * math.sqrt(2 / 8)
    d[0] /= math.sqrt(2)
    return d.float()


def jpeg_roundtrip(y: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """y: [n, 1, h, w] floats in 0..255; q: [n] qualities 1..100. Baseline luminance DCT quantisation, then back."""
    n, _, h, w = y.shape
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    yp = F.pad(y, (0, pw, 0, ph), mode="replicate") - 128.0
    blocks = yp.unfold(2, 8, 8).unfold(3, 8, 8)  # [n, 1, H8, W8, 8, 8]
    d = _dct().to(y.device)
    coef = d @ blocks @ d.T
    scale = torch.where(q < 50, 5000.0 / q, 200.0 - 2.0 * q) / 100.0
    qt = torch.tensor(_QLUM, dtype=torch.float32, device=y.device).view(1, 1, 1, 1, 8, 8)
    qt = ((qt * scale.view(-1, 1, 1, 1, 1, 1) * 100.0 + 50.0) / 100.0).floor().clamp(1, 255)
    rec = d.T @ (torch.round(coef / qt) * qt) @ d
    out = rec.permute(0, 1, 2, 4, 3, 5).reshape(n, 1, yp.shape[2], yp.shape[3])[:, :, :h, :w] + 128.0
    return out.round().clamp(0, 255)


def _homography(offsets: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Homographies [n, 3, 3] taking output pixel positions to source positions; offsets [n, 4, 2] are the source corners'
    displacements in units of the image height (output corners: TL, TR, BR, BL)."""
    dst = torch.tensor([[0, 0], [w, 0], [w, h], [0, h]], dtype=torch.float64, device=offsets.device)
    src = dst.unsqueeze(0) + offsets.double() * h
    n = offsets.shape[0]
    a = torch.zeros(n, 8, 8, dtype=torch.float64, device=offsets.device)
    b = torch.zeros(n, 8, dtype=torch.float64, device=offsets.device)
    for i in range(4):
        x, y = dst[i]
        u, v = src[:, i, 0], src[:, i, 1]
        a[:, 2 * i, 0], a[:, 2 * i, 1], a[:, 2 * i, 2] = x, y, 1
        a[:, 2 * i, 6], a[:, 2 * i, 7] = -u * x, -u * y
        a[:, 2 * i + 1, 3], a[:, 2 * i + 1, 4], a[:, 2 * i + 1, 5] = x, y, 1
        a[:, 2 * i + 1, 6], a[:, 2 * i + 1, 7] = -v * x, -v * y
        b[:, 2 * i], b[:, 2 * i + 1] = u, v
    sol = torch.linalg.solve(a, b)
    return torch.cat([sol, torch.ones(n, 1, dtype=torch.float64, device=offsets.device)], 1).view(n, 3, 3).float()


def _warp(cov: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    n, _, h, w = cov.shape
    hm = _homography(offsets, h, w)
    ys, xs = torch.meshgrid(torch.arange(h, device=cov.device) + 0.5, torch.arange(w, device=cov.device) + 0.5, indexing="ij")
    pts = torch.stack([xs, ys, torch.ones_like(xs)], -1).view(1, h * w, 3)
    p = pts @ hm.transpose(1, 2)
    p = p[..., :2] / p[..., 2:3]
    grid = torch.stack([p[..., 0] / w * 2 - 1, p[..., 1] / h * 2 - 1], -1).view(n, h, w, 2)
    return F.grid_sample(cov, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _blur_kernels(sigma: torch.Tensor, radius: int) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, device=sigma.device).float().view(1, -1)
    k = torch.exp(-0.5 * (x / sigma.clamp(min=1e-3).view(-1, 1)) ** 2)
    return k / k.sum(1, keepdim=True)


def _box_kernels(length: torch.Tensor, radius: int) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, device=length.device).float().view(1, -1).abs()
    k = (length.view(-1, 1) / 2 - x + 0.5).clamp(0, 1)
    k[:, radius] = k[:, radius].clamp(min=1e-3)
    return k / k.sum(1, keepdim=True)


def _sep_conv(img: torch.Tensor, kx: torch.Tensor | None, ky: torch.Tensor | None) -> torch.Tensor:
    """Per-sample separable convolution of [n, c, h, w] with kernels [n, K] (replicate padding)."""
    n, c, h, w = img.shape
    x = img.reshape(1, n * c, h, w)
    for k, horizontal in ((kx, True), (ky, False)):
        if k is None:
            continue
        r = k.shape[1] // 2
        kk = k.repeat_interleave(c, 0)
        if horizontal:
            x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kk.view(n * c, 1, 1, -1), groups=n * c)
        else:
            x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), kk.view(n * c, 1, -1, 1), groups=n * c)
    return x.reshape(n, c, h, w)


_SUBPIXEL = [[0.0, 2 / 3, 1 / 3], [0.0, 1.0, 0.0], [1 / 3, 2 / 3, 0.0]]  # R, G, B: the coverage sampled a third of a pixel apart


def _degrade_group(cov: torch.Tensor, p: dict, g: torch.Generator, wh: int) -> torch.Tensor:
    """cov: [n, 1, 48, Wp] text coverage 0..1 of samples that share the intermediate height wh. Returns luma 0..255, [n, 1, wh, w]."""
    dev = cov.device
    n = cov.shape[0]
    if wh != IN_H:
        cov = F.interpolate(cov, size=(wh, max(8, round(cov.shape[3] * wh / IN_H))), mode="bilinear", antialias=True, align_corners=False)
    h, w = cov.shape[2:]
    # corner jitter: vertical moves are full strength, horizontal ones a third of it (a line is far wider than it is high)
    cov = _warp(cov, p["persp_offsets"] * p["persp"].view(-1, 1, 1) * torch.tensor([0.35, 1.0], device=dev))
    if "dropout" in p:
        keep = (torch.rand(cov.shape, generator=g, device=dev) >= p["dropout"].view(-1, 1, 1, 1)).float()
        cov = cov * keep
    cov3 = cov.expand(-1, 3, -1, -1)
    sp = F.conv2d(F.pad(cov, (1, 1, 0, 0), mode="replicate"), torch.tensor(_SUBPIXEL, device=dev).view(3, 1, 1, 3))
    cov3 = torch.where(p["subpixel"].view(-1, 1, 1, 1), sp, cov3)
    bg, fg = p["bg"].view(n, 3, 1, 1) / 255.0, p["fg"].view(n, 3, 1, 1) / 255.0
    img = bg * (1 - cov3) + fg * cov3

    # uneven light: a tilted plane plus a coarse random field, scaled by the sample's strength
    ys = torch.linspace(-1, 1, h, device=dev).view(1, 1, h, 1)
    xs = torch.linspace(-1, 1, w, device=dev).view(1, 1, 1, w)
    d = p["light_dir"].view(n, 2, 1, 1)
    coarse = F.interpolate(torch.randn(n, 1, 3, 6, generator=g, device=dev), size=(h, w), mode="bilinear", align_corners=False)
    field = 1 + p["light"].view(n, 1, 1, 1) * (0.6 * (d[:, 0:1] * xs + d[:, 1:2] * ys) + 0.4 * coarse)
    field = field.clamp(min=0.55)
    img = img * field / field.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)  # shading darkens; the brightest spot stays at paper white

    if "streak" in p:
        gain = 1 + p["streak"].view(n, 1, 1, 1) * F.avg_pool2d(
            torch.randn(n, 1, h, 1, generator=g, device=dev), (3, 1), 1, (1, 0), count_include_pad=False
        )
        img = img * gain

    # blur: Gaussian sigma follows the working height, so it is the same fraction of the crop whatever the resolution
    sigma = p["blur_rel"] * wh / CROP_PER_EM
    if bool((sigma > 0.05).any()):
        r = max(1, min(14, math.ceil(3 * float(sigma.max()))))
        k = _blur_kernels(sigma, r)
        img = _sep_conv(img, k, k)
    if bool((p["motion"] > 0.5).any()):
        r = max(1, min(8, math.ceil(float(p["motion"].max()) / 2) + 1))
        img = _sep_conv(img, _box_kernels(p["motion"], r), None)

    # noise: read noise in grey levels, shot noise proportional to the square root of the signal
    img = img + p["noise"].view(n, 1, 1, 1) / 255.0 * torch.randn(img.shape, generator=g, device=dev)
    img = img + p["shot"].view(n, 1, 1, 1) * img.clamp(0, 1).sqrt() * torch.randn(img.shape, generator=g, device=dev)
    luma = (0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]).clamp(0, 1) * 255.0
    on = p["jpeg_q"] > 0
    if bool(on.any()):
        luma = luma.clone()
        luma[on] = jpeg_roundtrip(luma[on].round(), p["jpeg_q"][on])
    if "resample" in p:
        for k in range(2):
            m = p["resample"] > k
            if bool(m.any()):
                luma = luma.clone()
                luma[m] = _resample(luma[m], p["rs_angle"][m, k], p["rs_shift"][m, k])
    return luma


def _resample(img: torch.Tensor, angle: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Bilinear resampling of img [n, 1, h, w] about its centre by `angle` radians and a sub-pixel `shift` [n, 2] (x, y) in pixels, edge
    pixels repeated: what reading a line along its own axis does to the page's pixels (docs/spec/detector-v1.md, Sampling)."""
    n, _, h, w = img.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=img.device) + 0.5, torch.arange(w, device=img.device) + 0.5, indexing="ij")
    dx, dy = xs.view(1, h, w) - w / 2, ys.view(1, h, w) - h / 2
    c, s = torch.cos(angle).view(n, 1, 1), torch.sin(angle).view(n, 1, 1)
    xi = c * dx + s * dy + w / 2 + shift[:, 0].view(n, 1, 1)
    yi = -s * dx + c * dy + h / 2 + shift[:, 1].view(n, 1, 1)
    grid = torch.stack([2 * xi / w - 1, 2 * yi / h - 1], -1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _extent(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """First and last True index along the last axis of mask [n, L], and whether any is True."""
    ln = mask.shape[1]
    has = mask.any(1)
    first = mask.float().argmax(1)
    last = ln - 1 - mask.flip(1).float().argmax(1)
    return first, last, has


def _shift_gather(cov: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor, wd: torch.Tensor, wrap: bool) -> torch.Tensor:
    """Copy of cov [n, 1, h, w] moved by (dy, dx) pixels per sample (zeros where nothing moves in); wrap=True rolls the columns
    inside each sample's own width instead."""
    n, _, h, w = cov.shape
    dev = cov.device
    rows = torch.arange(h, device=dev).view(1, h, 1).float() - dy.view(n, 1, 1)
    cols = torch.arange(w, device=dev).view(1, 1, w).float() - dx.view(n, 1, 1)
    if wrap:
        cols = torch.remainder(cols, wd.view(n, 1, 1).float())
    ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    r = rows.round().long().clamp(0, h - 1).expand(n, h, w)
    c = cols.round().long().clamp(0, w - 1).expand(n, h, w)
    flat = cov[:, 0].reshape(n, h * w).gather(1, (r * w + c).reshape(n, h * w)).view(n, 1, h, w)
    return flat * ok.unsqueeze(1)


def add_distractors(cov: torch.Tensor, widths: torch.Tensor, P: dict, g: torch.Generator) -> torch.Tensor:
    """Ink that real line crops carry besides the line itself, drawn into the coverage before the camera simulation:
      sliver_v  per side: a bit of the neighbouring line, a shifted copy of this line showing in the top or bottom margin
      sliver_h  per side: the end of a neighbouring word or cell, a shifted copy showing in the left or right margin
      rule      one of: a vertical rule near one end (cell or form-field border), a horizontal rule above or below the text,
                an underline, or a box (both vertical rules)
    Every distractor stays inside the margin (clear of the line's own ink) so the label stays exact. cov: [n, 1, 48, Wp]."""
    n, _, h, w = cov.shape
    dev = cov.device
    wd = widths.to(dev).clamp(max=w)
    rowmax, colmax = cov.amax(dim=(1, 3)), cov.amax(dim=(1, 2))
    top, bot, has_r = _extent(rowmax > 0.25)
    left, right, has_c = _extent(colmax > 0.25)
    has = has_r & has_c
    rnd = lambda: torch.rand(n, generator=g, device=dev)  # noqa: E731
    out = cov
    zero = torch.zeros(n, device=dev)
    # neighbouring line above / below
    for side in (-1, 1):
        room = (top - 2).float() if side < 0 else (h - 1 - bot - 2).float()
        on = (rnd() < P.get("sliver_v", 0.0)) & has & (room >= 3)
        vis = (0.3 + 0.7 * rnd()) * room.clamp(min=0)
        dy = vis - bot.float() if side < 0 else (h - vis) - top.float()
        dx = (rnd() - 0.5) * wd.float()
        sl = _shift_gather(cov, torch.where(on, dy, zero), torch.where(on, dx, zero), wd, True) * (0.7 + 0.3 * rnd()).view(n, 1, 1, 1)
        out = torch.maximum(out, sl * on.view(n, 1, 1, 1))
    # neighbouring word or cell to the left / right
    for side in (-1, 1):
        room = (left - 2).float() if side < 0 else (wd - 1 - right - 2).float()
        on = (rnd() < P.get("sliver_h", 0.0)) & has & (room >= 3)
        vis = (0.3 + 0.7 * rnd()) * room.clamp(min=0)
        dx = vis - right.float() if side < 0 else (wd.float() - vis) - left.float()
        sl = _shift_gather(cov, zero, torch.where(on, dx, zero), wd, False) * (0.7 + 0.3 * rnd()).view(n, 1, 1, 1)
        out = torch.maximum(out, sl * on.view(n, 1, 1, 1))
    # rules
    on = (rnd() < P.get("rule", 0.0)) & has
    kind = (rnd() * 4).long().clamp(max=3)  # 0 vertical at one end, 1 horizontal above, 2 underline / below, 3 box
    thick = (1.0 + 2.0 * rnd()) * h / 48.0
    inten = (0.5 + 0.5 * rnd()).view(n, 1, 1, 1)
    rows = torch.arange(h, device=dev).view(1, 1, h, 1).float()
    cols = torch.arange(w, device=dev).view(1, 1, 1, w).float()
    t4 = thick.view(n, 1, 1, 1)
    line = lambda d: (t4 / 2 + 0.5 - d.abs()).clamp(0, 1)  # noqa: E731
    pos_l = (left.float() * (0.3 + 0.5 * rnd())).view(n, 1, 1, 1)
    pos_r = (right.float() + (wd - 1 - right).float() * (0.3 + 0.5 * rnd())).view(n, 1, 1, 1)
    pos_t = (top.float() * (0.3 + 0.5 * rnd())).view(n, 1, 1, 1)
    pos_b = (bot.float() + (h - 1 - bot).float() * (0.3 + 0.5 * rnd())).view(n, 1, 1, 1)
    inside = (cols < wd.view(n, 1, 1, 1)).float()
    ends_left = (rnd() < 0.5).view(n, 1, 1, 1)
    vert_one = line(cols - torch.where(ends_left, pos_l, pos_r))
    vert_both = torch.maximum(line(cols - pos_l), line(cols - pos_r))
    vert = torch.where((kind == 3).view(n, 1, 1, 1), vert_both, vert_one) * torch.ones_like(rows)
    hor = torch.where((kind == 1).view(n, 1, 1, 1), line(rows - pos_t), line(rows - pos_b)) * inside
    rule = torch.where((kind <= 0).view(n, 1, 1, 1) | (kind == 3).view(n, 1, 1, 1), vert, hor)
    # a rule is only drawn where the margin is wide enough to hold it clear of the text
    room_ok = torch.where(kind == 1, top >= 4, torch.where(kind == 2, h - 1 - bot >= 4, (left >= 4) & (wd - 1 - right >= 4)))
    return torch.maximum(out, rule * inten * (on & room_ok).view(n, 1, 1, 1))


@torch.no_grad()
def degrade(clean: torch.Tensor, widths: torch.Tensor, side: dict, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """clean: uint8 [B, 1, 48, Wp] on the device (255 = paper, padded anywhere right of each width); widths: [B] clean widths.

    Returns (uint8 [B, 1, 32, W32] padded by repeating each crop's last column, widths [B] of the 32 px crops, parameters).
    """
    dev = clean.device
    b, _, _, wp = clean.shape
    params = sample_params(side, b, g, dev)
    cov = 1.0 - clean.float() / 255.0
    if any(k in side["p"] for k in ("rule", "sliver_v", "sliver_h")):  # off in the configs of the earlier line sets
        cov = add_distractors(cov, widths, side["p"], g)
    w32p = out_width(wp)
    out = torch.empty(b, 1, OUT_H, w32p, dtype=torch.uint8, device=dev)
    wh_all = params["src_h"].long()
    for wh in sorted(set(wh_all.tolist())):
        idx = (wh_all == wh).nonzero().squeeze(1)
        sub = {k: v[idx] for k, v in params.items()}
        luma = _degrade_group(cov[idx], sub, g, wh)
        up = F.interpolate(luma, size=(OUT_H, w32p), mode="bicubic", antialias=True, align_corners=False)
        out[idx] = up.round().clamp(0, 255).to(torch.uint8)
    w32 = torch.tensor([out_width(int(w)) for w in widths], device=dev).clamp(max=w32p)
    col = torch.arange(w32p, device=dev).view(1, -1).minimum((w32 - 1).view(-1, 1))
    out = out.gather(3, col.view(b, 1, 1, w32p).expand(-1, 1, OUT_H, -1))
    return out, w32.cpu(), params


def param_records(params: dict[str, torch.Tensor], ids: list[str]) -> list[dict]:
    """JSON-ready per-sample records of the scalar parameters (corner offsets and light direction included)."""
    cpu = {k: v.detach().cpu() for k, v in params.items()}
    rows = []
    for i, sid in enumerate(ids):
        r: dict = {"id": sid}
        for k, v in cpu.items():
            x = v[i]
            r[k] = (
                (round(float(x), 4) if x.dtype.is_floating_point else (bool(x) if x.dtype == torch.bool else int(x)))
                if x.dim() == 0
                else [round(float(t), 3) for t in x.flatten()]
            )
        rows.append(r)
    return rows
