"""Seeded scan and camera simulation on a rendered page, with truth boxes moved along with the pixels.

Geometry is always one homography per page: the same 3x3 matrix warps the image (PIL perspective transform, which
takes the inverse map) and moves every word's four corners. A word's `quad` is those four corners (top-left,
top-right, bottom-right, bottom-left of the original box); `box` is the axis-aligned box around the quad.
All random draws come from the page's own generators, so a page is reproducible from (set, seed, index) alone.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageFilter


def homography(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> np.ndarray:
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    h = np.linalg.solve(np.array(a, float), np.array(b, float))
    return np.append(h, 1.0).reshape(3, 3)


def apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:3]


def warp(img: Image.Image, h: np.ndarray, size: tuple[int, int], fill) -> Image.Image:
    inv = np.linalg.inv(h)
    inv /= inv[2, 2]
    return img.transform(size, Image.Transform.PERSPECTIVE, tuple(inv.flatten()[:8]), Image.Resampling.BICUBIC, fillcolor=fill)


def move_words(words: list[dict], h: np.ndarray, scale: float = 1.0) -> list[dict]:
    """Transform each truth word's box corners with `h` then `scale`; adds `quad`, rewrites `box`.

    The ink box (`ink`, when present) moves the same way: `ink_quad` is its transformed corners, `ink` their box.
    """

    def move(box):
        x0, y0, x1, y1 = box
        q = apply_h(h, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)) * scale
        flat = [round(float(v), 1) for v in q.flatten()]
        aabb = [round(float(q[:, 0].min()), 1), round(float(q[:, 1].min()), 1),
                round(float(q[:, 0].max()), 1), round(float(q[:, 1].max()), 1)]
        return flat, aabb

    out = []
    for w in words:
        d = dict(w)
        d["quad"], d["box"] = move(w["box"])
        if "ink" in w:
            d["ink_quad"], d["ink"] = move(w["ink"])
        out.append(d)
    return out


def jpeg(img: Image.Image, quality: int, dpi: int | None = None) -> bytes:
    buf = io.BytesIO()
    opts = {"dpi": (dpi, dpi)} if dpi else {}
    img.save(buf, "JPEG", quality=quality, **opts)
    return buf.getvalue()


def degrade_scan(img: Image.Image, words: list[dict], rng, nrng: np.random.Generator) -> tuple[bytes, list[dict], dict]:
    """Flatbed-scan degradation of a 300 DPI grayscale page: skew, faint ink, low DPI, blur, noise, JPEG."""
    w, h = img.size
    skew = round(rng.uniform(-2.0, 2.0), 2)
    dpi = rng.choice([150, 200])
    faint = round(rng.uniform(0.3, 0.9), 2) if rng.random() < 0.5 else 1.0
    paper = rng.randint(238, 255)
    blur = round(rng.uniform(0.0, 0.8), 2)
    sigma = rng.choice([0.0, 2.0, 4.0, 7.0, 11.0])
    speckle = rng.choice([0.0, 0.0, 2e-4, 6e-4])
    quality = rng.randint(30, 85)

    c = np.array([w / 2, h / 2])
    t = math.radians(skew)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    moved = (corners - c) @ rot.T + c
    hm = homography([tuple(p) for p in corners], [tuple(p) for p in moved])
    page = warp(img, hm, (w, h), 255)

    a = np.asarray(page, np.float32)
    a = 255.0 - (255.0 - a) * faint  # lighter ink
    a = a * (paper / 255.0)  # greyish paper
    s = dpi / 300.0
    small = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize(
        (round(w * s), round(h * s)), Image.Resampling.BOX
    )
    if blur > 0:
        small = small.filter(ImageFilter.GaussianBlur(blur))
    b = np.asarray(small, np.float32)
    if sigma > 0:
        b = b + nrng.normal(0, sigma, b.shape).astype(np.float32)
    if speckle > 0:
        m = nrng.random(b.shape)
        b = np.where(m < speckle, 0.0, b)
        b = np.where(m > 1 - speckle, 255.0, b)
    out = Image.fromarray(np.clip(np.rint(b), 0, 255).astype(np.uint8))
    params = {
        "skew_deg": skew, "dpi": dpi, "ink_factor": faint, "paper_level": paper, "blur_px": blur,
        "noise_sigma": sigma, "speckle": speckle, "jpeg_quality": quality,
    }
    return jpeg(out, quality, dpi), move_words(words, hm, s), params


def _smooth_field(nrng: np.random.Generator, h: int, w: int, cells: int = 4) -> np.ndarray:
    g = nrng.random((cells, cells)).astype(np.float32)
    return np.asarray(Image.fromarray(g).resize((w, h), Image.Resampling.BICUBIC), np.float32)


def phone_capture(
    img: Image.Image, words: list[dict], rng, nrng: np.random.Generator, canvas: tuple[int, int] = (2016, 2688)
) -> tuple[bytes, list[dict], dict]:
    """Phone photo of a flat page: placement and perspective, desk, uneven light, blur, sensor noise, JPEG."""
    cw, ch = canvas
    pw0, ph0 = img.size
    frac = rng.uniform(0.62, 0.82)
    s = frac * cw / pw0
    pw, ph = pw0 * s, ph0 * s
    rot = math.radians(rng.uniform(-5, 5))
    jit = rng.uniform(0.01, 0.045)
    cx = cw / 2 + rng.uniform(-0.04, 0.04) * cw
    cy = ch / 2 + rng.uniform(-0.03, 0.03) * ch
    base = np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]])
    r = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    pts = base @ r.T
    pts = pts + np.array([[rng.uniform(-jit, jit) * pw, rng.uniform(-jit, jit) * ph] for _ in range(4)]) + [cx, cy]
    if pts.min() < 8 or pts[:, 0].max() > cw - 8 or pts[:, 1].max() > ch - 8:
        raise ValueError("page leaves the canvas")
    hm = homography([(0, 0), (pw0, 0), (pw0, ph0), (0, ph0)], [tuple(p) for p in pts])

    tint = np.array([rng.uniform(244, 255), rng.uniform(240, 255), rng.uniform(225, 250)], np.float32) / 255.0
    rgb = np.asarray(img.convert("RGB"), np.float32) * tint
    page = warp(Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)), hm, (cw, ch), (0, 0, 0))
    mask = warp(Image.new("L", img.size, 255), hm, (cw, ch), 0)

    desk = np.array(rng.choice([(96, 70, 48), (60, 62, 66), (150, 135, 110), (35, 45, 40), (120, 120, 125)]), np.float32)
    bg = desk[None, None, :] * (0.75 + 0.5 * _smooth_field(nrng, ch, cw, 6))[:, :, None]
    m = np.asarray(mask, np.float32)[:, :, None] / 255.0
    a = np.asarray(page, np.float32) * m + bg * (1 - m)

    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    x0, y0 = rng.uniform(0.25, 0.75) * cw, rng.uniform(0.2, 0.7) * ch
    vign = 1.0 - rng.uniform(0.1, 0.35) * (((xx - x0) ** 2 + (yy - y0) ** 2) / (cw**2 + ch**2) * 2.0)
    gx, gy = rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25)
    plane = 1.0 + gx * (xx / cw - 0.5) + gy * (yy / ch - 0.5)
    shadow_w = 0.0
    if rng.random() < 0.5:  # soft shadow band, e.g. from a hand or a lamp edge
        ang = rng.uniform(0, math.pi)
        d = (xx - cw / 2) * math.cos(ang) + (yy - ch / 2) * math.sin(ang)
        edge = rng.uniform(-0.3, 0.3) * cw
        shadow_w = round(rng.uniform(0.12, 0.3), 2)
        plane = plane * (1.0 - shadow_w / (1.0 + np.exp(-(d - edge) / (0.05 * cw))))
    a = a * (vign * plane)[:, :, None] * rng.uniform(0.9, 1.1)

    blur = round(rng.uniform(0.6, 2.2), 2)
    img2 = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
    b = np.asarray(img2, np.float32)
    sigma = round(rng.uniform(2.0, 7.0), 1)
    b = b + nrng.normal(0, sigma * 0.7, b.shape[:2]).astype(np.float32)[:, :, None]
    b = b + nrng.normal(0, sigma * 0.5, b.shape).astype(np.float32)
    out = Image.fromarray(np.clip(np.rint(b), 0, 255).astype(np.uint8))
    quality = rng.randint(55, 88)
    params = {
        "page_fraction": round(frac, 3), "rotation_deg": round(math.degrees(rot), 2), "corner_jitter": round(jit, 4),
        "blur_px": blur, "noise_sigma": sigma, "shadow": shadow_w, "vignette_centre": [round(x0), round(y0)],
        "jpeg_quality": quality, "canvas": [cw, ch], "dst_quad": [round(float(v), 1) for v in pts.flatten()],
    }
    return jpeg(out, quality), move_words(words, hm), params
