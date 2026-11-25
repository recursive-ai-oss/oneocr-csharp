"""Capture recipes of the financial sets: flatbed scan with photocopy wear, G3 fax, phone photo.

Like `degrade.py`, every recipe is one homography (skew) plus per-pixel effects drawn from the page's own generators, and the
truth boxes move with the pixels: `degrade.move_words` takes the full 3x3 matrix, so the anisotropic scaling of a fax
(200 x 100 dots per inch) is part of the matrix. Parameters are returned for the truth file so a report can slice by them.

Scan: skew 0.5 to 8 degrees, 150/200/300 DPI, photocopy wear (generation loss, toner speckle, streaks, edge shadow,
contrast loss) on most pages, JPEG. Fax: 1728 dots across the page; standard mode (200 x 100) or fine mode (200 x 200), a global
threshold that wanders across the page, scan-line dropouts, black line noise, jittered lines, written as a CCITT Group 3
(T.4) TIFF the way a fax arrives. Phone: the camera simulation of `degrade.phone_capture`.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageFilter

from .degrade import _smooth_field, jpeg, move_words, phone_capture, warp

FAX_WIDTH = 1728


def _rotation_fit(words: list[dict], size: tuple[int, int], deg: float, margin: float = 0.008) -> np.ndarray:
    """Rotation about the page centre, shrunk just enough that every word stays on the sheet (a scanner's page never clips text)."""
    w, h = size
    t = math.radians(deg)
    c = np.array([w / 2, h / 2])
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    pts = np.array([p for k in words for p in ((k["box"][0], k["box"][1]), (k["box"][2], k["box"][1]), (k["box"][2], k["box"][3]), (k["box"][0], k["box"][3]))], float)
    moved = (pts - c) @ rot.T
    s = min(1.0, (w / 2 * (1 - margin)) / max(1e-9, np.abs(moved[:, 0]).max()), (h / 2 * (1 - margin)) / max(1e-9, np.abs(moved[:, 1]).max()))
    a = np.eye(3)
    a[:2, :2] = rot * s
    a[:2, 2] = c - (rot * s) @ c
    return a


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def photocopy(a: np.ndarray, rng, nrng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Wear of a photocopy of a photocopy, on a 300 DPI float page (0 ink .. 255 paper)."""
    h, w = a.shape
    gens = rng.choice([1, 1, 2, 3])
    for _ in range(gens):  # generation loss: soften, then push to contrast again, which thickens ink and breaks hairlines
        blur = rng.uniform(0.6, 1.5)
        a = np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur)), np.float32)
        thr, k = rng.uniform(120, 175), rng.uniform(22, 45)
        a = 255.0 * _sigmoid((a - thr) / k)
    ink_floor = rng.uniform(10, 70)  # ink never goes fully black
    paper = rng.uniform(222, 250)
    a = ink_floor + (a / 255.0) * (paper - ink_floor)
    density = rng.choice([0.0, 5e-5, 1.5e-4, 4e-4])
    if density:
        m = nrng.random(((h + 1) // 2, (w + 1) // 2)) < density
        m = np.kron(m, np.ones((2, 2), bool))[:h, :w]
        a = np.where(m, np.minimum(a, nrng.uniform(0, 90, a.shape)), a)
    streaks = []
    for _ in range(rng.choice([0, 0, 1, 2, 3])):
        x, wd = rng.randrange(w), rng.randint(1, 5)
        y0 = rng.randrange(h // 2) if rng.random() < 0.5 else 0
        y1 = h if rng.random() < 0.5 else rng.randrange(y0 + h // 4, h + 1)
        alpha = rng.uniform(0.08, 0.45)
        tone = rng.choice([0.0, 255.0])
        a[y0:y1, x:x + wd] = a[y0:y1, x:x + wd] * (1 - alpha) + tone * alpha
        streaks.append([x, wd, round(alpha, 2)])
    edge = rng.uniform(0, 45) if rng.random() < 0.5 else 0.0
    if edge:  # copier shadow along one long edge
        ramp = np.clip(1 - np.arange(w) / (0.04 * w), 0, 1)[None, :]
        a = a - edge * (ramp if rng.random() < 0.5 else ramp[:, ::-1])
    return a, {"generations": int(gens), "ink_floor": round(ink_floor), "paper": round(paper), "toner_speckle": density,
               "streaks": streaks, "edge_shadow": round(edge)}


def scan(img: Image.Image, words: list[dict], rng, nrng: np.random.Generator) -> tuple[bytes, list[dict], dict]:
    """Flatbed scan of a 300 DPI grey page: skew, photocopy wear, resolution loss, noise, JPEG."""
    w, h = img.size
    skew = round(rng.uniform(0.5, 8.0) * rng.choice([-1, 1]), 2)
    hm = _rotation_fit(words, (w, h), skew)
    a = np.asarray(warp(img, hm, (w, h), 255), np.float32)
    params: dict = {"capture": "scan", "skew_deg": skew}
    if rng.random() < 0.75:
        a, params["photocopy"] = photocopy(a, rng, nrng)
    else:
        params["photocopy"] = None
        faint = round(rng.uniform(0.55, 1.0), 2)
        a = 255.0 - (255.0 - a) * faint
        params["ink_factor"] = faint
    dpi = rng.choice([150, 200, 300])
    s = dpi / 300.0
    small = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if s < 1:
        small = small.resize((round(w * s), round(h * s)), Image.Resampling.BOX)
    blur = round(rng.uniform(0.0, 0.7), 2)
    if blur:
        small = small.filter(ImageFilter.GaussianBlur(blur))
    b = np.asarray(small, np.float32)
    sigma = rng.choice([0.0, 2.0, 4.0, 7.0])
    if sigma:
        b = b + nrng.normal(0, sigma, b.shape).astype(np.float32)
    out = Image.fromarray(np.clip(np.rint(b), 0, 255).astype(np.uint8))
    quality = rng.randint(25, 80)
    params.update(dpi=dpi, blur_px=blur, noise_sigma=sigma, jpeg_quality=quality)
    return jpeg(out, quality, dpi), move_words(words, hm, s), params


def fax(img: Image.Image, words: list[dict], rng, nrng: np.random.Generator, mode: str) -> tuple[bytes, list[dict], dict]:
    """G3 fax of a 300 DPI grey page. `mode` is "std" (200 x 100 dpi, rows half as dense) or "fine" (200 x 200). Returns the TIFF bytes."""
    w, h = img.size
    skew = round(rng.uniform(0.2, 2.0) * rng.choice([-1, 1]), 2)
    hm = _rotation_fit(words, (w, h), skew)
    page = warp(img, hm, (w, h), 255)
    sx = FAX_WIDTH / w
    sy = sx * (0.5 if mode == "std" else 1.0)
    th = round(h * sy)
    small = page.resize((FAX_WIDTH, th), Image.Resampling.BOX)
    optics = round(rng.uniform(0.25, 0.7), 2)
    a = np.asarray(small.filter(ImageFilter.GaussianBlur(optics)), np.float32)
    sigma = rng.choice([2.0, 5.0, 9.0])
    a = a + nrng.normal(0, sigma, a.shape).astype(np.float32)
    thr0 = rng.uniform(120, 185)  # the fax machine's contrast setting
    wander = rng.choice([0.0, 12.0, 25.0])
    field = (_smooth_field(nrng, th, FAX_WIDTH, 5) - 0.5) * 2 * wander
    bits = a > (thr0 + field)  # True = white
    dropouts = rng.choice([0.0, 0.0008, 0.003])
    noise_lines = rng.choice([0.0, 0.0005, 0.002])
    jitter = rng.choice([0.0, 0.004, 0.015])
    n_drop = n_noise = 0
    for y in range(th):
        if dropouts and nrng.random() < dropouts:  # the sensor misses part of the scan line: ink disappears
            x0 = int(nrng.integers(0, FAX_WIDTH - 40))
            bits[y, x0:x0 + int(nrng.integers(30, 400))] = True
            n_drop += 1
        if noise_lines and nrng.random() < noise_lines:  # line noise: a black run on one scan line
            x0 = int(nrng.integers(0, FAX_WIDTH - 40))
            bits[y, x0:x0 + int(nrng.integers(4, 60))] = False
            n_noise += 1
        if jitter and nrng.random() < jitter:  # the line is shifted by a pixel or two
            bits[y] = np.roll(bits[y], int(nrng.choice([-2, -1, 1, 2])))
    speck = rng.choice([0.0, 1e-5, 6e-5])
    if speck:
        bits &= ~(nrng.random(bits.shape) < speck)
    im1 = Image.fromarray(np.where(bits, 255, 0).astype(np.uint8)).convert("1", dither=Image.Dither.NONE)
    ydpi = 100 if mode == "std" else 200
    buf = io.BytesIO()
    im1.save(buf, format="TIFF", compression="group3", dpi=(200, ydpi))
    m = np.diag([sx, sy, 1.0]) @ hm
    params = {"capture": "fax", "fax_mode": mode, "dots_per_inch": [200, ydpi], "skew_deg": skew, "optics_blur_px": optics,
              "noise_sigma": sigma, "threshold": round(thr0), "threshold_wander": wander, "dropout_lines": n_drop,
              "noise_lines": n_noise, "line_jitter": jitter, "speckle": speck, "size": [FAX_WIDTH, th]}
    return buf.getvalue(), move_words(words, m, 1.0), params


def fax_as_png(tiff: bytes) -> bytes:
    """The same 1-bit bitmap as a PNG (a lossless equivalent for readers without CCITT support)."""
    with Image.open(io.BytesIO(tiff)) as im:
        im1 = im.convert("1")
    buf = io.BytesIO()
    im1.save(buf, format="PNG", compress_level=6, dpi=(200, 100))
    return buf.getvalue()


def phone(img: Image.Image, words: list[dict], rng_factory, nrng_factory, canvas: tuple[int, int] = (2016, 2688)) -> tuple[bytes, list[dict], dict]:
    """The camera simulation; a placement that leaves the canvas is drawn again (as in the phone set)."""
    for attempt in range(60):
        try:
            data, w, params = phone_capture(img, words, rng_factory(attempt), nrng_factory(attempt), canvas)
        except ValueError:
            continue
        params["attempt"] = attempt
        return data, w, {"capture": "phone", **params}
    raise RuntimeError("no placement fit the canvas")
