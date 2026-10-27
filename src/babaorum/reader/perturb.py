"""Négatifs vrais sur un recadrage de ligne (protocole READER_GATE_PROTOCOL).

Modes d'E18 (axel src/axel/image/perturb.py) appliqués au recadrage lui-même,
avec une graine tirée de la clé de la ligne et d'un suffixe neuf (#r-gate) :
blanc, brouillé par tuiles de 28 px, flou fort (rayon = hauteur / 12), bruit
fort (σ = 70). Sans dépendance à axel ; PIL et numpy seulement.

Modes à densité appariée (docs/READER_NEG_DENSITE_PREREG.txt, ajoutés sans toucher aux quatre
premiers) : bandes verticales de pleine hauteur brassées (largeur = hauteur / 2), occultation partielle
par les colonnes d'une autre ligne (``donor``), bruit plus fort (σ = 100). Même taille d'image.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np
from PIL import Image, ImageFilter

MODES = ("blank", "scramble", "blur_strong", "noise_strong")
MODES_DENSITE = ("strips", "occlude_donor", "noise_xstrong")


def perturb(img: Image.Image, mode: str, key: str, donor: Image.Image | None = None) -> Image.Image:
    img = img.convert("L")
    seed = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
    if mode == "blank":
        return Image.new("L", img.size, 255)
    if mode == "blur_strong":
        return img.filter(ImageFilter.GaussianBlur(max(2.0, img.height / 12)))
    arr = np.asarray(img).copy()
    if mode == "noise_strong":
        rng = np.random.default_rng(seed)
        return Image.fromarray(np.clip(arr.astype(np.float32) + rng.normal(0, 70, arr.shape), 0, 255).astype(np.uint8))
    if mode == "scramble":
        h, w = arr.shape[0] // 28, arr.shape[1] // 28
        if h == 0 or w == 0:
            return Image.fromarray(arr[:, ::-1].copy())
        tiles = [arr[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28].copy() for r in range(h) for c in range(w)]
        random.Random(seed).shuffle(tiles)
        for k, t in enumerate(tiles):
            r, c = divmod(k, w)
            arr[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = t
        return Image.fromarray(arr)
    if mode == "noise_xstrong":
        rng = np.random.default_rng(seed)
        return Image.fromarray(np.clip(arr.astype(np.float32) + rng.normal(0, 100, arr.shape), 0, 255).astype(np.uint8))
    if mode == "strips":
        sw = max(2, arr.shape[0] // 2)
        strips = [arr[:, c:c + sw].copy() for c in range(0, arr.shape[1], sw)]
        random.Random(seed).shuffle(strips)
        return Image.fromarray(np.concatenate(strips, axis=1))
    if mode == "occlude_donor":
        assert donor is not None, "occlude_donor : ligne donneuse requise"
        d = np.asarray(donor.convert("L").resize(img.size, Image.BICUBIC))
        rng = random.Random(seed)
        w = arr.shape[1]
        n = rng.randint(3, 5)
        total = rng.uniform(0.30, 0.50) * w
        widths = [total / n] * n
        for k in range(n):  # n rectangles de pleine hauteur, chacun dans sa n-ième de la ligne
            lo, hi = k * w // n, (k + 1) * w // n
            rw = int(min(widths[k], hi - lo))
            x0 = rng.randint(lo, max(lo, hi - rw))
            arr[:, x0:x0 + rw] = d[:, x0:x0 + rw]
        return Image.fromarray(arr)
    raise ValueError(mode)
