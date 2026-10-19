#!/usr/bin/env python3
"""
colorimetro_foto: estimativa de cor do dente (CIELAB, D65) a partir de uma foto
com cartao cinza na mesma cena.

Autor: Marcelo Borille (Protocolo Borille). Licenca MIT.

Metodo
------
1. Le a foto como sRGB e converte para RGB linear.
2. Mede o cartao cinza (media aparada) e calcula um ganho por canal que
   leva o cartao a um cinza neutro com luminancia relativa igual a sua
   refletancia nominal (padrao 0,18). Isso corrige balanco de branco e
   exposicao em um unico passo.
3. Aplica o mesmo ganho a regiao do dente, converte para XYZ (D65, 2 graus)
   e depois para CIELAB.
4. Compara com uma cor-alvo por CIEDE2000 (Sharma, Wu e Dalal, 2005).

Limites (declarar em qualquer publicacao)
-----------------------------------------
Este e um estimador por camera, nao um espectrofotometro. A curva de tons do
JPEG, o perfil de cor da camera, reflexos especulares e a desigualdade da luz
entre cartao e dente limitam a exatidao. Use foto sem filtro de beleza, sem
HDR, com luz constante e o cartao no mesmo plano do dente. Valide contra uma
medicao instrumental antes de usar os numeros como dado de pesquisa.

Dependencias: numpy, Pillow.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field, asdict

import numpy as np
from PIL import Image

__version__ = "0.1.0"

# Limiares de Paravina et al., J Esthet Restor Dent 2015;27 Suppl 1:S1-9
PERCEPTIBILITY_THRESHOLD = 0.8
ACCEPTABILITY_THRESHOLD = 1.8

# Branco de referencia D65, observador 2 graus (Y = 1)
WHITE_D65 = (0.95047, 1.0, 1.08883)

_M_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


# ----------------------------------------------------------------------------
# Conversoes de cor
# ----------------------------------------------------------------------------
def srgb_to_linear(c8: np.ndarray) -> np.ndarray:
    """sRGB 0..255 -> linear 0..1."""
    c = np.asarray(c8, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    """linear 0..1 -> sRGB 0..255 (com clipping)."""
    lin = np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255.0), 0, 255)


def linear_rgb_to_xyz(lin: np.ndarray) -> np.ndarray:
    return np.asarray(lin) @ _M_RGB2XYZ.T


def xyz_to_lab(xyz: np.ndarray, white=WHITE_D65) -> np.ndarray:
    t = np.asarray(xyz, dtype=np.float64) / np.asarray(white)
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def linear_rgb_to_lab(lin: np.ndarray) -> np.ndarray:
    return xyz_to_lab(linear_rgb_to_xyz(lin))


def lab_to_srgb8(lab) -> tuple[int, int, int]:
    """CIELAB (D65) -> sRGB 8 bits, para desenhar amostras."""
    L, a, b = [float(x) for x in lab]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    eps, kappa = 216 / 24389, 24389 / 27

    def finv(f):
        return f**3 if f**3 > eps else (116 * f - 16) / kappa

    x = finv(fx) * WHITE_D65[0]
    y = (L / kappa if L <= kappa * eps else ((L + 16) / 116) ** 3) * WHITE_D65[1]
    z = finv(fz) * WHITE_D65[2]
    lin = np.linalg.solve(_M_RGB2XYZ, np.array([x, y, z]))
    return tuple(int(v) for v in linear_to_srgb(lin))


# ----------------------------------------------------------------------------
# CIEDE2000 (Sharma, Wu e Dalal, 2005)
# ----------------------------------------------------------------------------
def delta_e00(lab1, lab2, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> float:
    L1, a1, b1 = [float(x) for x in lab1]
    L2, a2, b2 = [float(x) for x in lab2]

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cbar**7 / (Cbar**7 + 25**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)

    def hue(ap, b):
        if ap == 0 and b == 0:
            return 0.0
        h = math.degrees(math.atan2(b, ap))
        return h + 360 if h < 0 else h

    h1p, h2p = hue(a1p, b1), hue(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        d = h2p - h1p
        if d > 180:
            d -= 360
        elif d < -180:
            d += 360
        dhp = d
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))

    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbp = (h1p + h2p + 360) / 2
    else:
        hbp = (h1p + h2p - 360) / 2

    T = (
        1
        - 0.17 * math.cos(math.radians(hbp - 30))
        + 0.24 * math.cos(math.radians(2 * hbp))
        + 0.32 * math.cos(math.radians(3 * hbp + 6))
        - 0.20 * math.cos(math.radians(4 * hbp - 63))
    )
    dTheta = 30 * math.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(Cbp**7 / (Cbp**7 + 25**7))
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / math.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -math.sin(math.radians(2 * dTheta)) * Rc

    return math.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )


def classify(de: float) -> str:
    if de <= PERCEPTIBILITY_THRESHOLD:
        return "imperceptivel"
    if de <= ACCEPTABILITY_THRESHOLD:
        return "perceptivel, clinicamente aceitavel"
    return "acima do limiar de aceitabilidade"


# ----------------------------------------------------------------------------
# Medicao
# ----------------------------------------------------------------------------
@dataclass
class RegionStats:
    n_pixels: int
    n_used: int
    mean_linear: tuple
    mean_srgb8: tuple
    clipped_fraction: float
    luminance_cv: float


def _trim_by_luminance(lin: np.ndarray, trim: float):
    """Remove 'trim' de cada ponta de luminancia (reflexos e sombras)."""
    y = lin @ _M_RGB2XYZ[1]
    if trim <= 0 or len(y) < 20:
        return lin, np.ones(len(y), bool)
    lo, hi = np.quantile(y, [trim, 1 - trim])
    keep = (y >= lo) & (y <= hi)
    return lin[keep], keep


def crop(img: Image.Image, box) -> np.ndarray:
    """box = (x, y, largura, altura) em pixels da imagem. Retorna (N,3) sRGB 0..255."""
    x, y, w, h = [int(round(v)) for v in box]
    arr = np.asarray(img.convert("RGB"))
    H, W = arr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"regiao fora da imagem: {box}")
    return arr[y0:y1, x0:x1].reshape(-1, 3)


def region_stats(px8: np.ndarray, trim: float = 0.10) -> RegionStats:
    lin = srgb_to_linear(px8)
    used, keep = _trim_by_luminance(lin, trim)
    mean = used.mean(axis=0)
    y = used @ _M_RGB2XYZ[1]
    cv = float(y.std() / y.mean()) if y.mean() > 0 else float("nan")
    clipped = float(np.mean(np.any(px8 >= 254, axis=1) | np.all(px8 <= 1, axis=1)))
    return RegionStats(
        n_pixels=len(px8),
        n_used=int(keep.sum()),
        mean_linear=tuple(mean),
        mean_srgb8=tuple(linear_to_srgb(mean)),
        clipped_fraction=clipped,
        luminance_cv=cv,
    )


@dataclass
class Measurement:
    L: float
    a: float
    b: float
    sd_L: float
    sd_a: float
    sd_b: float
    srgb8: tuple
    gains: tuple
    card_stats: RegionStats
    tooth_stats: RegionStats
    target: tuple | None = None
    delta_e00: float | None = None
    verdict: str | None = None
    warnings: list = field(default_factory=list)

    @property
    def lab(self):
        return (self.L, self.a, self.b)


def measure(
    img: Image.Image,
    card_box,
    tooth_box,
    target_lab=None,
    card_reflectance: float = 0.18,
    trim: float = 0.10,
) -> Measurement:
    card_px = crop(img, card_box)
    tooth_px = crop(img, tooth_box)

    card = region_stats(card_px, trim)
    tooth = region_stats(tooth_px, trim)

    card_lin = np.array(card.mean_linear)
    gains = card_reflectance / np.maximum(card_lin, 1e-6)

    tooth_lin = srgb_to_linear(tooth_px)
    tooth_used, _ = _trim_by_luminance(tooth_lin, trim)
    corrected = tooth_used * gains
    lab_px = linear_rgb_to_lab(np.clip(corrected, 0, None))
    mean_lab = linear_rgb_to_lab(np.clip(corrected.mean(axis=0), 0, None))
    sd = lab_px.std(axis=0)
    srgb8 = tuple(int(v) for v in linear_to_srgb(corrected.mean(axis=0)))

    m = Measurement(
        L=float(mean_lab[0]),
        a=float(mean_lab[1]),
        b=float(mean_lab[2]),
        sd_L=float(sd[0]),
        sd_a=float(sd[1]),
        sd_b=float(sd[2]),
        srgb8=srgb8,
        gains=tuple(float(g) for g in gains),
        card_stats=card,
        tooth_stats=tooth,
    )
    if target_lab is not None:
        m.target = tuple(float(v) for v in target_lab)
        m.delta_e00 = delta_e00(m.lab, m.target)
        m.verdict = classify(m.delta_e00)
    m.warnings = quality_warnings(m)
    return m


def quality_warnings(m: Measurement) -> list:
    w = []
    c, t = m.card_stats, m.tooth_stats
    if c.n_used < 400:
        w.append("Cartao com poucos pixels: selecione uma area maior.")
    if t.n_used < 400:
        w.append("Dente com poucos pixels: selecione uma area maior.")
    if c.clipped_fraction > 0.02:
        w.append("Cartao com pixels estourados ou pretos: reduza a exposicao.")
    if t.clipped_fraction > 0.02:
        w.append("Dente com pixels estourados ou pretos: reflexo ou exposicao.")
    if c.luminance_cv > 0.08:
        w.append("Luz desigual sobre o cartao (CV de luminancia > 8%).")
    if max(m.gains) / min(m.gains) > 2.0:
        w.append("Dominante de cor forte na cena: a correcao e grande e menos confiavel.")
    if t.luminance_cv > 0.15:
        w.append("Area do dente muito heterogenea: evite bordas, gengiva e reflexos.")
    return w


# ----------------------------------------------------------------------------
# Tabela de cores no formato do dentcolor (shade-table)
# ----------------------------------------------------------------------------
def load_shade_table(path: str) -> list:
    import json

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("shades", [])


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
CSV_FIELDS = [
    "arquivo", "caso", "dente", "L", "a", "b", "sd_L", "sd_a", "sd_b",
    "alvo_L", "alvo_a", "alvo_b", "deltaE00", "veredito",
    "ganho_R", "ganho_G", "ganho_B", "px_cartao", "px_dente", "avisos",
]


def row_for(path, caso, dente, m: Measurement) -> dict:
    t = m.target or (None, None, None)
    return {
        "arquivo": path, "caso": caso, "dente": dente,
        "L": round(m.L, 2), "a": round(m.a, 2), "b": round(m.b, 2),
        "sd_L": round(m.sd_L, 2), "sd_a": round(m.sd_a, 2), "sd_b": round(m.sd_b, 2),
        "alvo_L": t[0], "alvo_a": t[1], "alvo_b": t[2],
        "deltaE00": None if m.delta_e00 is None else round(m.delta_e00, 2),
        "veredito": m.verdict,
        "ganho_R": round(m.gains[0], 4), "ganho_G": round(m.gains[1], 4),
        "ganho_B": round(m.gains[2], 4),
        "px_cartao": m.card_stats.n_used, "px_dente": m.tooth_stats.n_used,
        "avisos": " | ".join(m.warnings),
    }


def _box(s: str):
    v = [float(x) for x in s.split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("use x,y,largura,altura")
    return tuple(v)


def _lab(s: str):
    v = [float(x) for x in s.split(",")]
    if len(v) != 3:
        raise argparse.ArgumentTypeError("use L,a,b")
    return tuple(v)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Colorimetro por foto (cartao cinza + CIELAB + dE00)")
    p.add_argument("foto", nargs="+", help="uma ou mais fotos (mesmas caixas em todas)")
    p.add_argument("--cartao", type=_box, required=True, help="x,y,largura,altura do cartao cinza")
    p.add_argument("--dente", type=_box, required=True, help="x,y,largura,altura do dente")
    p.add_argument("--alvo", type=_lab, help="cor-alvo L,a,b")
    p.add_argument("--refletancia", type=float, default=0.18, help="refletancia do cartao (padrao 0,18)")
    p.add_argument("--aparar", type=float, default=0.10, help="fracao aparada em cada ponta (padrao 0,10)")
    p.add_argument("--caso", default="")
    p.add_argument("--elemento", default="")
    p.add_argument("--csv", help="grava (ou acrescenta) as medidas neste CSV")
    a = p.parse_args(argv)

    rows = []
    for path in a.foto:
        img = Image.open(path)
        m = measure(img, a.cartao, a.dente, a.alvo, a.refletancia, a.aparar)
        rows.append(row_for(path, a.caso, a.elemento, m))
        print(f"{path}: L*={m.L:.2f} a*={m.a:.2f} b*={m.b:.2f}", end="")
        if m.delta_e00 is not None:
            print(f"  dE00={m.delta_e00:.2f} ({m.verdict})", end="")
        print()
        for w in m.warnings:
            print(f"  aviso: {w}", file=sys.stderr)

    if a.csv:
        import os

        novo = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            if novo:
                wr.writeheader()
            wr.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
