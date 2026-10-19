"""Testes: vetores de Sharma (CIEDE2000) e recuperacao de cor em foto sintetica."""
import numpy as np
from PIL import Image
import colorimetro_foto as cf

SHARMA = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((50.0, 2.5, 0.0), (61.0, -5.0, 29.0), 22.8977),
    ((50.0, 2.5, 0.0), (56.0, -27.0, -3.0), 31.9030),
    ((50.0, 2.5, 0.0), (58.0, 24.0, 15.0), 19.4535),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]

def test_sharma():
    for l1, l2, esperado in SHARMA:
        assert abs(cf.delta_e00(l1, l2) - esperado) < 1e-3, (l1, l2)
        assert abs(cf.delta_e00(l2, l1) - esperado) < 1e-3

def test_roundtrip_lab_srgb():
    for rgb in [(200, 190, 170), (255, 255, 255), (30, 60, 90)]:
        lab = cf.linear_rgb_to_lab(cf.srgb_to_linear(np.array(rgb)))
        assert tuple(cf.lab_to_srgb8(lab)) == rgb

def _foto(lab_dente, ganhos, expo):
    lin_dente = np.array(cf.lab_to_srgb8(lab_dente), float)
    lin_dente = cf.srgb_to_linear(lin_dente)
    cartao = np.array([0.18, 0.18, 0.18])
    img = np.zeros((200, 400, 3), np.uint8)
    for (x0, lin) in [(0, cartao), (200, lin_dente)]:
        c = cf.linear_to_srgb(lin * np.array(ganhos) * expo)
        img[:, x0:x0 + 200] = c.astype(np.uint8)
    return Image.fromarray(img)

def test_recuperacao_sob_luz_colorida():
    # alvo = Lab do dente ja quantizado em 8 bits (a foto neutra deve devolver ~0)
    rgb = cf.lab_to_srgb8((74.0, 2.0, 18.0))
    alvo = tuple(cf.linear_rgb_to_lab(cf.srgb_to_linear(np.array(rgb))))
    casos = [((1, 1, 1), 1.0, 0.25), ((0.8, 1.0, 1.25), 1.0, 0.6),
             ((1.2, 1.0, 0.7), 1.3, 0.6), ((0.8, 1.0, 1.25), 0.7, 1.3)]
    # tolerancias maiores quando a luz escura quantiza canais em poucos niveis
    for ganhos, expo, tol in casos:
        m = cf.measure(_foto((74.0, 2.0, 18.0), ganhos, expo), (20, 20, 160, 160), (220, 20, 160, 160), alvo)
        assert m.delta_e00 < tol, (ganhos, expo, m.delta_e00)
        assert not m.warnings

def test_aviso_cartao_estourado():
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    m = cf.measure(img, (0, 0, 50, 50), (50, 50, 50, 50))
    assert any("estourados" in w for w in m.warnings)

if __name__ == "__main__":
    for nome, f in list(globals().items()):
        if nome.startswith("test_"):
            f(); print("ok", nome)
