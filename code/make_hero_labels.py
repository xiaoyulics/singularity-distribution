"""Typeset the labels of the hero figure in the paper's fonts (PGF backend).

Usage:  python make_hero.py && python make_hero_labels.py   -> ../figs/fig_hero.pdf
"""
import json
import math
import os

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figs")
m = json.load(open(os.path.join(FIG, "hero_meta.json")))
W, H, HOR, SKY_TOP = m["W"], m["H"], m["HOR"], m["SKY_TOP"]
FAN_W, TMAX, YMAX = m["FAN_W"], m["TMAX"], m["YMAX"]
CX, CY, RPX = m["CX"], m["CY"], m["RPX"]
EPS, G, B = 1.40 - 1.01, math.log(2) / 4.5, m["b"]
Q0 = G * B * EPS


def xpix(t): return t / TMAX * FAN_W
def ypix(y): return HOR - (y / YMAX) * (HOR - SKY_TOP)


plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
    "pgf.preamble": "\n".join([r"\usepackage[T1]{fontenc}", r"\usepackage{amsmath}",
                               r"\usepackage[lining]{cochineal}", r"\usepackage[cochineal,vvarbb]{newtxmath}",
                               r"\usepackage[scale=0.86,semibold]{plex-sans}"]),
    "font.family": "serif", "font.size": 6.4})

art = np.asarray(Image.open(os.path.join(FIG, "hero_art.png")))
wid = 5.748                                     # 146 mm
fig = plt.figure(figsize=(wid, wid * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(art, extent=[0, W, H, 0], interpolation="lanczos")
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
CREAM, DIM, ICE, GOLD = "#efe6d6", "#b3a894", "#d7e3ff", "#ffe2ad"


def label(x, y, s, color=CREAM, size=6.2, **kw):
    ax.text(x, y, s, color=color, fontsize=size, **kw)


label(44, 96, r"\textsf{\bfseries OUTSIDE VIEW}", size=6.6)
label(44, 146, r"\textsf{capability (doublings) against calendar time}", color=DIM, size=5.9)
label(CX - RPX - 30, 96, r"\textsf{\bfseries INSIDE VIEW}", size=6.6)
label(CX - RPX - 30, 146, r"\textsf{radius: time left to the singularity}", color=DIM, size=5.9)
label(CX - RPX - 30, 190, r"\textsf{angle: capability, one turn per five doublings}", color=DIM, size=5.9)
Tst = 1 / (EPS * G)
label(xpix(Tst) + 24, ypix(9.0), r"\textsf{fluid ODE: one date}", color=ICE, size=5.9)
mode = B / Q0                                   # mode of Gamma(B+1)/Q0
label(xpix(mode) + 120, HOR - 0.20 * (HOR - SKY_TOP) - 14, r"\textsf{law of} $\zeta$", size=6.3)
Ez = m["Ez"]
label(xpix(Ez) + 16, HOR + 66, r"$\mathbb{E}\zeta$", color=GOLD, size=6.5)
label(16, HOR + 66, r"\textsf{now}", color=DIM, size=5.9)
label(CX + 18, CY - 16, r"$\zeta$", color="#fff4dc", size=7.0)
fig.savefig(os.path.join(FIG, "fig_hero.pdf"), dpi=480)
print("fig_hero.pdf written")
