"""Hero figure: 'The singularity has a distribution', rendered as a long-exposure light painting.

Left (outside view): histories of a Good process with Lomax idea sizes (Corollary 3.9) drawn
as additive light in calendar time; each leaves the sky at its own explosion time.  The
horizon glows with the exact law of zeta (Gamma(b+1)/q0), and the ticks under it mark E T_n,
which converge geometrically to E zeta (the q-ruler of the paper's design).

Right (inside view): the same histories in polar coordinates, radius = time remaining until
the history's own explosion (zeta - t), angle = capability (one turn per two doublings).
Between ideas a history moves radially inwards; at each idea it turns through an arc, so every
history is a stepped logarithmic spiral falling into the centre (Theorem 4.1: in units of the
current waiting time all of them look alike).  The reflection in the water is decorative.

Usage:  python make_hero.py   -> ../figs/hero_art.png, ../figs/hero_meta.json
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.stats import gamma as gamma_dist

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figs")

# ------------------------------------------------------------------ model
EPS = 1.40 - 1.01
G = math.log(2) / 4.5          # local log drift (convention B), per month
B = 6.0                        # Lomax shape: log(1+H) ~ Exp(B*EPS), zbar ~ Gamma(B+1)
Q0 = G * B * EPS               # g = q0 E log(1+H) = q0/(B eps)
Y1 = B / (B + 1)               # E(1+H)^(-eps): contraction of E T_n towards E zeta
NPATH = 1400
rng = np.random.default_rng(1310)

paths = []
for _ in range(NPATH):
    S, t = 0.0, 0.0
    T, D = [0.0], [0.0]
    while S < 70:
        q = Q0 * math.exp(EPS * S)
        t += rng.exponential() / q
        S += rng.exponential() / (B * EPS)
        T.append(t); D.append(S / math.log(2))
    paths.append((np.array(T), np.array(D), t))
zetas = np.array([p[2] for p in paths])
rank = np.argsort(np.argsort(zetas))
u_of = 1 - rank / (NPATH - 1)   # 1 = earliest explosion
HEROES = [int(np.argmin(np.abs(zetas - np.quantile(zetas, p)))) for p in (0.1, 0.5, 0.9)]

# ------------------------------------------------------------------ canvas geometry
W, H = 2800, 1120
HOR = int(0.75 * H)             # horizon row
SKY_TOP = 34
FAN_W = 1640                    # width of the calendar-time panel
TMAX = 36.0                     # months shown in that panel
YMAX = 13.0                     # doublings shown
CX, CY, RPX = 2215, 415, 405    # vortex centre and radius in pixels
RMONTHS = 27.0                  # remaining time shown at radius RPX
TURN = 5.0                      # doublings per full turn


def xpix(t):  return t / TMAX * FAN_W
def ypix(y):  return HOR - (y / YMAX) * (HOR - SKY_TOP)


stops = np.array([[0.33, 0.07, 0.20], [0.68, 0.24, 0.12], [0.92, 0.41, 0.20],
                  [0.98, 0.70, 0.32], [1.00, 0.95, 0.83]])


def colour(u):
    x = u * (len(stops) - 1)
    i = min(int(x), len(stops) - 2)
    f = x - i
    return stops[i] * (1 - f) + stops[i + 1] * f


class Light:
    """Collect points with weights and colours; splat them bilinearly, additively, in one pass."""
    def __init__(self):
        self.xs, self.ys, self.ws, self.cs = [], [], [], []

    def add(self, xs, ys, w, col):
        xs = np.asarray(xs, float); ys = np.asarray(ys, float)
        self.xs.append(xs); self.ys.append(ys)
        self.ws.append(np.broadcast_to(np.asarray(w, float), xs.shape).copy())
        self.cs.append(np.repeat(np.asarray(col, float)[None, :], len(xs), axis=0))

    def render(self):
        out = np.zeros((3, H, W))
        if not self.xs:
            return out
        xs = np.concatenate(self.xs); ys = np.concatenate(self.ys)
        ws = np.concatenate(self.ws); cs = np.concatenate(self.cs)
        ok = (xs >= 0) & (xs < W - 1) & (ys >= 0) & (ys < H - 1)
        xs, ys, ws, cs = xs[ok], ys[ok], ws[ok], cs[ok]
        x0 = np.floor(xs).astype(np.int64); y0 = np.floor(ys).astype(np.int64)
        fx = xs - x0; fy = ys - y0
        for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                           (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            idx = (y0 + dy) * W + (x0 + dx)
            for c in range(3):
                out[c] += np.bincount(idx, weights=wt * ws * cs[:, c], minlength=W * H).reshape(H, W)
        return out


def polyline(L, X, Y, w, col, step=0.7):
    seg = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
    if seg[-1] <= 0:
        return
    sg = np.arange(0.0, seg[-1], step)
    L.add(np.interp(sg, seg, X), np.interp(sg, seg, Y), w, col)


# ------------------------------------------------------------------ left panel: staircases in calendar time
fan = Light(); heroes = Light()
for i, ((T, D, z), u) in enumerate(zip(paths, u_of)):
    X = xpix(T); Yp = ypix(D)
    xs, ys = [X[0]], [Yp[0]]
    for k in range(len(T) - 1):
        xs += [X[k + 1], X[k + 1]]; ys += [Yp[k], Yp[k + 1]]
        if Yp[k + 1] < 0 or X[k + 1] > FAN_W + 40:
            break
    if i in HEROES:
        polyline(heroes, np.array(xs), np.array(ys), 0.55, 0.5 + 0.5 * colour(u))
    polyline(fan, np.array(xs), np.array(ys), 0.040, colour(u))
    if xpix(z) < FAN_W:                                       # spark where it leaves the sky
        fan.add([xpix(z)], [SKY_TOP + rng.normal(0, 2.5)], 0.7, colour(u))
fan_layer = fan.render()

# fluid ODE: one deterministic blow-up date
Tst = 1 / (EPS * G)
tt = np.linspace(0, Tst * (1 - 1e-7), 8000)
xs, ys = xpix(tt), ypix(-np.log2(1 - EPS * G * tt) / EPS)
seg = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
sg = np.arange(0, seg[-1], 0.5)
dd = (sg % 24) < 13
ode = Light(); ode.add(np.interp(sg, seg, xs)[dd], np.interp(sg, seg, ys)[dd], 1.3, np.array([0.80, 0.89, 1.0]))
ode_layer = ode.render()

# ------------------------------------------------------------------ right panel: the vortex
vort = Light()
for i, ((T, D, z), u) in enumerate(zip(paths, u_of)):
    R = (z - T) / RMONTHS * RPX            # remaining time -> radius (pixels)
    TH = math.pi + 2 * math.pi * D / TURN   # capability -> angle; start pointing left
    xs, ys = [], []
    for k in range(len(T) - 1):
        if R[k] < 0.8:
            break
        # radial segment at angle TH[k] from R[k] to R[k+1]
        xs += [CX + R[k] * math.cos(TH[k]), CX + R[k + 1] * math.cos(TH[k])]
        ys += [CY - R[k] * math.sin(TH[k]), CY - R[k + 1] * math.sin(TH[k])]
        # arc at radius R[k+1] from TH[k] to TH[k+1]
        na = max(2, int(R[k + 1] * abs(TH[k + 1] - TH[k]) / 0.7))
        a = np.linspace(TH[k], TH[k + 1], na)
        xs += list(CX + R[k + 1] * np.cos(a)); ys += list(CY - R[k + 1] * np.sin(a))
    if len(xs) > 2:
        polyline(vort, np.array(xs), np.array(ys), 0.022, colour(u))
        if i in HEROES:
            polyline(heroes, np.array(xs), np.array(ys), 0.45, 0.5 + 0.5 * colour(u))
vort_layer = vort.render()
hero_layer = heroes.render()

# ------------------------------------------------------------------ tone mapping and bloom
def bloom(L, a=(0.30, 0.18, 0.12), s=(2.5, 12, 45)):
    return sum(ai * np.stack([gaussian_filter(L[c], si) for c in range(3)]) for ai, si in zip(a, s))


light = fan_layer + ode_layer + vort_layer + hero_layer
lit = 1 - np.exp(-1.0 * (light + bloom(light)))

# ------------------------------------------------------------------ background
Yg = np.arange(H)[:, None]
Xg = np.arange(W)[None, :]
sky_top, sky_hor = np.array([0.030, 0.038, 0.062]), np.array([0.085, 0.080, 0.115])
water_top, water_bot = np.array([0.048, 0.048, 0.072]), np.array([0.018, 0.020, 0.032])
fr = np.clip(Yg / HOR, 0, 1) ** 1.6
wr = np.clip((Yg - HOR) / (H - HOR), 0, 1)
bg = np.stack([np.where(Yg < HOR, sky_top[c] * (1 - fr) + sky_hor[c] * fr,
                        water_top[c] * (1 - wr) + water_bot[c] * wr) * np.ones((1, W)) for c in range(3)])
# a faint halo behind the vortex
halo = np.exp(-(((Xg - CX) ** 2 + (Yg - CY) ** 2) / (2 * (RPX * 0.9) ** 2)))
bg = bg + np.stack([c * 0.05 * halo for c in (0.55, 0.42, 0.62)])

ns = 300
sx = rng.uniform(0, W, ns); sy = rng.uniform(0, HOR * 0.85, ns); sb = rng.uniform(0.15, 0.6, ns) ** 2
stars = np.zeros((H, W)); np.add.at(stars, (sy.astype(int), sx.astype(int)), sb)
stars = gaussian_filter(stars, 0.8) * 3.0

# horizon glow and silhouette: the exact law of zeta, Gamma(B+1)/Q0, on the calendar axis
tgrid = Xg[0] / FAN_W * TMAX
dens = gamma_dist(B + 1, scale=1 / Q0).pdf(tgrid) * (Xg[0] <= FAN_W + 200) * np.clip((FAN_W + 200 - Xg[0]) / 200, 0, 1)
dn = dens / dens.max()
glow = np.stack([c * 0.55 * np.exp(-((Yg - HOR) / 36.0) ** 2) * dn[None, :] for c in (0.93, 0.42, 0.18)])
curve_y = HOR - 0.20 * (HOR - SKY_TOP) * dn
sil = Light()
xc = np.linspace(0, FAN_W + 200, 9000)
sil.add(xc, np.interp(xc, np.arange(W), curve_y), 1.5 * np.interp(xc, np.arange(W), np.clip(dn * 8, 0, 1)),
        np.array([1.0, 0.86, 0.62]))
sil_layer = sil.render()
sil_lit = 1 - np.exp(-(sil_layer + 0.6 * np.stack([gaussian_filter(sil_layer[c], 5) for c in range(3)])))
fill = ((Yg >= curve_y[None, :]) & (Yg <= HOR)).astype(float) * 0.05 * (dn[None, :] > 1e-3)

# q-ruler on the horizon: E T_n = (1 - y1^n)/(q0 (1 - y1)) -> E zeta
Ez = 1 / (Q0 * (1 - Y1))
ruler = np.zeros((H, W))
for n in range(48):
    x = int(round(xpix((1 - Y1 ** n) / (Q0 * (1 - Y1)))))
    L = int(4 + 18 * Y1 ** (n / 2))
    ruler[HOR + 3: HOR + 3 + L, max(x - 1, 0): x + 1] += 0.7
ruler[HOR + 1: HOR + 3, : int(xpix(Ez))] += 0.32
ruler += 1.4 * np.exp(-(((Xg - xpix(Ez)) ** 2 + (Yg - (HOR + 2)) ** 2) / (2 * 3.2 ** 2)))
ruler = gaussian_filter(ruler, 0.7)
ruler_col = np.array([1.0, 0.88, 0.65])

# the singularity at the vortex centre
core = np.exp(-(((Xg - CX) ** 2 + (Yg - CY) ** 2) / (2 * 2.2 ** 2))) * 1.6 \
     + np.exp(-(((Xg - CX) ** 2 + (Yg - CY) ** 2) / (2 * 14 ** 2))) * 0.35


def screen(a, b):
    return 1 - (1 - a) * (1 - b)


img = bg.copy()
sky = (Yg < HOR)
for c in range(3):
    img[c] = screen(img[c], glow[c] * (Yg < HOR + 60))
    img[c] = screen(img[c], stars * sky)
    img[c] = img[c] + fill * 0.9 * ruler_col[c]
    img[c] = screen(img[c], lit[c] * sky)
    img[c] = screen(img[c], 0.85 * sil_lit[c] * (Yg <= HOR + 2))
    img[c] = screen(img[c], core * (1.0, 0.95, 0.85)[c])

# ------------------------------------------------------------------ reflection (decorative)
sky_only = np.stack([(lit[c] + 0.6 * sil_lit[c] + core * 0.6) * sky for c in range(3)])
depth = H - HOR
rr = np.arange(depth)
src_y = HOR - 1 - rr * (HOR / depth) * 0.52
ripple = 6.0 * np.sin(rr / 5.0) * (0.3 + rr / depth) + 2.2 * np.sin(rr / 1.6)
for c in range(3):
    Yc = np.repeat(src_y[:, None], W, axis=1)
    Xc = np.arange(W)[None, :] + ripple[:, None]
    refl = map_coordinates(sky_only[c], [Yc, Xc], order=1, mode="constant")
    refl = gaussian_filter(refl, (2.5, 4.0))
    fade = (0.40 * np.exp(-rr / (0.55 * depth)))[:, None]
    img[c, HOR:] = screen(img[c, HOR:], refl * fade)
    img[c, HOR:] = screen(img[c, HOR:], 0.5 * glow[c, HOR:])
    img[c] = screen(img[c], ruler * ruler_col[c])

vx = np.linspace(-1, 1, W)[None, :]; vy = np.linspace(-1, 1, H)[:, None]
img = np.clip(img * (1 - 0.25 * (0.6 * vx ** 2 + 0.5 * vy ** 2)), 0, 1)

Image.fromarray((np.transpose(img, (1, 2, 0)) * 255 + 0.5).astype(np.uint8)).save(
    os.path.join(FIG, "hero_art.png"), dpi=(480, 480))
meta = dict(W=W, H=H, HOR=HOR, SKY_TOP=SKY_TOP, FAN_W=FAN_W, TMAX=TMAX, YMAX=YMAX, CX=CX, CY=CY, RPX=RPX,
            RMONTHS=RMONTHS, Ez=Ez, xEz=float(xpix(Ez)), Tstar=Tst, xTstar=float(xpix(Tst)),
            zeta_q05=float(np.quantile(zetas, 0.05)), zeta_q95=float(np.quantile(zetas, 0.95)), npath=NPATH, b=B)
json.dump(meta, open(os.path.join(FIG, "hero_meta.json"), "w"), indent=1)
print(meta)
