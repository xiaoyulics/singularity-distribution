"""Regenerate every figure of the paper with LaTeX-rendered text in the paper's fonts.

Usage:  python make_figures.py            (uses cached exact detection thresholds if present)
        python make_figures.py --recompute
Seeds are fixed; outputs go to ../figs/.
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from scipy.stats import norm

import timing as T

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figs")
GEN = os.path.join(HERE, "..", "generated")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, REF = "#1b1e25", "#5d5952", "#e6e3dc", "#8f8a82"
plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
    "pgf.preamble": "\n".join([r"\usepackage[T1]{fontenc}", r"\usepackage{amsmath}",
                               r"\usepackage[lining]{cochineal}", r"\usepackage[cochineal,vvarbb]{newtxmath}",
                               r"\usepackage[scale=0.86,semibold]{plex-sans}"]),
    "text.usetex": False, "font.family": "serif", "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
    "lines.linewidth": 1.4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W2, H2 = 6.2, 2.45   # two-panel figure size (inches) for a 146 mm text block


def title(a, s):
    a.set_title(r"\textsf{\bfseries " + s + "}", fontsize=8.3, loc="left", color=INK)


EPS, G = T.EPS, T.G
TFL = T.ode_blowup(G)
rng = np.random.default_rng(2026)


def path(h, rng, nmax=4000):
    q0 = T.q0_logdrift(h); rho = (1 + h) ** EPS
    k = np.arange(nmax)
    Tn = np.concatenate([[0], np.cumsum(rng.exponential(size=nmax) / (q0 * rho ** k))])
    return Tn, np.arange(nmax + 1) * np.log2(1 + h), Tn[-1]


# =========================================================== Figure 1: paths and laws
fig, ax = plt.subplots(1, 2, figsize=(W2, H2))
a = ax[0]
tt = np.linspace(0, TFL * 0.9999, 2000)
for h, col, lab in [(1.0, BLUE, r"$h=1$"), (0.05, ORANGE, r"$h=0.05$")]:
    for j in range(3):
        Tn, y, z = path(h, rng)
        a.step(Tn, y, where="post", color=col, lw=1.0, label=lab if j == 0 else None)
a.plot(tt, -np.log2(1 - EPS * G * tt) / EPS, color=INK, lw=1.1, ls="--", label="fluid ODE")
a.axvline(TFL, color=REF, lw=0.8, ls=":")
a.text(TFL + 0.6, 1.0, r"$T^\ast$", color=INK2)
a.set_xlim(0, 42); a.set_ylim(0, 16)
a.set_xlabel("months"); a.set_ylabel(r"$\log_2(A_t/A_0)$")
title(a, "(a) sample paths")
a.legend(loc="lower right", fontsize=7.5, handlelength=1.6, frameon=True, facecolor="white",
         edgecolor="none", framealpha=1.0)
a = ax[1]
t = np.linspace(0.05, 70, 360)
for h, col, lab in [(0.1, AQUA, r"$h=0.1$"), (1.0, BLUE, r"$h=1$"), (9.0, ORANGE, r"$h=9$")]:
    q = T.q_of(h); q0 = T.q0_logdrift(h)
    f = np.array([float(T.dens(x * q0, q)) * q0 for x in t])
    a.plot(t, f, color=col, lw=1.4, label=lab)
    if h == 1.0:
        K = 200; w = q ** np.arange(K) / q0
        z = rng.exponential(size=(200000, K)) @ w
        hh, ed = np.histogram(z, bins=np.linspace(0, 70, 71), density=True)
        a.bar(ed[:-1], hh, width=1.0, align="edge", color=col, alpha=0.18, lw=0, label=r"Monte Carlo ($h=1$)")
a.axvline(TFL, color=REF, lw=0.8, ls=":")
a.set_xlim(0, 70); a.set_ylim(0, None)
a.set_xlabel(r"explosion time $\zeta$ (months)"); a.set_ylabel("density")
title(a, r"(b) exact law of $\zeta$")
a.legend(loc="upper right", fontsize=7.5)
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIG, "fig_paths_law.pdf")); plt.close(fig)
print("fig 1", flush=True)

# =========================================================== Figure 2: forecasting + log-periodic structure
fig, ax = plt.subplots(1, 2, figsize=(W2, H2))
a = ax[0]
h = 1.0; q = T.q_of(h); q0 = T.q0_logdrift(h)
lo_u, hi_u, mean_u = T.quantile(0.05, q), T.quantile(0.95, q), 1 / (1 - q)
Tn, y, z = path(h, np.random.default_rng(31))
n_show = 14
tg = np.linspace(0, Tn[n_show], 1200)
N = np.searchsorted(Tn, tg, side="right") - 1
qN = q0 * (1 / q) ** N
a.fill_between(tg, tg + lo_u / qN, tg + hi_u / qN, color=BLUE, alpha=0.18, lw=0, label="conditional 90\\% band")
a.plot(tg, tg + mean_u / qN, color=BLUE, lw=1.3, label=r"$t+\mathbb{E}[\zeta-t\mid\mathcal{F}_t]$")
a.axhline(z, color=INK, lw=0.9, ls="--", label=r"realised $\zeta$")
a.plot(tg, tg, color=REF, lw=0.7, ls=":")
a.set_xlabel(r"current time $t$ (months)"); a.set_ylabel("forecast of $\\zeta$ (months)")
title(a, "(a) forecasting with known parameters")
a.legend(loc="upper right", fontsize=7.0)
a.set_xlim(0, Tn[n_show]); a.set_ylim(0, max(45, z * 1.3))
a = ax[1]
x = np.linspace(0, 2, 400)
for ell, col in [(1.2, AQUA), (math.log(10), BLUE), (3.0, ORANGE)]:
    p = 1 + sum(2 * np.real(complex(mp.gamma(1 - 2j * np.pi * kk / ell)) * np.exp(2j * np.pi * kk * x)) for kk in range(1, 7))
    a.plot(x, p, color=col, lw=1.4, label=rf"$\ell={ell:.2f}$")
qq = 0.1; ell = math.log(10); R, K = 200000, 40
e = rng.exponential(size=(R, K)); w = qq ** np.arange(K)
tail = np.cumsum((e * w)[:, ::-1], axis=1)[:, ::-1]
xs = np.log(tail[:, 4:18]).ravel(); xs = xs[(xs > -38) & (xs < -8)]
hist, ed = np.histogram(np.mod(xs, ell) / ell, bins=24)
cen = 0.5 * (ed[:-1] + ed[1:]); dd = hist / hist.mean()
a.plot(np.concatenate([cen, cen + 1]), np.concatenate([dd, dd]), "o", ms=3.0, color=BLUE, mfc="white", mew=0.9,
       label=r"Monte Carlo, $\ell=\log 10$")
a.set_xlabel(r"$\log(\zeta-T_n)$ in units of $\ell$ (mod 2)"); a.set_ylabel(r"mean intensity $\times\,\ell$")
title(a, "(b) explosion-centred mean intensity")
a.set_ylim(0.6, 1.45); a.set_xlim(0, 2)
a.legend(loc="upper right", fontsize=6.8, ncol=2, columnspacing=0.8)
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIG, "fig_forecast.pdf")); plt.close(fig)
print("fig 2", flush=True)

# =========================================================== Figure 3: parameter uncertainty
P = T.LognormalPrior()
fig, a = plt.subplots(1, 1, figsize=(3.3, 2.45))
tg = np.logspace(0, 3.3, 200)
a.plot(tg, [P.surv_fluid(t) for t in tg], color=BLUE, lw=1.5, label=r"fluid $T^\ast$, illustrative prior on $r$")
a.plot(tg, np.minimum(1, P.pareto_constant() / tg), color=REF, lw=1.0, ls="--", label=r"limit $c/t$ (Prop.\ 4.6)")
q1, q01 = T.q_of(1.0), T.q0_logdrift(1.0)
tg2 = tg[tg < 150]
a.plot(tg2, [float(T.surv(t * q01, q1)) for t in tg2], color=ORANGE, lw=1.4, label=r"$r$ known, $h=1$")
a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1e-3, 1.3); a.set_xlim(1, 2000)
a.set_xlabel(r"$t$ (months)"); a.set_ylabel(r"$\mathbb{P}(\cdot>t\mid r>1)$")
a.legend(loc="upper right", fontsize=6.8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_uncertainty.pdf")); plt.close(fig)
print("fig 3", flush=True)

# =========================================================== Figure 4: detection
alpha = T.ALPHA
ells = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
cache = os.path.join(GEN, "nstar_grid.json")
if os.path.exists(cache) and "--recompute" not in sys.argv:
    grid = {float(k): v for k, v in json.load(open(cache)).items()}
else:
    grid = {}
    for l in ells:
        n, prev, cur = T.nstar(l)
        grid[l] = dict(n=n, err_prev=float(prev.err), err_n=float(cur.err))
        print("  n*", l, n, flush=True)
    json.dump({str(k): v for k, v in grid.items()}, open(cache, "w"), indent=1)
ns = np.array([grid[l]["n"] for l in ells]); ells_a = np.array(ells)
fig, ax = plt.subplots(1, 2, figsize=(W2, H2))
a = ax[0]
lg = np.logspace(-3, 0, 200)
a.plot(lg, [T.detection_bounds(l)[1] * l ** (2 / 3) for l in lg], color=REF, lw=1.0, ls="--", label="upper bound, Thm 5.1(ii)")
a.axhline(T.lan_constant(), color=BLUE, lw=1.2, label=r"limit $(12z_\alpha^2)^{1/3}$, Thm 5.1(iii)")
a.axhline((6 * np.log(1 / (4 * alpha))) ** (1 / 3), color=REF, lw=1.0, ls="-.", label="lower bound, Thm 5.1(i)")
a.plot(ells_a, ns * ells_a ** (2 / 3), "o", ms=4, color=BLUE, mfc="white", mew=1.0, label=r"exact integer $n^\ast$")
a.set_xscale("log"); a.set_ylim(0, 8)
a.set_xlabel(r"log-rate increment $\ell=\varepsilon\log(1+h)$"); a.set_ylabel(r"$n^\ast\,\ell^{2/3}$")
title(a, r"(a) fixed-sample threshold, $\alpha=0.05$")
a.legend(loc="upper left", fontsize=6.8)
a = ax[1]
a.plot(lg, np.exp(-(12 * norm.ppf(1 - alpha) ** 2 * lg) ** (1 / 3)), color=BLUE, lw=1.2,
       label=r"$\exp\{-(12z_\alpha^2\ell)^{1/3}\}$")
a.plot(ells_a, np.exp(-ells_a * ns), "o", ms=4, color=BLUE, mfc="white", mew=1.0, label=r"exact $W^\ast=q^{n^\ast}$")
for h, txt in [(0.01, r"$h=1\%$"), (0.1, r"$h=10\%$"), (1.0, r"$h=1$"), (9.0, r"$h=9$")]:
    l = EPS * np.log(1 + h)
    a.axvline(l, color=GRID, lw=0.8, zorder=0)
    a.text(l * 1.07, 0.97, txt, fontsize=7, color=INK2, rotation=90, va="top")
a.set_xscale("log"); a.set_ylim(0, 1)
a.set_xlabel(r"log-rate increment $\ell$"); a.set_ylabel(r"remaining mean / initial mean")
title(a, "(b) normalised expected remaining duration")
a.legend(loc="lower left", fontsize=7.0)
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIG, "fig_detection.pdf")); plt.close(fig)
print("fig 4", flush=True)

# =========================================================== Figure 5: deterministic delay and race regimes
fig, ax = plt.subplots(1, 2, figsize=(W2, H2))
a = ax[0]
tt = np.linspace(0, TFL * (1 - 1e-9), 3000)
a.plot(tt, -np.log(1 - EPS * G * tt) / EPS, color=INK, lw=1.1, ls="--", label=r"$D=0$: blow-up at $T^\ast$")
for D, col in [(1.0, ORANGE), (3.0, BLUE)]:
    t, lA = T.lagged_logA(D, 60.0, 1e-3)
    a.plot(t, lA, color=col, lw=1.4, label=rf"$D={D:.0f}$ month" + ("s" if D > 1 else ""))
a.set_yscale("log"); a.set_ylim(0.05, 1e3); a.set_xlim(0, 60)
a.set_xlabel("months"); a.set_ylabel(r"$\log(A_t/A_0)$")
title(a, "(a) deterministic delayed model")
a.legend(loc="lower right", fontsize=7.2)
a = ax[1]
e0 = EPS
LS = np.linspace(e0, 2.4, 400)
a.fill_between([0, e0], 0, 2, color=ORANGE, alpha=0.16, lw=0)
a.fill_between(LS, 0, np.clip(LS - e0, 0, 2), color=AQUA, alpha=0.22, lw=0)
a.fill_between(LS, np.clip(LS - e0, 0, 2), 2, color=BLUE, alpha=0.13, lw=0)
a.plot(LS, LS - e0, color=INK, lw=1.0)
a.axvline(e0, color=INK2, lw=0.8)
a.text(0.03, 1.05, "$S$ stays\nbounded", fontsize=7.0, color=INK)
a.text(0.55, 1.78, r"ratio $S/A\to0$", fontsize=7.0, color=INK)
a.text(1.55, 0.22, r"ratio $S/A\to\infty$", fontsize=7.0, color=INK)
a.annotate("critical equality " + r"$\varepsilon_S=\varepsilon$" + ":\nlimit depends on $\\theta$", xy=(1.35, 1.35 - e0),
           xytext=(0.62, 1.22), fontsize=6.8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
a.set_xlim(0, 2.4); a.set_ylim(0, 2)
a.set_xlabel(r"oversight returns to labour $\lambda_S$"); a.set_ylabel(r"oversight difficulty growth $\beta_S$")
title(a, r"(b) asymptotic regimes, fixed $\theta$")
a.grid(False)
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIG, "fig_frictions.pdf")); plt.close(fig)
print("fig 5 -- done", flush=True)
