"""Compute every number quoted in the paper and write generated/numbers.json and
generated/numbers.tex (LaTeX macros used by main.tex).

Usage:  python make_numbers.py
"""
from __future__ import annotations

import json
import math
import os
import time

import mpmath as mp
import numpy as np
from scipy.stats import norm

import timing as T

OUT = os.path.join(os.path.dirname(__file__), "..", "generated")
os.makedirs(OUT, exist_ok=True)

res: dict = {"meta": {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "lambda": T.LAM, "beta": T.BETA, "eps": T.EPS, "r": T.R,
                      "d0": T.D0, "g": T.G, "alpha": T.ALPHA}}

# ---------------------------------------------------------------- calibration box
cal = {
    "g": T.G,
    "ode_blowup_logdrift": T.ode_blowup(T.G),                         # 16.646
    "ode_first_doubling_logdrift": T.ode_first_doubling(T.G),         # 3.943
    "ode_blowup_firstdoubling_fixed": T.D0 / (1 - 2 ** (-T.EPS)),     # 18.998
    "leveldrift_blowup_g_1_over_d0": T.D0 / T.EPS,                    # 11.538
    "mean_first_doubling": {str(h): T.mean_first_doubling_lattice(h, T.q0_logdrift(h))
                            for h in (0.01, 0.05, 0.1, 1.0, 9.0)},
}
res["calibration"] = cal

# ---------------------------------------------------------------- Table 2
rows = {}
for h in T.H_GRID:
    q = T.q_of(h); ell = T.ell_of(h); q0 = T.q0_logdrift(h)
    q05 = T.quantile(0.05, q); q95 = T.quantile(0.95, q)
    n, prev, cur = T.nstar(ell)
    W = q ** n
    Ez = T.mean_zbar(q) / q0
    row = dict(h=h, q=q, ell=ell, q0=q0,
               Ezeta=Ez, CV=T.cv_zbar(q),
               Q05=q05 / q0, Q95=q95 / q0,
               P_le_12=float(T.cdf(12 * q0, q)),
               nstar=n, err_prev=float(prev.err), err_n=float(cur.err),
               type1=float(cur.type1), type2=float(cur.type2),
               Wstar=W, ET_nstar=(1 - W) * Ez, Erem=W * Ez,
               RemQ05=W * q05 / q0, RemQ95=W * q95 / q0)
    rows[str(h)] = row
    print(f"h={h}: n*={n} err=({row['err_prev']:.10f},{row['err_n']:.10f}) "
          f"E={Ez:.6f} Q05={row['Q05']:.6f} Q95={row['Q95']:.6f} W={W:.8f} rem={row['Erem']:.6f} "
          f"[{row['RemQ05']:.4f},{row['RemQ95']:.4f}] P12={row['P_le_12']:.6g}", flush=True)
res["table2"] = rows

# ---------------------------------------------------------------- h = 1 forecasting constants
q1 = T.q_of(1.0)
z05 = T.quantile(0.05, q1); z95 = T.quantile(0.95, q1)
res["forecast_h1"] = dict(mean_units=1 / (1 - q1), q05_units=z05, q95_units=z95,
                          lead12_min_doubling=12.0 / z05,
                          ET9=T.D0 * (1 - q1 ** 9) / (1 - q1), Ezeta=T.D0 / (1 - q1))

# ---------------------------------------------------------------- detection constants
res["detection"] = dict(lan=T.lan_constant(), lower_const=(6 * math.log(1 / (4 * T.ALPHA))) ** (1 / 3),
                        z_alpha=float(norm.ppf(1 - T.ALPHA)))

# ---------------------------------------------------------------- prior on r
P = T.LognormalPrior()
res["prior"] = dict(mu=P.mu, sigma=P.sigma, p_explode=P.p_explode(), median=P.median_fluid(),
                    p_gt_60=P.surv_fluid(60.0), p_gt_120=P.surv_fluid(120.0),
                    p_gt_24=P.surv_fluid(24.0), pareto_c=P.pareto_constant())

# ---------------------------------------------------------------- log-periodic amplitudes (2|Gamma(1+2 pi i/ell)|)
amp = {}
for name, ell in [("h1", T.ell_of(1.0)), ("ell12", 1.2), ("log10", math.log(10.0))]:
    amp[name] = 2 * float(abs(mp.gamma(1 + 2j * math.pi / ell)))
res["logperiodic"] = amp

# ---------------------------------------------------------------- lag, diversion
res["lag"] = dict(logA_doubling_D3=3.0 * math.log(2) / math.log(T.R))
gstar = 1 - 1 / T.R
res["diversion"] = dict(gamma_star=gstar, share_10=1 - 10 ** (-gstar), share_1000=1 - 1000 ** (-gstar))

# ---------------------------------------------------------------- pathwise warning fraction (Monte Carlo, h = 1, n = 9)
rng = np.random.default_rng(20261013)
K = 200
w = q1 ** np.arange(K)
fr = []
for _ in range(20):
    e = rng.exponential(size=(10000, K))
    tot = e @ w
    rem = e[:, 9:] @ w[9:]
    fr.append(rem / tot)
fr = np.concatenate(fr)
res["pathwise_fraction_h1_n9"] = dict(mean=float(fr.mean()), se=float(fr.std(ddof=1) / math.sqrt(fr.size)),
                                      ratio_of_means=q1 ** 9, paths=int(fr.size))

with open(os.path.join(OUT, "numbers.json"), "w") as f:
    json.dump(res, f, indent=1)

# ---------------------------------------------------------------- LaTeX macros
def f(x, d):
    return f"{x:.{d}f}"

def sci(x, d=1):
    m, e = f"{x:.{d}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"

M = {}
M["NumEps"] = f(T.EPS, 2)
M["NumR"] = f(T.R, 3)
M["NumG"] = f(T.G, 4)
M["NumTfluid"] = f(cal["ode_blowup_logdrift"], 2)
M["NumODEdouble"] = f(cal["ode_first_doubling_logdrift"], 2)
M["NumTfluidC"] = f(cal["ode_blowup_firstdoubling_fixed"], 2)
M["NumTlevel"] = f(cal["leveldrift_blowup_g_1_over_d0"], 2)
for key, name in [("0.01", "A"), ("0.05", "Ab"), ("0.1", "B"), ("1.0", "C"), ("9.0", "D")]:
    M[f"NumFirstDbl{name}"] = f(cal["mean_first_doubling"][key], 2)
for key, name in [("0.01", "A"), ("0.1", "B"), ("1.0", "C"), ("9.0", "D")]:
    r = rows[key]
    M[f"NumEll{name}"] = f(r["ell"], 4 if r["ell"] < 0.01 else 3)
    M[f"NumEz{name}"] = f(r["Ezeta"], 1)
    M[f"NumCV{name}"] = f(r["CV"], 3)
    M[f"NumQlo{name}"] = f(r["Q05"], 1)
    M[f"NumQhi{name}"] = f(r["Q95"], 1)
    M[f"NumPtwelve{name}"] = (sci(r["P_le_12"]) if r["P_le_12"] < 1e-3 else f(r["P_le_12"], 3))
    M[f"NumNstar{name}"] = str(r["nstar"])
    M[f"NumErrPrev{name}"] = f(r["err_prev"], 4)
    M[f"NumErrN{name}"] = f(r["err_n"], 4)
    M[f"NumW{name}"] = f(r["Wstar"], 3)
    M[f"NumETn{name}"] = f(r["ET_nstar"], 1)
    M[f"NumRem{name}"] = f(r["Erem"], 2)
    M[f"NumRemLo{name}"] = f(r["RemQ05"], 2)
    M[f"NumRemHi{name}"] = f(r["RemQ95"], 2)
fc = res["forecast_h1"]
M["NumMeanUnits"] = f(fc["mean_units"], 2)
M["NumZlo"] = f(fc["q05_units"], 3)
M["NumZhi"] = f(fc["q95_units"], 3)
M["NumLeadTwelve"] = f(fc["lead12_min_doubling"], 2)
M["NumETnine"] = f(fc["ET9"], 1)
M["NumEzetaC"] = f(fc["Ezeta"], 2)
M["NumLAN"] = f(res["detection"]["lan"], 2)
M["NumLowerConst"] = f(res["detection"]["lower_const"], 2)
pr = res["prior"]
M["NumPexplode"] = f(pr["p_explode"], 3)
M["NumPriorMedian"] = f(pr["median"], 1)
M["NumPgtFive"] = f(pr["p_gt_60"], 3)
M["NumPgtTen"] = f(pr["p_gt_120"], 3)
M["NumParetoC"] = f(pr["pareto_c"], 2)
M["NumAmpHone"] = sci(amp["h1"])
M["NumAmpTwelve"] = sci(amp["ell12"])
M["NumAmpLogTen"] = f(100 * amp["log10"], 0)
M["NumLagDouble"] = f(res["lag"]["logA_doubling_D3"], 2)
M["NumGammaStar"] = f(gstar, 3)
M["NumShareTen"] = f(100 * res["diversion"]["share_10"], 0)
M["NumShareThousand"] = f(100 * res["diversion"]["share_1000"], 0)
pf = res["pathwise_fraction_h1_n9"]
M["NumPathFrac"] = f(pf["mean"], 4)
M["NumPathFracSE"] = f(pf["se"], 5)
M["NumRatioMeans"] = f(pf["ratio_of_means"], 4)

with open(os.path.join(OUT, "numbers.tex"), "w") as fh:
    fh.write("% generated by code/make_numbers.py -- do not edit by hand\n")
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("wrote", len(M), "macros")
