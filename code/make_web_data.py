"""Precompute the exact curves and numbers used by the project page (docs/index.html).

Usage (from code/):  python make_web_data.py   -> ../docs/assets/data.js
"""
import json
import math
import os

import numpy as np

import timing as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "assets")
os.makedirs(OUT, exist_ok=True)

data = {"eps": T.EPS, "g": T.G, "d0": T.D0, "r": T.R, "Tfluid": T.ode_blowup(T.G)}

# exact densities of zeta (months) under convention B
grid = np.round(np.arange(0.0, 70.01, 0.25), 2)
laws = []
for h in (0.1, 0.25, 0.5, 1.0, 3.0, 9.0):
    q = T.q_of(h); q0 = T.q0_logdrift(h)
    dens = [0.0 if t == 0 else float(T.dens(t * q0, q)) * q0 for t in grid]
    laws.append(dict(h=h, q=q, ell=T.ell_of(h), q0=q0,
                     mean=T.mean_zbar(q) / q0, cv=T.cv_zbar(q),
                     q05=T.quantile(0.05, q) / q0, q95=T.quantile(0.95, q) / q0,
                     p12=float(T.cdf(12 * q0, q)),
                     density=[round(v, 6) for v in dens]))
    print("law", h, flush=True)
data["grid"] = grid.tolist()
data["laws"] = laws

# detection: exact thresholds (cached grid + Table 3) and the remaining-time quantiles
num = json.load(open(os.path.join(HERE, "..", "generated", "numbers.json")))
data["table3"] = [dict(h=float(k), **{x: v[x] for x in ("ell", "nstar", "Wstar", "ET_nstar", "Erem", "RemQ05", "RemQ95")})
                  for k, v in num["table2"].items()]
grid_n = json.load(open(os.path.join(HERE, "..", "generated", "nstar_grid.json")))
data["nstar_grid"] = sorted([[float(k), v["n"]] for k, v in grid_n.items()])
data["lan"] = T.lan_constant()

# forecasting constants (h = 1)
data["forecast"] = num["forecast_h1"]

# parameter uncertainty: fluid survival under the illustrative prior, the 1/t limit, intrinsic h = 1
P = T.LognormalPrior()
tg = np.logspace(0, 3.3, 120)
q1, q01 = T.q_of(1.0), T.q0_logdrift(1.0)
data["prior"] = dict(t=[round(float(x), 4) for x in tg],
                     fluid=[P.surv_fluid(float(x)) for x in tg],
                     c=P.pareto_constant(),
                     intrinsic=[float(T.surv(float(x) * q01, q1)) if x < 160 else None for x in tg],
                     p_explode=P.p_explode(), median=P.median_fluid(),
                     p5=P.surv_fluid(60.0), p10=P.surv_fluid(120.0))

# deterministic delayed model
lag = {}
for D in (1.0, 3.0):
    t, lA = T.lagged_logA(D, 60.0, 1e-3)
    idx = np.linspace(0, len(t) - 1, 300).astype(int)
    lag[str(int(D))] = [[round(float(t[i]), 3), (None if not np.isfinite(lA[i]) else round(float(lA[i]), 5))] for i in idx]
data["lag"] = lag
data["diversion"] = num["diversion"]
data["checks"] = sum(1 for line in open(os.path.join(HERE, "..", "generated", "checks_report.txt")) if "[PASS]" in line)

with open(os.path.join(OUT, "data.js"), "w") as fh:
    fh.write("window.PAPER = " + json.dumps(data, separators=(",", ":")) + ";\n")
print("wrote data.js", os.path.getsize(os.path.join(OUT, "data.js")), "bytes; checks:", data["checks"])
