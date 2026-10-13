"""Regression, stability and cross-checks for every numerical claim in the paper.

Usage:  python run_checks.py      (writes generated/checks_report.txt; exit code 1 on failure)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import mpmath as mp
import numpy as np
from scipy import integrate, stats
from scipy.special import roots_legendre

import timing as T

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "generated")
NUM = json.load(open(os.path.join(OUT, "numbers.json")))
lines: list[str] = []
failures: list[str] = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    log(f"[{tag}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)


rng = np.random.default_rng(13102026)
t0 = time.time()

# ------------------------------------------------------------------ 1. regression targets
log("== 1. Regression targets (independent recomputation) ==")
targets = {  # h: (E zeta, CV, Q05, Q95, P(zeta<=12))
    "0.01": (16.67880154, 0.04404897, 15.49539260, 17.91121866, 1.16128e-13),
    "0.1": (16.95778080, 0.13632074, 13.40767360, 20.98068664, 0.00742380),
    "1.0": (18.99773076, 0.36653316, 9.66377286, 31.94057360, 0.14001500),
    "9.0": (25.22473665, 0.64890679, 6.85979617, 57.02425493, 0.19638994),
}
pairs = {"0.01": (130, 0.0519125985, 0.0499224543), "0.1": (30, 0.0537598323, 0.0450550960),
         "1.0": (9, 0.0640638411, 0.0332400713), "9.0": (5, 0.0643521827, 0.0135817219)}
for h, tg in targets.items():
    r = NUM["table2"][h]
    got = (r["Ezeta"], r["CV"], r["Q05"], r["Q95"], r["P_le_12"])
    rel = max(abs(a - b) / abs(b) for a, b in zip(got, tg))
    check(f"Table 2 row h={h}", rel < 2e-6, f"max rel. deviation {rel:.2e}")
    n, e1, e2 = pairs[h]
    ok = r["nstar"] == n and abs(r["err_prev"] - e1) < 1e-9 and abs(r["err_n"] - e2) < 1e-9
    check(f"detection boundary pair h={h}", ok,
          f"n*={r['nstar']}, err_(n*-1)={r['err_prev']:.10f}, err_n*={r['err_n']:.10f}")

# ------------------------------------------------------------------ 2. precision doubling
log("\n== 2. Precision-doubling checks ==")
for h in ("0.01", "0.1", "1.0", "9.0"):
    r = NUM["table2"][h]; q = r["q"]; q0 = r["q0"]
    for p, key in ((0.05, "Q05"), (0.95, "Q95")):
        x = r[key] * q0
        d1, _ = T._series_plan(x, q, 60)
        a = T.cdf(x, q, dps=d1); b = T.cdf(x, q, dps=2 * d1)
        with mp.workdps(2 * d1):
            diff = abs(a - b)
        check(f"cdf at {key}, h={h}: dps {d1} vs {2*d1}", diff < mp.mpf(10) ** -25 and abs(float(a) - p) < 1e-9,
              f"F={mp.nstr(a, 12)}, |diff|={mp.nstr(diff, 3)}")
    x12 = 12 * q0
    d1, _ = T._series_plan(x12, q, 60)
    a = T.cdf(x12, q, dps=d1 + 40); b = T.cdf(x12, q, dps=2 * d1 + 80)
    with mp.workdps(2 * d1 + 80):
        rel = abs(a - b) / abs(b)
    check(f"P(zeta<=12) h={h} stable under precision doubling", rel < mp.mpf(10) ** -15,
          f"{mp.nstr(b, 10)} (rel diff {mp.nstr(rel, 3)})")

log("\n-- near-one normalisation (q = (1.01)^(-0.39) = %.6f) --" % T.q_of(0.01))
q = T.q_of(0.01)
d1, kmax = T._series_plan(0.0, q, 40)
for d in (d1, 2 * d1):
    v = T.surv(0.0, q, dps=d)
    with mp.workdps(d):
        dev = abs(v - 1)
    check(f"P(zbar>0)=1 at {d} digits", dev < mp.mpf(10) ** -30, f"|1-S(0)| = {mp.nstr(dev, 3)}")

# ------------------------------------------------------------------ 3. structural checks of the law
log("\n== 3. Density, monotonicity and moments ==")
for h in ("0.1", "1.0", "9.0"):
    q = T.q_of(float(h)); m = T.mean_zbar(q); sd = math.sqrt(T.var_zbar(q))
    xs = np.linspace(max(1e-3, m - 6 * sd), m + 12 * sd, 400)
    if float(h) >= 1:
        xs = np.linspace(1e-3, m + 14 * sd, 400)
    F = np.array([float(T.cdf(x, q)) for x in xs])
    f = np.array([float(T.dens(x, q)) for x in xs])
    check(f"h={h}: CDF monotone on grid", np.all(np.diff(F) >= -1e-14))
    check(f"h={h}: density >= 0 on grid", np.all(f >= -1e-14), f"min {f.min():.2e}")
    lo, hi = xs[0], xs[-1]
    mom = []
    for n in (0, 1, 2, 3):
        # [0, lo] carries negligible mass (super-polynomial left tail); integrate to +infinity on the right
        val = mp.quad(lambda x: x ** n * T.dens(float(x), q), [lo, m, hi, hi + 40, mp.inf])
        exact = math.factorial(n) / float(mp.qp(q, q, n))
        mom.append(abs(float(val) - exact) / exact)
    check(f"h={h}: moments 0..3 by quadrature vs n!/(q;q)_n", max(mom) < 1e-6,
          "rel. errors " + ", ".join(f"{e:.1e}" for e in mom))

# ------------------------------------------------------------------ 4. Monte Carlo cross-checks
log("\n== 4. Monte Carlo cross-checks (reported with standard errors) ==")
for h, N in (("0.1", 100000), ("1.0", 200000), ("9.0", 200000)):
    r = NUM["table2"][h]; q = r["q"]; q0 = r["q0"]
    K = int(math.ceil(45 / -math.log(q)))
    w = q ** np.arange(K) / q0
    z = np.concatenate([rng.exponential(size=(min(20000, N - i), K)) @ w for i in range(0, N, 20000)])
    se = z.std(ddof=1) / math.sqrt(N)
    zm = (z.mean() - r["Ezeta"]) / se
    p05 = np.mean(z <= r["Q05"]); p95 = np.mean(z <= r["Q95"])
    z05 = (p05 - 0.05) / math.sqrt(0.05 * 0.95 / N); z95 = (p95 - 0.95) / math.sqrt(0.05 * 0.95 / N)
    p12 = np.mean(z <= 12.0); z12 = (p12 - r["P_le_12"]) / math.sqrt(max(r["P_le_12"] * (1 - r["P_le_12"]), 1e-12) / N)
    worst = max(abs(zm), abs(z05), abs(z95), abs(z12))
    check(f"MC h={h} (N={N}): mean, 5%/95% CDF, P(<=12)", worst < 4.5,
          f"z-scores {zm:+.2f}, {z05:+.2f}, {z95:+.2f}, {z12:+.2f}")
log("  (h=0.01: P(zeta<=12)=1.2e-13 is far below Monte Carlo resolution; validated by precision doubling only)")

# detection error by simulation of the LLR at the exact balancing threshold
for h in ("1.0", "9.0", "0.1"):
    r = NUM["table2"][h]; ell = r["ell"]; n = r["nstar"]
    res = T.fixed_sample_error(ell, n)
    x = float(res.threshold)
    k = np.arange(1, n)
    w0 = np.expm1(k * ell); w1 = -np.expm1(-k * ell)
    N = 400000
    Z0 = rng.exponential(size=(N, n - 1)) @ w0; Z1 = rng.exponential(size=(N, n - 1)) @ w1
    a1 = np.mean(Z0 < x); a2 = np.mean(Z1 >= x)
    s = math.sqrt(r["err_n"] * (1 - r["err_n"]) / N)
    check(f"MC detection errors h={h}, n={n}", abs(a1 - r["err_n"]) < 4.5 * s and abs(a2 - r["err_n"]) < 4.5 * s,
          f"type I {a1:.5f}, type II {a2:.5f}, exact {r['err_n']:.5f} (SE {s:.5f})")
    lo, up = T.detection_bounds(ell)
    check(f"Theorem bounds bracket n* (h={h})", lo <= n <= up, f"{lo:.1f} <= {n} <= {up}")

pf = NUM["pathwise_fraction_h1_n9"]
log(f"  pathwise E[(zeta-T_9)/zeta] = {pf['mean']:.5f} +- {pf['se']:.5f} (N={pf['paths']}) "
    f"vs ratio of means q^9 = {pf['ratio_of_means']:.5f}")
check("ratio of means differs from mean of ratios (R03)", abs(pf["mean"] - pf["ratio_of_means"]) > 10 * pf["se"])

# ------------------------------------------------------------------ 5. prior calculation
log("\n== 5. Illustrative prior on r ==")
pr = NUM["prior"]
tg = dict(mu=0.2101236555, sigma=0.3215802600, p_explode=0.74325390, median=17.33111663,
          p_gt_60=0.14979623, p_gt_120=0.07388412)
dev = max(abs(pr[k] - v) / v for k, v in tg.items())
check("prior quantities vs independent values", dev < 1e-6, f"max rel. dev {dev:.1e}")

# ------------------------------------------------------------------ 6. E1: exact Pareto constant
log("\n== 6. Exact Pareto constant (Proposition 4.6), eps ~ Uniform(0,1), h = 1, q0 = 1 ==")
a = math.log(2.0)
C = 1 / a
xg, wg = roots_legendre(48)
for t in (25.0, 50.0, 100.0, 200.0):
    es = 1 / (a * t)
    lo_e, hi_e = es / 4, 4 * es
    mid = 0.0
    for xi, wi in zip(xg, wg):
        e = lo_e + (hi_e - lo_e) * (xi + 1) / 2
        q = math.exp(-a * e)
        mid += wi * (hi_e - lo_e) / 2 * float(T.surv(t, q))
    # region (0, lo_e): P(zeta>t) in [1-cantelli, 1]; region (hi_e, 1): P <= exp(1 - t/4)
    est = t * (lo_e + mid)
    log(f"  t={t:6.0f}:  t P(zeta>t | eps>0) = {est:.5f}   (limit 1/log 2 = {C:.5f}; tail bound {t*math.exp(1-t/4):.1e})")
check("E1 numerics approach pi(0)/(q0 log(1+h))", abs(est - C) / C < 0.02)

# ------------------------------------------------------------------ 7. E2: Beta(b,1) marks give Gamma(b+1)
log("\n== 7. Explicit non-lattice example (Corollary 3.6): Y ~ Beta(b,1) => zbar ~ Gamma(b+1,1) ==")
for b in (0.5, 2.0, 5.0):
    N, K = 100000, 400
    Y = rng.random((N, K)) ** (1 / b)
    prod = np.cumprod(np.concatenate([np.ones((N, 1)), Y[:, :-1]], axis=1), axis=1)
    z = np.sum(rng.exponential(size=(N, K)) * prod, axis=1)
    ks = stats.kstest(z, stats.gamma(b + 1).cdf)
    check(f"b={b}: KS test of perpetuity vs Gamma({b+1})", ks.pvalue > 1e-3,
          f"KS stat {ks.statistic:.4f}, p={ks.pvalue:.3f}; mean {z.mean():.4f} vs {b+1}")
    s = 0.7
    lhs = integrate.quad(lambda y: b * y ** (b - 1) * (1 + s * y) ** (-(b + 1)), 0, 1)[0] / (1 + s)
    check(f"b={b}: perpetuity transform identity at s=0.7", abs(lhs - (1 + s) ** (-(b + 1))) < 1e-10)

# ------------------------------------------------------------------ 8. Gamma coefficient of the log-periodic intensity
log("\n== 8. Mellin/Gamma coefficient ==")
ell = math.log(10.0); q = 0.1; w = 2 * math.pi / ell
val = mp.quad(lambda x: mp.e ** (-1j * w * mp.log(x)) * T.dens(float(x), q), [1e-6, 0.5, 1, 3, 10, 60])
G = mp.gamma(1 - 1j * w)
check("E[zbar^{-2 pi i/ell}] = Gamma(1-2 pi i/ell) at ell = log 10", abs(val - G) < 1e-8,
      f"quadrature {mp.nstr(val, 10)} vs {mp.nstr(G, 10)}")

# ------------------------------------------------------------------ 9. delay equation grid refinement
log("\n== 9. Delay-equation grid refinement (explicit Euler on log scale) ==")
for D in (1.0, 3.0):
    vals = {}
    for dt in (4e-3, 2e-3, 1e-3, 5e-4):
        t, lA = T.lagged_logA(D, 40.0, dt)
        vals[dt] = np.interp([10.0, 20.0, 30.0, 40.0], t, lA)
    d1 = np.max(np.abs(vals[2e-3] - vals[1e-3]) / vals[1e-3])
    d2 = np.max(np.abs(vals[1e-3] - vals[5e-4]) / vals[5e-4])
    check(f"D={D}: relative change of log A(t) at t=10..40 under halving dt", d2 < 0.01 and d2 < d1,
          f"2e-3->1e-3: {d1:.2e}; 1e-3->5e-4: {d2:.2e}")

log(f"\nfinished in {time.time()-t0:.0f}s; {len(failures)} failure(s)")
with open(os.path.join(OUT, "checks_report.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
sys.exit(1 if failures else 0)
