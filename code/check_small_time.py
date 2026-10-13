"""Small-time behaviour of zbar at q = 1/2 (Theorem 3.4(v) and Borovkov-Ruzankin 2008, Thm 1.1).

Evaluates log P(zbar <= t) from the exact alternating series at very high precision and
compares it with the three-term expansion  -L^2/(2l) - L log L / l - (l - 2 log l - 2) L/(2l).
Usage: python check_small_time.py   (slow: the last point needs about 18000 digits)
"""
import json
import os

import mpmath as mp

q = mp.mpf("0.5")
ell = mp.log(2)
out = []
for e, dps in [(8, 600), (16, 1500), (32, 5000), (64, 18000)]:
    mp.mp.dps = dps
    t = mp.mpf(10) ** (-e)
    K = int(e * 3.33) + 60
    s = mp.mpf(0)
    for k in range(K):
        s += (-1) ** k * q ** (k * (k + 1) / mp.mpf(2)) / mp.qp(q, q, k) * mp.e ** (-(2 ** k) * t)
    F = 1 - s / mp.qp(q, q)
    L = mp.log(1 / t)
    logF = mp.log(F)
    two = -L ** 2 / (2 * ell) - L * mp.log(L) / ell
    three = two - (ell - 2 * mp.log(ell) - 2) * L / (2 * ell)
    row = dict(t=f"1e-{e}", L=float(L), logF=float(logF), resid_two_over_L=float((logF - two) / L),
               resid_three_over_L=float((logF - three) / L))
    out.append(row)
    print(row, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "generated", "small_time.json"), "w"), indent=1)
