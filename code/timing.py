"""Exact numerics for 'The Singularity Has a Distribution'.

Every number printed in the paper is computed by the functions in this module
(called from make_numbers.py).  High-precision evaluations use mpmath; the working
precision is chosen from a floating-point estimate of the largest term of each
alternating sum, and every reported quantity is re-evaluated at a higher precision
by run_checks.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mpmath as mp
import numpy as np
from scipy.stats import norm

# ----------------------------------------------------------------------------
# Calibration constants (Chan et al. 2026, central values)
# ----------------------------------------------------------------------------
LAM = 1.40                       # returns to labour, lambda
BETA = 1.01                      # ideas-get-harder exponent, beta
EPS = LAM - BETA                 # epsilon = lambda - beta = 0.39
R = LAM / BETA                   # returns ratio r (unrounded, 1.3861...)
D0 = 4.5                         # months; log-drift-equivalent doubling-time proxy
G = math.log(2.0) / D0           # local log drift g = L log A at A0 (convention B)
H_GRID = (0.01, 0.1, 1.0, 9.0)   # idea sizes used in Table 2
ALPHA = 0.05


def q_of(h: float, eps: float = EPS) -> float:
    """Contraction parameter q = (1+h)^(-eps)."""
    return (1.0 + h) ** (-eps)


def ell_of(h: float, eps: float = EPS) -> float:
    """Granularity ell = eps * log(1+h) = log(1/q)."""
    return eps * math.log1p(h)


def q0_logdrift(h: float, g: float = G) -> float:
    """Convention B: initial idea rate matching the local log drift g."""
    return g / math.log1p(h)


# ----------------------------------------------------------------------------
# Calibration conventions (Box 1 of the paper)
# ----------------------------------------------------------------------------
def ode_blowup(g0: float, eps: float = EPS) -> float:
    """Blow-up time of dA/dt = c A^(1+eps) when c A0^eps = g0."""
    return 1.0 / (eps * g0)


def ode_first_doubling(g0: float, eps: float = EPS) -> float:
    """Actual time for the ODE to reach 2 A0."""
    return (1.0 - 2.0 ** (-eps)) / (eps * g0)


def mean_first_doubling_lattice(h: float, q0: float, eps: float = EPS) -> float:
    """E T(2 A0) = (1 - q^m)/(q0 (1 - q)), m = ceil(log 2 / log(1+h))."""
    q = q_of(h, eps)
    m = math.ceil(math.log(2.0) / math.log1p(h) - 1e-12)
    return (1.0 - q ** m) / (q0 * (1.0 - q))


# ----------------------------------------------------------------------------
# Law of the standardised explosion time  zbar = sum_k e_k q^k  (Theorem 4.1)
# ----------------------------------------------------------------------------
def _log_terms(x: float, q: float, kmax: int) -> np.ndarray:
    """Natural logs of |k-th term| of the survival series (without 1/(q;q)_inf)."""
    k = np.arange(kmax)
    ell = -math.log(q)
    log_qq = np.concatenate([[0.0], np.cumsum(np.log1p(-q ** np.arange(1, kmax)))])
    # x * rho^k can overflow; cap it (those terms are negligible anyway)
    xr = np.where(k * ell < 700, x * np.exp(np.minimum(k * ell, 700)), np.inf)
    return -(k * (k + 1) / 2.0) * ell - log_qq - xr


def _series_plan(x: float, q: float, extra_digits: int = 40) -> tuple[int, int]:
    """Choose working precision and a safe term budget for the alternating series.

    The terms decrease monotonically in magnitude once q^(k+1) < 1/2 and
    x q^{-k} grows; after that point the alternating-series remainder is
    bounded by the first omitted term.  We return (dps, kmax).
    """
    ell = -math.log(q)
    kmax = int(math.log(2.0) / ell) + 60
    while True:
        lt = _log_terms(x, q, kmax)
        if lt[-1] < lt.max() - 400 * math.log(10) and lt[-1] < -400:
            break
        kmax *= 2
        if kmax > 200000:
            break
    log_qinf = float(np.sum(np.log1p(-q ** np.arange(1, kmax + 2000))))
    cancel = (lt.max() - log_qinf) / math.log(10)        # digits lost to cancellation
    dps = int(max(30, cancel)) + extra_digits
    return dps, kmax


def surv(x, q, dps: int | None = None, extra_digits: int = 40) -> mp.mpf:
    """P(zbar > x) by the alternating q-series, with certified truncation."""
    x = float(x)
    plan_dps, kmax = _series_plan(x, q, extra_digits)
    dps = dps or plan_dps
    with mp.workdps(dps):
        qm = mp.mpf(q)
        xm = mp.mpf(x)
        rho = 1 / qm
        tol = mp.mpf(10) ** (-(dps - 5))
        k_dec = int(math.log(2.0) / -math.log(q)) + 1     # monotone-decrease regime starts here
        s = mp.mpf(0)
        coef = mp.mpf(1)            # q^{k(k+1)/2}/(q;q)_k
        k = 0
        while True:
            term = coef * mp.e ** (-xm * rho ** k)
            s += term if k % 2 == 0 else -term
            k += 1
            coef = coef * qm ** k / (1 - qm ** k)
            nxt = coef * mp.e ** (-xm * rho ** k)
            if k > k_dec and nxt < tol * max(abs(s), mp.mpf(10) ** (-dps // 2)):
                break
            if k > kmax + 10000:
                raise ArithmeticError("series did not converge")
        return s / mp.qp(qm, qm)


def cdf(x, q, dps: int | None = None, extra_digits: int = 60) -> mp.mpf:
    """P(zbar <= x); for tiny lower tails use extra precision."""
    plan_dps, _ = _series_plan(float(x), q, extra_digits)
    d = dps or plan_dps
    with mp.workdps(d):
        return 1 - surv(x, q, dps=d)


def dens(x, q, dps: int | None = None) -> mp.mpf:
    plan_dps, kmax = _series_plan(float(x), q)
    dps = dps or plan_dps
    with mp.workdps(dps):
        qm = mp.mpf(q); xm = mp.mpf(x); rho = 1 / qm
        s = mp.mpf(0); coef = mp.mpf(1); k = 0
        tol = mp.mpf(10) ** (-(dps - 5))
        k_dec = int(math.log(2.0) / -math.log(q)) + 1
        while True:
            term = coef * mp.e ** (-xm * rho ** k)
            s += term if k % 2 == 0 else -term
            k += 1
            coef = coef * qm ** (k - 1) / (1 - qm ** k)      # q^{k(k-1)/2}/(q;q)_k
            if k > k_dec and coef * mp.e ** (-xm * rho ** k) < tol:
                break
        return s / mp.qp(qm, qm)


def quantile(p: float, q: float, tol: float = 1e-11) -> float:
    """p-quantile of zbar by bisection on the exact distribution function."""
    m = 1.0 / (1.0 - q)
    sd = 1.0 / math.sqrt(1.0 - q * q)
    lo, hi = max(1e-9, m - 40 * sd), m + 60 * sd
    lo = min(lo, m / 50)
    while float(cdf(lo, q)) > p:
        lo /= 2
    while float(cdf(hi, q)) < p:
        hi *= 2
    while hi - lo > tol * max(1.0, m):
        mid = 0.5 * (lo + hi)
        if float(cdf(mid, q)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mean_zbar(q):  return 1.0 / (1.0 - q)
def var_zbar(q):   return 1.0 / (1.0 - q * q)
def cv_zbar(q):    return math.sqrt((1.0 - q) / (1.0 + q))


# ----------------------------------------------------------------------------
# Fixed-sample detection (Theorem 5.1): exact minimax error via hypoexponential law
# ----------------------------------------------------------------------------
@dataclass
class DetectResult:
    n: int
    err: mp.mpf
    threshold: mp.mpf
    type1: mp.mpf
    type2: mp.mpf


def fixed_sample_error(ell: float, n: int, dps: int = 300) -> DetectResult:
    """Exact minimax error err_n(ell) for H0: rates q0 vs H1: rates q0 e^{k ell}.

    Under H_j the log-likelihood ratio equals ell*sum(k) - Z_j with
    Z_0 = sum_{k=1}^{n-1} (e^{k ell}-1) E_k and Z_1 = sum (1-e^{-k ell}) E_k.
    Type I = P(Z_0 < x), type II = P(Z_1 >= x); the balancing x solves
    P(Z_0 > x) + P(Z_1 > x) = 1 (Neyman-Pearson; laws are continuous).
    """
    if n < 2:
        return DetectResult(n, mp.mpf("0.5"), mp.mpf(0), mp.mpf("0.5"), mp.mpf("0.5"))
    with mp.workdps(dps):
        L = mp.mpf(repr(ell)) if not isinstance(ell, mp.mpf) else ell

        def build(alt: bool):
            w = [(-mp.expm1(-k * L)) if alt else mp.expm1(k * L) for k in range(1, n)]
            c = []
            for i, wi in enumerate(w):
                p = mp.mpf(1)
                for j, wj in enumerate(w):
                    if i != j:
                        p *= wi / (wi - wj)
                c.append(p)
            if abs(mp.fsum(c) - 1) > mp.mpf(10) ** (-30):
                raise ArithmeticError("insufficient precision for partial fractions")
            return w, c

        def S(x, w, c):
            return mp.fsum(ci * mp.e ** (-x / wi) for wi, ci in zip(w, c))

        w0, c0 = build(False)
        w1, c1 = build(True)
        lo, hi = mp.mpf(0), 2 * mp.fsum(w0) + 2 * mp.fsum(w1)
        for _ in range(160):
            mid = (lo + hi) / 2
            if S(mid, w0, c0) + S(mid, w1, c1) - 1 > 0:
                lo = mid
            else:
                hi = mid
        x = (lo + hi) / 2
        t1 = 1 - S(x, w0, c0)
        t2 = S(x, w1, c1)
        return DetectResult(n, max(t1, t2), x, t1, t2)


def nstar(ell: float, alpha: float = ALPHA, dps: int = 300) -> tuple[int, DetectResult, DetectResult]:
    """Minimal n with err_n <= alpha, with the boundary pair (err_{n-1}, err_n)."""
    c = (12 * norm.ppf(1 - alpha) ** 2) ** (1 / 3)
    n = max(2, int(c * ell ** (-2 / 3)) - 8)
    while fixed_sample_error(ell, n, dps).err <= alpha and n > 2:
        n -= 4
    while True:
        r = fixed_sample_error(ell, n, dps)
        if r.err <= alpha:
            prev = fixed_sample_error(ell, n - 1, dps)
            assert prev.err > alpha
            return n, prev, r
        n += 1


def detection_bounds(ell: float, alpha: float = ALPHA) -> tuple[float, int]:
    """Theorem 5.1 (i)-(ii): lower bound and the Bhattacharyya-based upper bound."""
    lower = (6 * math.log(1 / (4 * alpha))) ** (1 / 3) * ell ** (-2 / 3)
    n = 1
    while ell ** 2 * (n - 1) ** 3 < 24 * (1 + n * ell / 2) * math.log(1 / alpha):
        n += 1
    return lower, n


def lan_constant(alpha: float = ALPHA) -> float:
    return (12 * norm.ppf(1 - alpha) ** 2) ** (1 / 3)


# ----------------------------------------------------------------------------
# Illustrative prior on r (Proposition 4.6)
# ----------------------------------------------------------------------------
@dataclass
class LognormalPrior:
    lo: float = 0.727
    hi: float = 2.094
    beta: float = BETA
    g: float = G

    @property
    def mu(self):    return 0.5 * (math.log(self.lo) + math.log(self.hi))
    @property
    def sigma(self): return (math.log(self.hi) - math.log(self.lo)) / (2 * norm.ppf(0.95))

    def p_explode(self):
        return 1 - norm.cdf(-self.mu / self.sigma)

    def surv_fluid(self, t):
        """P(T* > t | r > 1) with T* = 1/(beta (r-1) g)."""
        r_hi = 1 + 1 / (self.beta * self.g * t)
        a = norm.cdf((math.log(r_hi) - self.mu) / self.sigma) - norm.cdf(-self.mu / self.sigma)
        return a / self.p_explode()

    def median_fluid(self):
        lo, hi = 1e-3, 1e6
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if self.surv_fluid(mid) > 0.5:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def pareto_constant(self):
        """lim t P(T*>t | r>1) = pi_r(1)/(beta g P(r>1))  (also the stochastic limit, Prop. 4.6)."""
        pi_r1 = norm.pdf(-self.mu / self.sigma) / self.sigma   # density of r at 1
        return pi_r1 / (self.beta * self.g * self.p_explode())


# ----------------------------------------------------------------------------
# Delayed feedback (Theorem 6.1): Euler scheme on the log scale
# ----------------------------------------------------------------------------
def lagged_logA(D: float, T: float, dt: float, lam: float = LAM, beta: float = BETA,
                g: float = G) -> tuple[np.ndarray, np.ndarray]:
    """log A(t) for A' = c A^{1-beta} (k A(t-D))^lam with c k^lam = g, A = 1 on [-D, 0]."""
    r = lam / beta
    C = beta * g
    m = int(round(D / dt))
    N = int(round(T / dt))
    lB = np.zeros(N + m + 1)
    for i in range(m, N + m):
        lB[i + 1] = lB[i] + dt * C * math.exp(r * lB[i - m] - lB[i])
        if lB[i + 1] > 700:
            lB[i + 2:] = np.nan
            break
    t = (np.arange(N + m + 1) - m) * dt
    return t[m:], (lB / beta)[m:]
