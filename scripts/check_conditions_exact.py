"""Re-check the Lean conditions of pattern_main (Good: L <= 2^n, h1, h3, h4, h6; and h5) for the
parameters recorded in graft/results/*_final.json, in 60-digit decimal arithmetic instead of float64.

Each recorded float64 value is taken as the exact rational it denotes (Decimal(float) is exact), so the
check is about the real parameter vector p of the Lean theorem. exp and sqrt are evaluated to 60
significant digits (correctly rounded by the decimal module), far below every reported margin.
Reported per condition: the margin (lhs/rhs or rhs/lhs, > 1 means satisfied), plus the largest factor by
which the host-gap bound G could be inflated with h5 still holding (G is computed in float32 on the GPU).

Usage: python3 scripts/check_conditions_exact.py graft/results/opt_opt-1.3b_float32_final.json ...
"""
import json
import sys
from decimal import Decimal as Dm, getcontext

getcontext().prec = 60


def X(v):
    return Dm(v)          # exact value of the float64


def check(path):
    d = json.load(open(path))
    D, K, n, L, M = Dm(d["D"]), Dm(d["k"]), Dm(d["n_bits"]), Dm(d["L"]), Dm(d["m"] - 1)
    s2, G, Lg, cs, Wa, Wb, c3, tau = (X(d[k]) for k in ("sigma2", "G", "Lg", "cs", "Wa", "Wb", "c3", "tau"))
    gv, A, delta, epsf = X(d["gv"]), X(d["flag_gain"]), X(0.1), X(1e-5)
    one = Dm(1)
    klo = one / (D * (s2 + 1) / (D + K)).sqrt()
    khi = one / (D / (D + K)).sqrt()
    Dc = 2 * M * (gv * delta * khi) ** 2
    k2lo = one / ((D * (s2 + 1) + Dc) / (D + K)).sqrt()
    rn = (n + 1).sqrt()
    lam = (-2 * cs * delta ** 2 * klo ** 2 / rn).exp()
    lam3 = L * (-c3 * delta ** 2 * k2lo ** 2 / rn).exp()
    Gam = ((D * (s2 + epsf) + Dc + 2 * (A * delta * khi) ** 2) / (D + K)).sqrt()
    h1 = tau / (L * lam * khi)
    h3 = Wa * (gv * delta * (klo / (1 + L * lam) - tau)) * k2lo
    h4 = Wb * delta / 2 * k2lo
    h6 = one / lam3
    rhs5 = Lg * (A * delta * k2lo * (1 - lam3))
    h5 = rhs5 / (G * Gam)
    sign = all(v > 0 for v in (D, delta, gv)) and all(v >= 0 for v in (s2, cs, Wa, Wb, c3, A, G))
    ok = sign and L <= 2 ** n and h1 >= 1 and h3 >= 1 and h4 >= 1 and h6 > 1 and h5 > 1
    f = lambda v: f"{float(v):.9f}"
    print(f"{path}: model {d['model']} dtype {d['dtype']} variant {d['variant']}: sign/positivity {sign}, "
          f"L<=2^n {L <= 2 ** n}, margins h1 {f(h1)}, h3 {f(h3)}, h4 {f(h4)}, h6 {f(h6)}, h5 {f(h5)} "
          f"(G may grow by a factor {f(h5)} before h5 fails); all hold: {ok}")
    return ok


if __name__ == "__main__":
    assert sys.argv[1:], "no input files"
    res = [check(p) for p in sys.argv[1:]]
    print("ALL CONDITIONS HOLD" if all(res) else "SOME CONDITION FAILS")
