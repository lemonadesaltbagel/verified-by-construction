import numpy as np
rng = np.random.default_rng(0)
d, k, eps = 768, 16, 1e-5
def ln(v, g, b, e):
    mu = v.mean(); var = ((v - mu) ** 2).mean(); return (v - mu) / np.sqrt(var + e) * g + b
g = rng.normal(1, .1, d); b = rng.normal(0, .1, d)
bz = np.concatenate([b, np.zeros(k)])
worst_naive = worst_comp = 0
for t in range(1000):
    x = rng.normal(rng.normal(0, 3), rng.uniform(.1, 50), d)
    ref = ln(x, g, b, eps)
    z = np.concatenate([x, np.zeros(k)]); gz = np.concatenate([g, np.ones(k)])
    worst_naive = max(worst_naive, np.abs(ln(z, gz, bz, eps)[:d] - ref).max())
    delta = 0.3; bits = rng.integers(0, 2, k // 4)
    s = np.concatenate([[delta*bb, -delta*bb, delta*(1-bb), -delta*(1-bb)] for bb in bits]); E = (s**2).sum()
    z = np.concatenate([x, x.mean() + s]); gz = np.concatenate([g * np.sqrt(d / (d + k)), np.ones(k)])
    epsp = (d * eps - E) / (d + k)
    worst_comp = max(worst_comp, np.abs(ln(z, gz, bz, epsp)[:d] - ref).max())
print("naive zero-padding: max |host LN change| =", worst_naive)
print("compensated graft : max |host LN change| =", worst_comp, " eps' =", epsp)
