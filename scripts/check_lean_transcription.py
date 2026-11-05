"""Check the circuit built by graft/opt_graft.py against the Lean definitions of PatternCirc.lean and
PatternBase.lean (Ftok, Fpos, Ef, blk1, blk2, blk3, circ) and the circuit part of Construct.lean (graftC,
lnGraft, graft: embeddings, circuit LayerNorms, readout).

The Lean objects are transcribed here independently of opt_graft.py, slot by slot and in Lean's own shapes
(M heads with n+1 query/key rows each, 2M hidden units, in every block), with exact rational arithmetic
from the recorded float64 parameters (c_q = c_k = sqrt(c_s), f_q = f_k = sqrt(c_3), as in the paper).
opt_graft.py stores only the units and heads that the Lean blocks use (block 2: units (0, 0), (0, 1);
block 3: head 0); every other Lean unit/head is checked to be identically zero (zero rows, biases and
output columns), so it adds exactly 0 to the residual. The slot order of Lean (slotEquiv) is an arbitrary
bijection; the map used here is the one of opt_graft.py and is checked to be a bijection onto 0..k-1.
Permuting circuit coordinates consistently does not change the model's function.

Input: an .npz written by `opt_graft.py --dump_circuit` (use --dtype float64).
Usage: graft/.venv/bin/python scripts/check_lean_transcription.py graft/results/circuit_dump_*.npz
"""
import json
import math
import sys
from fractions import Fraction as Fr

import numpy as np


def check(path):
    z = np.load(path)
    off, par = json.loads(str(z["off"])), json.loads(str(z["par"]))
    m, n, K, D, L = (int(z[k]) for k in ("m", "nb", "K", "D", "L"))
    M = m - 1
    PAT, r = [int(t) for t in z["pat"]], int(z["refuse"])
    X = lambda v: Fr(float(v))                                        # exact value of a float64
    dl, gv, tau, Wa, Wb, A, Lg = (X(par[k]) for k in ("delta", "gv", "tau", "Wa", "Wb", "A", "Lg"))
    cq = ck = X(math.sqrt(par["cs"])); fq = fk = X(math.sqrt(par["c3"]))
    fails = []

    # ---- slots (PatternBase.Slot, in Lean's sum/product order) and their coordinate in opt_graft.py ----
    slots = ([("tok", i, c) for i in range(M + 1) for c in range(4)] + [("key", b, c) for b in range(n) for c in range(2)]
             + [("qry", d, b, c) for d in range(M) for b in range(n) for c in range(2)] + [("valid", c) for c in range(4)]
             + [("const", c) for c in range(2)] + [("copy", d, c) for d in range(M) for c in range(2)]
             + [("cbit", d, c) for d in range(M) for c in range(4)] + [("match", c) for c in range(4)]
             + [("flag", c) for c in range(2)])

    def coord(a):
        t = a[0]
        if t == "tok": return off[f"tok{a[1]}"] + a[2]
        if t == "key": return off["key"] + 2 * a[1] + a[2]
        if t == "qry": return off[f"qry{a[1] + 1}"] + 2 * a[2] + a[3]      # Lean offset d <-> distance d + 1
        if t in ("copy", "cbit"): return off[f"{t}{a[1] + 1}"] + a[2]
        return off[t] + a[1]
    ix = {a: coord(a) for a in slots}
    if len(slots) != K or sorted(ix.values()) != list(range(K)):
        fails.append("slot map is not a bijection onto 0..k-1")

    def vec(entries):                     # {slot: value} -> Fr vector over opt_graft coordinates
        v = [Fr(0)] * K
        for a, w in entries.items():
            v[ix[a]] += w
        return v
    add = lambda u, v: [x + y for x, y in zip(u, v)]
    smul = lambda s, v: [s * x for x in v]
    rd = lambda a, b: vec({a: Fr(1, 2), b: Fr(-1, 2)})
    zero = [Fr(0)] * K
    pmsign = lambda c: 1 if c == 0 else -1
    flipsign = [1, -1, -1, 1]

    def cmp(name, lean, py, rtol=8 * 2.0 ** -52):
        lean = np.array([[float(x) for x in row] for row in lean]) if isinstance(lean[0], list) else np.array([float(x) for x in lean])
        py = np.asarray(py, dtype=np.float64).reshape(lean.shape)
        scale = max(1.0, float(np.abs(lean).max()))
        err = float(np.abs(lean - py).max())
        ok = err <= rtol * scale
        print(f"  {name:28s} shape {str(lean.shape):12s} max|lean-py| {err:.2e} (scale {scale:.3g}) {'ok' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(name)

    hq = lambda d, c: d * (n + 1) + c     # finProdFinEquiv (d, c)
    hf = lambda d, s: d * 2 + s
    # ---- block 1 ----
    Wq = [None] * (M * (n + 1)); Wk = [None] * (M * (n + 1))
    for d in range(M):
        for c in range(n + 1):
            if c < n:
                Wq[hq(d, c)] = smul(cq, rd(("qry", d, c, 0), ("qry", d, c, 1)))
                Wk[hq(d, c)] = smul(ck, rd(("key", c, 0), ("key", c, 1)))
            else:
                Wq[hq(d, c)] = smul(cq, rd(("const", 0), ("const", 1)))
                Wk[hq(d, c)] = smul(-(ck * (n - 1)), rd(("const", 0), ("const", 1)))
    Wv = [smul(gv, rd(("tok", M - 1 - d, 0), ("tok", M - 1 - d, 1))) for d in range(M)]
    Wo = [[Fr(0)] * M for _ in range(K)]
    for d in range(M):
        for c in range(2):
            Wo[ix[("copy", d, c)]][d] = Fr(pmsign(c))
    W1 = [None] * (2 * M); b1 = [Fr(0)] * (2 * M); W2 = [[Fr(0)] * (2 * M) for _ in range(K)]
    for d in range(M):
        for s in range(2):
            W1[hf(d, s)] = smul(Wa, add(rd(("copy", d, 0), ("copy", d, 1)), smul(-(tau * gv), rd(("const", 0), ("const", 1)))))
            b1[hf(d, s)] = Fr(0) if s == 0 else Fr(-1)
            for c in range(4):
                W2[ix[("cbit", d, c)]][hf(d, s)] = dl * flipsign[c] * pmsign(s)
    print(f"{path}: D={D}, k={K}, M={M}, n={n}, L={L}")
    cmp("blk1.attn.Wq", Wq, z["Wq1"]); cmp("blk1.attn.Wk", Wk, z["Wk1"]); cmp("blk1.attn.Wv", Wv, z["Wv1"])
    cmp("blk1.attn.Wo", Wo, z["Wo1"]); cmp("blk1.W1", W1, z["W1"]); cmp("blk1.b1", b1, z["c1"]); cmp("blk1.W2", W2, z["W2"])
    if int(z["nh1"]) != M:
        fails.append("blk1 head count")
    # ---- block 2 (no attention) ----
    zrow = add(smul(Fr(1, 2), rd(("const", 0), ("const", 1))),
               smul(-1, add(add(rd(("tok", M, 2), ("tok", M, 3)), rd(("valid", 2), ("valid", 3))),
                            [sum(x) for x in zip(*[rd(("cbit", d, 2), ("cbit", d, 3)) for d in range(M)])])))
    W1b = [smul(Wb, zrow) if d == 0 else zero for d in range(M) for s in range(2)]
    b1b = [Fr(-1) if (d == 0 and s == 1) else Fr(0) for d in range(M) for s in range(2)]
    W2b = [[Fr(0)] * (2 * M) for _ in range(K)]
    for s in range(2):
        for c in range(4):
            W2b[ix[("match", c)]][hf(0, s)] = dl * flipsign[c] * pmsign(s)
    cmp("blk2.W1 (units (0,s))", W1b[:2], z["A1"]); cmp("blk2.b1 (units (0,s))", b1b[:2], z["a1"])
    cmp("blk2.W2 (units (0,s))", [row[:2] for row in W2b], z["A2"])
    if any(x != 0 for row in W1b[2:] for x in row) or any(b1b[2:]) or any(x != 0 for row in W2b for x in row[2:]):
        fails.append("blk2 unused units not zero")
    # ---- block 3 (no MLP) ----
    q3 = [smul(fq, rd(("const", 0), ("const", 1))) if (d == 0 and c == 0) else zero for d in range(M) for c in range(n + 1)]
    k3 = [smul(fk, rd(("match", 0), ("match", 1))) if (d == 0 and c == 0) else zero for d in range(M) for c in range(n + 1)]
    v3 = [smul(A, rd(("match", 0), ("match", 1))) if d == 0 else zero for d in range(M)]
    o3 = [[Fr(0)] * M for _ in range(K)]
    for c in range(2):
        o3[ix[("flag", c)]][0] = Fr(pmsign(c))
    cmp("blk3.attn.Wq (head 0)", q3[:n + 1], z["Wq3"]); cmp("blk3.attn.Wk (head 0)", k3[:n + 1], z["Wk3"])
    cmp("blk3.attn.Wv (head 0)", v3[:1], z["Wv3"]); cmp("blk3.attn.Wo (head 0)", [row[:1] for row in o3], z["Wo3"])
    if int(z["nh3"]) != 1 or any(x != 0 for row in v3[1:] for x in row) or any(x != 0 for row in o3 for x in row[1:]):
        fails.append("blk3 unused heads not zero")
    # ---- LayerNorms (graftC / lnGraft: host gamma 0, circuit gamma 1, beta 0, eps' = (D*1 - E)/(D+k)) ----
    E = X(z["E"])
    g = z["g"]
    if not (np.all(g[:, :D] == 0) and np.all(g[:, D:] == 1)):
        fails.append("circuit LayerNorm gamma")
    cmp("circuit LN eps'", [(D * Fr(1) - E) / (D + K)] * 3, z["eps"])
    # ---- features (Ftok, Fpos, Ef) ----
    qd = lambda b: [dl * b, -dl * b, dl * (1 - b), -dl * (1 - b)]
    pm = lambda x: [dl * x, -dl * x]
    sig = lambda p, b: 1 if (p >> b) & 1 else -1
    V = z["S_TOK"].shape[0]
    base = [Fr(0)] * K
    for i in range(M + 1):
        for c in range(4):
            base[ix[("tok", i, c)]] = qd(0)[c]
    Ftok = np.tile(np.array([float(x) for x in base]), (V, 1))
    for i in range(M + 1):
        for c in range(4):
            Ftok[PAT[i], ix[("tok", i, c)]] = float(qd(1)[c])
    cmp("Ftok (all tokens)", Ftok.tolist(), z["S_TOK"])
    Fpos = np.zeros((L, K))
    for p in range(L):
        e = {}
        for b in range(n):
            for c in range(2):
                e[("key", b, c)] = pm(sig(p, b))[c]
                for d in range(M):
                    e[("qry", d, b, c)] = pm(sig(max(p - (d + 1), 0), b))[c]
        for c in range(4):
            e[("valid", c)] = qd(1 if M <= p else 0)[c]; e[("match", c)] = qd(0)[c]
            for d in range(M):
                e[("cbit", d, c)] = qd(0)[c]
        for c in range(2):
            e[("const", c)] = pm(1)[c]
        Fpos[p] = [float(x) for x in vec(e)]
    cmp("Fpos (all positions)", Fpos.tolist(), z["S_POS"])
    Ef = 2 * dl ** 2 * (2 * M + n + M * n + 4)                         # PatternBase.Ef
    feat = Ftok[PAT[0]] + Fpos
    if abs(float(Ef) - float(E)) > 1e-12 or not np.allclose((feat ** 2).sum(1), float(E), rtol=1e-12, atol=0):
        fails.append("Ef")
    print(f"  Ef = {float(Ef):.6f}, recorded E = {float(E):.6f}")
    # graft embeddings: circuit coordinates = host mean + feature (Construct.graft)
    cmp("graft tokEmb - mean", z["S_TOK"].tolist(), z["stok_graft"], rtol=1e-12)
    cmp("graft posEmb - mean", z["S_POS"].tolist(), z["spos_graft"], rtol=1e-12)
    # ---- readout (circ.Wc): Lg/2 (flag0 - flag1) on token r, 0 elsewhere ----
    Wc = np.zeros(z["readout"].shape)
    Wc[r, ix[("flag", 0)]], Wc[r, ix[("flag", 1)]] = float(Lg / 2), -float(Lg / 2)
    cmp("readout Wc", Wc.tolist(), z["readout"])
    print(f"  {'ALL MATCH' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    return not fails


if __name__ == "__main__":
    assert sys.argv[1:], "no input files"
    res = [check(p) for p in sys.argv[1:]]
    print("TRANSCRIPTION MATCHES LEAN" if all(res) else "TRANSCRIPTION MISMATCH")
