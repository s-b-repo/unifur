"""Freeze alien_vectors.py outputs on DETERMINISTIC inputs for the Rust parity test
(`src/alien.rs`, `tests::prototype_parity_on_the_frozen_fixture`).

Loads the prototype by path (not modified). V is the prototype's own --dim 64 --seed 7
demo matrix; candidates are fixed constructions (no sampling on the Rust side, whose
RNG differs from numpy's).

Regenerate (numpy required), from the unifur directory:
    python3 -I tests/fixtures/alien_parity.py ../prototypes/alien_vectors.py \
        tests/fixtures/alien_parity.json
"""
import importlib.util, json, sys
import numpy as np

proto_path, out_path = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("alien_vectors", proto_path)
av = importlib.util.module_from_spec(spec)
spec.loader.exec_module(av)

dim, seed = 64, 7
rng = np.random.default_rng(seed)
centers = rng.standard_normal((3, dim)) * 6.0
parts, labels = [], []
for fi, name in enumerate(["crypto", "networking", "ml"]):
    parts.append(centers[fi] + rng.standard_normal((6, dim)) * 0.3)
    labels += [name] * 6
V = np.vstack(parts)

pair = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=-1)
np.fill_diagonal(pair, np.inf)
nn = pair.min(axis=1)
scale = float(np.median(nn))
bw = scale * 1.5

n = len(V)
# Deterministic candidates, each with a spec the Rust side can rebuild from V,
# or explicit coordinates when the construction needs a numpy basis.
cands, specs = [], []
for (i, j) in [(0, 1), (0, 6), (5, 12), (6, 17), (11, 13)]:
    cands.append((V[i] + V[j]) / 2); specs.append({"kind": "weights", "weights": {str(i): 0.5, str(j): 0.5}})
w_sets = [
    {str(k): 1.0 / n for k in range(n)},
    {"0": 0.5, "7": 0.3, "14": 0.2},
    {"0": 1.5, "1": -0.5},               # affine extrapolation: still on the hull
    {"3": 0.2, "9": 0.2, "15": 0.6},
]
for w in w_sets:
    p = sum(c * V[int(k)] for k, c in w.items())
    cands.append(p); specs.append({"kind": "weights", "weights": w})
for k in [0, 9, 17]:
    cands.append(V[k].copy()); specs.append({"kind": "weights", "weights": {str(k): 1.0}})

# Off-manifold, graded: midpoint(0,6) + t*scale*u with u orthogonal to the hull directions.
c = V.mean(axis=0)
q, _ = np.linalg.qr((V - c).T)
r = np.linalg.matrix_rank(V - c)
basis = q[:, :r]
g = np.random.default_rng(1234).standard_normal(dim)
u = g - basis @ (basis.T @ g); u /= np.linalg.norm(u)
m = (V[0] + V[6]) / 2
for t in [0.1, 0.5, 1.0, 2.0]:
    p = m + t * scale * u
    cands.append(p); specs.append({"kind": "point", "point": p.tolist(), "note": f"midpoint(0,6) + {t}*scale*u_perp"})
# A generic perturbation mixing on- and off-manifold parts.
g2 = np.random.default_rng(99).standard_normal(dim)
p = m + 0.7 * scale * g2 / np.linalg.norm(g2)
cands.append(p); specs.append({"kind": "point", "point": p.tolist(), "note": "midpoint(0,6) + 0.7*scale*random unit"})
nulls = av.nullspace_controls(V, 3, np.random.default_rng(11))
for p in nulls:
    cands.append(p); specs.append({"kind": "point", "point": p.tolist(), "note": "prototype nullspace control"})

C = np.array(cands)
resid = [av.affine_residual(p, V) for p in C]
order, coh, lik_norm, gap = av.rank_alien(C, V, scale, bw)
kde = [av.kde_likelihood(p, V, bw) for p in C]

fi = av.farthest_interior(V, av.midpoints(V), k=12)
mids = av.midpoints(V)
pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
chosen = []
for p in fi:
    idx = int(np.argmin(np.linalg.norm(mids - p, axis=1)))
    chosen.append(list(pairs[idx]))

a, b = V[0], V[-1]
slerps = {str(t): av.slerp(a, b, t).tolist() for t in [0.25, 0.5]}
pa, pb = a / (abs(a).max() * dim), b / (abs(b).max() * dim)
pm = av.poincare_midpoint(pa, pb)

out = {
    "source": "prototypes/alien_vectors.py --dim 64 --seed 7 (V); deterministic candidates",
    "numpy": np.__version__,
    "labels": labels,
    "V": V.tolist(),
    "nn": nn.tolist(),
    "scale": scale,
    "bandwidth": bw,
    "affine_rank": int(r),
    "candidates": specs,
    "residual": resid,
    "coherence": coh.tolist(),
    "kde": kde,
    "likelihood_norm": lik_norm.tolist(),
    "gap": gap.tolist(),
    "farthest_midpoint_pairs_k12": chosen,
    "slerp_v0_v17": slerps,
    "poincare_prototype_point": {"a": pa.tolist(), "b": pb.tolist(), "out": pm.tolist()},
}
with open(out_path, "w") as f:
    json.dump(out, f)
print("candidates", len(C), "rank", r, "scale", scale, "bw", bw)
for k, s in enumerate(specs):
    print(k, s.get("note", s.get("weights") if len(s.get("weights", {})) < 5 else "uniform"), f"res={resid[k]:.6g} coh={coh[k]:.6g} kde={kde[k]:.6g} lik={lik_norm[k]:.6g} gap={gap[k]:.6g}")
print("farthest pairs", chosen)
