"""E3: geodesic versus Euclidean neighbourhoods on a tightly wound Swiss roll."""
import numpy as np, json
from gtd import *; from manifolds import *

rng = np.random.default_rng(3)
X, T, info = swiss_roll(20000, rng, a=0.25, height=10.0)
P, Q = proj_from_basis(T)
G, _ = knn_graph(X, 20)
print("gap", info["gap"], "components", connected_components(G, directed=False)[0])
hs = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
res = {"h": hs, "gap": info["gap"]}
maps = {}
for name in ["GTD", "GTD-E", "LPCA"]:
    res[name] = dict(mean=[], p90=[], fail=[], terr=[], nor=[], dimok=[])
for h in hs:
    inner = info["dist_bd"] > h
    og = gtd(X, 2, 20, h, kernel_dist="graph", G=G, return_full=True)
    oe = gtd(X, 2, 20, h, kernel_dist="euclid", G=G, return_full=True)
    for name, o in [("GTD", og), ("GTD-E", oe)]:
        F = o["full"][inner]
        res[name]["terr"].append(float(tensor_error(F, P[inner]).mean()))
        res[name]["nor"].append(float(np.einsum("nij,nij->n", np.eye(3) - P[inner], F).mean()))
        res[name]["dimok"].append(float((intrinsic_dim_rule(o["eig"][inner]) == 2).mean()))
        if h in (1.0, 2.0):
            maps[f"{name}_nor_{h}"] = np.einsum("nij,nij->n", np.eye(3) - P, o["full"]).tolist()
    kp = int(round(og["m"].mean()))
    op = local_pca(X, 2, kp)
    for name, o in [("GTD", og), ("GTD-E", oe), ("LPCA", op)]:
        err = tangent_error_from_basis(o["U"][:, :, :2], Q)[inner]
        res[name]["mean"].append(float(err.mean()))
        res[name]["p90"].append(float(np.percentile(err, 90)))
        res[name]["fail"].append(float((err > 0.5).mean()))
        if h in (1.0, 2.0):
            maps[f"{name}_{h}"] = tangent_error_from_basis(o["U"][:, :, :2], Q).tolist()
    print(h, kp, {k: round(res[k]["mean"][-1], 4) for k in ["GTD", "GTD-E", "LPCA"]},
          {k: (round(res[k]["terr"][-1], 3), round(res[k]["nor"][-1], 3), res[k]["dimok"][-1]) for k in ["GTD", "GTD-E"]}, flush=True)
np.savez("results/exp3_maps.npz", s=0.5 * 0.25 * (info["t"] * np.sqrt(1 + info["t"] ** 2) + np.arcsinh(info["t"])),
         y=info["y"], X=X, dist_bd=info["dist_bd"], **{k: np.array(v) for k, v in maps.items()})
json.dump(res, open("results/exp3.json", "w"))
