"""E1: validation of the bias expansion (Theorem 4.3) on circle, sphere and torus."""
import numpy as np, json
from gtd import *; from manifolds import *

rng = np.random.default_rng(1)
res = {}

# ---- circle and sphere: blocks as functions of h (oracle and graph distances)
for name, gen, d, D, geo, hs, k in [
    ("circle", lambda: circle(20000, rng), 1, 2, circle_geo(1.0), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8], 30),
    ("sphere", lambda: sphere(20000, rng), 2, 3, sphere_geo(1.0), [0.15, 0.2, 0.3, 0.4, 0.5, 0.6], 40),
]:
    X, T, _ = gen()
    P, _ = proj_from_basis(T)
    G, _ = knn_graph(X, k)
    out = {"h": hs}
    for mode in ["oracle", "graph"]:
        tan, nor, tan_sd, nor_sd = [], [], [], []
        for h in hs:
            o = gtd(X, d, k, h, kernel_dist=mode, geo_fn=geo, return_full=True, G=G)
            F = o["full"]
            lt = np.einsum("nij,nij->n", P, F) / d
            ln = np.einsum("nij,nij->n", np.eye(D) - P, F) / (D - d)
            tan.append(lt.mean()); nor.append(ln.mean())
            tan_sd.append(lt.std() / np.sqrt(len(lt))); nor_sd.append(ln.std() / np.sqrt(len(ln)))
            print(name, mode, h, o["ncomp"], round(tan[-1], 5), round(nor[-1], 6), flush=True)
        out[mode] = dict(tan=tan, nor=nor, tan_se=tan_sd, nor_se=nor_sd)
    res[name] = out

# ---- torus: per-point blocks in the principal frame, binned in v
R, r = 2.0, 1.0
n, k, h = 40000, 40, 0.5
X, T, info = torus(n, rng, R, r)
u, v = info["u"], info["v"]
e1, e2 = T[:, :, 0], T[:, :, 1]
nrm = np.c_[np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)]  # outward normal
o = gtd(X, 2, k, h, kernel_dist="graph", return_full=True)
F = o["full"]
b11 = np.einsum("ni,nij,nj->n", e1, F, e1)
b22 = np.einsum("ni,nij,nj->n", e2, F, e2)
bnn = np.einsum("ni,nij,nj->n", nrm, F, nrm)
b2n = np.einsum("ni,nij,nj->n", e2, F, nrm)
bins = np.linspace(0, 2 * np.pi, 41)
idx = np.digitize(v, bins) - 1
emp = {key: [float(arr[idx == b].mean()) for b in range(40)] for key, arr in
       dict(b11=b11, b22=b22, bnn=bnn, b2n=b2n).items()}
se = {key: [float(arr[idx == b].std() / np.sqrt((idx == b).sum())) for b in range(40)] for key, arr in
      dict(b11=b11, b22=b22, bnn=bnn, b2n=b2n).items()}
res["torus"] = dict(vc=list(0.5 * (bins[1:] + bins[:-1])), emp=emp, se=se, h=h, n=n, k=k,
                    m=float(o["m"].mean()))
json.dump(res, open("results/exp1.json", "w"))
print("torus m", o["m"].mean())
