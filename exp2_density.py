"""E2: density invariance (Theorem 4.2) and the alpha = 0 contamination (Remark 4.2)."""
import numpy as np, json
from gtd import *; from manifolds import *

rng = np.random.default_rng(2)
res = {}
hs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# ---------------- circle, p ∝ 1 + 0.9 cos(theta)
X, T, info = circle(20000, rng, density="nonuniform", a=0.9)
th = info["theta"]
e = T[:, :, 0]; nrm = X.copy()  # outward normal (r = 1)
G, _ = knn_graph(X, 30)
bins = np.linspace(0, 2 * np.pi, 37); idx = np.digitize(th, bins) - 1; thc = 0.5 * (bins[1:] + bins[:-1])
circ = {"h": hs, "thc": list(thc)}
for alpha in [0.0, 1.0]:
    tilt_b, tan_b, rms_tilt, rms_tan = [], [], [], []
    for h in hs:
        o = gtd(X, 1, 30, h, alpha=alpha, return_full=True, G=G)
        F = o["full"]
        tilt = np.einsum("ni,nij,nj->n", e, F, nrm)
        tan = np.einsum("ni,nij,nj->n", e, F, e)
        tb = np.array([tilt[idx == b].mean() for b in range(36)])
        ab = np.array([tan[idx == b].mean() for b in range(36)])
        tilt_b.append(list(tb)); tan_b.append(list(ab))
        rms_tilt.append(float(np.sqrt(np.mean(tb ** 2))))
        rms_tan.append(float(np.sqrt(np.mean((ab - (1 - h * h / 7)) ** 2))))  # deviation from uniform-density limit
        print("circle", alpha, h, rms_tilt[-1], rms_tan[-1], flush=True)
    circ[f"a{int(alpha)}"] = dict(tilt=tilt_b, tan=tan_b, rms_tilt=rms_tilt, rms_tan=rms_tan)
res["circle"] = circ

# ---------------- sphere, p ∝ exp(kappa z)
kappa = 2.0
X, T, info = sphere(20000, rng, density="nonuniform", kappa=kappa)
P, Q = proj_from_basis(T)
z = X[:, 2]
grad = np.c_[0 * z, 0 * z, 1 + 0 * z] - z[:, None] * X  # tangential gradient of z
gn = np.linalg.norm(grad, axis=1); ghat = grad / np.maximum(gn, 1e-12)[:, None]
G, _ = knn_graph(X, 40)
pol = np.arccos(np.clip(z, -1, 1))
bins = np.linspace(0, np.pi, 19); idx = np.digitize(pol, bins) - 1; pc = 0.5 * (bins[1:] + bins[:-1])
sph = {"h": hs[1:], "pc": list(pc)}
for alpha in [0.0, 1.0]:
    tilt_b, terr, rms_tilt = [], [], []
    for h in hs[1:]:
        o = gtd(X, 2, 40, h, alpha=alpha, return_full=True, G=G)
        F = o["full"]
        tilt = np.einsum("ni,nij,nj->n", ghat, F, X)
        tb = [float(tilt[idx == b].mean()) if (idx == b).sum() > 30 else float("nan") for b in range(18)]
        tilt_b.append(tb)
        terr.append(float(tangent_error(o["U"], P, 2).mean()))
        valid = np.isfinite(tb)
        rms_tilt.append(float(np.sqrt(np.nanmean(np.array(tb) ** 2))))
        print("sphere", alpha, h, rms_tilt[-1], terr[-1], flush=True)
    sph[f"a{int(alpha)}"] = dict(tilt=tilt_b, terr=terr, rms_tilt=rms_tilt)
res["sphere"] = sph
json.dump(res, open("results/exp2.json", "w"))
