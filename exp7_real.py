"""E7: real data.  (a) image manifolds generated from a real photograph (rotation; rotation x scale),
with ground-truth tangents from finite differences; (b) handwritten digits: intrinsic dimension and
tangent-distance classification."""
import numpy as np, json
from skimage import data, transform, filters
from sklearn.datasets import load_digits
from gtd import *

res = {}
img = data.camera().astype(float) / 255.0
crop = transform.resize(filters.gaussian(img[60:316, 130:386], sigma=2.0), (128, 128), anti_aliasing=True)
yy, xx = np.mgrid[-1:1:32j, -1:1:32j]
mask = np.clip((0.95 - np.sqrt(xx ** 2 + yy ** 2)) / 0.1, 0, 1)  # smooth disk mask


def render(theta, s=1.0):
    tf = transform.SimilarityTransform(scale=s, rotation=theta)
    c = np.array([64.0, 64.0])
    M = transform.SimilarityTransform(translation=-c) + tf + transform.SimilarityTransform(translation=c)
    w = transform.warp(crop, M.inverse, order=3, mode="constant")
    return (transform.resize(w, (32, 32), anti_aliasing=True) * mask).ravel()


def image_manifold_cached(tag, n, seed, two_d=False, nonuniform=False):
    import os
    fn = f"cache/{tag}_{seed}.npz"
    if os.path.exists(fn):
        z = np.load(fn); return z["X"], z["Q"], z["inner"]
    X, Q, inner = image_manifold(n, np.random.default_rng(seed), two_d, nonuniform)
    os.makedirs("cache", exist_ok=True); np.savez(fn, X=X, Q=Q, inner=inner)
    return X, Q, inner


def image_manifold(n, rng, two_d=False, nonuniform=False):
    if nonuniform:
        th = []
        while len(th) < n:
            c = rng.uniform(0, 2 * np.pi, 2 * n); u = rng.uniform(0, 1.8, 2 * n)
            th.extend(c[u < 1 + 0.8 * np.cos(c)])
        th = np.array(th[:n])
    else:
        th = rng.uniform(0, 2 * np.pi, n)
    sc = rng.uniform(0.75, 1.25, n) if two_d else np.ones(n)
    X = np.array([render(a, b) for a, b in zip(th, sc)])
    de = 1e-3
    Tth = np.array([(render(a + de, b) - render(a - de, b)) / (2 * de) for a, b in zip(th, sc)])
    cols = [Tth]
    if two_d:
        Ts = np.array([(render(a, b + de) - render(a, b - de)) / (2 * de) for a, b in zip(th, sc)])
        cols.append(Ts)
    T = np.stack(cols, axis=2)
    Q = np.linalg.qr(T)[0]
    inner = np.abs(sc - 1.0) < 0.15 if two_d else np.ones(n, bool)
    return X, Q, inner


k, ch = 30, 1.0
for name, tag, n, two_d, nonu, nseed in [("Rotation (uniform)", "rotu", 2500, False, False, 3),
                                         ("Rotation (non-uniform)", "rotn", 2500, False, True, 3),
                                         ("Rotation $\\times$ scale", "rots", 5000, True, False, 2)]:
    d = 2 if two_d else 1
    out = {}
    for sd in range(nseed):
        X, Q, inner = image_manifold_cached(tag, n, 70 + sd, two_d, nonu)
        G, rk = knn_graph(X, k)
        h = ch * np.median(rk)
        runs = {"GTD": gtd(X, d, k, h, G=G), "GTD-c": gtd(X, d, k, h, G=G, calibrate=True),
                "GTD-ctr": gtd(X, d, k, h, G=G, center=True),
                "GTD ($\\alpha=0$)": gtd(X, d, k, h, alpha=0.0, G=G),
                "GTD-E": gtd(X, d, k, h, kernel_dist="euclid", G=G)}
        mbar = int(round(runs["GTD"]["m"].mean()))
        runs["LPCA ($k=\\bar m$)"] = local_pca(X, d, mbar)
        runs["LPCA ($k=30$)"] = local_pca(X, d, 30)
        for mth, o in runs.items():
            out.setdefault(mth, []).append(float(tangent_error_from_basis(o["U"][:, :, :d], Q)[inner].mean()))
        o1 = gtd(X, 1, k, h, G=G)
        dg, dloc = fixed_point_dim(o1["eig"])
        out.setdefault("dim", []).append(dg)
        out.setdefault("dimfrac", []).append(float((dloc[inner] == d).mean()))
        out.setdefault("mle", []).append(float(levina_bickel(X, 20)))
        out.setdefault("m", []).append(mbar)
        print(name, sd, {m: round(v[-1], 4) for m, v in out.items()}, flush=True)
    res[name] = out
    res[name]["d"] = d

# ---------------- digits
dg = load_digits()
Xd, yd = dg.data / 16.0, dg.target


def tangents_per_class(Xtr, ytr, method, dfix=None):
    Us, dims = {}, {}
    for c in np.unique(ytr):
        Xc = Xtr[ytr == c]
        G, rk = knn_graph(Xc, 10)
        h = 3.0 * np.median(rk)
        o1 = gtd(Xc, 1, 10, h, G=G)
        dc, _ = fixed_point_dim(o1["eig"])
        dims[int(c)] = dc
        dd = dfix if dfix is not None else dc
        if method == "GTD":
            o = gtd(Xc, dd, 10, h, G=G)
            U = o["U"]
            if U.shape[2] < dd:
                U = np.concatenate([U, np.zeros((U.shape[0], U.shape[1], dd - U.shape[2]))], 2)
            Us[int(c)] = U[:, :, :dd] if dd > 0 else None
        else:
            mbar = int(round(gtd(Xc, 1, 10, h, G=G)["m"].mean())) if method == "LPCA-m" else 10
            o = local_pca(Xc, max(dd, 1), max(mbar, dd + 1))
            Us[int(c)] = o["U"][:, :, :dd]
    return Us, dims


def td_classify(Xtr, ytr, Xte, Us):
    best = np.full(len(Xte), np.inf); pred = np.zeros(len(Xte), int)
    for c, U in Us.items():
        Xc = Xtr[ytr == c]
        for i in range(len(Xc)):
            R = Xte - Xc[i]
            if U is not None and U.shape[2] > 0:
                R = R - (R @ U[i]) @ U[i].T
            dist = np.linalg.norm(R, axis=1)
            upd = dist < best
            best[upd] = dist[upd]; pred[upd] = c
    return pred


dig = {"dims": [], "acc": {}}
for sp in range(10):
    rng = np.random.default_rng(500 + sp)
    perm = rng.permutation(len(Xd)); ntr = int(0.7 * len(Xd))
    tr, te = perm[:ntr], perm[ntr:]
    Xtr, ytr, Xte, yte = Xd[tr], yd[tr], Xd[te], yd[te]
    acc = dig["acc"]
    # Euclidean 1-NN
    D2 = ((Xte[:, None, :] - Xtr[None]) ** 2).sum(-1)
    acc.setdefault("1-NN (Euclidean)", []).append(float((ytr[D2.argmin(1)] == yte).mean()))
    for dfix in [None, 2, 4, 6, 8]:
        for method in ["GTD", "LPCA-m", "LPCA-10"]:
            Us, dims = tangents_per_class(Xtr, ytr, method, dfix)
            key = f"{method}|{'auto' if dfix is None else dfix}"
            acc.setdefault(key, []).append(float((td_classify(Xtr, ytr, Xte, Us) == yte).mean()))
            if dfix is None and method == "GTD":
                dig["dims"].append(dims)
    print("digits split", sp, {kk: round(v[-1], 4) for kk, v in acc.items()}, flush=True)
res["digits"] = dig
json.dump(res, open("results/exp7.json", "w"))
