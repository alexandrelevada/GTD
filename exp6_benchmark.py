"""E6: benchmark across manifolds (tangent and tensor errors) and intrinsic-dimension estimation."""
import numpy as np, json
from gtd import *; from manifolds import *


def datasets(rng, n):
    X, T, _ = circle(n, rng); yield "Circle", X, T, 1, None
    X, T, _ = sphere(n, rng); yield "Sphere $S^2$", X, T, 2, None
    X, T, _ = sphere(n, rng, density="nonuniform", kappa=4.0); yield "Sphere $S^2$ (vMF)", X, T, 2, None
    X, T, _ = torus(n, rng); yield "Torus $T^2$", X, T, 2, None
    X, T, _ = ellipsoid(n, rng); yield "Ellipsoid", X, T, 2, None
    X, T, info = swiss_roll(n, rng, a=0.5, height=10.0); yield "Swiss roll", X, T, 2, info["dist_bd"]
    X0, T0, _ = sphere(n, rng, d=3); X, T, _ = embed(X0, T0, 10, rng); yield "$S^3\\subset\\mathbb{R}^{10}$", X, T, 3, None
    X0, T0, _ = sphere(2 * n, rng, d=5); X, T, _ = embed(X0, T0, 20, rng); yield "$S^5\\subset\\mathbb{R}^{20}$", X, T, 5, None


def lpca_gap_dim(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, ind = nn.kneighbors(X)
    ds = []
    for i in range(0, X.shape[0], 5):
        Y = X[ind[i]] - X[ind[i]].mean(0)
        s = np.linalg.svd(Y, compute_uv=False) ** 2
        s = s[: min(len(s), 12)]
        ds.append(int(np.argmax(s[:-1] / np.maximum(s[1:], 1e-12))) + 1)
    return int(np.median(ds))


n, k, ch = 5000, 30, 2.0
seeds = range(5)
methods = ["GTD", "GTD-c", "GTD-ctr", "GTD ($\\alpha=0$)", "GTD-E", "GTD + self", "LPCA ($k=\\bar m$)", "LPCA ($k=30$)"]
tab = {}
dimtab = {}
for sd in seeds:
    rng = np.random.default_rng(1000 + sd)
    for name, X, T, d, dbd in datasets(rng, n):
        P, Q = proj_from_basis(T)
        G, rk = knn_graph(X, k)
        h = ch * np.median(rk)
        mask = np.ones(len(X), bool) if dbd is None else dbd > h
        smallD = X.shape[1] <= 20
        runs = {
            "GTD": gtd(X, d, k, h, G=G, return_full=smallD),
            "GTD-c": gtd(X, d, k, h, G=G, return_full=smallD, calibrate=True),
            "GTD-ctr": gtd(X, d, k, h, G=G, return_full=smallD, center=True),
            "GTD ($\\alpha=0$)": gtd(X, d, k, h, alpha=0.0, G=G, return_full=smallD),
            "GTD-E": gtd(X, d, k, h, kernel_dist="euclid", G=G, return_full=smallD),
            "GTD + self": gtd(X, d, k, h, exclude_self=False, G=G, return_full=smallD),
        }
        mbar = int(round(runs["GTD"]["m"].mean()))
        runs["LPCA ($k=\\bar m$)"] = local_pca(X, d, mbar)
        for cc in (1.5, 3.0):
            oc = gtd(X, d, k, cc * np.median(rk), G=G)
            runs[f"GTD c{cc}"] = oc
            runs[f"LPCA c{cc}"] = local_pca(X, d, int(round(oc["m"].mean())))
        runs["LPCA ($k=30$)"] = local_pca(X, d, 30)
        tab.setdefault(name, {"d": d, "D": X.shape[1], "h": [], "m": []})
        tab[name]["h"].append(float(h)); tab[name]["m"].append(mbar)
        for mth, o in runs.items():
            te = tangent_error_from_basis(o["U"][:, :, :d], Q)[mask].mean()
            tab[name].setdefault(mth, {"tan": [], "ten": []})
            tab[name][mth]["tan"].append(float(te))
            if "full" in o and o["full"] is not None:
                tab[name][mth]["ten"].append(float(tensor_error(o["full"][mask], P[mask]).mean()))
        # intrinsic dimension
        nb_ = graph_neighbors(G, np.full(len(X), h))
        o1 = diffusion_tensors(X, *nb_, 1, np.full(len(X), h))
        o1c = diffusion_tensors(X, *nb_, 1, np.full(len(X), h), center=True)
        dg, dloc = fixed_point_dim(o1["eig"])
        dgc, dlocc = fixed_point_dim(o1c["eig"])
        dimtab.setdefault(name, {"d": d, "gtd": [], "frac": [], "gtdc": [], "fracc": [], "mle": [], "lpca": []})
        dimtab[name]["gtd"].append(dg)
        dimtab[name]["frac"].append(float((dloc[mask] == d).mean()))
        dimtab[name]["gtdc"].append(dgc)
        dimtab[name]["fracc"].append(float((dlocc[mask] == d).mean()))
        ev = o1c["eig"]
        gaps = np.argmax(ev[:, :-1] / np.maximum(ev[:, 1:], 1e-12), axis=1) + 1
        dimtab[name].setdefault("gap", []).append(int(np.median(gaps)))
        dimtab[name]["mle"].append(float(levina_bickel(X, 20)))
        dimtab[name]["lpca"].append(lpca_gap_dim(X, mbar))
        print(sd, name, round(h, 3), mbar, {m: round(tab[name][m]["tan"][-1], 4) for m in methods},
              dg, round(dimtab[name]["frac"][-1], 3), dgc, round(dimtab[name]["fracc"][-1], 3), round(dimtab[name]["mle"][-1], 2), dimtab[name]["lpca"][-1],
              flush=True)
json.dump(dict(tab=tab, dim=dimtab), open("results/exp6.json", "w"))
