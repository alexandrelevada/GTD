"""E4: consistency and rates as n grows (Corollary 4.5), with timing."""
import numpy as np, json, time
from gtd import *; from manifolds import *

ns = [1000, 2000, 4000, 8000, 16000, 32000]
seeds = [0, 1, 2]
res = {}
for name, gen, ch, geo in [("sphere", lambda n, r: sphere(n, r), 1.1, sphere_geo(1.0)),
                           ("torus", lambda n, r: torus(n, r), 1.6, None)]:
    out = {k: [] for k in ["n", "h", "k", "m", "terr", "tanerr", "pca", "terr_or", "t_knn", "t_dij", "t_ten"]}
    for n in ns:
        h = ch * (np.log(n) / n) ** (1 / 6)
        k = int(np.ceil(2 * n ** (1 / 3)))
        acc = {key: [] for key in ["terr", "tanerr", "pca", "terr_or", "m", "t_knn", "t_dij", "t_ten"]}
        for sd in seeds:
            rng = np.random.default_rng(100 + sd)
            X, T, _ = gen(n, rng)
            P, Q = proj_from_basis(T)
            t0 = time.time(); G, _ = knn_graph(X, k); t1 = time.time()
            nbr, dst = graph_neighbors(G, np.full(n, h)); t2 = time.time()
            o = diffusion_tensors(X, nbr, dst, 2, np.full(n, h), return_full=True); t3 = time.time()
            acc["t_knn"].append(t1 - t0); acc["t_dij"].append(t2 - t1); acc["t_ten"].append(t3 - t2)
            acc["terr"].append(tensor_error(o["full"], P).mean())
            acc["tanerr"].append(tangent_error(o["U"], P, 2).mean())
            acc["m"].append(o["m"].mean())
            op = local_pca(X, 2, int(round(o["m"].mean())))
            acc["pca"].append(tangent_error(op["U"], P, 2).mean())
            if geo is not None:
                oo = gtd(X, 2, k, h, kernel_dist="oracle", geo_fn=geo, return_full=True, G=G)
                acc["terr_or"].append(tensor_error(oo["full"], P).mean())
        out["n"].append(n); out["h"].append(h); out["k"].append(k)
        for key in acc:
            out[key].append([float(np.mean(acc[key])), float(np.std(acc[key]))] if acc[key] else None)
        print(name, n, round(h, 3), k, {key: (round(out[key][-1][0], 4) if out[key][-1] else None)
                                       for key in acc}, flush=True)
    res[name] = out
json.dump(res, open("results/exp4.json", "w"))
