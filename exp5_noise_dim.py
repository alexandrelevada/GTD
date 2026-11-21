"""E5: ambient dimension and noise. (a) sphere S^2 in R^D with Gaussian noise;
(b) runtime breakdown for S^3 embedded in R^D."""
import numpy as np, json, time
from gtd import *; from manifolds import *

res = {"noise": {}, "runtime": {}}
# (a) noise
n, k, h, d = 8000, 30, 0.35, 2
sigmas = [0.0, 0.005, 0.01, 0.02, 0.04]
for D in [3, 20, 100]:
    out = dict(sigma=sigmas, tan=[], pca=[], nor=[], nor_pred=[], dimok=[], tan_c=[], tan_e=[], dimok_c=[], m=[])
    for sg in sigmas:
        rng = np.random.default_rng(5)
        X0, T0, _ = sphere(n, rng)
        X, T, Qe = embed(X0, T0, D, rng) if D > 3 else (X0, T0, None)
        _, Q = proj_from_basis(T)
        Xn = X + sg * rng.normal(size=X.shape)
        o = gtd(Xn, d, k, h)
        op = local_pca(Xn, d, int(round(o["m"].mean())))
        out["tan"].append(float(tangent_error_from_basis(o["U"][:, :, :d], Q).mean()))
        out["pca"].append(float(tangent_error_from_basis(op["U"][:, :, :d], Q).mean()))
        nor = (o["trace"] - o["eig"][:, :d].sum(1)) / (D - d)
        out["nor"].append(float(nor.mean()))
        out["nor_pred"].append(float((h * h / 4) / (D - d) + 2 * sg ** 2 / (c_K(d) * h * h)))
        out["dimok"].append(float((intrinsic_dim_rule(o["eig"]) == d).mean()))
        out["m"].append(float(o["m"].mean()))
        oc = gtd(Xn, d, k, h, calibrate=True)
        oe = gtd(Xn, d, k, h, kernel_dist="euclid")
        out["tan_c"].append(float(tangent_error_from_basis(oc["U"][:, :, :d], Q).mean()))
        out["tan_e"].append(float(tangent_error_from_basis(oe["U"][:, :, :d], Q).mean()))
        out["dimok_c"].append(float((intrinsic_dim_rule(oc["eig"]) == d).mean()))
        print("noise", D, sg, out["tan"][-1], out["pca"][-1], out["nor"][-1], out["nor_pred"][-1],
              out["dimok"][-1], flush=True)
    res["noise"][str(D)] = out

# (b) runtime vs D, S^3 (d = 3)
n, k, h, d = 5000, 30, 0.6, 3
rng = np.random.default_rng(6)
X0, T0, _ = sphere(n, rng, d=3)
for D in [4, 16, 64, 256, 1024]:
    r2 = np.random.default_rng(7)
    X, T, _ = embed(X0, T0, D, r2) if D > 4 else (X0, T0, None)
    Q = T
    t0 = time.time(); G, _ = knn_graph(X, k); t1 = time.time()
    nbr, dst = graph_neighbors(G, np.full(n, h)); t2 = time.time()
    o = diffusion_tensors(X, nbr, dst, d, np.full(n, h)); t3 = time.time()
    err = float(tangent_error_from_basis(o["U"][:, :, :d], Q).mean())
    res["runtime"][str(D)] = dict(knn=t1 - t0, dij=t2 - t1, ten=t3 - t2, err=err, m=float(o["m"].mean()),
                                  dimok=float((intrinsic_dim_rule(o["eig"]) == d).mean()))
    print("runtime", D, res["runtime"][str(D)], flush=True)
json.dump(res, open("results/exp5.json", "w"))
