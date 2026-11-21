"""E8: bandwidth sweep on the image-rotation manifold + illustrative figure of tangent images."""
import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gtd import *

src = open("exp7_real.py").read().split("k, ch = 30")[0]
exec(src)  # defines render, image_manifold

res = {}
for name, tag, n, two_d in [("rot", "rotu", 2500, False), ("rotscale", "rots", 5000, True)]:
    d = 2 if two_d else 1
    X, Q, inner = image_manifold_cached(tag, n, 70, two_d, False)
    G, rk = knn_graph(X, 30)
    out = dict(ch=[], m=[], gtd=[], ctr=[], pca=[], dimok=[])
    for ch in [0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
        h = ch * np.median(rk)
        o = gtd(X, d, 30, h, G=G)
        oc = gtd(X, d, 30, h, G=G, center=True)
        mb = int(round(o["m"].mean()))
        op = local_pca(X, d, max(mb, d + 2))
        out["ch"].append(ch); out["m"].append(mb)
        out["gtd"].append(float(tangent_error_from_basis(o["U"][:, :, :d], Q)[inner].mean()))
        out["ctr"].append(float(tangent_error_from_basis(oc["U"][:, :, :d], Q)[inner].mean()))
        out["pca"].append(float(tangent_error_from_basis(op["U"][:, :, :d], Q)[inner].mean()))
        out["dimok"].append(float((intrinsic_dim_rule(oc["eig"])[inner] == d).mean()))
        print(name, ch, mb, out["gtd"][-1], out["ctr"][-1], out["pca"][-1], out["dimok"][-1], flush=True)
        if name == "rot" and ch == 1.0:
            keep = dict(X=X, Q=Q, U=oc["U"][:, :, 0], Up=op["U"][:, :, 0], eig=oc["eig"])
    res[name] = out
json.dump(res, open("results/exp8.json", "w"))

# illustrative figure: images, true tangent, GTD tangent, LPCA tangent for 3 samples
X, Q, U, Up = keep["X"], keep["Q"], keep["U"], keep["Up"]
idx = [5, 400, 1200]
fig, ax = plt.subplots(3, 4, figsize=(5.6, 4.3))
for r_, i in enumerate(idx):
    t = Q[i, :, 0]
    u = U[i] * np.sign(U[i] @ t); up = Up[i] * np.sign(Up[i] @ t)
    v = np.abs(t).max()
    for c_, (img, title, cm, vmin, vmax) in enumerate([(X[i], "sample $x_i$", "gray", 0, 1),
                                                        (t, "true tangent", "RdBu_r", -v, v),
                                                        (u, "GTD-ctr tangent", "RdBu_r", -v, v),
                                                        (up, "LPCA tangent", "RdBu_r", -v, v)]):
        a = ax[r_, c_]
        a.imshow(img.reshape(32, 32), cmap=cm, vmin=vmin, vmax=vmax); a.set_xticks([]); a.set_yticks([])
        if r_ == 0: a.set_title(title, fontsize=8)
    err_g = np.sqrt(max(0, 1 - (u @ t) ** 2)); err_p = np.sqrt(max(0, 1 - (up @ t) ** 2))
    ax[r_, 2].set_xlabel(f"error {err_g:.3f}", fontsize=7); ax[r_, 3].set_xlabel(f"error {err_p:.3f}", fontsize=7)
fig.tight_layout()
fig.savefig("figs/fig_image_tangents.pdf", bbox_inches="tight"); fig.savefig("figs/fig_image_tangents.png", dpi=170, bbox_inches="tight")
