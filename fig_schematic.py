import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from gtd import *
from manifolds import *
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9, "savefig.bbox": "tight", "pdf.fonttype": 42})
BL, OR, GR = "#2a78d6", "#eb6834", "#888888"
rng = np.random.default_rng(3)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
# (a) spiral: Euclidean ball vs geodesic neighbourhood
t = np.sort(rng.uniform(1.3 * np.pi, 5.8 * np.pi, 1100)); a = 0.12
X = np.c_[a * t * np.cos(t), a * t * np.sin(t)]
i0 = np.argmin(np.abs(t - 3.5 * np.pi)); h = 0.95
G, _ = knn_graph(X, 8)
nbr, dst = graph_neighbors(G, np.full(len(X), h))
eu = np.linalg.norm(X - X[i0], axis=1) <= h
geo = np.zeros(len(X), bool); geo[nbr[i0]] = True
A = ax[0]
A.plot(X[:, 0], X[:, 1], ".", color="#cfcfcf", ms=2.5)
A.plot(X[eu & ~geo, 0], X[eu & ~geo, 1], ".", color=OR, ms=4, label="Euclidean only")
A.plot(X[geo, 0], X[geo, 1], ".", color=BL, ms=4, label="geodesic neighbours")
A.add_patch(Circle(X[i0], h, fill=False, ls="--", color=OR, lw=1))
A.plot(*X[i0], "k*", ms=9)
A.set_aspect("equal"); A.axis("off"); A.set_title("(a) geodesic vs Euclidean ball")
A.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=1)
# (b) weighted displacements and the tensor ellipse on a circle
th = rng.uniform(0, 2 * np.pi, 160); Xc = np.c_[np.cos(th), np.sin(th)]
j0 = np.argmin(np.abs(th - np.pi / 2)); hb = 0.5
o = gtd(Xc, 1, 8, hb, return_full=True)
nb = np.nonzero(np.linalg.norm(Xc - Xc[j0], axis=1) < hb * 1.05)[0]
G2, _ = knn_graph(Xc, 8); nbr2, dst2 = graph_neighbors(G2, np.full(len(Xc), hb))
w = K(dst2[j0] ** 2 / hb ** 2); w = w / w.max()
B = ax[1]
tt = np.linspace(np.pi / 2 - 0.9, np.pi / 2 + 0.9, 200)
B.plot(np.cos(tt), np.sin(tt), "-", color="#cfcfcf", lw=1)
for j, ww in zip(nbr2[j0], w):
    B.annotate("", xy=Xc[j], xytext=Xc[j0], arrowprops=dict(arrowstyle="-|>", color=BL, alpha=0.25 + 0.75 * ww, lw=0.6 + 0.8 * ww, mutation_scale=6))
P = o["full"][j0]; ev, V = np.linalg.eigh(P)
ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
sc = 0.35
B.add_patch(Ellipse(Xc[j0], 2 * sc * np.sqrt(ev[1]), 2 * sc * max(np.sqrt(ev[0]), 0.02), angle=ang, fill=False, color=OR, lw=1.6))
B.plot(*Xc[j0], "k*", ms=9)
B.text(Xc[j0, 0] + 0.02, Xc[j0, 1] + 0.09, "$x_i$", fontsize=9)
B.text(0.05, 0.72, "ellipse of $\\widehat P_i$", color=OR, fontsize=7.5)
B.set_xlim(-0.6, 0.6); B.set_ylim(0.78, 1.12); B.set_aspect("equal"); B.axis("off")
B.set_title("(b) weighted steps $x_j-x_i$")
# (c) normalized spectrum: S^3 in R^10
X0, T0, _ = sphere(4000, rng, d=3); X3, T3, _ = embed(X0, T0, 10, rng)
G3, rk = knn_graph(X3, 30)
o3 = gtd(X3, 3, 30, 2 * np.median(rk), G=G3, center=True)
med = np.median(o3["eig"], 0)[:7]
q1, q3 = np.percentile(o3["eig"], [10, 90], axis=0)[:, :7]
C = ax[2]
cols = [BL] * 3 + [GR] * 4
C.bar(np.arange(1, 8), med, color=cols, width=0.6, yerr=[med - q1, q3 - med], error_kw=dict(lw=0.8, capsize=2, ecolor="#555"))
C.axhline(0.5, color=OR, ls="--", lw=1); C.text(7.4, 0.53, "threshold $1/2$", color=OR, fontsize=7, ha="right")
C.axhline(1.0, color="#555", ls=":", lw=0.8)
C.set_xticks(range(1, 8)); C.set_xlabel("eigenvalue index $r$"); C.set_ylabel("$\\hat\\lambda_{i,r}$")
C.text(2, 1.08, "tangent", color=BL, ha="center", fontsize=8); C.text(5.5, 0.12, "normal", color=GR, ha="center", fontsize=8)
C.set_ylim(0, 1.2); C.spines[["top", "right"]].set_visible(False)
C.set_title("(c) spectrum of $\\widehat P_i$ ($S^3\\subset\\mathbb{R}^{10}$)")
fig.tight_layout()
fig.savefig("figs/fig_schematic.pdf"); fig.savefig("figs/fig_schematic.png", dpi=170)
