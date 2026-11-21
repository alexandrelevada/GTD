import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from theory import torus_theory

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "lines.markersize": 4.5, "savefig.bbox": "tight", "pdf.fonttype": 42})
C = {"GTD": "#2a78d6", "GTD-E": "#eb6834", "LPCA": "#1baf7a", "a0": "#eda100", "GTD-c": "#e87ba4",
     "oracle": "#008300", "theory": "#333333"}
F = "figs/"
which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7"]


def save(fig, name):
    fig.savefig(F + name + ".pdf"); fig.savefig(F + name + ".png", dpi=170); plt.close(fig)


# ------------------------------------------------------------------ E1
if "1" in which:
    r = json.load(open("results/exp1.json"))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for j, (name, d, tan_c, nor_c) in enumerate([("circle", 1, -1 / 7, 3 / 28), ("sphere", 2, -7 / 36, 1 / 4)]):
        h = np.array(r[name]["h"]); hh = np.linspace(0, h.max() * 1.05, 100)
        a = ax[j]
        a.plot(hh, 1 + tan_c * hh ** 2, "--", color=C["theory"], lw=1.1, label="theory $1+h^2B_{\\tan}$")
        a.plot(h, r[name]["oracle"]["tan"], "s", color=C["oracle"], mfc="none", label="GTD, exact $d_{\\mathcal{M}}$")
        a.plot(h, r[name]["graph"]["tan"], "o", color=C["GTD"], ms=3.5, label="GTD, graph $d_G$")
        a2 = a.twinx() if False else None
        a.set_xlabel("$h$"); a.set_title(f"({'ab'[j]}) {'circle $S^1$' if d == 1 else 'sphere $S^2$'}: tangent block")
        a.set_ylabel("$\\mathrm{tr}(P_x\\widehat P)/d$")
        if j == 0: a.legend(frameon=False, loc="lower left", fontsize=7)
    a = ax[2]
    for name, nor_c, mk in [("circle", 3 / 28, "o"), ("sphere", 1 / 4, "^")]:
        h = np.array(r[name]["h"]); hh = np.linspace(0, h.max() * 1.05, 100)
        a.plot(hh, nor_c * hh ** 2, "--", color=C["theory"], lw=1.1)
        a.plot(h, r[name]["oracle"]["nor"], mk, color=C["oracle"], mfc="none")
        a.plot(h, r[name]["graph"]["nor"], mk, color=C["GTD"], ms=3.5)
        a.annotate("$S^1$: $\\frac{3}{28}h^2$" if name == "circle" else "$S^2$: $\\frac{1}{4}h^2$",
                   (h[-1], nor_c * h[-1] ** 2), textcoords="offset points", xytext=(-62, -4) if name == "sphere" else (-72, 4))
    a.set_xlabel("$h$"); a.set_ylabel("$\\mathrm{tr}(P_x^{\\perp}\\widehat P)/(D-d)$"); a.set_title("(c) normal block")
    fig.tight_layout(); save(fig, "fig_bias_curves")

    t = r["torus"]; v = np.array(t["vc"]); vv = np.linspace(0, 2 * np.pi, 300)
    th = torus_theory(vv, t["h"])
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.1))
    labs = [("b11", "$e_1^{\\top}\\widehat P e_1$ (direction $u$)"), ("b22", "$e_2^{\\top}\\widehat P e_2$ (direction $v$)"),
            ("bnn", "$\\nu^{\\top}\\widehat P\\nu$ (normal)"), ("b2n", "$e_2^{\\top}\\widehat P\\nu$ (mixed)")]
    for j, (key, lab) in enumerate(labs):
        a = ax[j]
        a.plot(vv, th[j], "--", color=C["theory"], lw=1.1, label="theory")
        a.errorbar(v, t["emp"][key], yerr=2 * np.array(t["se"][key]), fmt="o", color=C["GTD"], ms=2.8,
                   elinewidth=0.7, label="GTD")
        a.set_title(lab, fontsize=8); a.set_xlabel("$v$")
        a.set_xticks([0, np.pi, 2 * np.pi]); a.set_xticklabels(["0", "$\\pi$", "$2\\pi$"])
    ax[3].legend(frameon=False, loc="lower left", fontsize=7)
    fig.tight_layout(); save(fig, "fig_torus_blocks")

# ------------------------------------------------------------------ E2
if "2" in which:
    r = json.load(open("results/exp2.json"))
    c = r["circle"]; hs = np.array(c["h"]); thc = np.array(c["thc"])
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35))
    j4 = hs.tolist().index(0.4); h = 0.4
    tt = np.linspace(0, 2 * np.pi, 300)
    pred0 = h * h * (3 / 14) * 0.9 * np.sin(tt) / (1 + 0.9 * np.cos(tt))
    a = ax[0]
    a.plot(tt, pred0, "--", color=C["theory"], lw=1.1, label="theory ($\\alpha=0$)")
    a.plot(thc, c["a0"]["tilt"][j4], "o", color=C["a0"], ms=3, label="$\\alpha=0$")
    a.plot(thc, c["a1"]["tilt"][j4], "s", color=C["GTD"], ms=3, label="$\\alpha=1$ (GTD)")
    a.axhline(0, color="#999", lw=0.6)
    a.set_xticks([0, np.pi, 2 * np.pi]); a.set_xticklabels(["0", "$\\pi$", "$2\\pi$"])
    a.set_xlabel("$\\theta$"); a.set_ylabel("$e^{\\top}\\widehat P\\nu$"); a.set_title("(a) $S^1$, mixed block, $h=0.4$")
    a.legend(frameon=False, fontsize=7, loc="lower left")
    a = ax[1]
    pt0 = 1 + h * h * (-1 / 7 + (4 / 35) * (-0.9 * np.cos(tt)) / (1 + 0.9 * np.cos(tt)))
    a.plot(tt, pt0, "--", color=C["theory"], lw=1.1)
    a.axhline(1 - h * h / 7, color=C["GTD"], lw=0.9, ls=":")
    a.plot(thc, c["a0"]["tan"][j4], "o", color=C["a0"], ms=3)
    a.plot(thc, c["a1"]["tan"][j4], "s", color=C["GTD"], ms=3)
    a.set_xticks([0, np.pi, 2 * np.pi]); a.set_xticklabels(["0", "$\\pi$", "$2\\pi$"])
    a.set_xlabel("$\\theta$"); a.set_ylabel("$e^{\\top}\\widehat P e$"); a.set_title("(b) $S^1$, tangent block, $h=0.4$")
    a = ax[2]
    for key, col, mk, lab in [("a0", C["a0"], "o", "$\\alpha=0$"), ("a1", C["GTD"], "s", "$\\alpha=1$ (GTD)")]:
        a.loglog(hs, c[key]["rms_tilt"], mk + "-", color=col, label=lab)
    hh = np.array([0.1, 0.6])
    a.loglog(hh, 0.3 * hh ** 2, "--", color=C["theory"], lw=0.9); a.text(0.13, 0.3 * 0.13 ** 2 * 1.6, "slope 2", fontsize=7)
    a.loglog(hh, 0.03 * hh ** 3, ":", color=C["theory"], lw=0.9); a.text(0.13, 0.03 * 0.13 ** 3 * 0.35, "slope 3", fontsize=7)
    a.set_xlabel("$h$"); a.set_ylabel("RMS of binned mixed block"); a.set_title("(c) density contamination vs $h$")
    a.legend(frameon=False, fontsize=7, loc="lower right")
    a.set_xticks([0.1, 0.2, 0.3, 0.4, 0.6]); a.set_xticklabels(["0.1", "0.2", "0.3", "0.4", "0.6"]); a.minorticks_off()
    fig.tight_layout(); save(fig, "fig_density")

    s = r["sphere"]; pc = np.array(s["pc"]); hs = np.array(s["h"])
    fig, ax = plt.subplots(1, 2, figsize=(5.0, 2.3))
    j = hs.tolist().index(0.4); h = 0.4
    pp = np.linspace(0, np.pi, 200)
    ax[0].plot(pp, -h * h * 2.0 * np.sin(pp) / 4, "--", color=C["theory"], lw=1.1, label="theory ($\\alpha=0$)")
    ax[0].plot(pc, s["a0"]["tilt"][j], "o", color=C["a0"], ms=3, label="$\\alpha=0$")
    ax[0].plot(pc, s["a1"]["tilt"][j], "s", color=C["GTD"], ms=3, label="$\\alpha=1$ (GTD)")
    ax[0].axhline(0, color="#999", lw=0.6)
    ax[0].set_xlabel("polar angle $\\varphi$"); ax[0].set_ylabel("$\\hat g^{\\top}\\widehat P\\nu$")
    ax[0].set_title("(a) $S^2$, vMF density, $h=0.4$")
    ax[0].legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=7, loc=(0.52, 0.32))
    ax[1].plot(hs, s["a0"]["terr"], "o-", color=C["a0"], label="$\\alpha=0$")
    ax[1].plot(hs, s["a1"]["terr"], "s-", color=C["GTD"], label="$\\alpha=1$ (GTD)")
    ax[1].set_xlabel("$h$"); ax[1].set_ylabel("mean tangent error"); ax[1].set_title("(b) tangent error")
    ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); save(fig, "fig_density_sphere")

# ------------------------------------------------------------------ E3
if "3" in which:
    r = json.load(open("results/exp3.json")); h = np.array(r["h"])
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for name, col, mk, lab in [("GTD", C["GTD"], "o", "GTD (geodesic)"), ("GTD-E", C["GTD-E"], "s", "GTD-E (Euclidean)"),
                               ("LPCA", C["LPCA"], "^", "LPCA")]:
        ax[0].plot(h, r[name]["mean"], mk + "-", color=col, label=lab)
        if name != "LPCA":
            ax[1].plot(h, r[name]["terr"], mk + "-", color=col, label=lab)
            ax[2].plot(h, r[name]["dimok"], mk + "-", color=col, label=lab)
    for a in ax: a.axvline(r["gap"], color="#999", lw=0.8, ls=":"); a.set_xlabel("$h$")
    ax[0].text(r["gap"] * 1.02, 0.11, "layer gap", fontsize=7, color="#666")
    ax[0].set_ylabel("mean tangent error"); ax[0].set_title("(a) tangent spaces"); ax[0].legend(frameon=False, fontsize=7)
    ax[1].set_ylabel("mean $\\|\\widehat P-P_x\\|$"); ax[1].set_title("(b) metric tensors")
    ax[2].set_ylabel("fraction with $\\hat d_i=2$"); ax[2].set_title("(c) dimension rule"); ax[2].set_ylim(0, 1.05)
    fig.tight_layout(); save(fig, "fig_swiss_curves")

    m = np.load("results/exp3_maps.npz")
    fig = plt.figure(figsize=(7.2, 2.6))
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    sub = np.random.default_rng(0).choice(len(m["X"]), 4000, replace=False)
    ax0.scatter(m["X"][sub, 0], m["X"][sub, 2], m["X"][sub, 1], c=m["s"][sub], s=0.6, cmap="viridis")
    ax0.set_title("(a) tight Swiss roll", fontsize=8); ax0.set_axis_off(); ax0.view_init(55, -60)
    vmax = 0.6
    for j, (key, lab) in enumerate([("GTD_nor_2.0", "(b) GTD, $h=2$"), ("GTD-E_nor_2.0", "(c) GTD-E, $h=2$")]):
        a = fig.add_subplot(1, 3, j + 2)
        sc = a.scatter(m["s"], m["y"], c=np.clip(m[key], 0, vmax), s=0.4, cmap="magma_r", vmin=0, vmax=vmax, rasterized=True)
        a.set_title(lab, fontsize=8); a.set_xlabel("arc length $s$")
        if j == 0: a.set_ylabel("height")
        a.grid(False)
    cb = fig.colorbar(sc, ax=fig.axes[1:], shrink=0.85, pad=0.02); cb.set_label("$\\mathrm{tr}(P_x^{\\perp}\\widehat P)$")
    save(fig, "fig_swiss_maps")

# ------------------------------------------------------------------ E4
if "4" in which:
    r = json.load(open("results/exp4.json"))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for j, name in enumerate(["sphere", "torus"]):
        o = r[name]; n = np.array(o["n"])
        get = lambda k: np.array([x[0] for x in o[k]])
        a = ax[j]
        a.loglog(n, get("terr"), "o-", color=C["GTD"], label="GTD $\\|\\widehat P-P_x\\|$")
        if o["terr_or"][0] is not None:
            a.loglog(n, get("terr_or"), "s--", color=C["oracle"], mfc="none", label="GTD exact $d_{\\mathcal{M}}$")
        a.loglog(n, get("tanerr"), "o-", color=C["GTD"], alpha=0.55, ms=3, label="GTD tangent")
        a.loglog(n, get("pca"), "^-", color=C["LPCA"], label="LPCA tangent")
        ref = get("terr")[0] * (np.log(n) / n / (np.log(n[0]) / n[0])) ** (1 / 3)
        a.loglog(n, ref, ":", color=C["theory"], lw=1, label="$(\\log n/n)^{1/3}$")
        a.set_xlabel("$n$"); a.set_title(f"({'ab'[j]}) {'sphere $S^2$' if name=='sphere' else 'torus $T^2$'}")
        if j == 0: hl = a.get_legend_handles_labels()
        pass
    a = ax[2]
    for name, mk in [("sphere", "o"), ("torus", "^")]:
        o = r[name]; n = np.array(o["n"])
        tot = np.array([o["t_knn"][i][0] + o["t_dij"][i][0] + o["t_ten"][i][0] for i in range(len(n))])
        m = np.array([x[0] for x in o["m"]])
        a.loglog(n, tot, mk + "-", color=C["GTD"], label=f"{name} (total)")
        a.loglog(n, [x[0] for x in o["t_dij"]], mk + ":", color=C["GTD-E"], label=f"{name} (Dijkstra)")
    a.set_xlabel("$n$"); a.set_ylabel("seconds"); a.set_title("(c) runtime"); a.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(*hl, frameon=False, fontsize=7, loc="lower center", ncol=5, bbox_to_anchor=(0.36, 0.0))
    save(fig, "fig_convergence")

# ------------------------------------------------------------------ E5
if "5" in which:
    r = json.load(open("results/exp5.json"))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35))
    cols = {"3": C["GTD"], "20": C["GTD-E"], "100": C["LPCA"]}
    for D, o in r["noise"].items():
        s = np.array(o["sigma"])
        ax[0].semilogy(s[1:], o["tan"][1:], "o-", color=cols[D], label=f"GTD, $D={D}$")
        ax[0].semilogy(s[1:], o["tan_e"][1:], "s--", color=cols[D], ms=3.5, mfc="none", label=f"GTD-E, $D={D}$")
        ax[0].semilogy(s[1:], o["pca"][1:], "^:", color=cols[D], ms=3.5, label=f"LPCA, $D={D}$")
        ok = np.array(o["dimok"]) > 0.5
        ax[1].semilogy(s[ok], np.array(o["nor"])[ok], "o", color=cols[D], label=f"$D={D}$")
        ax[1].semilogy(s, o["nor_pred"], "--", color=cols[D], lw=0.9)
    ax[0].set_xlabel("noise $\\sigma$"); ax[0].set_ylabel("mean tangent error"); ax[0].set_title("(a) noise robustness")
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], color=cols[D], lw=2, label=f"$D={D}$") for D in cols] + \
         [Line2D([], [], color="#555", ls="-", marker="o", label="GTD"),
          Line2D([], [], color="#555", ls="--", marker="s", mfc="none", label="GTD-E"),
          Line2D([], [], color="#555", ls=":", marker="^", label="LPCA")]
    ax[0].set_ylim(2.5e-3, 1.6)
    ax[0].legend(handles=hd, frameon=True, framealpha=0.9, edgecolor="none", fontsize=6, ncol=2, loc="lower right")
    ax[1].set_xlabel("noise $\\sigma$"); ax[1].set_ylabel("mean normal eigenvalue"); ax[1].set_title("(b) normal block vs prediction")
    ax[1].legend(frameon=False, fontsize=6.5)
    rt = r["runtime"]; Ds = np.array([int(k) for k in rt])
    ax[2].loglog(Ds, [rt[k]["knn"] for k in rt], "o-", color=C["GTD-E"], label="$k$-NN graph")
    ax[2].loglog(Ds, [rt[k]["dij"] for k in rt], "s-", color=C["LPCA"], label="truncated Dijkstra")
    ax[2].loglog(Ds, [rt[k]["ten"] for k in rt], "^-", color=C["GTD"], label="local tensors (SVD)")
    ax[2].set_xlabel("ambient dimension $D$"); ax[2].set_ylabel("seconds"); ax[2].set_title("(c) runtime, $S^3$, $n=5000$")
    ax[2].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); save(fig, "fig_noise_runtime")

# ------------------------------------------------------------------ E7/E8 real data
if "7" in which:
    r8 = json.load(open("results/exp8.json")); r7 = json.load(open("results/exp7.json"))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for j, (key, title) in enumerate([("rot", "(a) rotation manifold ($d=1$)"), ("rotscale", "(b) rotation $\\times$ scale ($d=2$)")]):
        o = r8[key]; ch = np.array(o["ch"])
        ax[j].plot(ch, o["gtd"], "o-", color=C["GTD"], label="GTD")
        ax[j].plot(ch, o["ctr"], "D-", color=C["GTD-c"], label="GTD-ctr")
        ax[j].plot(ch, o["pca"], "^-", color=C["LPCA"], label="LPCA ($k=\\bar m$)")
        ax[j].set_xlabel("$c_h$  ($h=c_h\\,\\mathrm{median}\\,r_k$)"); ax[j].set_title(title)
        for x, y, m in zip(ch[::2], np.array(o["pca"])[::2], np.array(o["m"])[::2]):
            ax[j].annotate(f"$\\bar m$={m}", (x, y), textcoords="offset points", xytext=(-6, 6), fontsize=6, color="#666")
    ax[0].set_ylabel("mean tangent error"); ax[0].legend(frameon=False, fontsize=7)
    acc = r7["digits"]["acc"]; ds = [2, 4, 6, 8]
    base = np.mean(acc["1-NN (Euclidean)"])
    for key, col, mk, lab in [("GTD", C["GTD"], "o", "TD with GTD tangents"), ("LPCA-m", C["LPCA"], "^", "TD with LPCA ($k=\\bar m$)"),
                              ("LPCA-10", C["GTD-E"], "s", "TD with LPCA ($k=10$)")]:
        mu = [100 * np.mean(acc[f"{key}|{d}"]) for d in ds]; sd = [100 * np.std(acc[f"{key}|{d}"]) / np.sqrt(10) for d in ds]
        ax[2].errorbar(ds, mu, yerr=sd, fmt=mk + "-", color=col, label=lab, capsize=2, ms=4)
    ax[2].axhline(100 * base, color="#666", ls=":", lw=1); ax[2].text(5.0, 100 * base + 0.03, "Euclidean 1-NN", fontsize=7, color="#666", va="bottom")
    ax[2].set_xlabel("tangent dimension $d$"); ax[2].set_ylabel("test accuracy (\\%)"); ax[2].set_title("(c) digits, tangent-distance 1-NN")
    ax[2].set_ylim(98.25, 99.45); ax[2].legend(frameon=False, fontsize=6.2, loc="upper left"); ax[2].set_xticks(ds)
    fig.tight_layout(); save(fig, "fig_real")

if "8" in which:
    from gtd import gtd, local_pca, knn_graph
    z = np.load("cache/rotu_70.npz"); X, Q = z["X"], z["Q"]
    G, rk = knn_graph(X, 30); h = np.median(rk)
    oc = gtd(X, 1, 30, h, G=G, center=True); mb = int(round(oc["m"].mean()))
    op = local_pca(X, 1, mb)
    U, Up = oc["U"][:, :, 0], op["U"][:, :, 0]
    idx = [5, 400, 1200]
    fig, ax = plt.subplots(3, 4, figsize=(5.4, 4.3))
    for r_, i in enumerate(idx):
        t = Q[i, :, 0]; u = U[i] * np.sign(U[i] @ t); up = Up[i] * np.sign(Up[i] @ t)
        v = np.abs(t).max()
        for c_, (img, title, cm, vmin, vmax) in enumerate([(X[i], "sample $x_i$", "gray", 0, 1),
                                                            (t, "true tangent", "RdBu_r", -v, v),
                                                            (u, "GTD-ctr tangent", "RdBu_r", -v, v),
                                                            (up, "LPCA tangent", "RdBu_r", -v, v)]):
            a = ax[r_, c_]; a.imshow(img.reshape(32, 32), cmap=cm, vmin=vmin, vmax=vmax)
            a.set_xticks([]); a.set_yticks([]); a.grid(False); a.spines[:].set_visible(True)
            if r_ == 0: a.set_title(title, fontsize=8)
        eg = np.sqrt(max(0, 1 - (u @ t) ** 2)); ep = np.sqrt(max(0, 1 - (up @ t) ** 2))
        ax[r_, 2].set_xlabel(f"error {eg:.3f}", fontsize=7); ax[r_, 3].set_xlabel(f"error {ep:.3f}", fontsize=7)
    fig.tight_layout(); save(fig, "fig_image_tangents")
