import json, numpy as np

r = json.load(open("results/exp6.json"))
tab, dim = r["tab"], r["dim"]
out = []
cols = ["GTD", "GTD-ctr", "GTD-c", "GTD ($\\alpha=0$)", "GTD-E", "GTD + self", "LPCA ($k=\\bar m$)", "LPCA ($k=30$)"]
hdr = ["GTD", "GTD-ctr", "GTD-c", "$\\alpha=0$", "GTD-E", "+self", "LPCA$_{\\bar m}$", "LPCA$_{30}$"]


def fmt_row(vals, sds, best_set):
    s = []
    for j, (v, sd) in enumerate(zip(vals, sds)):
        t = f"{100 * v:.2f}" if np.isfinite(v) else "--"
        if j in best_set: t = "\\textbf{" + t + "}"
        s.append(t)
    return s


# ---- Table: tangent errors
lines = ["\\begin{table}[t]",
         "\\caption{Mean tangent-space error $\\|\\widehat U_i\\widehat U_i^{\\top}-P_{x_i}\\|$ ($\\times10^{2}$), averaged over "
         "5 independent samples of size $n=5000$ ($n=10000$ for $S^5$); $k=30$, $h=2\\,\\mathrm{median}_i r_k(x_i)$. "
         "Standard deviations across samples are at most $0.11$ for $d\\le2$, $0.30$ for $S^3$ and $0.67$ for $S^5$. Bold: best entry among the "
         "estimators that use the same neighbourhood size (all columns except LPCA$_{30}$).}\\label{tab:tangent}",
         "\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}",
         "\\begin{tabular}{@{}lcc" + "c" * len(cols) + "@{}}", "\\toprule",
         "Manifold & $d$ & $\\bar m$ & " + " & ".join(hdr) + "\\\\", "\\midrule"]
for name, t in tab.items():
    vals = [np.mean(t[c]["tan"]) for c in cols]; sds = [np.std(t[c]["tan"]) for c in cols]
    best = {int(np.argmin(vals[:-1]))}
    lines.append(f"{name} & {t['d']} & {int(np.mean(t['m']))} & " + " & ".join(fmt_row(vals, sds, best)) + "\\\\")
lines += ["\\botrule", "\\end{tabular}", "\\end{table}"]
out.append("\n".join(lines))

# ---- Table: tensor errors
tc = cols[:6]
lines = ["\\begin{table}[t]",
         "\\caption{Mean metric-tensor error $\\|\\widehat P_i-P_{x_i}\\|$ ($\\times10^{2}$) for the normalized estimators, "
         "same setting as Table~\\ref{tab:tangent}.}\\label{tab:tensor}",
         "\\centering\\footnotesize", "\\begin{tabular}{@{}l" + "c" * len(tc) + "@{}}", "\\toprule",
         "Manifold & " + " & ".join(hdr[:6]) + "\\\\", "\\midrule"]
for name, t in tab.items():
    vals = [np.mean(t[c]["ten"]) if t[c]["ten"] else np.nan for c in tc]
    best = {int(np.nanargmin(vals))}
    lines.append(f"{name} & " + " & ".join(fmt_row(vals, [0] * len(vals), best)) + "\\\\")
lines += ["\\botrule", "\\end{tabular}", "\\end{table}"]
out.append("\n".join(lines))

# ---- Table: bandwidth sensitivity (GTD vs LPCA at matched neighbourhood size)
lines = ["\\begin{table}[t]",
         "\\caption{Bandwidth sensitivity: mean tangent error ($\\times10^{2}$) of GTD and local PCA "
         "with the \\emph{same} average neighbourhood size, for $h=c_h\\,\\mathrm{median}_i r_k(x_i)$.}\\label{tab:bandwidth}",
         "\\centering\\footnotesize", "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{$c_h=1.5$} & \\multicolumn{2}{c}{$c_h=2$} & \\multicolumn{2}{c}{$c_h=3$}\\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
         "Manifold & GTD & LPCA & GTD & LPCA & GTD & LPCA\\\\", "\\midrule"]
for name, t in tab.items():
    row = [name]
    for g, l in [("GTD c1.5", "LPCA c1.5"), ("GTD", "LPCA ($k=\\bar m$)"), ("GTD c3.0", "LPCA c3.0")]:
        gm, lm = np.mean(t[g]["tan"]), np.mean(t[l]["tan"])
        mm = "--"
        row += [f"{100 * gm:.2f}", f"{100 * lm:.2f}"]
    lines.append(" & ".join(row) + "\\\\")
lines += ["\\botrule", "\\end{tabular}", "\\end{table}"]
bw = "\n".join(lines)

# ---- Table: dimension
lines = ["\\begin{table}[t]",
         "\\caption{Intrinsic-dimension estimation (5 samples). GTD: largest self-consistent dimension of the threshold "
         "rule (Remark~\\ref{rem:dimension}), with the uncentred and the centred tensor; ``\\%'' is the fraction of "
         "samples $x_i$ whose local count equals $d$; GTD-gap: median largest eigen-ratio gap of the centred tensor; "
         "MLE: Levina--Bickel estimator ($k=20$); LPCA: median largest eigen-ratio gap of local PCA.}\\label{tab:dimension}",
         "\\centering\\footnotesize", "\\begin{tabular}{@{}lcccccccc@{}}", "\\toprule",
         "Manifold & $d$ & GTD & \\% & GTD-ctr & \\% & GTD-gap & MLE & LPCA\\\\", "\\midrule"]
for name, t in dim.items():
    md = lambda a: int(np.median(a))
    lines.append(f"{name} & {t['d']} & {md(t['gtd'])} & {100 * np.mean(t['frac']):.1f} & {md(t['gtdc'])} & "
                 f"{100 * np.mean(t['fracc']):.1f} & {md(t['gap'])} & {np.mean(t['mle']):.2f} & {md(t['lpca'])}\\\\")
lines += ["\\botrule", "\\end{tabular}", "\\end{table}"]
out.append("\n".join(lines))
out.insert(2, bw)
open("results/tables_exp6.tex", "w").write("\n\n".join(out))
for name, txt in zip(["tangent", "tensor", "bandwidth", "dimension"], out):
    open(f"results/tables_exp6_{name}.tex", "w").write(txt + "\n")
print("\n\n".join(out))
# numbers for the bandwidth table (with m values)
for name, t in tab.items():
    print(name, [(round(np.mean(t[g]["tan"]) * 100, 2), round(np.mean(t[l]["tan"]) * 100, 2)) for g, l in
                 [("GTD c1.5", "LPCA c1.5"), ("GTD", "LPCA ($k=\\bar m$)"), ("GTD c3.0", "LPCA c3.0")]])
