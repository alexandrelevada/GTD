# Geodesic Tensor Diffusion (GTD)

Python implementation and reproducibility package for the paper

> **Geodesic Tensor Diffusion: Consistent Estimation of the Riemannian Metric Tensor from Point Clouds**  
> Alexandre L. M. Levada — Computing Department, Federal University of São Carlos (UFSCar)

GTD estimates the Riemannian metric tensor (and the tangent space) at every sample of a point cloud
lying on an unknown $d$-dimensional manifold $\mathcal{M}\subset\mathbb{R}^D$. At each sample $x_i$, it computes
the normalized second moment of a short random step $x_j - x_i$ whose transition probabilities

1. use **shortest-path distances on a k-nearest-neighbour graph** (geodesic, not Euclidean, neighbourhoods);
2. are corrected by an **$\alpha = 1$ density normalization** (the result does not depend on how the data were sampled);
3. **exclude the point itself** and may use **graph-smoothed bandwidths**.

The tensor is divided by $c_K h^2$, so its tangent eigenvalues are close to 1 and its normal eigenvalues are close to 0.
This gives a metric estimate with an absolute scale, an estimate of the tangent space, and a threshold rule for the
intrinsic dimension.

---

## Contents

| File | Description |
|---|---|
| `gtd.py` | The estimator (GTD and its variants), the local PCA baseline, error measures, the dimension rule |
| `manifolds.py` | Samplers for the synthetic manifolds, with exact tangent spaces (and exact geodesics where available) |
| `theory.py` | Closed-form predictions of the bias expansion on the torus |
| `exp1_bias.py` … `exp8_real_sweep.py` | One script per experiment (see the table below) |
| `plots.py` | Produces the figures from the saved results |
| `fig_schematic.py` | Produces Figure 1 (illustration of the method) |
| `make_tables.py` | Produces the LaTeX tables of the benchmark |
| `run_all.sh` | Runs the whole pipeline |
| `requirements.txt` | Python dependencies |

---

## 1. Installation

Requires Python ≥ 3.10. The paper's results were obtained with Python 3.13, NumPy 2.5, SciPy 1.18,
scikit-learn 1.9, scikit-image 0.26 and Matplotlib 3.11.

```bash
git clone https://github.com/<user>/<repository>.git
cd <repository>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No external datasets need to be downloaded. The real-data experiments use the `camera` photograph distributed with
scikit-image and the handwritten digits distributed with scikit-learn.

## 2. Quick start

```python
import numpy as np
from gtd import knn_graph, gtd, tangent_error_from_basis, fixed_point_dim
from manifolds import sphere, proj_from_basis

rng = np.random.default_rng(0)
X, T, _ = sphere(3000, rng)                 # 3000 points on the unit sphere S^2 in R^3
_, Q = proj_from_basis(T)                   # exact tangent bases (for evaluation only)

G, rk = knn_graph(X, k=30)                  # symmetric k-NN graph and k-NN radii
h = 2 * np.median(rk)                       # bandwidth: h = c_h * median r_k, with c_h = 2

out = gtd(X, d=2, k=30, h=h, G=G, center=True)   # centred GTD (recommended for tangent spaces)
print(out["eig"][:3, :4])                    # eigenvalues of each tensor: ~1 (tangent), ~0 (normal)
print(tangent_error_from_basis(out["U"][:, :, :2], Q).mean())   # mean tangent error (about 0.012)

# intrinsic dimension when d is unknown (self-consistent threshold rule)
d_hat, _ = fixed_point_dim(gtd(X, d=1, k=30, h=h, G=G)["eig"])
print(d_hat)                                 # 2
```

Main options of `gtd(X, d, k, h, ...)`:

| Argument | Meaning | Default |
|---|---|---|
| `alpha` | density normalization exponent (1 = GTD, 0 = no normalization) | `1.0` |
| `kernel_dist` | `"graph"` (GTD), `"euclid"` (Euclidean kernel), `"oracle"` (exact geodesics, needs `geo_fn`) | `"graph"` |
| `center` | subtract the weighted mean step (centred variant) | `False` |
| `exclude_self` | exclude $x_i$ from its own neighbourhood | `True` |
| `s`, `t` | pilot scales and number of graph-smoothing steps for the bandwidths | `None`, `0` |
| `calibrate` | heuristic rescaling of graph distances (GTD-c) | `False` |
| `return_full` | also return the full $D\times D$ tensors (small $D$ only) | `False` |

The output is a dictionary with the eigenvalues `eig`, the leading eigenvectors `U` (tangent basis in `U[:, :, :d]`),
the neighbourhood sizes `m`, the degrees `q`, and, if requested, the full tensors `full`.

## 3. Reproducing the paper

### Step 1 — create the output folders

```bash
mkdir -p results figs cache
```

All scripts must be run **from the repository root**. Results are stored in `results/` (JSON), figures in `figs/`
(PDF and PNG), and the rendered image manifolds in `cache/`.

### Step 2 — run the experiments

Each script is independent, except where noted, and prints its progress to the terminal. Typical running times are
for a 2-core machine with 8 GB of RAM.

| Script | Paper | What it does | Time |
|---|---|---|---|
| `exp1_bias.py` | Sec. 5.2, Figs. 2–3 | Bias expansion on $S^1$, $S^2$ (exact and graph distances) and on the torus | ~10 min |
| `exp2_density.py` | Sec. 5.2, Fig. 4 | Density invariance: $\alpha=1$ vs $\alpha=0$ on $S^1$ and $S^2$ | ~30 min |
| `exp3_swiss.py` | Sec. 5.2, Fig. 5 | Geodesic vs Euclidean neighbourhoods on a tight Swiss roll | ~10 min |
| `exp4_convergence.py` | Sec. 5.2, Fig. 6 | Convergence rates and runtime for $n$ from 1,000 to 32,000 | ~20 min |
| `exp5_noise_dim.py` | Sec. 5.3, Fig. 7 | Noise robustness in $\mathbb{R}^D$ and runtime versus $D$ | ~15 min |
| `exp6_benchmark.py` | Sec. 5.3, Table 3 | Benchmark on 8 manifolds, bandwidth sweep, metric error, intrinsic dimension | ~30 min |
| `exp7_real.py` | Sec. 5.4, Table 4, Fig. 9(c) | Image manifolds from a photograph; tangent-distance classification of digits | ~20 min |
| `exp8_real_sweep.py` | Sec. 5.4, Fig. 9(a)–(b) | Bandwidth sweep on the image manifolds (reuses `cache/` from `exp7_real.py`) | ~20 min |

```bash
python exp1_bias.py
python exp2_density.py
python exp3_swiss.py
python exp4_convergence.py
python exp5_noise_dim.py
python exp6_benchmark.py
python exp7_real.py
python exp8_real_sweep.py      # run after exp7_real.py
```

> Tip: the scripts are single-threaded at the Python level, so on a machine with several cores you can run
> independent experiments in parallel (for example `exp1`–`exp6` in separate terminals).

### Step 3 — figures and tables

```bash
python plots.py 1 2 3 4 5 7 8   # Figs. 2-9 (the argument selects which experiment to plot)
python fig_schematic.py         # Fig. 1
python make_tables.py           # LaTeX tables -> results/tables_exp6_*.tex
```

| Output | Paper |
|---|---|
| `figs/fig_schematic.pdf` | Figure 1 |
| `figs/fig_bias_curves.pdf`, `figs/fig_torus_blocks.pdf` | Figures 2 and 3 |
| `figs/fig_density.pdf` | Figure 4 (and `fig_density_sphere.pdf`, quoted in the text) |
| `figs/fig_swiss_curves.pdf` | Figure 5 (and `fig_swiss_maps.pdf`, not used in the paper) |
| `figs/fig_convergence.pdf` | Figure 6 |
| `figs/fig_noise_runtime.pdf` | Figure 7 |
| `figs/fig_image_tangents.pdf` | Figure 8 |
| `figs/fig_real.pdf` | Figure 9 |
| `results/tables_exp6_tangent.tex` | Table 3 |
| `results/tables_exp6_{tensor,bandwidth,dimension}.tex` | Numbers quoted in Section 5.3 |
| `results/exp7.json` | Table 4 and the digits results (printed by `exp7_real.py`) |

### Everything at once

```bash
bash run_all.sh
```

## 4. Notes on reproducibility

- All random number generators are seeded, so repeated runs give the same numbers. Timings depend on the machine.
- The real-data scripts render the image manifolds once and save them in `cache/` (about 0.5 GB). Delete this folder
  to force re-rendering.
- Truncated geodesic distances are computed with `scipy.sparse.csgraph.dijkstra`, which returns dense rows. Memory
  grows as (batch size) × $n$. If you run out of memory with large $n$, reduce `batch` in `graph_neighbors` (`gtd.py`).
- Always check that the $k$-NN graph is connected (`out["ncomp"] == 1`). In one dimension a small $k$ can disconnect
  the graph and bias the estimates.

## 5. Practical recommendations

- Use the **uncentred** tensor (`center=False`) when the metric (diffusion) tensor itself is needed, and the
  **centred** one (`center=True`) to estimate tangent spaces.
- Choose $k$ large enough to keep the graph distortion small ($k\approx 20$–$40$ worked well), and
  $h = c_h\,\mathrm{median}_i\, r_k(x_i)$ with $c_h\in[1,3]$, using the smaller values on strongly curved data.
- Graph geodesics are sensitive to noise in high ambient dimension. Denoise the data, or use
  `kernel_dist="euclid"`, when the noise energy $D\sigma^2$ is comparable to the squared graph resolution.

## 6. Citation

If you use this code, please cite:

```bibtex
@article{levada_gtd,
  author  = {Levada, Alexandre L. M.},
  title   = {Geodesic Tensor Diffusion: Consistent Estimation of the Riemannian Metric Tensor from Point Clouds},
  journal = {Computational and Applied Mathematics},
  year    = {2026},
  note    = {Submitted}
}
```

## 7. License

Released under the MIT License (see `LICENSE`).

## Contact

Alexandre L. M. Levada — alexandre.levada@ufscar.br
