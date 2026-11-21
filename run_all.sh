#!/usr/bin/env bash
# Reproduces every experiment, figure and table of the paper (several hours on a laptop).
set -e
mkdir -p results figs cache
python exp1_bias.py          # Figs. 2-3  (bias expansion: circle, sphere, torus)
python exp2_density.py       # Fig. 4     (density invariance)
python exp3_swiss.py         # Fig. 5     (geodesic vs Euclidean neighbourhoods)
python exp4_convergence.py   # Fig. 6     (convergence rates, runtime)
python exp5_noise_dim.py     # Fig. 7     (noise, ambient dimension, runtime)
python exp6_benchmark.py     # Table 3 + bandwidth / metric / dimension results
python exp7_real.py          # Table 4, Fig. 9(c) (image manifolds, digits)
python exp8_real_sweep.py    # Fig. 9(a)-(b)       (bandwidth sweep on image manifolds)
python plots.py 1 2 3 4 5 7 8
python fig_schematic.py      # Fig. 1
python make_tables.py        # LaTeX tables in results/
