"""Geodesic Tensor Diffusion (GTD) and baseline local tensor estimators.

Kernel profile: K(t) = (1 - t)_+ (Epanechnikov-type, Lipschitz, compact support).
For intrinsic dimension d:  c_K = 1/(d+4),  rho_K = m4/m2 = 1/(d+6).
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from sklearn.neighbors import NearestNeighbors


def K(t):
    return np.clip(1.0 - t, 0.0, None)


def c_K(d):
    return 1.0 / (d + 4.0)


def rho_K(d):
    return 1.0 / (d + 6.0)


# ----------------------------------------------------------------------------
# graph
# ----------------------------------------------------------------------------
def knn_graph(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, ind = nn.kneighbors(X)
    n = X.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = ind[:, 1:].ravel()
    vals = dist[:, 1:].ravel()
    G = csr_matrix((vals, (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)  # symmetric kNN graph
    rk = dist[:, -1]
    return G.tocsr(), rk


def graph_neighbors(G, radius, batch=1000):
    """Truncated Dijkstra: for each i, neighbors j != i with d_G(i,j) <= radius[i]."""
    n = G.shape[0]
    nbr, dst = [None] * n, [None] * n
    rmax = float(np.max(radius))
    for s in range(0, n, batch):
        idx = np.arange(s, min(n, s + batch))
        D = dijkstra(G, directed=False, indices=idx, limit=rmax)
        for r, i in enumerate(idx):
            row = D[r]
            j = np.nonzero(row <= radius[i])[0]
            j = j[j != i]
            nbr[i], dst[i] = j, row[j]
    return nbr, dst


def euclid_neighbors(X, radius):
    nn = NearestNeighbors().fit(X)
    dist, ind = nn.radius_neighbors(X, radius=float(np.max(radius)))
    nbr, dst = [], []
    for i in range(X.shape[0]):
        m = (ind[i] != i) & (dist[i] <= radius[i])
        nbr.append(ind[i][m]); dst.append(dist[i][m])
    return nbr, dst


def oracle_neighbors(X, radius, geo_fn, chord_factor=1.0):
    """Exact geodesic distances (for validation). geo_fn(X, i, J) -> geodesic distances."""
    nn = NearestNeighbors().fit(X)
    dist, ind = nn.radius_neighbors(X, radius=float(np.max(radius)) * chord_factor)
    nbr, dst = [], []
    for i in range(X.shape[0]):
        J = ind[i][ind[i] != i]
        g = geo_fn(X, i, J)
        m = g <= radius[i]
        nbr.append(J[m]); dst.append(g[m])
    return nbr, dst


# ----------------------------------------------------------------------------
# bandwidths
# ----------------------------------------------------------------------------
def smoothed_bandwidths(G, h, s, t):
    n = G.shape[0]
    A = (G > 0).astype(float) + csr_matrix(np.eye(n))
    deg = np.asarray(A.sum(1)).ravel()
    s = s / s.mean()
    for _ in range(t):
        s = (A @ s) / deg
    return h * s


# ----------------------------------------------------------------------------
# estimator
# ----------------------------------------------------------------------------
def diffusion_tensors(X, nbr, dst, d, hvec, alpha=1.0, exclude_self=True,
                      n_eig=None, return_full=False, center=False):
    """Compute GTD-type tensors from neighbor lists.

    Returns dict with eigenvalues (descending, length n_eig), top eigenvectors U (n, D, dmax),
    optionally full tensors (n, D, D) when D is small.
    """
    n, Dm = X.shape
    cK = c_K(d)
    if n_eig is None:
        n_eig = min(Dm, d + 6)
    # kernel values with symmetric bandwidth h_ij = sqrt(h_i h_j)
    Kv = []
    q = np.zeros(n)
    for i in range(n):
        hij2 = hvec[i] * hvec[nbr[i]]
        kv = K(dst[i] ** 2 / hij2)
        Kv.append(kv)
        q[i] = kv.sum() + (0.0 if exclude_self else K(0.0))
    eig = np.zeros((n, n_eig))
    trace = np.zeros(n)
    dmax = min(Dm, d + 3)
    U = np.zeros((n, Dm, dmax))
    full = np.zeros((n, Dm, Dm)) if return_full else None
    isolated = 0
    for i in range(n):
        j = nbr[i]
        kv = Kv[i]
        a = kv / np.maximum(q[j], 1e-300) ** alpha
        if exclude_self:
            tot = a.sum()
        else:
            tot = a.sum() + K(0.0) / q[i] ** alpha
        if tot <= 0 or len(j) == 0:
            isolated += 1
            continue
        w = a / tot
        dX = X[j] - X[i]
        if center:  # centred variant: subtract the weighted mean displacement
            dX = dX - w @ dX
        Z = (np.sqrt(w)[:, None] * dX) / (np.sqrt(cK) * hvec[i])
        if return_full:
            P = Z.T @ Z
            full[i] = P
            ev, V = np.linalg.eigh(P)
            ev, V = ev[::-1], V[:, ::-1]
        else:
            _, sv, Vt = np.linalg.svd(Z, full_matrices=False)
            ev, V = sv ** 2, Vt.T
        trace[i] = ev.sum()
        r = min(n_eig, len(ev))
        eig[i, :r] = ev[:r]
        rr = min(dmax, V.shape[1])
        U[i, :, :rr] = V[:, :rr]
    return dict(eig=eig, trace=trace, U=U, full=full, q=q, isolated=isolated,
                m=np.array([len(j) for j in nbr]))


def gtd(X, d, k, h, alpha=1.0, exclude_self=True, kernel_dist="graph",
        geo_fn=None, s=None, t=0, return_full=False, G=None, calibrate=False, center=False):
    if G is None:
        G, _ = knn_graph(X, k)
    ncomp = connected_components(G, directed=False)[0]
    if s is not None and t > 0:
        hvec = smoothed_bandwidths(G, h, s, t)
    else:
        hvec = np.full(X.shape[0], float(h))
    radius = np.sqrt(hvec * hvec.max())
    if kernel_dist == "graph" and calibrate:
        # heuristic: remove the global zig-zag inflation of graph distances
        nbr, dst = graph_neighbors(G, 1.15 * radius)
        ratio = np.concatenate([dst[i] / np.linalg.norm(X[nbr[i]] - X[i], axis=1)
                                for i in range(0, X.shape[0], 10)])
        c = float(np.median(ratio))
        for i in range(X.shape[0]):
            dd = dst[i] / c
            keep = dd <= radius[i]
            nbr[i], dst[i] = nbr[i][keep], dd[keep]
    elif kernel_dist == "graph":
        nbr, dst = graph_neighbors(G, radius)
    elif kernel_dist == "euclid":
        nbr, dst = euclid_neighbors(X, radius)
    elif kernel_dist == "oracle":
        nbr, dst = oracle_neighbors(X, radius, geo_fn)
    else:
        raise ValueError(kernel_dist)
    out = diffusion_tensors(X, nbr, dst, d, hvec, alpha, exclude_self,
                            return_full=return_full, center=center)
    out["h"] = hvec
    out["ncomp"] = ncomp
    return out


def local_pca(X, d, k):
    """Baseline: centred covariance of the k Euclidean nearest neighbours (local PCA)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, ind = nn.kneighbors(X)
    n, Dm = X.shape
    dmax = min(Dm, d + 3)
    U = np.zeros((n, Dm, dmax))
    for i in range(n):
        Y = X[ind[i]] - X[ind[i]].mean(0)
        _, _, Vt = np.linalg.svd(Y, full_matrices=False)
        rr = min(dmax, Vt.shape[0])
        U[i, :, :rr] = Vt[:rr].T
    return dict(U=U)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def tangent_error(U, Ptrue, d):
    """sin of largest principal angle = ||U U^T - P||_op, per sample."""
    Ud = U[:, :, :d]
    E = np.einsum("nia,nja->nij", Ud, Ud) - Ptrue
    return np.linalg.norm(E, ord=2, axis=(1, 2))


def tangent_error_from_basis(U, Tb):
    """U: (n,D,d) estimated orthonormal; Tb: (n,D,d) true orthonormal basis.  Memory-light."""
    M = np.einsum("nia,nib->nab", U, Tb)  # cosines of principal angles
    s = np.linalg.svd(M, compute_uv=False)
    return np.sqrt(np.clip(1.0 - s.min(1) ** 2, 0, 1))


def tensor_error(full, Ptrue):
    return np.linalg.norm(full - Ptrue, ord=2, axis=(1, 2))


def intrinsic_dim_rule(eig):
    return (eig >= 0.5).sum(1)


def levina_bickel(X, k=15):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    dist = dist[:, 1:]
    m = np.log(dist[:, -1:] / dist[:, :-1]).sum(1) / (k - 1)
    return 1.0 / np.mean(m)  # MacKay-Ghahramani averaging of inverse


def fixed_point_dim(eig1, dmax=None):
    """eig1: eigenvalues computed with the normalisation c_K(1) (columns sorted descending).
    Returns the largest self-consistent dimension d' (median count of eigenvalues >= 1/2 after
    renormalising with c_K(d') is at least d'), and the per-point counts at that d'."""
    dmax = eig1.shape[1] - 1 if dmax is None else dmax
    best = 0
    for dc in range(1, dmax + 1):
        cnt = int(np.median((eig1 * (dc + 4) / 5.0 >= 0.5).sum(1)))
        if cnt >= dc:
            best = dc
    dd = max(best, 1)
    return best, (eig1 * (dd + 4) / 5.0 >= 0.5).sum(1)
