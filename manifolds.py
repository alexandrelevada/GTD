"""Synthetic manifolds with exact tangent spaces (and exact geodesics where available)."""
import numpy as np


def proj_from_basis(T):
    """T: (n,D,d) -> orthonormalise and return projectors (n,D,D) and basis."""
    Q = np.linalg.qr(T)[0]
    return np.einsum("nia,nja->nij", Q, Q), Q


# ---------------------------------------------------------------- circle
def circle(n, rng, r=1.0, density="uniform", a=0.9):
    if density == "uniform":
        th = rng.uniform(0, 2 * np.pi, n)
    else:  # p(theta) ∝ 1 + a cos(theta)
        th = []
        while len(th) < n:
            c = rng.uniform(0, 2 * np.pi, 2 * n)
            u = rng.uniform(0, 1 + a, 2 * n)
            th.extend(c[u < 1 + a * np.cos(c)])
        th = np.array(th[:n])
    X = r * np.c_[np.cos(th), np.sin(th)]
    T = np.c_[-np.sin(th), np.cos(th)][:, :, None]
    return X, T, dict(theta=th, r=r)


def circle_geo(r):
    def f(X, i, J):
        c = np.clip(X[J] @ X[i] / r ** 2, -1, 1)
        return r * np.arccos(c)
    return f


# ---------------------------------------------------------------- sphere S^d
def sphere(n, rng, d=2, r=1.0, density="uniform", kappa=2.0):
    if density == "uniform":
        Y = rng.normal(size=(n, d + 1))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    else:  # p ∝ exp(kappa * x_last)
        Y = []
        while len(Y) < n:
            Z = rng.normal(size=(3 * n, d + 1))
            Z /= np.linalg.norm(Z, axis=1, keepdims=True)
            u = rng.uniform(0, 1, 3 * n)
            Y.extend(Z[u < np.exp(kappa * (Z[:, -1] - 1))])
        Y = np.array(Y[:n])
    X = r * Y
    # tangent basis: orthonormal complement of Y
    T = np.zeros((n, d + 1, d))
    for i in range(n):
        q = np.linalg.qr(np.c_[Y[i], np.eye(d + 1)[:, :d]])[0]
        T[i] = q[:, 1:d + 1]
    return X, T, dict(Y=Y, r=r)


def sphere_geo(r):
    def f(X, i, J):
        c = np.clip(X[J] @ X[i] / r ** 2, -1, 1)
        return r * np.arccos(c)
    return f


# ---------------------------------------------------------------- torus in R^3
def torus(n, rng, R=2.0, r=1.0):
    u, v = [], []
    while len(u) < n:  # uniform area measure: density ∝ (R + r cos v)
        uu = rng.uniform(0, 2 * np.pi, 2 * n)
        vv = rng.uniform(0, 2 * np.pi, 2 * n)
        acc = rng.uniform(0, R + r, 2 * n) < R + r * np.cos(vv)
        u.extend(uu[acc]); v.extend(vv[acc])
    u, v = np.array(u[:n]), np.array(v[:n])
    X = np.c_[(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)]
    Tu = np.c_[-np.sin(u), np.cos(u), 0 * u]
    Tv = np.c_[-np.sin(v) * np.cos(u), -np.sin(v) * np.sin(u), np.cos(v)]
    T = np.stack([Tu, Tv], axis=2)
    return X, T, dict(u=u, v=v)


# ---------------------------------------------------------------- ellipsoid
def ellipsoid(n, rng, axes=(1.0, 1.5, 2.0)):
    a = np.array(axes)
    Y = rng.normal(size=(n, 3)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    X = Y * a
    N = X / a ** 2; N /= np.linalg.norm(N, axis=1, keepdims=True)
    T = np.zeros((n, 3, 2))
    for i in range(n):
        q = np.linalg.qr(np.c_[N[i], np.eye(3)[:, :2]])[0]
        T[i] = q[:, 1:3]
    return X, T, dict()


# ---------------------------------------------------------------- tight Swiss roll
def swiss_roll(n, rng, a=0.5, t0=1.5 * np.pi, t1=4.5 * np.pi, height=10.0):
    """r(t) = a t; gap between consecutive layers = 2 pi a.  Sampled uniformly in area."""
    # arc length element ds = a sqrt(1+t^2) dt  -> rejection sampling on t
    t = []
    wmax = np.sqrt(1 + t1 ** 2)
    while len(t) < n:
        c = rng.uniform(t0, t1, 2 * n)
        acc = rng.uniform(0, wmax, 2 * n) < np.sqrt(1 + c ** 2)
        t.extend(c[acc])
    t = np.array(t[:n])
    y = rng.uniform(0, height, n)
    X = np.c_[a * t * np.cos(t), y, a * t * np.sin(t)]
    Tt = np.c_[a * (np.cos(t) - t * np.sin(t)), 0 * t, a * (np.sin(t) + t * np.cos(t))]
    Tt /= np.linalg.norm(Tt, axis=1, keepdims=True)
    Ty = np.c_[0 * t, 1 + 0 * t, 0 * t]
    T = np.stack([Tt, Ty], axis=2)
    # intrinsic coordinate along the roll (arc length) for interior masking
    s = 0.5 * a * (t * np.sqrt(1 + t ** 2) + np.arcsinh(t))
    s0 = 0.5 * a * (t0 * np.sqrt(1 + t0 ** 2) + np.arcsinh(t0))
    s1 = 0.5 * a * (t1 * np.sqrt(1 + t1 ** 2) + np.arcsinh(t1))
    dist_bd = np.minimum.reduce([s - s0, s1 - s, y, height - y])
    return X, T, dict(t=t, y=y, dist_bd=dist_bd, gap=2 * np.pi * a)


# ---------------------------------------------------------------- isometric embedding into R^D
def embed(X, T, D, rng):
    Q = np.linalg.qr(rng.normal(size=(D, X.shape[1])))[0]  # D x D0, orthonormal columns
    return X @ Q.T, np.einsum("ij,njk->nik", Q, T), Q
