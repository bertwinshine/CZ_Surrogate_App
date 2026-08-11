"""
pod_rbf.py — POD + RBF surrogate for the CZ melt-flow reference dataset.

Builds a reduced-order model from certified ANSYS Fluent snapshots.
Method: mean-subtract, SVD for the POD basis, RBF interpolation of the
POD coefficients against the sweep parameter, then reconstruct the field.

No ML framework needed — numpy and scipy only.
"""
import numpy as np
from scipy.interpolate import RBFInterpolator

FIELDS = ["u_r", "u_z", "u_swirl", "p", "T"]


class PODRBF:
    def __init__(self, n_modes=None, energy=0.9999, kernel="thin_plate_spline", degree=1):
        self.n_modes, self.energy = n_modes, energy
        self.kernel, self.degree = kernel, degree

    def fit(self, params, snapshots):
        """params: (n_cases, n_params). snapshots: dict field -> (n_nodes, n_cases)."""
        self.params = np.atleast_2d(np.asarray(params, float))
        if self.params.shape[0] == 1 and self.params.shape[1] != 1:
            self.params = self.params.T
        self.lo, self.hi = self.params.min(0), self.params.max(0)
        self.models = {}
        for f, X in snapshots.items():
            mu = X.mean(axis=1, keepdims=True)
            U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
            e = np.cumsum(S**2) / np.sum(S**2)
            k = self.n_modes or int(np.searchsorted(e, self.energy) + 1)
            k = min(k, X.shape[1] - 1)
            A = (np.diag(S[:k]) @ Vt[:k]).T          # coefficients, (n_cases, k)
            rbf = RBFInterpolator(self.params, A, kernel=self.kernel, degree=self.degree)
            self.models[f] = dict(mu=mu, U=U[:, :k], rbf=rbf, k=k, energy=e[:k])
        return self

    def predict(self, q):
        q = np.atleast_2d(np.asarray(q, float))
        return {f: (m["U"] @ m["rbf"](q).T + m["mu"]) for f, m in self.models.items()}

    def in_range(self, q):
        q = np.atleast_2d(np.asarray(q, float))
        return np.all((q >= self.lo) & (q <= self.hi), axis=1)


def loo(params, snapshots, **kw):
    """Leave-one-out error, % of each field's range. Returns dict field -> array."""
    params = np.asarray(params, float).reshape(len(params), -1)
    n = len(params)
    out = {f: np.zeros(n) for f in snapshots}
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        sub = {f: X[:, tr] for f, X in snapshots.items()}
        m = PODRBF(**kw).fit(params[tr], sub)
        pred = m.predict(params[i])
        for f, X in snapshots.items():
            t = X[:, i:i + 1]
            rng = t.max() - t.min()
            out[f][i] = 100 * np.sqrt(((pred[f] - t) ** 2).mean()) / (rng if rng else 1)
    return out
