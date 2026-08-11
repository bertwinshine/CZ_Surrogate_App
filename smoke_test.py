import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from matplotlib.tri import Triangulation

FIELDS = ["u_r", "u_z", "u_swirl", "p", "T"]
d = np.load("models.npz")
M = {"r": d["r"], "z": d["z"]}
for ax in ["temperature", "crucible", "crystal"]:
    P = d["%s__params" % ax].reshape(-1, 1)
    M[ax] = {"params": P}
    for f in FIELDS:
        M[ax][f] = dict(
            mu=d["%s__%s__mu" % (ax, f)],
            U=d["%s__%s__U" % (ax, f)],
            rbf=RBFInterpolator(P, d["%s__%s__A" % (ax, f)],
                                kernel="thin_plate_spline", degree=1))
tri = Triangulation(M["r"], M["z"])


def pred(a, v, f):
    e = M[a][f]
    return e["U"] @ e["rbf"](np.array([[v]])).ravel() + e["mu"]


n = 0
for ax in ["temperature", "crucible", "crystal"]:
    P = M[ax]["params"].ravel()
    for v in np.linspace(P.min(), P.max(), 7):
        for f in FIELDS:
            y = pred(ax, float(v), f)
            assert np.isfinite(y).all() and len(y) == 8181
            fig, a2 = plt.subplots(figsize=(7.2, 4.0))
            lim = np.abs(y).max()
            lv = (np.linspace(-lim, lim, 41) if f in ("u_r", "u_z", "u_swirl")
                  else np.linspace(y.min(), y.max(), 41))
            a2.tricontourf(tri, y, levels=lv, extend="both")
            plt.close(fig)
            n += 1

print("%d field+plot combinations rendered, all finite, 8181 nodes each" % n)
print("triangulation: %d triangles" % len(tri.triangles))
print()
for ax, unit, base in [("temperature", "K", 1745), ("crucible", "rpm", -3),
                       ("crystal", "rpm", 8)]:
    P = M[ax]["params"].ravel()
    print("  %-12s %8.4g .. %-8.4g %-4s baseline %-6g (%d cases)"
          % (ax, P.min(), P.max(), unit, base, len(P)))
