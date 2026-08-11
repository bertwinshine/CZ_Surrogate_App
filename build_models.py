#!/usr/bin/env python3
"""
build_models.py
===============

Reads the certified CZ case files, builds a POD basis per sweep axis per field,
and writes everything the app needs into one compressed archive.

Run once after adding or changing cases:

    python build_models.py

Writes models.npz. The app does not read the CSVs at all, so the archive is the
only thing that has to be deployed alongside it.
"""
import glob, os, re
import numpy as np
import pandas as pd

FIELDS = ["u_r", "u_z", "u_swirl", "p", "T"]
ENERGY = 0.9999          # POD modes kept, as a fraction of total energy
MAXK   = 6               # never keep more than this many modes

# Where the case files live. Each entry: axis name -> (folder, filename pattern,
# function pulling the parameter value out of the filename).
SOURCES = {
    "temperature": ("data/temperature", "*.csv",
                    lambda n: float(re.search(r"(\d{4})", n).group(1))),
    "crucible":    ("data/crucible",    "*.csv",
                    lambda n: -float(re.search(r"m(\d+(?:p\d+)?)rpm", n).group(1).replace("p", "."))),
    "crystal":     ("data/crystal",     "*.csv",
                    lambda n: float(re.search(r"_(\d+)rpm", n).group(1))),
}


def load_axis(folder, pattern, parse):
    """Load every case in a folder, sorted by parameter value."""
    rows = []
    for path in glob.glob(os.path.join(folder, pattern)):
        name = os.path.basename(path)
        rows.append((parse(name), pd.read_csv(path)))
    rows.sort(key=lambda t: t[0])
    if not rows:
        raise SystemExit("no case files found in %s" % folder)
    params = np.array([p for p, _ in rows], float)
    frames = [d for _, d in rows]

    # Every case has to sit on the same nodes in the same order, otherwise the
    # snapshot columns are not comparable and the POD basis is meaningless.
    r0, z0 = frames[0].r.values, frames[0].z.values
    for p, d in zip(params, frames):
        if not (np.allclose(d.r.values, r0) and np.allclose(d.z.values, z0)):
            raise SystemExit("node mismatch at %s = %g in %s" % (folder, p, folder))
    return params, frames, r0, z0


def main():
    out = {}
    coords_done = False
    summary = []

    for axis, (folder, pattern, parse) in SOURCES.items():
        params, frames, r, z = load_axis(folder, pattern, parse)

        if not coords_done:
            out["r"], out["z"] = r, z
            coords_done = True
        elif not (np.allclose(out["r"], r) and np.allclose(out["z"], z)):
            raise SystemExit("axis %s is on a different mesh from the others" % axis)

        out["%s__params" % axis] = params
        ks = []
        for f in FIELDS:
            X = np.column_stack([d[f].values for d in frames])
            mu = X.mean(axis=1, keepdims=True)
            U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
            e = np.cumsum(S**2) / np.sum(S**2)
            k = min(int(np.searchsorted(e, ENERGY) + 1), MAXK, X.shape[1] - 1)
            out["%s__%s__mu" % (axis, f)] = mu.ravel()
            out["%s__%s__U" % (axis, f)] = U[:, :k]
            out["%s__%s__A" % (axis, f)] = (np.diag(S[:k]) @ Vt[:k]).T
            ks.append(k)
        summary.append((axis, len(params), params.min(), params.max(), ks))

    np.savez_compressed("models.npz", **out)

    print("wrote models.npz  (%d nodes)\n" % len(out["r"]))
    print("%-13s %6s %10s %10s   %s" % ("axis", "cases", "min", "max", "modes " + "/".join(FIELDS)))
    for axis, n, lo, hi, ks in summary:
        print("%-13s %6d %10.4g %10.4g   %s" % (axis, n, lo, hi, "/".join(str(k) for k in ks)))


if __name__ == "__main__":
    main()
