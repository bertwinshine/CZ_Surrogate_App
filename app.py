"""
app.py — CZ melt flow surrogate
===============================

Interactive demo of a POD + RBF surrogate built on certified ANSYS Fluent
reference cases of the Czochralski silicon melt.

Move one slider and the predicted flow field updates immediately. The full CFD
case behind each of these takes minutes to solve; the surrogate reconstructs it
from a handful of POD modes in well under a second.

One parameter varies at a time. The reference dataset is three separate sweeps
through a common baseline, so there is no data anywhere off those three lines
and the surrogate has never seen two parameters move together. The app locks the
other two rather than pretending otherwise.

Run with:
    streamlit run app.py
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.interpolate import RBFInterpolator

FIELDS = ["u_r", "u_z", "u_swirl", "p", "T"]

LABEL = {
    "u_r":     ("Radial velocity", "m/s"),
    "u_z":     ("Axial velocity", "m/s"),
    "u_swirl": ("Swirl velocity", "m/s"),
    "p":       ("Pressure", "Pa"),
    "T":       ("Temperature", "K"),
}

AXES = {
    "temperature": dict(
        label="Crucible wall temperature",
        unit="K", step=1.0, baseline=1745.0, fmt="%.0f",
        others="crystal +8 rpm, crucible −3 rpm",
    ),
    "crucible": dict(
        label="Crucible rotation",
        unit="rpm", step=0.1, baseline=-3.0, fmt="%.1f",
        others="1745 K, crystal +8 rpm",
    ),
    "crystal": dict(
        label="Crystal rotation",
        unit="rpm", step=0.1, baseline=8.0, fmt="%.1f",
        others="1745 K, crucible −3 rpm",
    ),
}

# Leave-one-out error over interior cases, % of each field's range.
# From the validation runs; see README.
ACCURACY = {
    "temperature": dict(u_r=0.06, u_z=0.07, u_swirl=0.00, p=0.10, T=0.06),
    "crucible":    dict(u_r=0.23, u_z=0.27, u_swirl=0.04, p=0.47, T=0.25),
    "crystal":     dict(u_r=0.88, u_z=1.78, u_swirl=0.08, p=1.16, T=1.49),
}

# Regions where the flow changes structure and a linear reduced-order model
# degrades. Worth saying out loud rather than letting the plot imply precision
# it does not have.
CAUTION = {
    "crucible": (-4.6, -3.4,
                 "Velocity minimum near −4 rpm: rotation and buoyancy nearly "
                 "cancel here."),
    "crystal":  (6.0, 8.0,
                 "The meridional circulation reverses between 6 and 7 rpm. "
                 "A linear surrogate blends two opposed flow structures across "
                 "this gap, so treat predictions here as indicative."),
}


@st.cache_resource
def load():
    d = np.load("models.npz")
    m = {"r": d["r"], "z": d["z"]}
    for axis in AXES:
        P = d["%s__params" % axis].reshape(-1, 1)
        entry = {"params": P}
        for f in FIELDS:
            A = d["%s__%s__A" % (axis, f)]
            entry[f] = dict(
                mu=d["%s__%s__mu" % (axis, f)],
                U=d["%s__%s__U" % (axis, f)],
                rbf=RBFInterpolator(P, A, kernel="thin_plate_spline", degree=1),
            )
        m[axis] = entry
    m["tri"] = Triangulation(m["r"], m["z"])
    return m


def predict(m, axis, value, field):
    e = m[axis][field]
    return e["U"] @ e["rbf"](np.array([[value]])).ravel() + e["mu"]


def plot(m, values, field, axis, value):
    name, unit = LABEL[field]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    diverging = field in ("u_r", "u_z", "u_swirl")
    if diverging:
        lim = np.abs(values).max()
        levels = np.linspace(-lim, lim, 41)
        cmap = "RdBu_r"
    else:
        levels = np.linspace(values.min(), values.max(), 41)
        cmap = "inferno" if field == "T" else "viridis"

    c = ax.tricontourf(m["tri"], values, levels=levels, cmap=cmap, extend="both")
    fig.colorbar(c, ax=ax, label="%s [%s]" % (name, unit))

    ax.set_xlabel("r  [m]   (axis at left, crucible wall at right)")
    ax.set_ylabel("z  [m]")
    ax.set_title("%s   —   %s = %s %s" % (
        name, AXES[axis]["label"], AXES[axis]["fmt"] % value, AXES[axis]["unit"]))
    ax.set_aspect("equal")

    # crystal edge, where the free surface starts
    ax.axvline(0.18, color="w", lw=0.8, ls="--", alpha=0.6)
    ax.text(0.18, 0.153, " crystal edge", fontsize=7, color="0.3", va="bottom")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- interface --
st.set_page_config(page_title="CZ melt surrogate", layout="wide")
m = load()

st.title("Czochralski melt flow — surrogate model")
st.caption(
    "A reduced-order model built from certified ANSYS Fluent reference cases. "
    "Move the slider and the predicted field updates immediately."
)

left, right = st.columns([1, 2.4])

with left:
    axis = st.radio(
        "Parameter to vary",
        list(AXES),
        format_func=lambda a: AXES[a]["label"],
    )
    cfg = AXES[axis]
    P = m[axis]["params"].ravel()
    lo, hi = float(P.min()), float(P.max())

    value = st.slider(
        "%s [%s]" % (cfg["label"], cfg["unit"]),
        lo, hi, float(cfg["baseline"]), step=cfg["step"], format=cfg["fmt"],
    )
    st.caption("Held fixed: %s" % cfg["others"])

    field = st.selectbox(
        "Field", FIELDS, index=1,
        format_func=lambda f: "%s  [%s]" % (LABEL[f][0], LABEL[f][1]),
    )

    err = ACCURACY[axis][field]
    st.metric("Validated error, this axis", "%.2f %%" % err,
              help="Leave-one-out error over interior cases, as a percentage "
                   "of the field's range. Each case was predicted by a model "
                   "trained without it.")

    st.caption(
        "Trained on %d cases from %s to %s %s. The surrogate interpolates "
        "inside this range only." % (len(P), cfg["fmt"] % lo, cfg["fmt"] % hi, cfg["unit"])
    )

    if axis in CAUTION:
        c_lo, c_hi, msg = CAUTION[axis]
        if c_lo <= value <= c_hi:
            st.warning(msg)

    near_edge = min(abs(value - lo), abs(value - hi)) < 0.02 * (hi - lo)
    if near_edge:
        st.info(
            "At the edge of the sampled range. Accuracy is lowest here — the "
            "model has training data on one side only."
        )

with right:
    values = predict(m, axis, value, field)
    st.pyplot(plot(m, values, field, axis, value), use_container_width=True)

    a, b, c = st.columns(3)
    if field in ("u_r", "u_z", "u_swirl"):
        a.metric("max", "%.5f m/s" % values.max())
        b.metric("min", "%.5f m/s" % values.min())
        c.metric("mean |value|", "%.5f m/s" % np.abs(values).mean())
    else:
        u = LABEL[field][1]
        a.metric("max", "%.3f %s" % (values.max(), u))
        b.metric("min", "%.3f %s" % (values.min(), u))
        c.metric("range", "%.3f %s" % (values.max() - values.min(), u))

with st.expander("What this is, and what it is not"):
    st.markdown(
        """
The reference data is 33 steady 2D axisymmetric CFD cases of the melt, solved in
ANSYS Fluent and certified individually: every boundary condition is re-derived
from the case label and checked against the exported field, so a case whose setup
does not match its label is rejected rather than used.

The surrogate is Proper Orthogonal Decomposition plus radial basis function
interpolation — classical linear algebra, no neural network. Each field is
represented by a handful of POD modes; the interpolator maps the parameter to the
mode coefficients. It exists as a baseline: anything more elaborate has to beat
it to justify its complexity.

**Limits worth knowing.**

*One parameter at a time.* The reference cases are three separate sweeps through
a common baseline. There is no data anywhere off those lines, so nothing here
says how temperature and rotation interact — which in this problem is a real
effect, since it is buoyancy competing with forced convection. The other two
sliders are locked for that reason.

*No extrapolation.* Outside the trained range the model has no basis to build
from and will return something smooth and wrong. The sliders stop at the data.

*Simplified physics.* Laminar, Boussinesq, steady, with the free surface imposed
as a prescribed wall rather than solved. No turbulence model, no radiation, no
Marangoni convection, no species transport, no magnetic field. Fixed melt height
and geometry.

This is a benchmark for method development, not a model of a real puller.
        """
    )

st.caption(
    "Reference data: github.com/bertwinshine — CZ_Study_TempChange, "
    "CZ_study_Crucible_Sweep, CZ_Crystal_Sweep"
)
