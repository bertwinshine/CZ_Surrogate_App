# CZ melt flow surrogate

An interactive reduced-order model of the Czochralski silicon melt. Move a
slider, and the predicted velocity, pressure or temperature field appears
immediately. The full CFD case behind each prediction takes minutes to solve in
ANSYS Fluent; the surrogate reconstructs it in about 0.1 ms.

It is built on 33 certified reference cases across three parameter sweeps, and it
is deliberately honest about where it stops working.

## Try it

```
pip install -r requirements.txt
streamlit run app.py
```

Opens in a browser. Nothing else to configure.

## What it does

Three parameters, one moving at a time:

| Parameter | Range | Cases | Baseline |
|---|---|---|---|
| Crucible wall temperature | 1730 – 1785 K | 10 | 1745 K |
| Crucible rotation | −1 to −10 rpm | 12 | −3 rpm |
| Crystal rotation | 4 to 20 rpm | 11 | 8 rpm |

Pick one, move it, and the other two hold at baseline. The app locks them rather
than letting them move, and the reason is in the limits section below.

## How it works

Two steps, both classical linear algebra. No neural network.

**Proper Orthogonal Decomposition.** The reference cases are stacked into a
snapshot matrix and decomposed with an SVD. Almost all the variation across a
sweep turns out to live in a handful of spatial patterns — for the swirl velocity
a single mode carries 99.96 % of the energy. Each field is then described by two
to six numbers instead of 8181.

**Radial basis interpolation.** An RBF maps the parameter value to those few
coefficients. Reconstructing a field is one small matrix multiply, which is why
it is instant.

That is the whole model. It is meant as a baseline: a method with more machinery
in it, physics-informed or otherwise, has to beat this to justify the complexity.

## Accuracy

Leave-one-out: each case is predicted by a model trained without it, then
compared to the CFD result. Interior cases only, as a percentage of each field's
range.

| Axis | u_r | u_z | u_swirl | p | T |
|---|---|---|---|---|---|
| Temperature | 0.06 % | 0.07 % | 0.00 % | 0.10 % | 0.06 % |
| Crucible | 0.23 % | 0.27 % | 0.04 % | 0.47 % | 0.25 % |
| Crystal | 0.88 % | 1.78 % | 0.08 % | 1.16 % | 1.49 % |

The crystal axis is worse for a reason worth stating. Between 6 and 7 rpm the
meridional circulation reverses direction — the axial velocity correlation
between adjacent cases sits above 0.92 everywhere in that sweep except across
that one step, where it falls to 0.135. POD builds its basis from the snapshots
and the interpolator blends them, so predicting a point between two opposed
circulations returns something close to their average: a weak field resembling
neither. Adding cases at 5 and 7 rpm cut the error there roughly in half, but a
linear reduced-order model crossing a structural change keeps a residual. The app
flags that region rather than hiding it.

The crucible axis has a milder version of the same thing, a velocity minimum near
−4 rpm where rotation and buoyancy nearly cancel.

## Limits

**One parameter at a time.** The reference data is three sweeps through a common
baseline. Geometrically that is a star, not a volume — there is no data anywhere
off those three lines. So the model has never seen two parameters move together
and cannot know how temperature and rotation interact, which in this problem is a
real effect rather than a technicality: it is buoyancy competing with forced
convection. Letting all three sliders move would produce confident, smooth,
unverifiable answers. Joint sampling across the parameter box is what would fix
this, and it needs new cases, not a bigger model.

**No extrapolation.** Outside the trained range there is no basis to build from.
Holding out an endpoint instead of an interior case gives 12.5 % rather than
tenths of a percent. The sliders stop where the data stops.

**Simplified physics.** Laminar, Boussinesq, steady, free surface imposed as a
prescribed wall rather than solved. No turbulence model, no radiation, no
Marangoni convection, no species transport, no magnetic field. Fixed melt height
and geometry, 8181 nodes under the ANSYS Student licence.

This is a benchmark for method development, not a model of a real puller.

## Reference data

Each sweep lives in its own repository with the case files, the boundary
conditions, and a certification script that re-derives every boundary condition
from the case label and checks the exported field against it — so a case whose
setup does not match its label is rejected rather than quietly used.

- [CZ_Study_TempChange](https://github.com/bertwinshine/CZ_Study_TempChange)
- [CZ_study_Crucible_Sweep](https://github.com/bertwinshine/CZ_study_Crucible_Sweep)
- [CZ_Crystal_Sweep](https://github.com/bertwinshine/CZ_Crystal_Sweep)

## Files

```
app.py             the Streamlit app
build_models.py    reads the case files, writes models.npz
models.npz         POD bases and interpolation coefficients
pod_rbf.py         the surrogate as a reusable module
smoke_test.py      renders every slider position and field, checks all finite
data/              the 33 certified cases, one folder per sweep
```

To rebuild after adding or changing cases:

```
python build_models.py
python smoke_test.py
```

`build_models.py` refuses to proceed if the cases are not all on the same nodes
in the same order, since the snapshot columns would not be comparable and the POD
basis would be meaningless.

## Credit

The ANSYS modelling, the sweep design and the validation are mine. The Python was
written with AI assistance — I understand what it does and can walk through it.
