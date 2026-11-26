"""Checks that the table <-> per-axis-head index conventions used by the model match ShiftGrid / stage 4."""
import json, numpy as np
from pathlib import Path
from shiftgrid import ShiftGrid
from metrics import table_to_grid, flat_from_axes, axes_from_flat, expected_loss

cfg = json.loads(Path("work/stage4/config.json").read_text())
g = ShiftGrid(cfg["n_steps"], cfg["n_steps_z"], cfg["step"], cfg["min_dim"])
rng = np.random.default_rng(0)

# 1. flat <-> axes round trip and agreement with ShiftGrid.encode
f = rng.integers(0, g.total, 2000)
j = axes_from_flat(f, g.C)
assert np.array_equal(flat_from_axes(j, g.C), f)
sh = g.decode(f)
for k in range(3):
    lo = np.rint(sh[:, 2*k] / g.step).astype(int) + g.ns[k]
    hi = np.rint(sh[:, 2*k+1] / g.step).astype(int) + g.ns[k]
    assert np.array_equal(j[k], lo * g.n[k] + hi), f"axis {k}: head index != lo*n+hi"
print("flat <-> (j0, j1, j2) <-> decoded shifts: consistent")

# 2. one-hot distribution picks exactly the table entry (so the expected-loss head is indexed correctly)
tab = rng.random((4, g.total)).astype(np.float32)
L = table_to_grid(1 - tab, g.C)
fi = rng.integers(0, g.total, 4)
ji = axes_from_flat(fi, g.C)
p = [np.eye(g.C[k])[ji[k]] for k in range(3)]
e = expected_loss(p[0], p[1], p[2], L)
assert np.allclose(e, 1 - tab[np.arange(4), fi]), (e, 1 - tab[np.arange(4), fi])
print("one-hot expected loss == table entry: OK")

# 3. real stage-4 table: argmax from the per-axis grid equals stage-4's own best candidate score
t = np.load("work/stage5/val_tab_tol.npy")[0].astype(np.float32)
G = np.where(np.isnan(t), -np.inf, t).reshape(g.C)
best = np.unravel_index(np.argmax(G), g.C)
assert np.isclose(G[best], np.nanmax(t))
print("real table reshape: max preserved. DONE")
