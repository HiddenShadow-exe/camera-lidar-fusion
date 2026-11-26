#!/usr/bin/env python3
"""STAGE 2 - shift grid: every face of the CV box moves independently along its outward normal.

No data is written (the grid is fully determined by --n-steps / --step); this stage verifies the grid maths on the
real CV boxes of stage 1 so that the labels of stage 4 (shift indices) and the model outputs (6 shifts) are unambiguous.
"""
import argparse
from pathlib import Path

import numpy as np

from common import Gates, banner, obb_corners, step
from shiftgrid import ShiftGrid, add_grid_args, grid_from_args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    add_grid_args(ap)
    a = ap.parse_args()
    banner(2, "face-shift grid")
    g = grid_from_args(a)
    gates = Gates(2)
    step("grid: " + g.describe())
    z = np.load(Path(a.work) / "stage1" / "boxes.npz")
    C, D, R = z["center"], z["dims"], z["R"]
    rng = np.random.default_rng(0)

    zi = g.zero_index()
    gates.check("shift 0 is the identity", np.all(g.decode(zi) == 0) and all(
        np.allclose(g.box_from((C[i], D[i], R[i]), zi)[0], C[i]) and np.allclose(g.box_from((C[i], D[i], R[i]), zi)[1], D[i])
        for i in range(len(C))), f"zero index = {zi}; box_from(0) reproduces all {len(C)} CV boxes exactly")

    idx = rng.integers(0, g.total, 5000)
    gates.check("encode(decode(i)) == i", np.array_equal(g.encode(g.decode(idx)), idx), "5000 random indices")

    # independent check of the face positions through the corners of the shifted box.
    # Only VALID candidates (every dimension >= --min-dim): stage 4 scores the others as NaN, and for them hi - lo can
    # even be negative (thin boxes!), which would make this corner test meaningless.
    bad, n_checked, n_invalid_skipped = 0, 0, 0
    while n_checked < 300:
        i = int(rng.integers(len(C)))
        f = int(rng.integers(g.total))
        j = np.unravel_index(f, g.C)
        if not all(v[jk] for v, jk in zip(g.valid_mask(D[i]), j)):
            n_invalid_skipped += 1
            continue
        n_checked += 1
        sh = g.decode(f)
        c, d, Rm = g.box_from((C[i], D[i], R[i]), f)
        local = (obb_corners(c, d, Rm) - C[i]) @ R[i]            # shifted corners in the ORIGINAL box frame
        for k in range(3):
            lo_expected, hi_expected = -D[i][k] / 2 - sh[2 * k], D[i][k] / 2 + sh[2 * k + 1]
            if not (np.isclose(local[:, k].min(), lo_expected) and np.isclose(local[:, k].max(), hi_expected)):
                bad += 1
    gates.check("faces move along their own normal", bad == 0,
                f"{bad} mismatches in {n_checked} random valid candidates x 3 axes ({n_invalid_skipped} too-thin candidates skipped)")

    thin = sum(int(D[i][:2].min() < 0.12) for i in range(len(D)))
    gates.check("thin boxes", True, f"{thin}/{len(D)} CV boxes have an xy side < 12 cm; for them many (lo, hi) shift combinations "
                f"are invalid (< {g.min_dim * 100:.0f} cm) and are scored NaN", hard=False)
    inv = [int((~v).sum()) for v in g.valid_mask(D[0])]
    gates.check("valid candidates", True, f"first box: invalid (lo,hi) combos per axis {inv} (thinner than {g.min_dim * 100:.0f} cm)", hard=False)
    gates.check("best shift reachable", True,
                f"CV error beyond +-{g.max_offset[0] * 100:.0f} cm (xy) / +-{g.max_offset[2] * 100:.0f} cm (z) cannot be corrected -> "
                f"stage 4 reports how often the optimum sits on the grid boundary", hard=False)
    gates.finish()


if __name__ == "__main__":
    main()