#!/usr/bin/env python3
"""DIAGNOSTIC after stage 4 (not a pipeline stage).  Reads the finished tables; changes nothing.

 A. Grid range.  Stage 4 warns when many optima sit on the grid boundary.  Here, per face:
      * share of samples that CANNOT reach their best score without this face's extreme offset (loss > --eps), and which end
      * how much score is lost on those samples if the face is limited to its inner offsets
      * 'still climbing': the score at the extreme is higher than one step further in  -> the true optimum is beyond the grid
    plus the loss when ALL faces are limited to their inner offsets at once (= what a grid one step smaller would give).
 B. Odd/even consistency.  For GT points that are NOT shadowed (the occluded scan saw them too) the range of the matched
    occluded-scan ray minus the GT range, in cm.  Median ~0 and a small spread -> both recordings share one frame.
    A median of several cm -> the target / LiDAR moved between the recordings (a systematic error in every score).

    python s4c_report.py --work work

 C. Height.  CV box top minus the highest GT points of the target (99.5th percentile; the LiDAR sees the top edge of the
    front face).  Positive = the camera box is taller than the object.
 D. Tilt.  The GT points of the face that looks at the LiDAR are fitted with a line (plane offset against lateral position,
    box frame).  'tilt' = angle between that line and the CV box face (a yaw error of the CV fit; the shift grid cannot fix it,
    it only moves faces parallel to themselves), 'offset' = distance of the fitted plane from the CV face at the face centre.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from common import banner, progress, read_csv, step
from shiftgrid import ShiftGrid
import json

FACES = ["lo_x", "hi_x", "lo_y", "hi_y", "lo_z", "hi_z"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--kind", default="orig", choices=["orig", "aug", "all"])
    ap.add_argument("--eps", type=float, default=0.01, help="a score loss above this counts as 'needs the boundary'")
    ap.add_argument("--ang-deg", type=float, default=0.5, help="ray matching tolerance for part B (same as stage 0)")
    a = ap.parse_args()
    banner("4c", "grid-range and odd/even consistency report")
    work = Path(a.work)
    cfg = json.loads((work / "stage4" / "config.json").read_text())
    grid = ShiftGrid(cfg["n_steps"], cfg["n_steps_z"], cfg["step"], cfg["min_dim"])
    n = [int(x) for x in grid.n]
    s0 = {r["sample_id"]: r for r in read_csv(work / "stage0" / "index.csv")}
    files = sorted((work / "stage4" / "tables").glob("*/*.npz"))
    files = [f for f in files if f.stem in s0 and (a.kind == "all" or s0[f.stem]["kind"] == a.kind) and not s0[f.stem]["category"].lower().startswith("virtual")]
    step(f"{len(files)} samples ({a.kind}) | grid {grid.describe()}")

    loss_face = {f: [] for f in FACES}
    end_of = {f: [] for f in FACES}
    climbing = {f: [] for f in FACES}
    loss_all, best_all, ids = [], [], []
    off_med, off_p90, off_n, off_ids = [], [], [], []
    top_diff, tilt, plane_off, geo_ids = [], [], [], []
    per = {}                                                                   # sid -> dict for the per-sample table
    boxes = np.load(work / "stage1" / "boxes.npz")
    bidx = {s_: i for i, s_ in enumerate(boxes["ids"])}
    for f in progress(files, total=len(files), desc="samples"):
        sid = f.stem
        t = np.load(f)["score_tol"].astype(np.float64).reshape(n[0], n[0], n[1], n[1], n[2], n[2])
        t = np.where(np.isnan(t), -np.inf, t)
        m = t.max()
        if not np.isfinite(m):
            continue
        ids.append(sid)
        per[sid] = dict(best=float(m), cv=float(t[tuple(n_ // 2 for n_ in [n[0], n[0], n[1], n[1], n[2], n[2]])]))
        best_all.append(m)
        inner = t[1:-1, 1:-1, 1:-1, 1:-1, 1:-1, 1:-1] if min(n) > 2 else t
        loss_all.append(m - inner.max())
        for ax, name in enumerate(FACES):
            other = tuple(i for i in range(6) if i != ax)
            v = t.max(axis=other)                                              # best score as a function of this face's offset
            vi = v[1:-1].max()
            loss_face[name].append(m - vi)
            per[sid]["loss_" + name] = float(m - vi)
            end_of[name].append(-1 if v[0] > vi and v[0] >= v[-1] else (1 if v[-1] > vi else 0))
            climbing[name].append(bool((v[0] - v[1] > 0.005 and v[0] >= m - 1e-3) or (v[-1] - v[-2] > 0.005 and v[-1] >= m - 1e-3)))

        z = np.load(s0[sid]["path"])
        org, scan, gt, sh = z["origin"], z["scan"].astype(np.float64), z["gt"].astype(np.float64), z["gt_shadow"]
        # ---- C, D: height and tilt of the CV box against the GT of the target ------------------------------------------
        if sid in bidx:
            bi = bidx[sid]
            c0, d0, R0 = boxes["center"][bi], boxes["dims"][bi], boxes["R"][bi]
            lg = (gt - c0) @ R0
            foot = (np.abs(lg[:, 0]) < d0[0] / 2 + 0.03) & (np.abs(lg[:, 1]) < d0[1] / 2 + 0.03)
            if foot.sum() >= 30:
                top_diff.append(float((c0[2] + d0[2] / 2 - np.percentile(gt[foot, 2], 99.5)) * 100))
                per[sid]["top"] = top_diff[-1]
            lo_ = (org - c0) @ R0
            k = int(np.argmax(np.abs(lo_[:2]) / (d0[:2] / 2)))                    # face of the box that looks at the LiDAR
            sg = 1.0 if lo_[k] > 0 else -1.0
            lat_k = 1 - k
            near = (np.abs(lg[:, k] * sg - d0[k] / 2) < 0.08) & (np.abs(lg[:, lat_k]) < d0[lat_k] / 2 - 0.03) & (np.abs(lg[:, 2]) < d0[2] / 2)
            if near.sum() >= 40:
                x, y = lg[near, lat_k], lg[near, k] * sg - d0[k] / 2
                p = np.polyfit(x, y, 1)
                keep = np.abs(y - np.polyval(p, x)) <= np.percentile(np.abs(y - np.polyval(p, x)), 80)
                p = np.polyfit(x[keep], y[keep], 1)
                tilt.append(float(np.degrees(np.arctan(p[0]))))
                plane_off.append(float(p[1] * 100))
                geo_ids.append(sid)
                per[sid]["tilt"], per[sid]["off"] = tilt[-1], plane_off[-1]

        # ---- B: odd/even range consistency on non-shadowed GT points --------------------------------------------
        vs, rs = scan - org, np.linalg.norm(scan - org, axis=1)
        vg, rg = gt - org, np.linalg.norm(gt - org, axis=1)
        d, idx = cKDTree(vs / rs[:, None]).query(vg / rg[:, None])
        ok = (~sh) & (np.degrees(2 * np.arcsin(np.clip(d / 2, 0, 1))) <= a.ang_deg)
        if ok.sum() >= 15:
            diff = np.clip((rs[idx[ok]] - rg[ok]) * 100, -10, 10)
            off_med.append(float(np.median(diff)))
            off_p90.append(float(np.percentile(np.abs(diff), 90)))
            off_n.append(int(ok.sum()))
            off_ids.append(sid)

    ids = np.array(ids)
    best_all, loss_all = np.array(best_all), np.array(loss_all)
    print(f"\n  A. GRID RANGE   offsets xy {grid.offsets(0)[0] * 100:+.0f}..{grid.offsets(0)[-1] * 100:+.0f} cm, z {grid.offsets(2)[0] * 100:+.0f}..{grid.offsets(2)[-1] * 100:+.0f} cm")
    print(f"     oracle score (tol): median {np.median(best_all):.3f};  with every face limited to its inner offsets: median {np.median(best_all - loss_all):.3f} "
          f"(mean loss {loss_all.mean():.3f}, p90 {np.percentile(loss_all, 90):.3f}, {np.mean(loss_all > a.eps):.0%} of samples lose > {a.eps})")
    print(f"     {'face':<5} {'needs boundary':>14} {'at lower end':>13} {'at upper end':>13} {'mean loss':>10} {'p90 loss':>9} {'still climbing':>15}")
    for name in FACES:
        L, E = np.array(loss_face[name]), np.array(end_of[name])
        need = L > a.eps
        print(f"     {name:<5} {np.mean(need):>14.0%} {np.mean(need & (E == -1)):>13.0%} {np.mean(need & (E == 1)):>13.0%} {L.mean():>10.3f} {np.percentile(L, 90):>9.3f} {np.mean(climbing[name]):>15.0%}")
    worst = np.argsort(-loss_all)[:5]
    print("     most affected samples: " + ", ".join(f"{ids[i]} (-{loss_all[i]:.2f})" for i in worst))

    print(f"\n  B. ODD/EVEN CONSISTENCY   (r_occluded_scan - r_GT on GT points the occluded scan also saw, cm; + = occluded scan farther)")
    if off_med:
        om = np.array(off_med)
        print(f"     {len(om)} samples with >= 15 matched points | per-sample median offset: median {np.median(om):+.2f} cm, p10 {np.percentile(om, 10):+.2f}, p90 {np.percentile(om, 90):+.2f}")
        print(f"     per-sample p90 of |offset|: median {np.median(off_p90):.2f} cm")
        for i in np.argsort(-np.abs(om))[:3]:
            print(f"       largest: {off_ids[i]}: median {om[i]:+.2f} cm over {off_n[i]} points")
    else:
        print("     too few non-shadowed GT points to say anything (full occlusions have none).")

    print("\n  C. HEIGHT   CV box top minus top of the target's GT points [cm]  (+ = box taller than the object)")
    if top_diff:
        td = np.array(top_diff)
        print(f"     {len(td)} samples: median {np.median(td):+.1f}, p10 {np.percentile(td, 10):+.1f}, p90 {np.percentile(td, 90):+.1f}, share above +4 cm: {np.mean(td > 4):.0%}")

    print("\n  D. TILT of the LiDAR-facing face against the CV box")
    if tilt:
        ta, po = np.array(tilt), np.array(plane_off)
        print(f"     {len(ta)} samples: tilt median {np.median(ta):+.1f} deg, median |tilt| {np.median(np.abs(ta)):.1f} deg, p90 |tilt| {np.percentile(np.abs(ta), 90):.1f} deg;"
              f" plane offset at the face centre: median {np.median(po):+.1f} cm (+ = object face in FRONT of the CV face, nearer the LiDAR; - = behind it)")
        print(f"     a tilt of 4 deg moves the face {np.tan(np.radians(4)) * 50:.1f} cm over a 50 cm wide face")
        print("     caution: the tilt estimate is inflated when the LiDAR looks at the box near its diagonal (two faces mixed); check large ones in the picture.")

    print("\n  PER SAMPLE   (oracle/CV = tol score; height = box top - GT top [cm]; tilt [deg]; offset = object face vs CV face [cm], + = nearer the LiDAR; loss hi_z = what the z range costs)")
    print(f"     {'sample':<40} {'CV box':>6} {'oracle':>6} {'height':>7} {'tilt':>6} {'offset':>7} {'loss hi_z':>9}")
    for sid in ids:
        p = per[sid]
        g = lambda k, f: (f % p[k]) if k in p else "      -"
        print(f"     {sid:<40} {p['cv']:>6.2f} {p['best']:>6.2f} {g('top', '%+7.1f')} {g('tilt', '%+6.1f')} {g('off', '%+7.1f')} {p['loss_hi_z']:>9.3f}")


if __name__ == "__main__":
    main()