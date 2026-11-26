#!/usr/bin/env python3
"""DIAGNOSTIC (read-only): run stage 0's pair loading on the pairs that are NOT in work/stage0/index.csv and report why
each one drops out (exception, empty camera cluster, too little shadow, ...). Nothing is written.

    python s0_dryrun.py --input dataset --work work            # only pairs that are not used
    python s0_dryrun.py --input dataset --work work --all      # every pair (to compare used and unused ones)
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from common import find_pairs, read_csv

import s0_augment_pairs as s0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--traceback", action="store_true", help="print the full traceback of every failing pair")
    k = ap.parse_args()
    sys.argv = ["s0_augment_pairs.py", "--input", k.input]                # stage 0's own defaults
    a = s0.parse()
    used = {r["scene"] for r in read_csv(Path(k.work) / "stage0" / "index.csv")}
    pairs = [p for p in find_pairs(k.input) if k.all or p.scene not in used]
    print(f"\n  {len(pairs)} pairs checked with stage 0's loader (min shadow points {a.min_shadow_pts}, shadow margin {a.shadow_margin * 100:.0f} cm, "
          f"cluster min points {a.cluster_min_pts})\n")
    for p in pairs:
        tag = "used  " if p.scene in used else "UNUSED"
        try:
            units, diag = s0.load_pair(p, a)
        except Exception as e:
            print(f"  {tag} {p.scene:<30} LOAD ERROR: {e!r}")
            if k.traceback:
                print(traceback.format_exc())
            continue
        # how does the occluded scan compare with the GT, point by point?
        import json
        from common import read_pcd_xyz
        even = json.loads((p.even_dir / "recording_data.json").read_text())
        gt = np.asarray(even["3d_data_lidar"]["object_points"], float)[:, :3]
        gt = s0.clip_range(gt, a.max_range)
        scan = s0.clip_range(read_pcd_xyz(p.odd_dir / "pointcloud_lidar.pcd"), a.max_range)
        rg, rs = np.linalg.norm(gt, axis=1), np.linalg.norm(scan, axis=1)
        dist, idx = cKDTree(scan / rs[:, None]).query(gt / rg[:, None], k=1, distance_upper_bound=2 * np.sin(np.deg2rad(a.shadow_ang_deg) / 2))
        found = np.isfinite(dist)
        d = rs[idx[found]] - rg[found]
        near, closer, farther = int((np.abs(d) <= a.shadow_margin).sum()), int((d < -a.shadow_margin).sum()), int((d > a.shadow_margin).sum())
        us = " | ".join(f"u{u['idx']}: cluster {len(u['cluster'])} pts, shadow {int(u['gt_shadow'].sum())}/{len(u['gt'])} GT pts" for u in units) or "NO CLUSTER (camera points < min or no component)"
        print(f"  {tag} {p.scene:<30} {us}")
        print(f"         GT object points {len(gt)}: matched to an odd ray {int(found.sum())} | odd range within +-{a.shadow_margin * 100:.0f} cm of GT {near} | odd CLOSER (= shadow) {closer} | odd FARTHER {farther} "
              f"| json-vs-pcd {diag['json_vs_pcd_med_mm']:.1f} mm | calib shift {diag['calib_dpos_cm']:.1f} cm")


if __name__ == "__main__":
    main()