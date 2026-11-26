#!/usr/bin/env python3
"""DIAGNOSTIC (read-only): which recordings / pairs of the dataset folder are used, which are not, and why.

Lists per category the recording folders, the odd/even pairs found, and for every pair that is NOT in work/stage0/index.csv
the quick checks that usually explain it (missing files, empty point lists, shadow too small is only known to stage 0).

    python s0_inventory.py --input dataset --work work
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from common import find_pairs, read_csv


def n_points(p, *keys):
    try:
        d = json.loads(Path(p).read_text())
        for k in keys:
            d = d[k]
        return len(d)
    except Exception as e:
        return f"ERR {type(e).__name__}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="dataset")
    ap.add_argument("--work", default="work")
    a = ap.parse_args()
    root = Path(a.input)
    cats = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))
    print(f"\n  folders in {root}:")
    for c in cats:
        recs = sorted(p.name for p in c.iterdir() if p.is_dir() and p.name.startswith("recording_"))
        print(f"     {c.name:<28} {len(recs):>3} recording folders" + (f"   (e.g. {recs[0]} .. {recs[-1]})" if recs else ""))
    print()
    pairs = find_pairs(root)
    used = {r["scene"] for r in read_csv(Path(a.work) / "stage0" / "index.csv")}
    print(f"\n  pairs found (odd + following even recording): {len(pairs)} in {len({p.category for p in pairs})} categories: "
          + ", ".join(f"{k}={v}" for k, v in Counter(p.category for p in pairs).items()))
    print(f"  scenes that reached stage 0's index: {len(used)}")
    miss = [p for p in pairs if p.scene not in used]
    print(f"  pairs NOT used: {len(miss)}\n")
    for p in miss:
        need = ["recording_data.json", "pointcloud_lidar.pcd"]
        absent = [f"{d.name}/{f}" for d in (p.odd_dir, p.even_dir) for f in need if not (d / f).exists()]
        cam_o = n_points(p.odd_dir / "recording_data.json", "3d_data_camera", "object_points")
        cam_e = n_points(p.even_dir / "recording_data.json", "3d_data_camera", "object_points")
        lid_e = n_points(p.even_dir / "recording_data.json", "3d_data_lidar", "object_points")
        print(f"     {p.scene:<32} missing files: {absent or 'none'} | camera cluster points odd/even: {cam_o}/{cam_e} | even LiDAR object points: {lid_e}")
    if not miss:
        print("     (every pair is in the index; if you expected more scenes, the missing ones are not pairs of recording_odd + recording_even in this folder)")


if __name__ == "__main__":
    main()