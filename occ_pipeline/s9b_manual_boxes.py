#!/usr/bin/env python3
"""STAGE 9b - your manually labelled boxes (json key  3d_data_lidar.obb_3d,  the label of the CAMERA cloud) as pipeline boxes.

Approach 4 of the comparison: the hand-drawn box replaces the CV box. It is evaluated exactly like every other box
(s9_compare_methods.py): blocked LiDAR rays are extended to its front face, voxelised, compared with the GT.

For every real sample (= one camera cluster of the EVEN recording) the label is read from the recording json (even first, odd as fallback).
A json may hold one box (dict) or several (list): the label whose centre is nearest to the cluster (in xy) is used, within --max-match-dist.
Frame check: the label is expected in the LiDAR frame. For every label the fraction of the cluster points inside the label (+3 cm) is computed
twice, taking the label as it is and taking it as a camera-frame box, and the frame that fits better is used (reported).
Diagnostics are printed and written to <work>/stage9/manual_boxes.csv: coverage, bottom / top z, size and centre difference to the CV box.

    python s9b_manual_boxes.py --input dataset --work work_virt
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import Gates, Rig, banner, find_pairs, progress, read_csv, rot_z, step, write_csv

ALIASES = {"center": ("center", "centre", "c"), "dims": ("dimensions", "extent", "size", "dims"), "rot": ("rotation", "R", "rot", "rotation_matrix")}


def get_path(doc, dotted):
    for k in dotted.split("."):
        if not isinstance(doc, dict) or k not in doc:
            return None
        doc = doc[k]
    return doc


def pick(o, names):
    for n in names:
        if n in o:
            return o[n]
    return None


def parse_boxes(o):
    """json value -> list of (centre, dims, R). Accepts a dict (centre / dimensions / rotation 3x3 or 9 values, or yaw), or a list of them."""
    if o is None:
        return []
    if isinstance(o, list) and o and isinstance(o[0], dict):
        return [b for x in o for b in parse_boxes(x)]
    if not isinstance(o, dict):
        return []
    c, d, r = pick(o, ALIASES["center"]), pick(o, ALIASES["dims"]), pick(o, ALIASES["rot"])
    if c is None or d is None:
        return []
    if r is None:
        yaw = o.get("yaw_rad", o.get("yaw"))
        if yaw is None:
            return []
        R = rot_z(float(yaw))
    else:
        R = np.asarray(r, float).reshape(3, 3)
    return [(np.asarray(c, float).reshape(3), np.asarray(d, float).reshape(3), R)]


def coverage(points, box, margin=0.03):
    c, d, R = box
    return float(np.mean(np.all(np.abs((points - c) @ R) <= d / 2 + margin, axis=1)))


def to_lidar(box, rig):
    """camera-frame box -> LiDAR frame (q = R_rig (p - t))."""
    c, d, R = box
    return rig.R @ (c - rig.t), d, rig.R @ R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="the raw dataset root used for stage 0 (<category>/recording_XX/recording_data.json)")
    ap.add_argument("--work", default="work")
    ap.add_argument("--label-key", default="3d_data_lidar.obb_3d", help="dotted json path of the manual box")
    ap.add_argument("--max-match-dist", type=float, default=0.25, help="max xy distance between label centre and cluster centroid [m]")
    ap.add_argument("--allow-odd", action="store_true", help="fall back to the label of the ODD recording when the even one has none (default: off, odd labels are not trusted)")
    ap.add_argument("--out", default=None, help="default <work>/stage9")
    a = ap.parse_args()

    banner("9b", f"manual boxes from '{a.label_key}'")
    gates = Gates("9b")
    work = Path(a.work)
    out = Path(a.out) if a.out else work / "stage9"
    out.mkdir(parents=True, exist_ok=True)
    s0 = [r for r in read_csv(work / "stage0" / "index.csv") if r["kind"] == "orig"]
    cvb = np.load(work / "stage1" / "boxes.npz")
    bid = {s: i for i, s in enumerate(cvb["ids"])}
    pairs = {p.slug: p for p in find_pairs(a.input)}
    step(f"1/2 reading the labels of {len(s0)} real samples")
    res, rows, missing, no_cv = {}, [], [], []
    for r in progress(s0, total=len(s0), desc="labels"):
        sid = r["sample_id"]
        if sid not in bid or r["pair"] not in pairs:
            no_cv.append(sid)
            continue
        pair = pairs[r["pair"]]
        z = np.load(r["path"])
        cluster = z["cluster"].astype(np.float64)
        rig = Rig(z["cam_K"], z["cam_R"], z["cam_t"], z["hw"])
        cands, src = [], ""
        for tag, d_ in (("even", pair.even_dir), ("odd", pair.odd_dir))[:2 if a.allow_odd else 1]:
            doc = json.loads((d_ / "recording_data.json").read_text())
            cands = parse_boxes(get_path(doc, a.label_key))
            if cands:
                src = tag
                alt = parse_boxes(doc.get("obb_label"))
                break
        if not cands:
            missing.append(sid)
            continue
        cen = cluster.mean(0)
        best = None
        for box in cands:
            for frame, bx in (("lidar", box), ("camera", to_lidar(box, rig))):
                dist = float(np.linalg.norm(bx[0][:2] - cen[:2]))
                cov = coverage(cluster, bx)
                if best is None or (cov, -dist) > (best[0], -best[1]):
                    best = (cov, dist, frame, bx)
        cov, dist, frame, (c, d, Rm) = best
        row = {"sample_id": sid, "scene": r["scene"], "source": src, "frame_used": frame, "n_labels_in_json": len(cands),
               "xy_dist_to_cluster_cm": dist * 100, "cluster_inside_label_frac": cov,
               "label_bottom_z": float(c[2] - 0.5 * np.abs(Rm[2]) @ d), "label_top_z": float(c[2] + 0.5 * np.abs(Rm[2]) @ d),
               "tilt_deg": float(np.degrees(np.arccos(np.clip(np.abs(Rm[2]).max(), 0, 1))))}
        b = bid[sid]
        row["centre_diff_to_cv_cm"] = float(np.linalg.norm(c - cvb["center"][b]) * 100)
        row["dims_cm"] = " ".join(f"{x * 100:.0f}" for x in d)
        row["cv_dims_cm"] = " ".join(f"{x * 100:.0f}" for x in cvb["dims"][b])
        if alt:
            ca = alt[0]
            row["obb_label_vs_this_centre_cm"] = float(np.linalg.norm(ca[0] - c) * 100)
        if dist > a.max_match_dist:
            row["rejected"] = f"label centre {dist * 100:.0f} cm from the cluster"
            missing.append(sid)
        else:
            res[sid] = (c, d, Rm)
        rows.append(row)

    step("2/2 diagnostics")
    ids = list(res)
    np.savez(out / "manual_boxes.npz", ids=np.array(ids), center=np.array([res[i][0] for i in ids]), dims=np.array([res[i][1] for i in ids]),
             R=np.array([res[i][2] for i in ids]))
    write_csv(out / "manual_boxes.csv", rows)
    if rows:
        cov = np.array([r["cluster_inside_label_frac"] for r in rows])
        cd = np.array([r["centre_diff_to_cv_cm"] for r in rows])
        zb = np.array([r["label_bottom_z"] for r in rows])
        tilt = np.array([r["tilt_deg"] for r in rows])
        print(f"     {len(ids)}/{len(s0)} real samples have a usable manual box ({len(missing)} without a label: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}; {len(no_cv)} skipped because stage 1 has no CV box for them)")
        print(f"     label source: " + ", ".join(f"{k}={sum(1 for r in rows if r['source'] == k)}" for k in ("even", "odd")))
        print(f"     frames used: " + ", ".join(f"{k}={sum(1 for r in rows if r['frame_used'] == k)}" for k in ("lidar", "camera")))
        print(f"     camera cluster inside the label (+3 cm): median {np.median(cov):.0%}, min {cov.min():.0%}")
        print(f"     label centre vs CV box centre: median {np.median(cd):.1f} cm, max {cd.max():.1f} cm")
        print(f"     label bottom z: median {np.median(zb):.2f} m (min {zb.min():.2f}, max {zb.max():.2f}); floor of the CV box = {float(np.load(s0[0]['path'])['floor_z']):.2f} m; "
              f"tilt of the label: max {tilt.max():.1f} deg")
        gates.check("manual boxes found", len(ids) >= 3, f"{len(ids)} boxes")
        gates.check("label fits the camera cluster", float(np.median(cov)) > 0.8,
                    f"median {np.median(cov):.0%} of the cluster points lie inside the label (+3 cm); low = wrong key / frame / unit mix-up (see {out / 'manual_boxes.csv'})")
        gates.check("frames consistent", len({r['frame_used'] for r in rows}) == 1, "all labels in the same frame", hard=False)
        gates.check("labels stand on the floor", float(np.percentile(np.abs(zb - float(np.load(s0[0]['path'])['floor_z'])), 90)) < 0.05,
                    "bottom of the labels within 5 cm of the floor (the CV box always reaches the floor; a label that floats above it regenerates a different front face)",
                    hard=False)
    else:
        gates.check("manual boxes found", False, f"no '{a.label_key}' in any recording json")
    gates.finish()


if __name__ == "__main__":
    main()