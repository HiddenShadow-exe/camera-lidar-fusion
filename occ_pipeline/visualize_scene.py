#!/usr/bin/env python3
"""Look at one sample: LiDAR points (red), camera points (blue), the traditional-CV box, a few shifted boxes and what each scored.

    python visualize_scene.py --work work --list                       # which samples already have a stage-4 table (works while stage 4 is still running)
    python visualize_scene.py --work work                              # first finished real sample: CV box, oracle box and 4 others
    python visualize_scene.py --work work --sample full_occlusion_boxes_01_u0_aug003 --n-show 8
    python visualize_scene.py --work work --box=-6,0,-3,0,0,-3        # also draw your own candidate: lo_x,hi_x,lo_y,hi_y,lo_z,hi_z in cm (+ = outward;
                                                                      # write it with '=' - a leading minus sign confuses the argument parser)
    python visualize_scene.py --work work --viewer mpl --no-show       # matplotlib picture only -> work/viz/<sample>.png

What is drawn (all in the LiDAR frame)
    red    LiDAR returns of the OCCLUDED scan (the rays that get extended), cropped to --crop metres around the CV box
    blue   camera point cluster of the target box (the only input of the CV box fit)
    green  ground-truth LiDAR points of the target (scan with the occluder removed) - what the regenerated points are compared with
    white box = traditional CV box (shift 0) | orange box = oracle (best score in the stage-4 table) | other colours = candidates
           picked across the score range. Each box carries its score; points of the same colour as a box are the LiDAR points
           REGENERATED for that box (rays blocked by the occluder, extended to the box's front face).
    grey line  LiDAR origin -> box centre, grey rectangle = the floor assumed for the bottom face
Scores come straight from the stage-4 table (no re-computation): tol = tolerance score, exact = exact voxel hit/miss; both 1 = perfect.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import obb_corners, ray_box_interval, read_csv
from shiftgrid import ShiftGrid, best_candidate

FACES = ["lo_x", "hi_x", "lo_y", "hi_y", "lo_z", "hi_z"]
PALETTE = [("cyan", (0.0, 0.9, 0.9)), ("magenta", (1.0, 0.0, 1.0)), ("yellow", (1.0, 0.95, 0.0)), ("lime", (0.5, 1.0, 0.2)),
           ("pink", (1.0, 0.55, 0.75)), ("violet", (0.65, 0.45, 1.0)), ("brown", (0.8, 0.5, 0.3)), ("teal", (0.1, 0.6, 0.5))]
C_SCAN, C_CAM, C_GT = (1.0, 0.1, 0.1), (0.15, 0.35, 1.0), (0.0, 0.8, 0.25)
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]   # corner order of obb_corners


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def finished_tables(work):
    """sample_id -> table path for every sample whose stage-4 table exists (stage 4 may still be running)."""
    return {p.stem: p for p in sorted((Path(work) / "stage4" / "tables").glob("*/*.npz"))}


def load_table(path):
    try:
        z = np.load(path)
        return {k: z[k] for k in z.files}
    except Exception as e:                                      # the file may be in the middle of being written
        raise SystemExit(f"could not read {path} ({e!r}); if stage 4 is writing it right now, run again in a moment")


def list_samples(work, kind, limit, grid):
    tabs = finished_tables(work)
    s0 = {r["sample_id"]: r for r in read_csv(Path(work) / "stage0" / "index.csv")}
    l1, zero = grid.l1_all(), grid.zero_index()
    print(f"{len(tabs)} samples have a stage-4 table so far\n")
    print(f"{'sample_id':<44} {'kind':<5} {'CV box tol':>10} {'oracle tol':>10} {'CV box exact':>12} {'oracle exact':>12}  oracle shift [cm]  lo_x hi_x lo_y hi_y lo_z hi_z")
    n = 0
    for sid, p in tabs.items():
        if kind != "all" and s0.get(sid, {}).get("kind") != kind:
            continue
        t = load_table(p)
        st, se = t["score_tol"].astype(np.float32), t["score_exact"].astype(np.float32)
        b, mt, _ = best_candidate(st, l1)
        _, me, _ = best_candidate(se, l1)
        print(f"{sid:<44} {s0.get(sid, {}).get('kind', '?'):<5} {st[zero]:>10.3f} {mt:>10.3f} {se[zero]:>12.3f} {me:>12.3f}  {np.round(grid.decode(b) * 100, 1)}")
        n += 1
        if n >= limit:
            print(f"... (--limit {limit})")
            break


def pick_candidates(score, l1, zero, oracle, n_show, mode, rng, step):
    """Indices of the candidates to draw: CV box, oracle, then (n_show - 2) more."""
    chosen = [zero] + ([oracle] if oracle != zero else [])
    valid = np.flatnonzero(~np.isnan(score))
    k = max(0, n_show - len(chosen))
    if mode == "top":                                           # next best, but visibly different boxes
        order = valid[np.lexsort((l1[valid], -score[valid]))]
        for f in order:
            if len(chosen) >= n_show:
                break
            if all(np.abs(grid_decode(f) - grid_decode(c)).max() >= 2 * step - 1e-9 for c in chosen):
                chosen.append(int(f))
    elif mode == "random":
        pool = np.setdiff1d(valid, chosen)
        chosen += [int(x) for x in rng.choice(pool, min(k, len(pool)), replace=False)]
    else:                                                       # spread: one candidate per score quantile (5% ... 95%)
        for q in np.linspace(5, 95, k) if k else []:
            target = np.percentile(score[valid], q)
            tol_ = 0.01
            while True:
                pool = valid[(np.abs(score[valid] - target) <= tol_) & ~np.isin(valid, chosen)]
                if len(pool) or tol_ > 1:
                    break
                tol_ *= 2
            if len(pool):
                chosen.append(int(rng.choice(pool)))
    return chosen


grid_decode = None                                              # set in main (ShiftGrid.decode)


def regen_points(origin, scan, box, margin):
    """LiDAR points regenerated for one box: rays whose measured range ends BEFORE the box (blocked by the occluder),
    extended to the box's front face. Same rule as the stage-4 evaluator (entry > range + margin)."""
    c, d, R = box
    vec = scan - origin
    r = np.linalg.norm(vec, axis=1)
    ok = r > 1e-6
    vec, r = vec[ok], r[ok]
    dw = vec / r[:, None]
    tin, tout = ray_box_interval((origin - c) @ R, dw @ R, -d / 2, d / 2)
    blocked = (tout >= tin) & (tin > 0) & (tin > r + margin)
    return origin + tin[blocked][:, None] * dw[blocked]


def through_stats(origin, scan, box, margin):
    """Where do the LiDAR returns lie relative to a box?  For every ray that enters the box:
         blocked  entry > range + margin      (occluder in front -> these rays get regenerated)
         surface  |range - entry| <= margin   (the return sits on the box's front face: box surface == measured surface)
         inside   entry + margin < range <= exit + margin   (the return lies INSIDE the box volume: box is thicker than the object)
         through  range > exit + margin       (the ray passes completely through the box and hits something behind)
       inside + through = the evaluator's 'rays behind box'."""
    c, d, R = box
    vec = scan - origin
    r = np.linalg.norm(vec, axis=1)
    ok = r > 1e-6
    vec, r = vec[ok], r[ok]
    dw = vec / r[:, None]
    tin, tout = ray_box_interval((origin - c) @ R, dw @ R, -d / 2, d / 2)
    hit = (tout >= tin) & (tin > 0)
    blocked = hit & (tin > r + margin)
    surface = hit & (np.abs(r - tin) <= margin)
    inside = hit & (tin + margin < r) & (r <= tout + margin)
    through = hit & (r > tout + margin)
    depth = (r - tin)[inside | through] * 100                                  # cm behind the entry
    pts = scan[ok][inside | through]
    return dict(hit=int(hit.sum()), blocked=int(blocked.sum()), surface=int(surface.sum()), inside=int(inside.sum()),
                through=int(through.sum()), depth=depth, pts=pts)


def face_sensitivity(score, grid):
    """For every face: best achievable score if that face sits at offset s (maximum over all other faces)."""
    t = np.where(np.isnan(score), -np.inf, score).reshape(grid.n[0], grid.n[0], grid.n[1], grid.n[1], grid.n[2], grid.n[2])
    rows = []
    for ax, name in enumerate(FACES):
        other = tuple(a for a in range(6) if a != ax)
        v = t.max(axis=other)
        rows.append((name, grid.offsets(ax // 2) * 100, np.where(np.isfinite(v), v, np.nan)))
    return rows


# ----------------------------------------------------------------------------------------------
# viewers
# ----------------------------------------------------------------------------------------------
def show_open3d(scene, args):
    import open3d as o3d
    geoms, labels = [], []
    for g in scene["points"]:
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(g["xyz"])
        p.paint_uniform_color(g["color"])
        geoms.append({"name": g["name"], "geometry": p})
    for b in scene["boxes"]:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(b["corners"])
        ls.lines = o3d.utility.Vector2iVector(EDGES)
        ls.colors = o3d.utility.Vector3dVector([b["color"]] * len(EDGES))
        geoms.append({"name": b["name"], "geometry": ls})
        labels.append((b["label_pos"], b["label"]))
    for l in scene["lines"]:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(l["xyz"])
        ls.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(l["xyz"]) - 1)])
        ls.colors = o3d.utility.Vector3dVector([l["color"]] * (len(l["xyz"]) - 1))
        geoms.append({"name": l["name"], "geometry": ls})
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15).translate(scene["origin"])
    geoms.append({"name": "LiDAR origin", "geometry": frame})

    c = scene["center"]
    kw = dict(title=f"{scene['sample']}   (tick / untick items in the left panel)", width=1500, height=900, show_ui=True,
              bg_color=(0.07, 0.07, 0.09, 1.0), point_size=3,
              lookat=c, eye=c + np.array([-1.2, -1.0, 1.0]) * 1.0, up=[0, 0, 1])
    def on_init(vis):
        for pos, text in labels:
            vis.add_3d_label(pos, text)
    for attempt, extra in enumerate(({"on_init": on_init, "line_width": 3}, {"on_init": on_init}, {})):
        try:
            o3d.visualization.draw(geoms, **kw, **extra)
            if attempt:
                print("  (this Open3D version has no in-scene labels / line width: use the table above)" if "on_init" not in extra else "")
            return
        except TypeError:
            continue
        except Exception as e:
            print(f"  [warn] open3d.visualization.draw failed ({e!r}); falling back to the legacy viewer")
            break
    o3d.visualization.draw_geometries([g["geometry"] for g in geoms], window_name=scene["sample"], width=1500, height=900)


def show_matplotlib(scene, args):
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 7))
    ax3 = fig.add_subplot(1, 3, 1, projection="3d")
    axt = fig.add_subplot(1, 3, 2)
    axb = fig.add_subplot(1, 3, 3)
    rng = np.random.default_rng(0)
    allpts = []
    mc = lambda c: (0.0, 0.0, 0.0) if tuple(c) == (1.0, 1.0, 1.0) else c          # white is invisible on a white background
    for g in scene["points"]:
        xyz = g["xyz"]
        if len(xyz) > 5000:
            xyz = xyz[rng.choice(len(xyz), 5000, replace=False)]
        allpts.append(xyz)
        regen = g["name"].startswith("regenerated")
        cam = g["name"].startswith("camera")
        sz = {"regen": 14, "cam": 1.5}.get("regen" if regen else "cam" if cam else "", g.get("s", 2) * (2 if "GT" in g["name"] else 1))
        kw = dict(s=sz, c=[mc(g["color"])], label=g["name"].replace("(white)", "(black)"), alpha=0.25 if cam else 1.0, zorder=2 if cam else 4 if regen else 3,
                  edgecolors="k" if regen else "none", linewidths=0.3)
        ax3.scatter(*xyz.T, depthshade=False, **kw)
        axt.scatter(xyz[:, 0], xyz[:, 1], **kw)
    for b in scene["boxes"]:
        for i, j in EDGES:
            seg = b["corners"][[i, j]]
            ax3.plot(*seg.T, color=mc(b["color"]), lw=2)
        # footprint (top-down): the 4 bottom corners in loop order
        loop = [b["corners"][k] for k in (0, 2, 6, 4, 0)]
        axt.plot([p[0] for p in loop], [p[1] for p in loop], color=mc(b["color"]), lw=2, zorder=5)
        k = scene["boxes"].index(b)
        ck = (1, 3, 7, 5)[k % 4]                                 # a different top corner per box -> labels do not pile up
        ax3.text(*(b["corners"][ck] + np.array([0, 0, 0.015])), f"#{k}", color=mc(b["color"]), fontsize=9, fontweight="bold", ha="center")
        axt.text(*b["corners"][ck - 1][:2], f"#{k}", color=mc(b["color"]), fontsize=9, fontweight="bold",
                 ha="center", va="center", zorder=6, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    for l in scene["lines"]:
        ax3.plot(*l["xyz"].T, color=l["color"], lw=1, alpha=0.7)
        axt.plot(l["xyz"][:, 0], l["xyz"][:, 1], color=l["color"], lw=1, alpha=0.7)
    ax3.scatter(*scene["origin"], s=60, c="k", marker="^")
    axt.scatter(*scene["origin"][:2], s=60, c="k", marker="^")
    near = [g["xyz"] for g in scene["points"] if not g["name"].startswith("LiDAR occluded")]
    pts = np.vstack(near + [b["corners"] for b in scene["boxes"]])
    pad = 0.12
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    lo[2] = min(lo[2], float(scene["floor_z"]) - 0.01)
    ax3.set_xlim(lo[0], hi[0]); ax3.set_ylim(lo[1], hi[1]); ax3.set_zlim(lo[2], hi[2])
    ax3.set_box_aspect(tuple(np.maximum(hi - lo, 1e-3)))
    axt.set_xlim(lo[0], hi[0]); axt.set_ylim(lo[1], hi[1])
    ax3.view_init(elev=28, azim=-125)
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z"); ax3.set_title("perspective (LiDAR frame, ^ = LiDAR)")
    ax3.locator_params(nbins=4)
    axt.set_aspect("equal", adjustable="box"); axt.set_xlabel("x [m]"); axt.set_ylabel("y [m]"); axt.set_title("top view (zoomed on the boxes; LiDAR is off to the left)"); axt.grid(alpha=0.3)
    axt.legend(loc="upper right", fontsize=8, markerscale=4)
    names = [f"#{k} {b['short']}" for k, b in enumerate(scene["boxes"])]
    y = np.arange(len(names))[::-1]
    axb.barh(y + 0.2, [b["tol"] for b in scene["boxes"]], height=0.38, color=[mc(b["color"]) for b in scene["boxes"]], edgecolor="k", label="tol")
    axb.barh(y - 0.2, [b["exact"] for b in scene["boxes"]], height=0.38, color=[mc(b["color"]) for b in scene["boxes"]], edgecolor="k", hatch="//", label="exact")
    axb.set_yticks(y); axb.set_yticklabels(names, fontsize=8); axb.set_xlim(0, 1.0); axb.set_xlabel("score (solid = tolerance, hatched = exact)")
    axb.set_title("what each box scored"); axb.grid(axis="x", alpha=0.3)
    fig.suptitle(scene["sample"], fontsize=11)
    fig.tight_layout()
    out = Path(args.work) / "viz"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{scene['sample']}.png"
    fig.savefig(path, dpi=110)
    print(f"  picture saved: {path}")
    if not args.no_show:
        plt.show()


# ----------------------------------------------------------------------------------------------
def main():
    global grid_decode
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--sample", default=None, help="sample_id (default: first finished real sample)")
    ap.add_argument("--list", action="store_true", help="list the samples that already have a stage-4 table and exit")
    ap.add_argument("--kind", default="orig", choices=["orig", "aug", "all"], help="filter for --list / the default sample")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--which", choices=["tol", "exact"], default="tol", help="score used for 'oracle' and for picking candidates")
    ap.add_argument("--n-show", type=int, default=6, help="boxes drawn in total (CV box + oracle + others); do not go much above 8")
    ap.add_argument("--pick", choices=["spread", "top", "random"], default="spread",
                    help="spread: one box per score quantile (worst .. best); top: next best visibly-different boxes; random")
    ap.add_argument("--box", action="append", default=[], help="extra candidate 'lo_x,hi_x,lo_y,hi_y,lo_z,hi_z' in cm, repeatable; write --box=-3,0,0,0,0,-3 (with '=')")
    ap.add_argument("--regen", default="cv,oracle", help="whose regenerated points to draw: none | cv,oracle | all | comma list of box numbers")
    ap.add_argument("--crop", type=float, default=0.8, help="draw points within this margin [m] around the CV box (0 = everything)")
    ap.add_argument("--show-through", action="store_true", help="also draw the LiDAR returns that lie inside / behind the CV box (the 'rays behind box' of row #0) in grey")
    ap.add_argument("--no-gt", action="store_true", help="do not draw the ground-truth LiDAR points")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viewer", choices=["auto", "o3d", "mpl"], default="auto")
    ap.add_argument("--no-show", action="store_true", help="matplotlib: only save the picture")
    a = ap.parse_args()

    cfg = json.loads((Path(a.work) / "stage4" / "config.json").read_text())
    grid = ShiftGrid(cfg["n_steps"], cfg["n_steps_z"], cfg["step"], cfg["min_dim"])
    grid_decode = grid.decode
    l1, zero = grid.l1_all(), grid.zero_index()
    if a.list:
        list_samples(a.work, a.kind, a.limit, grid)
        return

    tabs = finished_tables(a.work)
    s0 = {r["sample_id"]: r for r in read_csv(Path(a.work) / "stage0" / "index.csv")}
    if not tabs:
        raise SystemExit("no stage-4 tables yet (work/stage4/tables is empty) - wait for the first sample to finish")
    sid = a.sample or next((s for s in tabs if a.kind == "all" or s0[s]["kind"] == a.kind), next(iter(tabs)))
    if sid not in tabs:
        raise SystemExit(f"{sid} has no finished stage-4 table yet (finished: {len(tabs)}; try --list)")
    boxes = np.load(Path(a.work) / "stage1" / "boxes.npz")
    bi = list(boxes["ids"]).index(sid)
    cv = (boxes["center"][bi], boxes["dims"][bi], boxes["R"][bi])
    z = np.load(s0[sid]["path"])
    origin, scan, cam, gt = z["origin"], z["scan"].astype(np.float64), z["cluster"].astype(np.float64), z["gt"].astype(np.float64)
    tab = load_table(tabs[sid])
    score = (tab["score_tol"] if a.which == "tol" else tab["score_exact"]).astype(np.float32)
    st, se = tab["score_tol"].astype(np.float32), tab["score_exact"].astype(np.float32)
    oracle, best, nties = best_candidate(score, l1)
    rng = np.random.default_rng(a.seed)

    cands = pick_candidates(score, l1, zero, oracle, a.n_show, a.pick, rng, grid.step)
    custom_f = set()
    for spec in a.box:
        try:
            sh = np.array([float(x) for x in spec.split(",")]) / 100.0
            assert len(sh) == 6
        except Exception:
            raise SystemExit(f"--box expects six numbers in cm, got '{spec}'")
        idx = np.rint(sh / grid.step).astype(int).reshape(3, 2) + np.array(grid.ns)[:, None]
        if not np.allclose(sh, np.rint(sh / grid.step) * grid.step, atol=1e-6) or np.any(idx < 0) or np.any(idx >= np.array(grid.n)[:, None]):
            raise SystemExit(f"--box {spec}: not on the grid (multiples of {grid.step * 100:.1f} cm, within +-{grid.max_offset[0] * 100:.0f} cm in x/y and "
                             f"+-{grid.max_offset[2] * 100:.0f} cm in z)")
        f = int(grid.encode(sh))
        if np.isnan(score[f]):
            raise SystemExit(f"--box {spec}: this candidate is thinner than --min-dim (invalid)")
        custom_f.add(f)
        if f not in cands:
            cands.append(f)

    # ---- crop + colours ------------------------------------------------------------------------------------
    c0, d0, R0 = cv
    def crop(p):
        if a.crop <= 0:
            return p
        return p[np.all(np.abs((p - c0) @ R0) <= d0 / 2 + a.crop, axis=1)]
    margin = cfg["block_margin"]
    which_regen = set()
    if a.regen == "all":
        which_regen = set(range(len(cands)))
    elif a.regen != "none":
        for tok in a.regen.split(","):
            tok = tok.strip()
            which_regen |= {0} if tok == "cv" else ({cands.index(oracle)} if tok == "oracle" else {int(tok)})
    boxes_out, points = [], [{"name": "LiDAR occluded scan (red)", "xyz": crop(scan), "color": C_SCAN, "s": 1},
                             {"name": "camera cluster (blue)", "xyz": cam, "color": C_CAM, "s": 3}]
    if not a.no_gt:
        points.append({"name": "GT LiDAR, occluder removed (green)", "xyz": crop(gt), "color": C_GT, "s": 2})
    print(f"\nsample {sid}   ({s0[sid]['kind']}, scene {s0[sid]['scene']})   stage-4 table: {int(tab['n_rays'])} rays, "
          f"{int(tab['nS'])} shadow voxels (S), {int(tab['nG'])} GT voxels (G); {cfg['voxel_frame']}-aligned voxels of {cfg['voxel_size'] * 100:.1f} cm")
    print(f"CV box: dims {np.round(d0, 3)} m  centre {np.round(c0, 3)}\n")
    print(f"{'#':>2} {'colour':<8} {'role':<10} {'tol':>6} {'exact':>6} {'regen vox':>9} {'rays behind box':>15}   shift [cm]  lo_x hi_x lo_y hi_y lo_z hi_z")
    ci = 0
    for k, f in enumerate(cands):
        role = "CV box" if f == zero else ("oracle" if f == oracle else ("custom" if f in custom_f else "candidate"))
        cname, col = ("white", (1.0, 1.0, 1.0)) if f == zero else (("orange", (1.0, 0.55, 0.0)) if f == oracle else PALETTE[ci % len(PALETTE)])
        if f not in (zero, oracle):
            ci += 1
        box = grid.box_from(cv, f)
        cor = obb_corners(*box)
        sh = grid.decode(f) * 100
        n_regen = None
        if k in which_regen:
            rp = regen_points(origin, scan, box, margin)
            n_regen = len(rp)
            points.append({"name": f"regenerated for #{k} ({cname})", "xyz": rp, "color": col, "s": 4})
        short = f"{role}  {np.round(sh, 0).astype(int).tolist()}"
        label = f"#{k} {role}: tol {st[f]:.2f} / exact {se[f]:.2f}"
        top_c = box[0] + box[2] @ np.array([0, 0, box[1][2] / 2 + 0.03 + 0.045 * k])
        boxes_out.append({"name": f"box #{k} {role} ({cname})", "corners": cor, "color": col, "label": label, "label_pos": top_c,
                          "short": short, "tol": float(st[f]), "exact": float(se[f])})
        print(f"{k:>2} {cname:<8} {role:<10} {st[f]:>6.3f} {se[f]:>6.3f} {int(tab['n_pred'][f]):>9} {int(tab['n_viol'][f]):>15}   {np.round(sh, 1)}")
    print(f"\n   oracle by '{a.which}' score = {best:.3f}  ({nties} tied candidates; the one with the smallest total shift is drawn).")
    print(f"   'rays behind box' = LiDAR rays that measured something BEHIND the box entry, i.e. rays that see through the box (should be few).")

    print("\n   where do the LiDAR returns lie relative to each shown box?  (rays that enter the box, margin " + f"{margin * 100:.0f} cm)")
    print(f"   {'#':>2} {'role':<10} {'enter':>6} {'blocked':>8} {'on surface':>10} {'inside box':>10} {'through':>8}   cm behind entry of inside/through returns: p10 / median / p90")
    for k, f in enumerate(cands):
        role = "CV box" if f == zero else ("oracle" if f == oracle else ("custom" if f in custom_f else "candidate"))
        ts = through_stats(origin, scan, grid.box_from(cv, f), margin)
        dp = np.percentile(ts["depth"], [10, 50, 90]) if len(ts["depth"]) else [float("nan")] * 3
        print(f"   {k:>2} {role:<10} {ts['hit']:>6} {ts['blocked']:>8} {ts['surface']:>10} {ts['inside']:>10} {ts['through']:>8}   {dp[0]:6.1f} / {dp[1]:6.1f} / {dp[2]:6.1f}")
        if k == 0 and a.show_through and len(ts["pts"]):
            points.append({"name": "returns inside/behind the CV box (grey)", "xyz": crop(ts["pts"]), "color": (0.55, 0.55, 0.55), "s": 5})

    print("\n   how much does each face matter?  best achievable '" + a.which + "' score when that face sits at the given offset (any other faces):")
    for name, off, v in face_sensitivity(score, grid):
        print(f"     {name}: " + "  ".join(f"{o:+.0f}:{x:.2f}" if np.isfinite(x) else f"{o:+.0f}:  - " for o, x in zip(off, v)))

    lines = [{"name": "LiDAR -> CV box centre", "xyz": np.array([origin, c0]), "color": (0.6, 0.6, 0.6)}]
    hx, hy = d0[0] / 2 + (a.crop or 0.5), d0[1] / 2 + (a.crop or 0.5)
    fl = np.array([c0 + R0 @ np.array([sx * hx, sy * hy, 0.0]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))])
    fl[:, 2] = float(z["floor_z"])
    lines.append({"name": "assumed floor", "xyz": fl, "color": (0.45, 0.45, 0.45)})
    scene = {"sample": sid, "points": points, "boxes": boxes_out, "lines": lines, "origin": origin, "center": c0, "floor_z": float(z["floor_z"])}

    viewer = a.viewer
    if viewer == "auto":
        try:
            import open3d  # noqa: F401
            viewer = "o3d"
        except ImportError:
            viewer = "mpl"
    print(f"\n  viewer: {viewer}")
    show_open3d(scene, a) if viewer == "o3d" else show_matplotlib(scene, a)


if __name__ == "__main__":
    main()