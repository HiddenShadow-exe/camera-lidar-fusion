#!/usr/bin/env python3
"""STAGE 6 - train the shift model (requires torch).

Model:  PointNet-style encoder over the camera cluster (in the CV box frame)  +  MLP over the context vector
        (LiDAR origin / camera centre in box frame, CV box dims)  ->  three heads, one per box axis k, each with
        C_k = n_k^2 logits = the joint (lo-face shift, hi-face shift) of that axis.
        P(candidate) = P0(j0) * P1(j1) * P2(j2)      -> the model outputs a distribution over all candidates of the table.

Candidates thinner than --min-dim are impossible: their logits are masked per axis (validity is separable per axis and
depends only on the CV box dims), so the model can never choose one.
Logits are bounded to +-8 (tanh) so the softmax can never saturate completely (a saturated softmax has zero gradient;
that froze the first version of this stage at the best CONSTANT shift).

Loss  --loss soft (default): the target is the Boltzmann distribution over ALL candidates of the sample,
        q(c) ~ exp(score(c) / tau)  (tau = --tau, 0.05); the model minimises KL(q || P) with P = P0*P1*P2. Every near-optimal
        candidate gets mass (plateaus are harmless) and the gradient never vanishes.
      --loss expected : E_p[1 - score] computed exactly against the table (einsum over the 3 heads).
      --loss ce       : cross-entropy to the oracle candidate (ties broken to the smallest shift).
      --score tol|exact|mix|graded selects which table is optimised (mix = mean of tol and exact; graded = the distance-weighted
        stage-4 table, needs stage 4 with --graded-sigma > 0 and stage 5 rerun).

--no-virtual : leave the virtual-occluder samples (category 'virtual...') out of training completely (ablation: do they help?).
--tag NAME   : suffix of the output files (pred_cv<NAME>.npz, history_cv<NAME>.csv, ...) so that ablation runs do not overwrite each other.

--cv K : scene-level K-fold cross-validation over ALL real scenes (train+val pooled, augmented copies stay with their
        scene). A fresh model per fold, fixed number of epochs (no epoch picking on the held-out fold). Writes
        work/stage6/pred_cv.npz; evaluate with  s7_evaluate.py --split cv  (best constant is then fitted out-of-fold, too).

Every epoch is evaluated on the val split with the metric that matters: REGRET = best score in the table - score of the
model's chosen candidate (argmax of the predicted distribution), next to the CV-box (shift 0) baseline.
Outputs: work/stage6/{best.pt, history.csv, curves.png, pred_train.npz, pred_val.npz}   (--cv: history_cv.csv, pred_cv.npz)
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from common import Gates, banner, progress, read_csv, step, write_csv
from metrics import clean_scores
from shiftgrid import ShiftGrid

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("stage 6 needs torch:  pip install torch   (stages 0-5 and 7 do not)")


class ShiftNet(nn.Module):
    def __init__(self, C, ctx_dim=9, width=128, hidden=512, pos_scale=0.5, dropout=0.1, bound=8.0):
        super().__init__()
        self.C, self.pos_scale, self.bound = tuple(C), pos_scale, bound
        self.point_mlp = nn.Sequential(
            nn.Linear(3, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, 2 * width), nn.LayerNorm(2 * width), nn.GELU(),
            nn.Linear(2 * width, 4 * width), nn.LayerNorm(4 * width), nn.GELU())
        self.ctx_mlp = nn.Sequential(nn.Linear(ctx_dim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(8 * width + 64, hidden), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList([nn.Linear(hidden, c) for c in self.C])
        for h in self.heads:                                   # start close to a uniform distribution
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, pts, ctx, masks=None):
        f = self.point_mlp(pts / self.pos_scale)               # (B, N, 4w)
        g = torch.cat([f.max(1).values, f.mean(1), self.ctx_mlp(ctx / self.pos_scale)], 1)
        h = self.trunk(g)
        out = []
        for k, head in enumerate(self.heads):
            l = self.bound * torch.tanh(head(h) / self.bound)  # bounded: the softmax can never saturate completely
            if masks is not None:
                l = l.masked_fill(~masks[k], -1e4)             # impossible (too thin) face pairs of this axis
            out.append(l)
        return out                                             # list of (B, C_k) logits


def build_scores(tol, exact, kind, graded=None):
    if kind == "graded":
        if graded is None or graded.shape[1] == 0:
            raise SystemExit("--score graded needs work/stage5/*_tab_graded.npy: run stage 4 (graded score on by default) and stage 5 again")
        return graded.float()
    t = tol.float() if kind == "tol" else exact.float() if kind == "exact" else 0.5 * (tol.float() + exact.float())
    return t


def expected_loss_t(logits, score_grid):
    p = [F.softmax(l, -1) for l in logits]
    return torch.einsum("bi,bj,bk,bijk->b", p[0], p[1], p[2], 1.0 - score_grid)


def soft_loss_t(logits, score, valid, C, tau):
    """KL(q || P) up to a constant; q = softmax(score / tau) over the valid candidates of each sample, P = P0*P1*P2.
    Because P is a product, only the three marginals of q matter -> three ordinary cross-entropies."""
    B = score.shape[0]
    s = torch.where(valid, score, torch.full_like(score, float("-inf")))
    q = F.softmax(s / tau, dim=1).reshape(B, *C)
    marg = (q.sum((2, 3)), q.sum((1, 3)), q.sum((1, 2)))
    return sum(-(m * F.log_softmax(l, -1)).sum(-1) for m, l in zip(marg, logits)).mean()


def choose(logits, C):
    j = [l.argmax(-1) for l in logits]
    return (j[0] * C[1] + j[1]) * C[2] + j[2]


def make_masks(ctx, grid, dev):
    """(B, C_k) bool per axis: face pair keeps the box at least --min-dim thick. ctx[:, 6:9] = CV box dims."""
    dims = ctx[:, 6:9].double().cpu().numpy()
    vm = [np.stack([grid.valid_mask(d)[k] for d in dims]) for k in range(3)]
    return [torch.from_numpy(v).to(dev) for v in vm]


class Split:
    def __init__(self, work, name, dev, kind, C, grid, limit=0):
        d = Path(work) / "stage5"
        self.pts = torch.from_numpy(np.load(d / f"{name}_points.npy")).to(dev)
        self.ctx = torch.from_numpy(np.load(d / f"{name}_ctx.npy")).to(dev)
        tol = torch.from_numpy(np.load(d / f"{name}_tab_tol.npy"))
        exa = torch.from_numpy(np.load(d / f"{name}_tab_exact.npy"))
        gp = d / f"{name}_tab_graded.npy"
        gra = torch.from_numpy(np.load(gp)) if gp.exists() else torch.zeros(tol.shape[0], 0)
        self.score = torch.nan_to_num(build_scores(tol, exa, kind, gra), nan=0.0).to(dev)      # invalid -> score 0 (loss 1)
        self.valid = (~torch.isnan(tol.float())).to(dev)
        self.tol, self.exa, self.gra = tol.float().to(dev), exa.float().to(dev), gra.float().to(dev)   # for the regret metrics (NaN kept)
        self.meta = read_csv(d / f"{name}_meta.csv")
        self.kind = np.array([m["kind"] for m in self.meta])
        self.ids = [m["sample_id"] for m in self.meta]
        if limit:
            sl = slice(0, limit)
            for k in ("pts", "ctx", "score", "valid", "tol", "exa", "gra"):
                setattr(self, k, getattr(self, k)[sl])
            self.meta, self.kind, self.ids = self.meta[sl], self.kind[sl], self.ids[sl]
        self.n, self.C = len(self.ids), C
        self.oracle = torch.tensor([int(m["best_graded"] if kind == "graded" else m["best_tol"] if kind != "exact" else m["best_exact"]) for m in self.meta], device=dev)
        self.vmask = make_masks(self.ctx, grid, dev)

    KEYS = ("pts", "ctx", "score", "valid", "tol", "exa", "gra", "oracle")

    def subset(self, idx):
        s = object.__new__(Split)
        t = torch.as_tensor(np.asarray(idx), device=self.pts.device)
        for k in self.KEYS:
            setattr(s, k, getattr(self, k)[t])
        s.vmask = [m[t] for m in self.vmask]
        s.meta, s.kind, s.ids = [self.meta[i] for i in idx], self.kind[idx], [self.ids[i] for i in idx]
        s.n, s.C = len(idx), self.C
        return s

    @staticmethod
    def concat(a, b):
        s = object.__new__(Split)
        for k in Split.KEYS:
            setattr(s, k, torch.cat([getattr(a, k), getattr(b, k)]))
        s.vmask = [torch.cat([x, y]) for x, y in zip(a.vmask, b.vmask)]
        s.meta, s.kind, s.ids = a.meta + b.meta, np.concatenate([a.kind, b.kind]), a.ids + b.ids
        s.n, s.C = a.n + b.n, a.C
        return s


def regret_stats(split, chosen, zero_idx, which):
    """regret of the chosen candidates against the table maximum, for 'tol', 'exact' or 'graded' scores (all samples)."""
    tab = {"tol": split.tol, "exact": split.exa, "graded": split.gra}[which]
    best = torch.nan_to_num(tab, nan=-1.0).max(1).values
    got = torch.nan_to_num(tab.gather(1, chosen[:, None])[:, 0], nan=-1.0)
    zero = torch.nan_to_num(tab[:, zero_idx], nan=-1.0)
    return (best - got), (best - zero)


@torch.no_grad()
def evaluate(model, split, C, zero_idx, kind, bs=64, n_pts=None):
    model.eval()
    chosen, losses = [], []
    for i in range(0, split.n, bs):
        sl = slice(i, i + bs)
        pts = split.pts[sl].float() if n_pts is None else split.pts[sl, :n_pts].float()
        lg = model(pts, split.ctx[sl], [m[sl] for m in split.vmask])
        chosen.append(choose(lg, C))
        losses.append(expected_loss_t(lg, split.score[sl].reshape(-1, *C)))
    chosen, loss = torch.cat(chosen), torch.cat(losses)
    out = {"loss": float(loss.mean())}
    for w in ("tol", "exact") + (("graded",) if split.gra.shape[1] else ()):
        rg, rz = regret_stats(split, chosen, zero_idx, w)
        out[f"regret_{w}"], out[f"zero_regret_{w}"] = float(rg.mean()), float(rz.mean())
        real = torch.from_numpy(split.kind == "orig").to(rg.device)
        if real.any():
            out[f"regret_{w}_real"], out[f"zero_regret_{w}_real"] = float(rg[real].mean()), float(rz[real].mean())
    return out, chosen


def fit(tr, va, a, C, zero, dev, sel_key, select, epochs, seed, tag="", ckpt=None, cfg=None):
    """Train one fresh model. select=True: return the epoch with the best val regret; False: return the LAST epoch."""
    torch.manual_seed(seed)
    model = ShiftNet(C, width=a.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    nb = max(1, int(np.ceil(tr.n / a.batch_size)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=epochs * nb, pct_start=0.1)
    hist, best_val, best_ep, best_state, t0 = [], float("inf"), -1, None, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(tr.n, device=dev)
        run, seen = 0.0, 0
        for b in range(nb):
            idx = perm[b * a.batch_size:(b + 1) * a.batch_size]
            P = tr.pts.shape[1]
            cols = torch.randperm(P, device=dev)[:min(a.train_points, P)]
            pts = tr.pts[idx][:, cols].float()
            if a.jitter > 0:
                pts = pts + a.jitter * torch.randn_like(pts)
            lg = model(pts, tr.ctx[idx], [m[idx] for m in tr.vmask])
            if a.loss == "soft":
                loss = soft_loss_t(lg, tr.score[idx], tr.valid[idx], C, a.tau)
            elif a.loss == "expected":
                loss = expected_loss_t(lg, tr.score[idx].reshape(-1, *C)).mean()
            else:
                j = np.unravel_index(tr.oracle[idx].cpu().numpy(), C)
                loss = sum(F.cross_entropy(l, torch.from_numpy(jk).to(dev)) for l, jk in zip(lg, j))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss.detach()) * len(idx)
            seen += len(idx)
        tr_m, _ = evaluate(model, tr, C, zero, a.score, n_pts=a.train_points)
        va_m, _ = evaluate(model, va, C, zero, a.score, n_pts=a.train_points)
        hist.append({"epoch": ep, "train_loss_running": run / seen, "train_loss": tr_m["loss"], "val_loss": va_m["loss"],
                     **{f"train_{k}": v for k, v in tr_m.items() if k != "loss"}, **{f"val_{k}": v for k, v in va_m.items() if k != "loss"},
                     "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0})
        crit = va_m[f"regret_{sel_key}"]
        star = ""
        if select and crit < best_val:
            best_val, best_ep, star = crit, ep, "  * best"
            best_state = copy.deepcopy(model.state_dict())
            if ckpt is not None:
                torch.save({"model": best_state, "C": C, "width": a.width, "cfg": cfg, "args": vars(a), "epoch": ep}, ckpt)
        if ep == 1 or ep == epochs or ep % a.print_every == 0 or star:
            print(f"  {tag}ep {ep:3d}  expected loss train {tr_m['loss']:.4f} val {va_m['loss']:.4f} (train objective {run / seen:.4f}) | val regret tol {va_m['regret_tol']:.4f} "
                  f"(CV box {va_m['zero_regret_tol']:.4f}) exact {va_m['regret_exact']:.4f} (CV box {va_m['zero_regret_exact']:.4f}){star}", flush=True)
    if select:
        model.load_state_dict(best_state)
    else:
        best_ep, best_val = epochs, hist[-1][f"val_regret_{sel_key}"]
    return model, hist, best_ep, best_val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--train-points", type=int, default=1024, help="random subset of the stored points per step")
    ap.add_argument("--jitter", type=float, default=0.003, help="gaussian point jitter [m] (train only)")
    ap.add_argument("--loss", choices=["soft", "expected", "ce"], default="soft")
    ap.add_argument("--tau", type=float, default=0.05, help="--loss soft: temperature of the target distribution over candidates (score units)")
    ap.add_argument("--score", choices=["tol", "exact", "mix", "graded"], default="tol", help="which stage-4 table is optimised")
    ap.add_argument("--select", choices=["tol", "exact", "graded"], default=None, help="regret used for choosing the best epoch (default = --score, mix -> tol)")
    ap.add_argument("--cv", type=int, default=0, help="K > 1: scene-level K-fold cross-validation over all real scenes (see top of file)")
    ap.add_argument("--no-virtual", action="store_true", help="exclude the virtual-occluder samples from training completely (ablation)")
    ap.add_argument("--tag", default="", help="suffix for the output files (pred_cv<tag>.npz, ...) so ablation runs do not overwrite each other")
    ap.add_argument("--print-every", type=int, default=5)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="2 epochs on 64 samples: only checks that everything runs")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sel_key = a.select or ("exact" if a.score == "exact" else "tol")

    banner(6, "train the shift model")
    out = Path(a.work) / "stage6"
    out.mkdir(parents=True, exist_ok=True)
    gates = Gates(6)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((Path(a.work) / "stage5" / "config.json").read_text())
    grid = ShiftGrid(cfg["n_steps"], cfg["n_steps_z"], cfg["step"], cfg["min_dim"])
    C, zero = grid.C, grid.zero_index()
    step(f"device {dev} | {grid.describe()} | loss {a.loss}" + (f" (tau {a.tau})" if a.loss == "soft" else "") + f" on '{a.score}' table | lr {a.lr}")

    lim = 64 if a.smoke else 0
    tr, va = Split(a.work, "train", dev, a.score, C, grid, lim), Split(a.work, "val", dev, a.score, C, grid, lim)
    is_virt = lambda sp: np.array([m["category"].lower().startswith("virtual") for m in sp.meta])
    if a.no_virtual and not (a.cv and a.cv > 1):
        step(f"--no-virtual: {int(is_virt(tr).sum())} virtual-occluder samples removed from training")
        tr = tr.subset(np.flatnonzero(~is_virt(tr)))
    epochs = 2 if a.smoke else a.epochs
    n_par = sum(p.numel() for p in ShiftNet(C, width=a.width).parameters())
    step(f"model: {n_par / 1e6:.2f} M parameters, heads {C}")

    # =========================================================================================================
    if a.cv and a.cv > 1:
        pool = Split.concat(tr, va)
        scenes = np.array([m["scene"] for m in pool.meta])
        virt = np.array([m["category"].lower().startswith("virtual") for m in pool.meta])          # make_virtual_pairs.py scenes
        uniq = np.array(sorted(set(scenes[~virt])))
        rng = np.random.default_rng(a.seed)
        fold_of_scene = {s: i % a.cv for i, s in enumerate(rng.permutation(uniq))}
        fold = np.array([-1 if v else fold_of_scene[s] for s, v in zip(scenes, virt)])                # -1 = always training data
        K = int(fold.max()) + 1
        step(f"{a.cv}-fold cross-validation over {len(uniq)} real scenes ({int((~virt).sum())} samples, {int(((pool.kind == 'orig') & ~virt).sum())} real boxes) "
             f"+ {int(virt.sum())} virtual-occluder samples that are only ever used for training; fixed {epochs} epochs per fold, no epoch picking")
        chosen_all = np.full(pool.n, zero, np.int64)                 # virtual samples keep 'shift 0' (never evaluated)
        hist_all, rows = [], []
        for f in range(K):
            tr_i, te_i = np.flatnonzero((fold != f) & ~(virt & a.no_virtual)), np.flatnonzero(fold == f)   # fold -1 (virtual) is always in tr_i unless --no-virtual
            trf, tef = pool.subset(tr_i), pool.subset(te_i)
            step(f"fold {f + 1}/{K}: train {trf.n} ({len(set(scenes[tr_i]))} scenes) | held out {tef.n} ({len(set(scenes[te_i]))} scenes: {', '.join(sorted(set(scenes[te_i])))})")
            model, hist, _, _ = fit(trf, tef, a, C, zero, dev, sel_key, False, epochs, a.seed + f, tag=f"[fold {f + 1}] ")
            m, ch = evaluate(model, tef, C, zero, a.score, n_pts=None)
            chosen_all[te_i] = ch.cpu().numpy()
            hist_all += [{"fold": f, **h} for h in hist]
            rows.append((f, tef.n, m))
            print(f"     fold {f + 1} held out: regret tol {m['regret_tol']:.4f} (CV box {m['zero_regret_tol']:.4f}) | exact {m['regret_exact']:.4f} (CV box {m['zero_regret_exact']:.4f})")
        write_csv(out / f"history_cv{a.tag}.csv", hist_all)
        np.savez(out / f"pred_cv{a.tag}.npz", idx=chosen_all, ids=np.array(pool.ids), fold=fold, use_virtual=np.array(not a.no_virtual))
        ch_t = torch.from_numpy(chosen_all).to(dev)
        ev = torch.from_numpy(~virt).to(dev)                         # evaluate on the real scenes only
        print("\n     POOLED out-of-fold regret (lower is better), real scenes only:")
        for w in ("tol", "exact") + (("graded",) if pool.gra.shape[1] else ()):
            rg, rz = regret_stats(pool, ch_t, zero, w)
            rg, rz = rg[ev], rz[ev]
            real = torch.from_numpy((pool.kind == "orig")[~virt]).to(dev)
            print(f"       {w:<5} all samples: model {float(rg.mean()):.4f} vs CV box {float(rz.mean()):.4f} | real only ({int(real.sum())}): model {float(rg[real].mean()):.4f} vs CV box {float(rz[real].mean()):.4f}")
        rg, rz = regret_stats(pool, ch_t, zero, sel_key)
        rg, rz = rg[ev], rz[ev]
        gates.check("loss is finite", all(np.isfinite(h["train_loss"]) for h in hist_all), "all folds")
        gates.check("model beats the CV box out-of-fold", float(rg.mean()) < float(rz.mean()), f"regret {float(rg.mean()):.4f} vs {float(rz.mean()):.4f}", hard=False)
        print(f"\n     next: python s7_evaluate.py --work {a.work} --split cv" + (f" --tag {a.tag}" if a.tag else ""))
        (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        gates.finish()
        return

    # =========================================================================================================
    step(f"train {tr.n} samples ({int((tr.kind == 'orig').sum())} real, {len(set(m['scene'] for m in tr.meta))} scenes) | "
         f"val {va.n} ({int((va.kind == 'orig').sum())} real, {len(set(m['scene'] for m in va.meta))} scenes) | best epoch by val regret ({sel_key})")
    model, hist, best_ep, best_val = fit(tr, va, a, C, zero, dev, sel_key, True, epochs, a.seed, ckpt=out / f"best{a.tag}.pt", cfg=cfg)
    write_csv(out / f"history{a.tag}.csv", hist)

    step("final predictions with the best epoch")
    for name, sp in (("train", tr), ("val", va)):
        m, ch = evaluate(model, sp, C, zero, a.score, n_pts=None)
        np.savez(out / f"pred_{name}{a.tag}.npz", idx=ch.cpu().numpy(), ids=np.array(sp.ids), use_virtual=np.array(not a.no_virtual))
        print(f"     {name}: regret tol {m['regret_tol']:.4f} (CV box {m['zero_regret_tol']:.4f}) | "
              f"regret exact {m['regret_exact']:.4f} (CV box {m['zero_regret_exact']:.4f})")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        e = [h["epoch"] for h in hist]
        ax[0].plot(e, [h["train_loss"] for h in hist], label="train"); ax[0].plot(e, [h["val_loss"] for h in hist], label="val")
        ax[0].set_title("expected loss (1 - score)"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(e, [h["val_regret_tol"] for h in hist], label="model, tol")
        ax[1].plot(e, [h["val_zero_regret_tol"] for h in hist], "--", label="CV box, tol")
        ax[1].plot(e, [h["val_regret_exact"] for h in hist], label="model, exact")
        ax[1].plot(e, [h["val_zero_regret_exact"] for h in hist], "--", label="CV box, exact")
        ax[1].set_title("val regret (lower is better)"); ax[1].set_xlabel("epoch"); ax[1].legend()
        fig.tight_layout(); fig.savefig(out / f"curves{a.tag}.png", dpi=120)
    except ImportError:
        print("     (matplotlib not installed: no curves.png, see history.csv)")

    last = hist[-1]
    gates.check("loss is finite", all(np.isfinite(h["train_loss"]) and np.isfinite(h["val_loss"]) for h in hist), f"last train {last['train_loss']:.4f}")
    gates.check("train loss decreased", hist[-1]["train_loss_running"] < hist[0]["train_loss_running"] - 1e-4 or a.smoke,
                f"training objective {hist[0]['train_loss_running']:.4f} -> {hist[-1]['train_loss_running']:.4f}")
    moved = abs(hist[-1]["train_loss_running"] - hist[-6]["train_loss_running"]) if len(hist) > 6 else 1.0
    gates.check("training is not frozen", moved > 1e-5 or a.smoke, f"objective changed by {moved:.2e} over the last 5 epochs (0 = zero gradient = saturated/dead model)", hard=False)
    gates.check("model beats the CV box on val (best epoch)", best_val < hist[best_ep - 1][f"val_zero_regret_{sel_key}"],
                f"val regret {best_val:.4f} vs CV box {hist[best_ep - 1][f'val_zero_regret_{sel_key}']:.4f} at epoch {best_ep}", hard=False)
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    gates.finish()


if __name__ == "__main__":
    main()