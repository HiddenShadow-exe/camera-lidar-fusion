# Occlusion completion with a learned box correction: pipeline and evaluation

## 1. Idea in one paragraph
The live script fits a box to the **camera** point cluster only (traditional CV box). LiDAR rays that were blocked by an occluder are
extended until they hit the front face of that box; the extended points are the *regenerated* points. The camera box is a few cm off
(systematic bias plus per-box error), so the regenerated points land slightly wrong. We therefore train a model that looks at the
camera cluster and predicts **how far to move each of the 6 box faces** before the rays are extended. The LiDAR is never used to
*fit* the box. It is only used for the ray extension and, with the ground truth, for scoring.

## 2. Data
- Recordings come in pairs: **odd = occluded scene, even = ground truth (GT)** of the same scene with the occluder removed.
  35 real scenes / 37 boxes in 4 categories (full, multi, partial, occluded). The `_cardboard box` folder is not a real pair and is excluded.
- Your 100 single box recordings (no pair) are used through **virtual occluders** (`make_virtual_pairs.py`): the original scan is the GT, a
  synthetic box occluder is placed between LiDAR and target and the blocked scan points are replaced by its surface. These are **training only**,
  never evaluated.
- Everything lives in the LiDAR frame (z up, floor at z ~ -0.4 m). Camera to LiDAR is `R = R_yaw @ diag(1,-1,-1)`, as in the live script.

## 3. Pipeline (stages)
| Stage | What it does |
|---|---|
| 0 `s0_augment_pairs` | One sample per target box (connected camera cluster of the GT recording). Stores cluster, LiDAR origin, occluded scan, GT, and `gt_shadow` (a GT point is *shadowed* if the nearest occluded-scan ray, within 0.5 deg, is more than 5 cm closer). Adds augmented copies (`--n-aug`: rotation about the vertical axis through the camera, small shift, sensor artifacts). Scene-level train/val split, so copies of one scene never straddle the split. |
| 1 `s1_cv_boxes` | The traditional CV box per sample: top plane, minimum-area rectangle, extended down to the floor. Gate: GT points must lie a few cm from the box surface (catches frame mix-ups). |
| 2 `s2_check_grid` | Checks the shift-grid maths. |
| 3 `s3_parity` | The fast evaluator must reproduce a slow reference (your Moller-Trumbore caster + voxel sets). |
| 4 `s4_loss_tables` | **Score every candidate shift of every sample** (see 4). Slow, resumable. |
| 5 `s5_build_dataset` | Packs the model inputs and the score tables. |
| 6 `s6_train` | Trains the model, 5-fold scene-level cross-validation (needs torch). |
| 7 `s7_evaluate` | Regret evaluation against baselines. |
| 8 `s8_tolerance_breakdown` | Exact / face / edge / vertex breakdown of the same choices. |

## 4. Candidates and scores (stage 4)
- **Candidate** = every face moved independently along its outward normal (+ grows the box): `[lo_x, hi_x, lo_y, hi_y, lo_z, hi_z]`,
  in steps of `--step` (3 cm in the last run), `n` steps each way (x/y +-9 cm, z +-6 cm, 60,025 candidates). Boxes thinner than 5 cm are invalid.
- **Regeneration per candidate:** a ray is *blocked* if it enters the box farther than its measured range plus a margin (3 cm); it gets a point
  `origin + t_in * dir` on the box's front face. The points are voxelised (3 cm, grid aligned to the CV box, so the score does not depend on how the scene sits in the world).
- **Sets:** P = regenerated voxels, G = all GT voxels, S = GT voxels hidden by the occluder (what has to be recovered).
- **Scores** (1 = perfect, 0 = nothing recovered):
  - **exact** = |P and S| / (|S| + |P without G|). Hit = same voxel.
  - **tol** = the same, but a hit may be within 1 voxel (cube, 26 neighbours).
  - **graded** = distance-weighted: exact 1.00, face 0.61, edge 0.37, vertex 0.22, 2 voxels straight 0.14, farther 0 (`--graded-sigma`, `--graded-radius`).
- **Oracle** = best candidate of the table (ties go to the smallest shift). It needs the GT, so it is a ceiling.

## 5. Model (stage 5 and 6)
- Input: the camera cluster (2048 points, in the CV-box frame) plus a 9-value context (LiDAR origin and camera centre in the box frame, CV dims).
- Output: three heads, one per axis, each a distribution over that axis' (lo, hi) shift pairs. P(candidate) = P_x * P_y * P_z. Too-thin pairs are masked.
- Loss: the target is `softmax(score / 0.05)` over all candidates of the sample, trained with KL (`--loss soft`). `--score tol|exact|mix|graded` picks the table.
- Cross-validation: `--cv 5` by scene, fixed number of epochs, no epoch picking on the held-out fold. Virtual samples are always training data
  (`--no-virtual` excludes them as an ablation, `--tag` keeps runs apart).

## 6. Evaluation (stage 7 and 8)
- **Regret** = oracle score - score of the chosen candidate (0 = optimal). Mean over the 37 real boxes, 95% CI by bootstrap over scenes.
- **Methods compared** (same tables for all):
  - **CV box**: shift 0.
  - **Best constant**: one shift for all boxes, fitted on the training scenes of each fold, limited to valid shifts per box. The "no per-box learning" baseline.
  - **Model**: argmax of the predicted distribution, out-of-fold.
  - **Oracle (tol / exact / graded)**: best shift by that score, uses the GT, ceiling only.
  - **Random**: expected score of a random valid candidate.
- Reported for tol, exact and graded, plus "optimal" (share of boxes where the choice is within 0.001 of the oracle) and the shift error to the nearest optimal candidate in cm.
- **Stage 8**: regenerates each method's choice and classifies every cell by its nearest neighbour in the other set
  (exact | face | edge | vertex | more than 1 voxel). P / R / F1 are given for exact, exact+face, +edge, +vertex.
  Precision side: regenerated cell to GT. Recall side: hidden GT cell (S) to regenerated cell.
- **Separate live-script protocol** (`occupancy_tolerance_eval.py`): evaluates the unchanged live pipeline against the full GT at 3 cm with the same tolerance levels.
- Hard gates stop a stage on errors (e.g. regret never negative, no invalid choice, recomputed score equals the table).

## 7. Results so far (37 real boxes, tol regret; lower is better)
| Run | CV box | Best constant | Model | Oracle score |
|---|---|---|---|---|
| work_all (2 cm step, no virtual) | 0.41 | 0.26 | 0.24 | 0.81 |
| work_virt (3 cm step, with virtual) | 0.27 | 0.26 | 0.19 | 0.90 |

The runs are **not directly comparable** (the grid also changed). In work_virt the model beats the constant by -0.076 (CI [-0.16, 0.00]) and the
CV box by -0.087 (CI [-0.17, -0.01]). Its gain comes from boxes where the CV box is bad (regret 0.56 to 0.28); on already-good boxes it is slightly
worse (0.10 to 0.13). On the **exact** score the model is *not* better than the constant (0.174 vs 0.161): it finds the right neighbourhood, not the exact voxel.

## 8. Open points
- Graded score and `--no-virtual` runs are implemented but not run yet (stages 4 to 8).
- Top and bottom faces probably barely influence the rays; the best constant only moves z faces. A z-flatness check is parked.
- The model over-corrects boxes that are already good. Candidates: ensembling, early stopping, a fallback to the CV box.
- `occluded_boxes_07_u1` (oracle 0.28) is a phantom or unsolvable unit.
- Final numbers should also be re-scored in the world-aligned 3 cm protocol of the live script. Yaw as a 7th shift is a possible later extension.