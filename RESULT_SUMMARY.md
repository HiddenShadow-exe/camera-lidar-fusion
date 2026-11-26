# Results summary (work_virt, 37 real boxes in 35 scenes, 5-fold scene-level cross-validation, 3 cm voxels)

## Main table: mean regret = best possible score - achieved score (lower is better)
| Method | tol | exact | graded |
|---|---|---|---|
| CV box (no shift) | 0.273 | 0.190 | 0.233 |
| Best constant shift (fitted incl. virtual samples) | 0.262 | 0.161 | 0.222 |
| Best constant shift (fitted on real scenes only) | 0.188 | 0.204 | 0.223 |
| Model, trained on tol | 0.185 | 0.174 | 0.185 |
| **Model, trained on graded score** | **0.150** | **0.150** | **0.154** |
| Model, tol, without virtual boxes | 0.210 | 0.183 | 0.200 |

Mean score of the graded-trained model vs CV box vs best possible: tol 0.745 / 0.622 / 0.895, exact 0.222 / 0.183 / 0.372, graded 0.487 / 0.408 / 0.641.

## Cell-level breakdown (graded-trained model, stage 8)
| | CV box | Best constant | Model | Oracle (graded) |
|---|---|---|---|---|
| F1 exact | 28 | 29 | 35 | 53 |
| F1 exact+face | 64 | 64 | 75 | 89 |
| F1 exact+face+edge | 71 | 72 | 83 | 93 |
| F1 exact+face+edge+vertex (1-voxel) | 72 | 73 | 84 | 93 |
| Regenerated cells > 1 voxel from GT | 18.6 % | 16.7 % | 7.2 % | 1.6 % |
| Hidden GT cells without a regenerated cell within 1 voxel | 32.7 % | 32.4 % | 19.3 % | 9.4 % |

Precision = share of regenerated cells that hit; recall = share of hidden GT cells that hit. Exact precision / recall: CV box 31 / 27, model 44 / 30.

## Statistics (paired, bootstrap over scenes, tol regret, 36 boxes in the pasted table)
- Graded model vs CV box: -0.126, 95% CI [-0.217, -0.044] (significant).
- Graded model vs best constant (real-only fit): -0.037, CI [-0.095, +0.021] (not significant).
- Graded model vs tol-trained model: -0.037, CI [-0.083, +0.002] (borderline).
- Boxes where the CV box is clearly off (23 boxes): regret 0.413 -> 0.176. Boxes where it is already good (13 boxes): 0.042 -> 0.114.

## Say it honestly
- The learned face shifts clearly beat the traditional CV box and remove more than half of the gross errors (cells more than one voxel off).
- The gain over one fixed correction for all boxes is small and not statistically established with 35 scenes.
- Virtual occluders (one run) and the graded loss (one run) both helped; neither effect is proven beyond one seed.
- The model over-corrects some boxes that were already good (4 clearly worse: multi_03, partial_03, occluded_13, occluded_19).
- 7 boxes remain far from optimal (regret > 0.3); occluded_07_u1 is probably unsolvable (best possible 0.28).
- Numbers use our 3 cm box-aligned scoring with recall on hidden cells only; they are not comparable with the live-script evaluation.
- Single seed, fixed 60 epochs, held-out regret fluctuates by about 0.05 between epochs; 29 % of oracles sit on the grid boundary.

## Next steps (not done)
Several seeds and an ensemble, early stopping chosen on an inner validation split, a fallback to the CV box for boxes it already fits well,
re-scoring in the world-aligned live-script protocol, yaw as an extra shift.