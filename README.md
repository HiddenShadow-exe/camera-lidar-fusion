# Learning to correct camera-based boxes for LiDAR occlusion completion

**Goal.** A warehouse robot has a LiDAR and an overhead RGB-D camera. When a box is partly hidden behind another object, the LiDAR only sees the occluder. We fit a box to the *camera* points of the hidden box and extend the blocked LiDAR rays to that box's front face, which fills the hidden part of the occupancy grid. The camera box is a few centimetres off, so the filled voxels land in the wrong place.

**Idea.** A small point-cloud network looks at the camera points and predicts how far to move each of the six faces of the box (independently, in both directions) before the rays are extended. The LiDAR is never used to fit the box; it is used only for the ray extension and for scoring against the ground truth (the same scene recorded again with the occluder removed).

**Evaluation.** 31 real scenes, 5-fold cross-validation by scene (a scene is never in training and test). The completed 3 cm voxels are compared with the ground-truth voxels at the exact level and with a 1-voxel tolerance. All methods use the same completion step.

## Result

![Comparison of four box sources on the final voxel completion](work_v3/stage9/figure_methods_compact.png)

*(a) F1 by tolerance, (b, c) where regenerated / hidden cells find their counterpart, (d) per-box F1, (e) worst distance per box, (f) summary. "Best shift in grid" uses the ground truth and is only an upper bound.*

| F1 in % (31 boxes) | exact voxel | within 1 voxel |
|---|---|---|
| Traditional CV box | 28.6 | 72.0 |
| PointNet box regression (trained on hand labels) | 33.0 | 81.5 |
| **CV box + learned face shifts (ours)** | **36.3** | **83.3** |
| Hand-labelled box (reference) | 42.4 | 89.2 |
| Best possible shift (uses ground truth) | 55.5 | 94.3 |

* Our shifts improve the traditional box by **+11 F1 points within 1 voxel** (95 % CI [+1, +22]) and cut the share of regenerated cells more than one voxel away from the ground truth from 18 % to 6 % (PointNet: 13 %).
* Ours and the PointNet baseline are statistically indistinguishable (+1.8 points within 1 voxel, CI [−4.9, +8.2]); PointNet's own gain over the traditional box is not significant (+9.4, CI [−1.4, +21.2]). Hand labels remain better than both.
* Limits: only 31 real scenes and one training seed. The model occasionally spoils a box that was already good (one box drops to 0 %).

## Where to find what

| Folder | Content |
|---|---|
| `occ_pipeline/` | All code of the project, one script per stage (`s0_…` to `s9_…`), `run_pipeline.py` runs stages 0–5. |
| `model/` | PointNet box-regression baseline (`train_obb.py`) and its data augmentation. |
| `dataset/` | Raw recordings (odd = occluded scene, even = ground truth; only the even ones carry correct hand labels) per category, plus generated "virtual" occluder scenes used for training only. |
| `work_v3/stage0` – `stage5` | Intermediate data: samples, camera boxes, scores of all 164,025 shift candidates per sample, model dataset. |
| `work_v3/stage6` | Cross-validated model predictions. |
| `work_v3/stage7_per_sample_graded.csv` | Per-box regret of every method. |
| `work_v3/stage8_graded/` | Exact / 1-voxel breakdown figures and tables. |
| `work_v3/stage9/` | **Final four-way comparison**: `figure_methods_compact.png/.pdf`, `parts/` (single panels), `comparison_summary.csv`, `comparison_per_box.csv`, `FIGURE_CAPTION.txt`. |
| `PROJECT_DOCUMENTATION.md` | Full technical documentation (method, every stage, commands, results, known issues). |

## Reproduce

```powershell
python .\occ_pipeline\run_pipeline.py --input dataset --work work_v3 --from-stage 0 --to-stage 5 --n-aug 3 --voxel-size 0.03 --n-steps 4 --n-steps-z 2 --step 0.03 --exclude-units occluded_boxes_07_u1
python .\occ_pipeline\s6_train.py --work work_v3 --cv 5 --epochs 60 --score graded --tag _graded
python .\occ_pipeline\s7_evaluate.py --work work_v3 --split cv --tag _graded
python .\occ_pipeline\s8_tolerance_breakdown.py --work work_v3 --split cv --tag _graded
python .\model\dataset_augmentation.py --input dataset --output dataset_aug_v3 --n-aug 12
python .\occ_pipeline\s9a_pointnet_boxes.py --work work_v3 --pred work_v3/stage6/pred_cv_graded.npz --aug-data dataset_aug_v3 --train-obb-dir model   # even recordings only (default)
python .\occ_pipeline\s9b_manual_boxes.py --input dataset --work work_v3
python .\occ_pipeline\s9_compare_methods.py --work work_v3 --pred work_v3/stage6/pred_cv_graded.npz
```
Requires Python with PyTorch (CUDA recommended: stage 4 takes about one hour on a GPU), NumPy, SciPy, Matplotlib.