# PathSnap — Member 2: ML Model Training & Evaluation

## What this part does

Member 2 trains and compares machine-learning models that score road-segment candidates for each GPS observation. The scores are handed to the sequence-decoding stage (Member 3) and can also be consumed by the application/integration stage (Member 4).

## Included files

- `train_models.py` — trains and evaluates five models using grouped 5-fold cross-validation by trip, then fits each model on the full training-trip set and creates scores for the student-test trips.
- `score_all.py` — applies a saved model to **all candidate rows** in a candidate CSV/GZIP CSV and normalizes candidate probabilities within each `(trip_id, point_index)` group. This script does not read labels.
- `results_zip/models/` — saved model bundles (`joblib`) for all five model families.
- `results_zip/fold_results.csv` — per-fold evaluation results.
- `results_zip/run_config.json` — seed, feature list, model configurations, selection rule, package versions, and label caveat.
- `results_zip/oof_scores_*.csv.gz` — out-of-fold candidate scores.
- `results_zip/student_scores_*.csv.gz` — student-test candidate scores for each model.
- `results_zip/mlp_loss_curve.csv` — MLP training loss history.

## Models compared

| Model | Configuration recorded in run config |
|---|---|
| Logistic Regression | C=1, L2, StandardScaler, max_iter=300 |
| Gaussian Naive Bayes | var_smoothing=1e-9 |
| Extra Trees | 100 trees, max_depth=16, min_samples_leaf=20 |
| HistGradientBoosting | learning_rate=0.1, max_iter=200, max_leaf_nodes=31 |
| MLP | hidden layers (64, 32), Adam, max_iter=40, StandardScaler |

## Cross-validation results

Means across the five grouped CV folds, computed from the included `fold_results.csv`:

| Model | Top-1 accuracy | Macro F1 | Log loss | Mean fit time (s) |
|---|---:|---:|---:|---:|
| **Extra Trees** | **84.25%** | 89.97% | 0.138 | 17.41 |
| HistGradientBoosting | 83.96% | **90.16%** | **0.129** | 2.68 |
| Logistic Regression | 83.84% | 88.63% | 0.152 | 4.08 |
| Gaussian Naive Bayes | 82.61% | 86.93% | 0.455 | 2.37 |
| MLP | 82.34% | 88.94% | 0.156 | 153.13 |

**Selection rule written in the experiment configuration:** highest grouped-CV Top-1 accuracy; ties are broken by Macro F1, then lower student-test inference time. Under this rule, **Extra Trees** is the selected model based on the CV results shown. HistGradientBoosting is very close in Top-1 accuracy and has a slightly higher Macro F1, lower log loss, and much shorter mean CV fit time.

These are cross-validation averages—not a claim of performance on the evaluator's hidden test set.

## Features used

The recorded feature list has 15 features:

`distance_m`, `candidate_rank`, `outside_radius`, `arc_length_m`, `arc_fraction`, `distance_from_arc_start_m`, `distance_to_arc_end_m`, `arc_bearing_deg`, `gps_heading_deg`, `heading_difference_deg`, `dt_s`, `gps_step_m`, `gps_speed_m_s`, `from_out_degree`, `to_in_degree`.

## Dataset and evaluation notes

- The training script documents 80 training trips, 5 folds grouped by trip, and a 10-trip student-test split.
- The saved results include candidate-score files for the student-test split (84,315 candidate rows per model).
- The ZIP does **not** include the input dataset/handoff CSVs, so training cannot be reproduced from this ZIP alone. Obtain Member 1's handoff data to rerun.
- The archive has fold-level results, but no consolidated `comparison.csv` / student-test metric summary. The student-test score files are present; do not report a student-test metric unless it is computed from the proper evaluation labels.
- Labels are described in `run_config.json` as approximate reference-route alignment, not independently verified point labels.
- No Bayesian hyperparameter optimization is evidenced in the included script/config; the listed model configurations are fixed settings.

## Re-run training (requires Member 1's handoff data)

Install compatible Python packages (the recorded run used Python 3.13.7, scikit-learn 1.9.1, pandas 3.1.1, and NumPy 2.5.5), then run:

```bash
python train_models.py --data-dir path/to/Member1/output/handoff --out-dir results
```

For a quick debug run only:

```bash
python train_models.py --data-dir path/to/Member1/output/handoff --out-dir results_debug --subsample 0.05
```

## Score candidates for Member 3 / integration

Example using the saved Extra Trees model:

```bash
python score_all.py --model results_zip/models/extra_trees.joblib --candidates path/to/student_test_candidates.csv.gz --out results/scores_all_student_test_extra_trees.csv.gz
```

The input candidate file must contain the model's feature columns plus `trip_id`, `point_index`, and `arc_id`. The output has `trip_id`, `point_index`, `arc_id`, `raw_score`, `candidate_probability`, and `model_version`. Probabilities are normalized within each GPS point.

## GitHub guidance

Upload the **extracted project folder contents** to your repository rather than relying only on the ZIP. The largest saved model is Extra Trees (~27 MB), which is below GitHub's 100 MB individual-file limit. The complete archive is large (~107 MB); check the repository's total size and consider Git LFS or excluding bulky score files if your team does not need to version them. Keep the scripts, README, config, fold results, and selected model artifacts available for review.
