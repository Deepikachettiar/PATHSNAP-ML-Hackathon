# Sixth classifier: Random Forest

The model uses 100 bootstrap trees, maximum depth 16, minimum leaf size 20,
square-root feature sampling, median imputation, and seed 42. Parameters were
chosen before evaluation. The same 15 features and five trip folds as the
original experiments were used. No parameters were tuned to match a desired score.

The earlier handoff was recovered from archive.zip into Member2/data_snapshot.
Its fold observation counts and student candidate keys match the original saved
experiments. snapshot_manifest.json records the input hashes. The current Member 1
handoff contains updated alignment labels, so it would produce a different experiment.

Results use approximate reference-route labels. Mean grouped-CV Top-1 is 84.0655%,
mean grouped-CV macro F1 is 0.901839, and mean grouped-CV log loss is 0.119157.
Student-test Top-1 is 89.0894%, macro F1 is 0.930052, and log loss is 0.090318.
Top-1 evaluation includes 13,189 usable reference points, including candidate misses.

- models/random_forest.joblib contains the fitted pipeline and feature metadata.
- comparison.csv records this model's metrics and timings.
- comparison_six.csv compares all six models on the same snapshot.
- fold_results.csv and fold_results_six.csv retain fold-level evidence.
- oof_scores_random_forest.csv.gz contains validation scores.
- student_scores_random_forest.csv.gz contains eligible evaluation-point scores.
- scores_all_student_test_random_forest.csv.gz scores all 130,923 candidate rows
  covering 19,160 GPS points for decoder use.
- run_config.json records the actual runtime and hyperparameters.
- scoring_audit.json records the completed export checks.

The original five experiments ran on Windows, so their timings are not directly
comparable with this local run. Extra Trees remains first under the written
CV Top-1 selection rule. Random Forest ranks second. Bayesian optimisation is
still a separate experiment to perform.

Run from the workspace root:

```bash
python Member2/train_models.py --data-dir Member2/data_snapshot --out-dir Member2/results_random_forest --models random_forest
python Member2/compare_six.py
python Member2/score_all.py --model Member2/results_random_forest/models/random_forest.joblib --candidates Member2/data_snapshot/student_test_candidates.csv.gz --out Member2/results_random_forest/scores_all_student_test_random_forest.csv.gz
```
