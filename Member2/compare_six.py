"""Compare saved model scores on the verified original reference-label snapshot."""
import json
from pathlib import Path

import pandas as pd

from train_models import evaluate


def main():
    root = Path(__file__).resolve().parent
    original = root / "results_zip"
    sixth = root / "results_random_forest"
    labels = pd.read_csv(root / "data_snapshot/student_test_candidates.csv.gz", dtype={"trip_id": str})
    labels = labels[labels.label_usable == 1].reset_index(drop=True)
    keys = ["trip_id", "point_index", "arc_id"]
    folds = pd.concat([pd.read_csv(original / "fold_results.csv"),
                       pd.read_csv(sixth / "fold_results.csv")], ignore_index=True)
    rows = []
    for name, results in folds.groupby("model", sort=False):
        directory = sixth if name == "random_forest" else original
        scores = pd.read_csv(directory / f"student_scores_{name}.csv.gz", dtype={"trip_id": str})
        pd.testing.assert_frame_equal(labels[keys], scores[keys])
        metrics = evaluate(labels, scores.raw_score.to_numpy())
        rows.append(dict(model=name, cv_top1_mean=results.top1.mean(),
                         cv_top1_std=results.top1.std(), cv_macro_f1_mean=results.macro_f1.mean(),
                         cv_log_loss_mean=results.log_loss.mean(), student_top1=metrics["top1"],
                         student_macro_f1=metrics["macro_f1"], student_log_loss=metrics["log_loss"],
                         student_eligible_points=metrics["eligible_points"]))
    comparison = pd.DataFrame(rows).sort_values(["cv_top1_mean", "cv_macro_f1_mean"], ascending=False)
    comparison.to_csv(sixth / "comparison_six.csv", index=False)
    folds.to_csv(sixth / "fold_results_six.csv", index=False)
    (sixth / "comparison_notes.json").write_text(json.dumps(dict(
        source_snapshot="Member2/data_snapshot/snapshot_manifest.json",
        all_student_keys_verified=True,
        labels="Approximate offline reference-route alignment",
        hyperparameters="Random Forest parameters chosen before its evaluation; no score-matching tuning",
        environments="Original five models ran on Windows; Random Forest ran locally. Timing is not directly comparable.",
        aggregation="CV values are unweighted averages of the five trip-grouped folds."), indent=2) + "\n")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
