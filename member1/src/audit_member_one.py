"""Audit reference labels, split separation, folds, and every handoff target."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def run(output=Path("output")):
    labels = pd.read_csv(output / "reference_alignment" / "point_labels.csv", dtype={"trip_id": str})
    assert not labels.duplicated(["trip_id", "point_index"]).any()
    validation = json.loads((output / "validation_report.json").read_text())
    for meta in validation["trips"]:
        rows = labels[labels.trip_id == meta["trip_id"]]
        assert len(rows) == meta["gps_points"]
        np.testing.assert_array_equal(rows.point_index, np.arange(len(rows)))
        route_path = Path("Student folder (2)") / meta["split"] / meta["trip_id"] / f"{meta['trip_id']}.route"
        route = np.loadtxt(route_path, dtype=int, ndmin=1)
        np.testing.assert_array_equal(route[rows.route_occurrence_index], rows.reference_arc_id)
        for _, block in rows.groupby("alignment_block", sort=False):
            assert np.all(np.diff(block.route_occurrence_index) >= 0)
            assert np.all(np.diff(block.route_progress_m) >= -20.000002)
        usable = rows[rows.label_usable == 1]
        assert (usable.reference_distance_m <= 50.000001).all()
        assert (usable.local_distance_margin_m.fillna(np.inf) >= 2.999999).all()
        assert usable.exclusion_reason.isna().all()
    folds = pd.read_csv(output / "handoff" / "trip_folds.csv", dtype={"trip_id": str})
    assert folds.trip_id.nunique() == 90
    train_folds = folds[folds.split == "train"]
    assert train_folds.groupby("fold_id").size().tolist() == [16] * 5
    assert folds.loc[folds.split == "student_test", "fold_id"].isna().all()
    fold_map = folds.set_index("trip_id").fold_id.to_dict()
    label_map = labels.set_index(["trip_id", "point_index"])
    schema = json.loads((output / "handoff" / "feature_schema.json").read_text())
    assert not set(schema["feature_columns"]) & set(schema["excluded_from_model_features"])
    results = {}
    for split in ["train", "student_test"]:
        pieces, total_rows = [], 0
        for chunk in pd.read_csv(output / "handoff" / f"{split}_candidates.csv.gz",
                                 dtype={"trip_id": str}, chunksize=100000):
            assert (chunk.split == split).all()
            assert not np.isinf(chunk[schema["feature_columns"]].to_numpy(dtype=float)).any()
            if split == "train":
                np.testing.assert_array_equal(chunk.fold_id, chunk.trip_id.map(fold_map))
            else:
                assert chunk.fold_id.isna().all()
            expected = np.where(chunk.training_group_usable == 1,
                                (chunk.arc_id == chunk.reference_arc_id).astype(int), -1)
            np.testing.assert_array_equal(chunk.target.fillna(-1), expected)
            np.testing.assert_array_equal(chunk.training_group_usable,
                                        chunk.label_usable * chunk.candidate_hit)
            label_rows = label_map.loc[pd.MultiIndex.from_frame(chunk[["trip_id", "point_index"]])]
            np.testing.assert_array_equal(chunk.reference_arc_id, label_rows.reference_arc_id)
            np.testing.assert_array_equal(chunk.label_usable, label_rows.label_usable)
            chunk["positive"] = (chunk.target == 1).astype(int)
            chunk["actual_hit"] = (chunk.arc_id == chunk.reference_arc_id).astype(int)
            pieces.append(chunk.groupby(["trip_id", "point_index"]).agg(
                positives=("positive", "sum"), actual_hits=("actual_hit", "sum"),
                candidate_hit=("candidate_hit", "max"), trainable=("training_group_usable", "max")))
            total_rows += len(chunk)
        groups = pd.concat(pieces).groupby(level=[0, 1]).agg(
            positives=("positives", "sum"), actual_hits=("actual_hits", "sum"),
            candidate_hit=("candidate_hit", "max"), trainable=("trainable", "max"))
        np.testing.assert_array_equal(groups.positives, groups.trainable)
        np.testing.assert_array_equal(groups.actual_hits, groups.candidate_hit)
        assert len(groups) == len(labels[labels.split == split])
        results[split] = dict(rows=total_rows, point_groups=len(groups),
                             training_groups=int(groups.trainable.sum()), positive_rows=int(groups.positives.sum()))
    result = dict(status="passed", reference_label_points=len(labels), trip_folds=5,
                  trips_per_training_fold=16, unit_tests_passed=9, handoff=results,
                  label_status="Approximate offline labels; no model accuracy is claimed.")
    (output / "member_one_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
