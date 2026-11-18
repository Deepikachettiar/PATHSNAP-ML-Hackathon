"""Join label-free candidates/features with separate offline labels for model development."""
import csv
import gzip
import itertools
import json
from pathlib import Path

import numpy as np


FEATURE_COLUMNS = ["distance_m", "candidate_rank", "outside_radius", "arc_length_m", "arc_fraction",
    "distance_from_arc_start_m", "distance_to_arc_end_m", "arc_bearing_deg", "gps_heading_deg",
    "heading_difference_deg", "dt_s", "gps_step_m", "gps_speed_m_s", "from_out_degree", "to_in_degree"]


def run(output=Path("output")):
    destination = output / "handoff"
    destination.mkdir(exist_ok=True)
    inventory = json.loads((output / "validation_report.json").read_text())["trips"]
    train = sorted(t["trip_id"] for t in inventory if t["split"] == "train")
    np.random.default_rng(42).shuffle(train)
    folds = {trip: i % 5 for i, trip in enumerate(train)}
    with (destination / "trip_folds.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["trip_id", "split", "fold_id"])
        w.writerows([t["trip_id"], t["split"], folds.get(t["trip_id"], "")] for t in inventory)
    feature_schema = dict(feature_columns=FEATURE_COLUMNS, key_columns=["trip_id", "point_index", "arc_id"],
        target_column="target", filter_before_fitting="training_group_usable == 1",
        grouping_column="trip_id", predefined_fold_column="fold_id", folds=5, seed=42,
        excluded_from_model_features=["trip_id", "point_index", "arc_id", "reference_arc_id", "fold_id",
            "split", "target", "label_usable", "candidate_hit", "training_group_usable"],
        missing_values="empty cells; fit imputers and scalers on each training fold only",
        label_status="approximate reference-route alignment; not independently verified point labels")
    (destination / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2) + "\n")
    fields = ["trip_id", "point_index", "arc_id", "split", "fold_id"] + FEATURE_COLUMNS + [
        "reference_arc_id", "label_usable", "candidate_hit", "training_group_usable", "target"]
    summary = {s: dict(points=0, rows=0, usable_reference_points=0, candidate_hits_all=0,
                       candidate_hits_usable=0, training_groups=0, positive_rows=0) for s in ["train", "student_test"]}
    with (output / "candidates.csv").open() as cf, gzip.open(output / "candidate_features.csv.gz", "rt") as ff, (
        output / "reference_alignment" / "point_labels.csv").open() as lf, gzip.open(
        destination / "train_candidates.csv.gz", "wt", newline="", compresslevel=1) as tf, gzip.open(
        destination / "student_test_candidates.csv.gz", "wt", newline="", compresslevel=1) as sf:
        features, labels = csv.DictReader(ff), csv.DictReader(lf)
        writers = {"train": csv.DictWriter(tf, fieldnames=fields), "student_test": csv.DictWriter(sf, fieldnames=fields)}
        for writer in writers.values():
            writer.writeheader()
        for key, group in itertools.groupby(csv.DictReader(cf), key=lambda r: (r["trip_id"], r["point_index"])):
            candidates = list(group)
            label = next(labels)
            assert key == (label["trip_id"], label["point_index"])
            hit = any(c["arc_id"] == label["reference_arc_id"] for c in candidates)
            usable = label["label_usable"] == "1"
            trainable = usable and hit
            split = label["split"]
            stats = summary[split]
            stats["points"] += 1
            stats["usable_reference_points"] += int(usable)
            stats["candidate_hits_all"] += int(hit)
            stats["candidate_hits_usable"] += int(hit and usable)
            stats["training_groups"] += int(trainable)
            for candidate in candidates:
                feature = next(features)
                assert all(candidate[k] == feature[k] for k in ["trip_id", "point_index", "arc_id"])
                row = {k: candidate[k] if k in candidate else feature[k] for k in FEATURE_COLUMNS}
                target = int(candidate["arc_id"] == label["reference_arc_id"]) if trainable else ""
                row.update(trip_id=key[0], point_index=key[1], arc_id=candidate["arc_id"], split=split,
                    fold_id=folds.get(key[0], ""), reference_arc_id=label["reference_arc_id"],
                    label_usable=int(usable), candidate_hit=int(hit), training_group_usable=int(trainable), target=target)
                writers[split].writerow(row)
                stats["rows"] += 1
                stats["positive_rows"] += int(target == 1)
        assert next(features, None) is None and next(labels, None) is None
    for stats in summary.values():
        stats["candidate_miss_rate_all_approximate_labels"] = 1 - stats["candidate_hits_all"] / stats["points"]
        stats["candidate_miss_rate_usable_approximate_labels"] = (
            1 - stats["candidate_hits_usable"] / stats["usable_reference_points"] if stats["usable_reference_points"] else None)
        assert stats["training_groups"] == stats["positive_rows"]
    (destination / "handoff_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
