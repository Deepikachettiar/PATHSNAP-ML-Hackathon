"""Reopen the model exports and check aliases, geometry, labels, and whole point groups."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from export_model_ready import ALIASES, FEATURES


def run(output=Path("output")):
    folder = output / "model_ready"
    summary = json.loads((folder / "export_summary.json").read_text())
    result = {}
    for name in ["candidates_with_features_labels", "train", "student_test"]:
        pieces, total = [], 0
        for chunk in pd.read_csv(folder / f"{name}.csv.gz", dtype={"trip_id": str}, chunksize=50000):
            assert not chunk.duplicated(["trip_id", "point_index", "arc_id"]).any()
            for alias, source in ALIASES.items():
                np.testing.assert_allclose(chunk[alias], chunk[source], equal_nan=True, atol=1e-8)
            assert np.isfinite(chunk[["distance_m", "offset_m", "arc_length_m", "projected_x_m", "projected_y_m"]]).all().all()
            assert (chunk.offset_m >= 0).all()
            assert (chunk.offset_m <= chunk.arc_length_m + 1e-6).all()
            assert chunk.heading_mismatch_deg.dropna().between(0, 180).all()
            assert not np.isinf(chunk[FEATURES].to_numpy(float)).any()
            good = chunk.training_group_usable == 1
            assert chunk.loc[good, "label"].isin([0, 1]).all()
            assert chunk.loc[~good, "label"].isna().all()
            np.testing.assert_array_equal(chunk.loc[good, "label"],
                (chunk.loc[good, "arc_id"] == chunk.loc[good, "reference_arc_id"]).astype(int))
            if name != "candidates_with_features_labels":
                assert good.all() and (chunk.split == name).all()
            chunk["positive"] = (chunk.label == 1).astype(int)
            pieces.append(chunk.groupby(["split", "trip_id", "point_index"]).agg(
                positive=("positive", "sum"), usable=("training_group_usable", "max")))
            total += len(chunk)
        groups = pd.concat(pieces).groupby(level=[0, 1, 2]).agg(positive=("positive", "sum"), usable=("usable", "max"))
        np.testing.assert_array_equal(groups.positive, groups.usable)
        expected = sum(t["all_rows"] for t in summary.values()) if name == "candidates_with_features_labels" else summary[name]["model_rows"]
        assert total == expected
        result[name] = dict(rows=total, point_groups=len(groups), eligible_groups=int(groups.usable.sum()),
                            positive_rows=int(groups.positive.sum()))
    result["status"] = "passed"
    (folder / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
