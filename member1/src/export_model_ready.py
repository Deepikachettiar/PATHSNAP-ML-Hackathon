"""Export self-contained candidate tables with geometry, features, and offline labels."""
import argparse
from contextlib import ExitStack
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


ALIASES = {
    "offset_m": "distance_from_arc_start_m",
    "heading_mismatch_deg": "heading_difference_deg",
    "displacement_m": "gps_step_m",
    "elapsed_s": "dt_s",
    "label": "target",
}
FEATURES = ["distance_m", "offset_m", "heading_mismatch_deg", "displacement_m", "elapsed_s",
            "arc_length_m", "arc_fraction", "distance_to_arc_end_m", "gps_speed_m_s",
            "from_out_degree", "to_in_degree", "outside_radius"]
KEYS = ["trip_id", "point_index", "arc_id"]


class RowBuffer:
    """Read the next n rows, even across a compressed CSV chunk boundary."""
    def __init__(self, iterator):
        self.iterator = iterator
        self.pending = pd.DataFrame()

    def take(self, n):
        parts = []
        while n:
            if self.pending.empty:
                self.pending = next(self.iterator)
            used = min(n, len(self.pending))
            parts.append(self.pending.iloc[:used])
            self.pending = self.pending.iloc[used:]
            n -= used
        return pd.concat(parts, ignore_index=True)

    def assert_exhausted(self):
        if not self.pending.empty or next(self.iterator, None) is not None:
            raise ValueError("Handoff has more rows than the geometric candidate table")


def export(output=Path("output"), chunksize=50000):
    destination = output / "model_ready"
    destination.mkdir(exist_ok=True)
    filenames = ["candidates_with_features_labels.csv.gz", "train.csv.gz", "student_test.csv.gz"]
    counts = {s: dict(all_rows=0, model_rows=0, positive_rows=0) for s in ["train", "student_test"]}
    written = set()
    with ExitStack() as stack:
        buffers = {}
        for split in counts:
            reader = stack.enter_context(pd.read_csv(output / "handoff" / f"{split}_candidates.csv.gz",
                dtype={"trip_id": str}, chunksize=chunksize))
            buffers[split] = RowBuffer(iter(reader))
        streams = {name: stack.enter_context(gzip.open(destination / (name + ".tmp"), "wt",
                   newline="", compresslevel=1)) for name in filenames}
        reader = stack.enter_context(pd.read_csv(output / "candidates.csv", dtype={"trip_id": str}, chunksize=chunksize))
        for geometry in reader:
            for split, subset in geometry.groupby("split", sort=False):
                subset = subset.reset_index(drop=True)
                handoff = buffers[split].take(len(subset))
                for key in KEYS + ["split"]:
                    if not subset[key].equals(handoff[key]):
                        raise ValueError(f"Candidate/handoff row mismatch: {split}, {key}")
                for column in set(subset.columns) & set(handoff.columns) - set(KEYS + ["split"]):
                    np.testing.assert_allclose(subset[column], handoff[column], equal_nan=True, atol=1e-9)
                combined = pd.concat([subset, handoff.drop(columns=list(set(subset.columns) & set(handoff.columns)))], axis=1)
                for alias, source in ALIASES.items():
                    combined[alias] = combined[source]
                good = combined.training_group_usable == 1
                if not combined.loc[good, "label"].isin([0, 1]).all() or combined.loc[~good, "label"].notna().any():
                    raise ValueError("Unexpected supervised-label eligibility")
                name = filenames[0]
                combined.to_csv(streams[name], index=False, header=name not in written, float_format="%.9f")
                written.add(name)
                usable = combined.loc[good].copy()
                usable["label"] = usable.label.astype(np.int8)
                usable["target"] = usable.target.astype(np.int8)
                name = f"{split}.csv.gz"
                usable.to_csv(streams[name], index=False, header=name not in written, float_format="%.9f")
                written.add(name)
                counts[split]["all_rows"] += len(combined)
                counts[split]["model_rows"] += len(usable)
                counts[split]["positive_rows"] += int(usable.label.sum())
        for buffer in buffers.values():
            buffer.assert_exhausted()
    # Publish only completely closed CSV/gzip files; interrupted runs leave previous exports intact.
    for name in filenames:
        (destination / (name + ".tmp")).replace(destination / name)
    schema = dict(feature_columns=FEATURES, key_columns=KEYS, target_column="label",
        aliases=ALIASES, label_status="approximate offline reference-route alignment",
        fitting_files=["train.csv.gz"], evaluation_files=["student_test.csv.gz"],
        full_file="candidates_with_features_labels.csv.gz",
        full_file_filter="training_group_usable == 1",
        missing_values="Keep NaN; fit imputation only inside each training fold.",
        notes=["offset_m is metres from the directed arc start to the snapped point; it is not arc_fraction.",
               "heading_mismatch_deg ranges from 0 to 180 and is missing when heading is undefined.",
               "candidate_rank is retained for compatibility but excluded from the default feature list.",
               "Use only feature_columns as X. Reference IDs, eligibility flags, target, and label are not features.",
               "All rows in train.csv.gz and student_test.csv.gz have a valid binary label.",
               "In the full file, excluded labels are blank, not zero.",
               "Group cross-validation by trip_id; use the supplied fold_id."])
    (destination / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    (destination / "export_summary.json").write_text(json.dumps(counts, indent=2) + "\n")
    (destination / "README.md").write_text('''# Model-ready candidate tables

Use `train.csv.gz` for model fitting and trip-grouped cross-validation. Use
`student_test.csv.gz` for development evaluation. These are ordinary gzip-compressed CSV files;
pandas reads them directly. No manual feature/label join is needed. Both include geometry,
all existing features, the compatibility columns below, binary `label`, and fold metadata.

| Requested column | Meaning / original column |
| --- | --- |
| `distance_m` | GPS-to-snapped-road distance in metres |
| `offset_m` | Position along directed arc in metres: `distance_from_arc_start_m` |
| `heading_mismatch_deg` | GPS/road heading difference: `heading_difference_deg` |
| `displacement_m` | Previous GPS step distance: `gps_step_m` |
| `elapsed_s` | Seconds since previous point: `dt_s` |
| `arc_length_m` | Length of the candidate arc in metres |
| `label` | 1 if candidate matches the usable approximate reference arc, otherwise 0; alias of `target` |

`candidates_with_features_labels.csv.gz` includes every original candidate, including uncertain
labels and missed-reference groups. Filter `training_group_usable == 1` before fitting on that file.
Unusable labels are blank. Never fill blank labels with zero. The filtered files already apply this rule.

```python
import json
import pandas as pd

folder = "output/model_ready/"
schema = json.load(open(folder + "schema.json"))
train = pd.read_csv(folder + "train.csv.gz", dtype={"trip_id": "string"})
test = pd.read_csv(folder + "student_test.csv.gz", dtype={"trip_id": "string"})
X_train = train[schema["feature_columns"]]
y_train = train["label"].astype(int)
groups = train["trip_id"]
fold_ids = train["fold_id"].astype(int)
X_test = test[schema["feature_columns"]]
y_test = test["label"].astype(int)
```

Only the explicit feature allowlist belongs in X. Do not use all numeric columns: reference arc,
labels, target, eligibility flags, or fold metadata would leak answers or partition information.
First-point time/displacement values and undefined headings remain missing. Fit any imputation
and scaling on each training fold, then transform its validation rows. Do not fit them on test data.

For each eligible `(trip_id, point_index)` group, exactly one candidate is positive. Arc IDs and
point indices are zero-based; arc IDs are local to each trip. `trip_id` must remain a string.
Reference labels are inferred from route alignment and remain approximate. Report label coverage
with evaluation scores. Excluded points remain available in the full table for end-to-end accounting.

`output/candidates.csv` remains the label-free geometry table for inference and the decoder.
Do not use offline reference labels in the inference pipeline.

Regenerate with `.venv/bin/python src/export_model_ready.py` and audit with
`.venv/bin/python src/audit_model_ready.py`.
''')
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    export(args.output)
