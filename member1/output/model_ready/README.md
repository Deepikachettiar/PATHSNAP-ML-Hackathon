# Model-ready candidate tables

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
