# Member 1: data preparation and road alignment

This directory contains the dataset validation, directed road graphs, GPS-to-road projection,
reference-route alignment, candidate generation, and feature preparation for PathSnap.

## Classifier files

Start with `output/model_ready/README.md` and `output/model_ready/schema.json`.

| File | Contents |
| --- | --- |
| `output/model_ready/train.csv.gz` | 875,231 candidate rows from 134,528 eligible training points |
| `output/model_ready/student_test.csv.gz` | 84,441 candidate rows from 13,213 eligible development-test points |
| `output/model_ready/candidates_with_features_labels.csv.gz` | All 1,490,874 candidates across 227,577 GPS points |

These tables include keys, features, projected coordinates, split/fold metadata, and `label`.
Requested features are named `distance_m`, `offset_m`, `heading_mismatch_deg`, `displacement_m`,
`elapsed_s`, and `arc_length_m`. Original feature names are retained for compatibility.
`offset_m` is distance in metres from the directed arc start to the snapped point.

The fitting tables contain eligible groups with binary labels and one positive candidate per
point. The full table retains excluded groups with blank labels; filter
`training_group_usable == 1` before fitting from that table. Never replace missing labels with zero.

Use only the explicit feature allowlist in `schema.json`. Reference IDs, labels, target,
eligibility flags, and fold IDs do not belong in the model inputs. Keep `trip_id` as a string
and use trip-based validation folds. Fit imputation/scaling inside each training fold only.

## Other outputs

- `output/graphs/*.npz`: all 90 directed road graphs with original node and arc IDs.
- `output/reference_alignment/`: approximate reference labels and per-trip alignment diagnostics.
- `output/handoff/trip_folds.csv`: five fixed folds of 16 whole training trips each.
- `output/figures/`: exploration charts, geographic overview/detail figures, and a Folium screenshot.
- `output/trip_inventory.csv` and `output/validation_report.json`: input and graph diagnostics.
- `output/member_one_report.md`: methodology, measured results, assumptions, and teammate handoff.
- `output/model_ready/audit.json`, `output/member_one_audit.json`, and `output/output_audit.json`: passed full-output checks.

## Installation and reproduction

Run from this `member1/` directory with Python 3.12 or later. Put the supplied
`Student folder (2)/train` and `Student folder (2)/student_test` folders here. Raw prerequisites
are not republished. Evaluator-testing data has not been used.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/build_candidates.py
.venv/bin/python src/audit_outputs.py
.venv/bin/python src/align_reference.py
.venv/bin/python src/prepare_handoff.py
.venv/bin/python src/explore_dataset.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python src/audit_member_one.py
.venv/bin/python src/write_member_one_report.py
.venv/bin/python src/export_model_ready.py
.venv/bin/python src/audit_model_ready.py
```

The pipeline can regenerate intermediate tables and Folium HTML maps that are omitted from
this package to avoid duplicate large exports. The included model-ready tables are the direct
classifier handoff. Folium basemaps normally require an internet connection; static figures work offline.

## Geometry and label limitations

GPS coordinates are treated as WGS84 longitude/latitude and transformed to a per-trip UTM CRS.
Each arc is a straight segment between its supplied nodes. Candidates are the nearest up to
eight directed arcs within 100 metres, with a flagged nearest-road fallback when needed.
Indices are zero-based and local to each trip. Opposite directions remain distinct arcs.

Supplied route rows are chronological road occurrences, not per-GPS labels. Offline sequence
alignment generates approximate point labels with distance, ambiguity, gap, and connectivity
checks. Those labels are not independently verified ground truth. Report retained-point coverage
with downstream evaluation scores and keep excluded points in end-to-end failure accounting.

The inference path uses candidate geometry/features and road graphs. It must not read the offline
reference labels. This stage includes no trained classifier or final HMM prediction claims.

## Validation

Nine unit tests passed. Full audits check candidate geometry, feature/label joins, alignment
order, whole-trip folds, label eligibility, feature aliases, exported row counts, and one positive
per eligible point group. See `output/test_results.txt` and the audit JSON files for retained evidence.
