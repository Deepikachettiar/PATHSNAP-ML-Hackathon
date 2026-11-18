# Member 1: data preparation and road alignment

## Completed scope

The implementation reads and validates the supplied development data, builds directed road graphs,
projects GPS observations onto road arcs, produces label-free candidates and features, aligns GPS
points to supplied reference-route occurrences for offline supervision, and prepares split-separated
tables for model development. Exploration figures and interactive Folium maps accompany the handoff.

| Responsibility | Implementation | Evidence/output |
| --- | --- | --- |
| Read and validate data | `src/build_candidates.py` | `trip_inventory.csv`, `validation_report.json` |
| Build road graphs | `src/build_candidates.py`, `src/road_graph.py` | `graphs/*.npz` |
| Align reference roads | `src/align_reference.py` | `reference_alignment/point_labels.csv` |
| Generate candidates/features | `src/build_candidates.py` | `candidates.csv`, `candidate_features.csv.gz` |
| Prepare model handoff | `src/prepare_handoff.py` | `handoff/*`, fixed trip folds, feature schema |
| Explore and visualise | `src/explore_dataset.py` | `figures/*`, `maps/*`, `feature_missingness.csv` |
| Verify outputs | unit tests and two audit scripts | `output_audit.json`, `member_one_audit.json` |

No classifier, trained-model result, or HMM decoding result is included in this stage. The nearest-road
map layer is a diagnostic baseline. Members 2 and 3 will supply model predictions and the final decoded
route for the final comparison maps and report.

## Dataset and validation findings

There are 80 training trips and 10 student-test trips, comprising **227,577 GPS
observations**. Evaluator-testing data is absent and has not been processed. No trip is split across
partitions or folds. The raw columns are longitude, latitude, timestamp for tracks; longitude, latitude
for nodes; directed source/destination node IDs for arcs; and chronological arc IDs for routes.
All indices are zero-based raw row positions, and IDs are local to a trip.

No missing, nonfinite, out-of-range coordinate, non-integer reference-ID, invalid node-reference, or
non-increasing timestamp rows were found. Input formats, bounds, SHA-256 hashes, and per-trip file sizes
are recorded. No GPS observations were dropped or reordered.

- 7,553 time intervals differ from one second; 1,260 exceed five seconds.
  The largest gap is 153,064 seconds.
- 151 GPS steps imply speeds above 70 m/s. This is a reported diagnostic, not a cleaning threshold.
- Per-trip graph totals include 106 zero-length arcs, 24 self-loops, and
  1,903 duplicate directed endpoint pairs. Arc identities remain intact.
- 89 networks have multiple weak components. None have isolated nodes.
- Reference routes contain 14 adjacent-arc discontinuities. They are retained and flagged.

These network totals are summed over the supplied trip networks, which may overlap geographically.
They are not global counts of unique physical roads.

## Geometry and candidate policy

WGS84 longitude/latitude is transformed to a per-trip UTM CRS selected from the road network's median
location. The output carries its EPSG code and also includes the snapped position in longitude/latitude.
All distances are computed in metres. The source roads provide endpoints only, so each directed arc is
represented by a straight segment; undisclosed road curvature is not invented.

For GPS point p and endpoints a and b, compute f = clip(((p-a) dot (b-a)) / |b-a|², 0, 1),
q = a + f(b-a), and distance = |p-q|. Degenerate arcs project to their endpoint.
An STRtree finds all directed arcs within 100 m; up to the nearest eight are retained.
If none are within the radius, the nearest arc is retained with `outside_radius=1`.

The completed table has **1,490,874 rows**. All GPS points have candidates;
1,413 require an out-of-radius fallback. The eight-candidate cap was applied
at 141,708 points. Search radius and cap are initial configurable choices,
not tuned or optimal values. Opposite directions remain distinct candidates. No route labels enter this calculation.

## Offline reference alignment and uncertainty

Route rows do not map directly to track rows. The aligner treats each route occurrence as a distinct
state, preserving repeated traversals. For each point, it shortlists the 32 closest route occurrences,
then uses dynamic programming over ordered occurrences. The emission cost is (distance / 20 m)².
The transition cost is |change in along-route position - observed GPS displacement| / 50 m.
This encourages geographical agreement and approximately consistent progress.

Within a block, occurrence indices cannot decrease, along-route progress can move backward at most
20 m to tolerate GPS noise, and route discontinuities cannot be crossed. A time gap over 60 s, an
unreachable shortlisted transition, or a point farther than 150 m from all reference roads starts a
new flagged block. When nearby route segments exist, more distant states beyond 150 m are disallowed.
The point after an outlier also begins a new block. These are documented heuristics; they have not
been tuned using student-test outcomes. The shortlist means this is an approximate sequence alignment.

All points receive an alignment record for inspection. A label is eligible for supervised use only if:

1. Its selected reference distance is at most 50 m.
2. The closest competing distinct route arc is at least 3 m farther away.
3. It is not beside an alignment restart and its reference arc has nonzero length.
4. The selected reference arc is actually present in the independently generated candidate group.

Repeated occurrences of the same arc are distinct alignment states but share the same per-point arc
class. The competing-arc margin is deliberately conservative: intersections, overlapping geometry,
and routes traversing both directions can lose many eligible labels. The exclusions are recorded;
no inferred label is claimed to be independently verified physical ground truth.

## Measured handoff coverage

| Split | GPS points | Candidate rows | Usable reference labels | Usable candidate groups | Miss rate, all inferred labels | Miss rate, usable inferred labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 208,417 | 1,359,951 | 135,090 | 134,528 | 11.546% | 0.416% |
| student_test | 19,160 | 130,923 | 13,220 | 13,213 | 0.465% | 0.053% |

These miss rates measure absence of the **inferred reference arc** from the candidate set. They are
not classifier accuracy and are not an independent ground-truth assessment. The conservative subset
excludes difficult points and must not be reported as performance on all observations. Subsequent
model reports should state both coverage and conditional scores; keep missing candidate groups in
end-to-end failure accounting rather than silently excluding them from the denominator.

## Features and missing values

| Features | Why they are included |
| --- | --- |
| GPS-to-arc distance, candidate rank, fallback flag | Geographical fit and candidate-set confidence |
| Arc length and fraction, distances to endpoints | Position along the directed road; useful for transitions |
| Directed bearing, previous GPS heading, heading difference | Directional compatibility, including reverse arcs |
| Previous time interval, displacement, speed | Sampling irregularity and observed movement |
| Source out-degree, destination in-degree | Local road connectivity context |

Angles are degrees clockwise from grid north. Headings are missing at the first point or for movement
under 1 m. Previous-step features are missing at trip starts. Missingness is tabulated in
`feature_missingness.csv`. No rows are dropped to conceal missing values; no imputer or scaler is
fitted here. Members 2 and 3 should fit learned preprocessing only on the training portion of each fold.

## Member 2 handoff

Use `handoff/train_candidates.csv.gz` for fitting and cross-validation and
`handoff/student_test_candidates.csv.gz` for development evaluation. Each row has the candidate key,
features, reference label, and eligibility flags. **Filter `training_group_usable == 1` before fitting**;
excluded targets are blank, not negative. Every eligible point group has exactly one positive target.
Predict candidate probabilities, then select the highest-scoring candidate per GPS point.

Use only columns listed in `handoff/feature_schema.json` as model inputs. Identifiers, reference arc,
target, split, folds, and label eligibility flags are not input features. Never train directly on arc ID
as a globally meaningful numeric class. The binary target indicates candidate correctness for its point.
`trip_folds.csv` gives five deterministic folds of 16 training trips, seed 42; student-test trips have no fold.

```python
import json
import pandas as pd
schema = json.load(open("output/handoff/feature_schema.json"))
table = pd.read_csv("output/handoff/train_candidates.csv.gz", dtype={"trip_id": "string"})
table = table[table.training_group_usable == 1].copy()
X = table[schema["feature_columns"]]
y = table.target.astype(int)
groups = table.trip_id
fold_ids = table.fold_id.astype(int)
# Fit preprocessing on the training rows of each fold, then transform its validation rows.
```

## Member 3 handoff

Use label-free `candidates.csv`, `candidate_features.csv.gz`, and `graphs/<trip_id>.npz` for inference.
The `src/road_graph.py` loader provides directed node distances and candidate-to-candidate distances,
including endpoint remainders and same-arc movement. Unreachable paths return infinity. Parallel arc
IDs are preserved, while shortest-path adjacency uses the minimum weight for duplicate node pairs
instead of summing duplicate sparse entries. Explicit zero-weight connections are supported.

```python
from src.road_graph import RoadGraph
g = RoadGraph.load("output/graphs/00000001.npz")
distance_m = g.candidate_distance_m(from_arc=0, from_fraction=.5, to_arc=1, to_fraction=.2)
```

Treat fallbacks, large time gaps, weak components, and zero-length roads explicitly in the decoder.
Offline alignment files must not be read by the inference path. They exist only for supervision and
permitted development evaluation.

## Figures and maps

- `figures/dataset_exploration.png`: measured sampling, trajectory size, distance, connectivity, and label coverage.
- `figures/*_roads.png`: geographic views of training trips 00000001 and 00000054 and student-test trip 00000000.
- `maps/*.html`: Folium maps with road, raw GPS, supplied reference route, nearest candidate, and offline-label layers.
- `figures/folium_train_00000001.jpg`: inspected Folium screenshot with candidate layer enabled.

Trip 00000054 is included as a difficult example because it contains many GPS points far from the supplied
reference geometry. The maps retain those disagreements. Folium uses external Leaflet assets and map
tiles, so an internet connection is normally needed to view the basemap. Static road figures work offline.

## Verification and reproduction

**9 unit tests passed.** They cover projection/clamping, reverse and degenerate arcs,
index retrieval against exhaustive geometry on 20 real points, ordered alignment, resets and reference
discontinuities, directed shortest paths, duplicate edges, zero weights, and backward candidate moves.
The candidate audit checks every exported geometry/key. The Member 1 audit checks every alignment row,
fold/split separation, every handoff target, feature/label joins, candidate-hit flags, and exactly one
positive in each eligible group. Both retained audits report `passed`.

```bash
.venv/bin/python src/build_candidates.py
.venv/bin/python src/audit_outputs.py
.venv/bin/python src/align_reference.py
.venv/bin/python src/prepare_handoff.py
.venv/bin/python src/explore_dataset.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python src/audit_member_one.py
.venv/bin/python src/write_member_one_report.py
```

The Folium screenshot is a separate visual QA artifact; regenerating HTML/plots does not recapture a
browser screenshot. The final classifier and decoded-route comparison are later team integration work.
