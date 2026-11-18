# PathSnap — Member 3: Route Decoder

## Role / contribution
Member 3 is responsible for route decoding and sequence reconstruction. The decoder takes candidate road segments and the probabilities/scores produced by Member 2's machine-learning model, then selects a road sequence while considering directed road-network connectivity and the vehicle's movement.

## Recommended ML scorer
**Extra Trees (Member 2)** is the recommended scorer for this package because it ranked highest in Member 2's reported 5-fold grouped-CV Top-1 accuracy (84.25%). It is the scorer whose candidate probabilities are bundled with this Member 3 package.

This is a model-selection statement based on Member 2's cross-validation comparison, not a claim that Extra Trees is universally best or that it produces the best decoded sequence on every trip.

## Decoder approach
- Candidate roads are represented by directed arcs in the road graph.
- Candidate probabilities from Member 2 are read as log scores.
- The decoder considers transitions from a previous candidate to current candidates.
- It searches directed paths through the road graph, subject to a movement-distance bound based on elapsed time.
- Transition scores consider how well road-network distance matches observed GPS movement.
- Dynamic scoring and backtracking select a sequence across observations (Viterbi-style sequence decoding).
- If candidates/scores are missing, a transition is impossible, or a time gap exceeds the supported interval, the chain can restart. These restarts are reported; they are not silently treated as connected.

## How to run (Windows)
1. Extract the ZIP fully.
2. Open the extracted `PathSnap_Final_Only` folder.
3. Open Command Prompt in that folder (Explorer address bar: type `cmd`, press Enter).
4. Run:
   ```bat
   py -3 final.py
   ```
   or:
   ```bat
   python final.py
   ```
5. Open `view.html` in a browser and select a trip JSON under `outputs/final/`.

The normal `final.py` run uses Python's standard library and does not need internet access or pip installation. `prepare.py` is optional regeneration tooling and requires pandas, NumPy, joblib and scikit-learn; it is not needed for the bundled final run.

## Output files
- `outputs/final/<trip_id>.json`: decoded points, baseline candidate, selected arc, status, lines and replay information.
- `outputs/final/summary.csv`: point-level baseline-vs-decoded accuracy and connection counts by trip.
- `outputs/final/sequence_metrics.csv`: sequence edit distance against each supplied reference route.
- `view.html`: interactive replay viewer.

## Bundled evaluation results (10 student-test trips)
All 19,160 observations across the 10 bundled student-test trips were counted using derived approximate point labels.

| Metric | Baseline (highest candidate score) | Decoded route |
|---|---:|---:|
| Point accuracy | 80.59% (15,441/19,160) | 78.67% (15,074/19,160) |
| Sequence edit distance (sum across trips) | 4,423 | 2,805 |
| Edit distance / reference entries (weighted) | 1.272 | 0.807 |
| Matched points | — | 19,160 / 19,160 |
| Changed point selections | — | 3,628 |
| Connected steps | — | 19,064 / 19,150 = 99.55% |

### Interpretation
The decoder improved sequence-level similarity: total sequence edit distance decreased from 4,423 to 2,805 (about 36.6% fewer edits), and the weighted edit-distance-per-reference-entry decreased from 1.272 to 0.807. However, point-level accuracy decreased from 80.59% to 78.67%. This trade-off must be reported honestly: the decoder made routes more sequence-consistent, but did not improve pointwise accuracy on these derived labels.

## Important evaluation limitations
- The reference point labels are derived by a monotone alignment of GPS observations to each ordered `.route` using projection distance. They are approximate labels, not an official independently verified point-level ground truth.
- The bundled scores are for the ten student-test trips. Do not present them as evaluator-test results.
- Member 2's original Extra Trees probabilities are supplied for 13,189 observations. For the remaining 5,971, `prepare.py` reconstructed features using a local metre approximation and applied the saved forest; these reconstructed features may differ from Member 1's original feature implementation.
- All 19,160 points are included in the reported point accuracy.
- Decoder chain restarts can occur for missing candidates/scores, impossible transitions or time gaps. The connection metric is not 100%; the observed connected-step ratio is 99.55%.

## Aggregate result reproduced
Running `python final.py` in the extracted package completed all ten trips and printed:
- Baseline accuracy: 15,441 / 19,160 = 80.59%
- Decoded arc accuracy: 15,074 / 19,160 = 78.67%

## Short viva explanation
“My role was the route-decoding component. Member 2's Extra Trees model gives probability scores to candidate roads. My decoder uses those scores together with the directed road network and movement between GPS observations to select a connected, sequence-aware route using dynamic scoring and backtracking. On the ten student-test trips, it reduced sequence edit distance by about 36.6% and achieved 99.55% connected steps, although point accuracy decreased from 80.59% to 78.67%. These labels are approximate route-alignment labels, so we report the results with that limitation.”
