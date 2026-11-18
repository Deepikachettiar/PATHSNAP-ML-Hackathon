#!/usr/bin/env python3
"""
Score EVERY candidate row with a saved model (for Member 3's decoder / Member 4's API).

train_models.py only scores label_usable points (needed for evaluation). The decoder needs a
score for every candidate of every GPS point, so use this script for that.

Usage:
  python score_all.py --model results/models/hist_gb.joblib ^
      --candidates data/output/handoff/student_test_candidates.csv.gz ^
      --out results/scores_all_student_test_hist_gb.csv.gz

Output columns: trip_id, point_index, arc_id, raw_score, candidate_probability, model_version
  candidate_probability is normalised within each (trip_id, point_index) with a 1e-6 floor,
  so it sums to 1 per GPS point. Ties are broken by lowest arc_id downstream.
This script only reads the track/network-derived feature columns; it never reads labels.
"""
import argparse
import joblib
import numpy as np
import pandas as pd

PROB_FLOOR = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    model, feats, name = bundle["model"], bundle["features"], bundle["name"]
    df = pd.read_csv(a.candidates, dtype={"trip_id": str})
    p = model.predict_proba(df[feats])[:, 1]
    out = df[["trip_id", "point_index", "arc_id"]].copy()
    out["raw_score"] = p
    s = pd.Series(np.maximum(p, PROB_FLOOR), index=out.index)
    out["candidate_probability"] = (
        s / s.groupby([out["trip_id"], out["point_index"]]).transform("sum")).values
    out["model_version"] = name
    out.to_csv(a.out, index=False)
    sums = out.groupby(["trip_id", "point_index"]).candidate_probability.sum()
    print(f"scored {len(out):,} rows, {len(sums):,} GPS points, model={name}; "
          f"probability sums min/max = {sums.min():.6f}/{sums.max():.6f}")


if __name__ == "__main__":
    main()
