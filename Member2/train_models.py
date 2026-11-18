#!/usr/bin/env python3
"""
PathSnap - Member 2: five-model comparison on Member 1's candidate tables.

What this script does
  1. Loads Member 1's handoff files (train / student_test candidates, fold ids, schema).
  2. For each of five models runs 5-fold CV GROUPED BY TRIP (folds come from Member 1's
     trip_folds / fold_id column, seed 42). Preprocessing (imputer / scaler) lives inside
     each pipeline, so it is fitted on the training folds only.
  3. Refits each model on all 80 train trips and evaluates on the 10 student_test trips.
  4. Writes comparison tables, per-fold results, out-of-fold scores, student-test scores,
     the MLP loss curve (real history only) and a run_config.json for reproducibility.

Denominators (state these in the report)
  * Fitting rows      : rows with training_group_usable == 1 (reference arc is among candidates).
  * Candidate metrics : macro F1 / precision / recall / log loss, computed on candidate rows of
                        training_group_usable == 1 groups. Positive class = "this arc is the
                        reference arc"; negative class = every other candidate. Threshold 0.5.
  * Top-1 accuracy    : computed over ALL label_usable == 1 points. If the reference arc is not
                        among the candidates the point counts as a MISS (wrong).
  * Labels are APPROXIMATE (route alignment by Member 1), not verified ground truth.

Run:
  python train_models.py --data-dir path/to/output/handoff --out-dir results
Quick debug run (uses a random 5% of point groups, keeps groups intact):
  python train_models.py --data-dir path/to/output/handoff --out-dir results_debug --subsample 0.05
"""
import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

KEYS = ["trip_id", "point_index"]
PROB_FLOOR = 1e-6          # floor used before normalising scores within each GPS point
THRESHOLD = 0.5            # for candidate-level precision / recall / F1

# Write your selection rule BEFORE looking at results (edit if your team decides otherwise).
SELECTION_RULE = ("Rank by grouped-CV observation Top-1 (higher is better); break ties with "
                  "grouped-CV macro F1, then with lower student-test inference time.")


# --------------------------------------------------------------------------- models
def build_models(seed):
    """name -> (family, estimator, key_config). Edit freely; keep the same features/folds for all."""
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "logreg": ("Linear", make_pipeline(
            imp(), StandardScaler(),
            LogisticRegression(C=1.0, max_iter=300, random_state=seed)),
            dict(C=1.0, penalty="l2", scaler="standard", max_iter=300)),
        "gaussian_nb": ("Probabilistic", make_pipeline(
            imp(), GaussianNB(var_smoothing=1e-9)),
            dict(var_smoothing=1e-9)),
        "extra_trees": ("Tree ensemble", make_pipeline(
            imp(), ExtraTreesClassifier(n_estimators=100, max_depth=16, min_samples_leaf=20,
                                        n_jobs=-1, random_state=seed)),
            dict(n_estimators=100, max_depth=16, min_samples_leaf=20)),
        "hist_gb": ("Boosting", HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=200, max_leaf_nodes=31, random_state=seed),
            dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31)),
        "mlp": ("Neural network", make_pipeline(
            imp(), StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), solver="adam", max_iter=40,
                          early_stopping=False, random_state=seed)),
            dict(hidden=(64, 32), solver="adam", max_iter=40, scaler="standard")),
    }


# --------------------------------------------------------------------------- metrics
def candidate_metrics(y, p):
    y = np.asarray(y).astype(int)
    pred = (p >= THRESHOLD).astype(int)
    return {
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "precision_pos": precision_score(y, pred, zero_division=0),
        "recall_pos": recall_score(y, pred, zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),
    }


def top1(df, p):
    """Observation-level Top-1 over every label_usable point. Ties -> lowest arc_id."""
    d = df[KEYS + ["arc_id", "reference_arc_id"]].copy()
    d["p"] = p
    d = d.sort_values(KEYS + ["p", "arc_id"], ascending=[True, True, False, True])
    first = d.drop_duplicates(KEYS, keep="first")
    return float((first["arc_id"] == first["reference_arc_id"]).mean()), len(first)


def normalised(df, p):
    s = pd.Series(np.maximum(p, PROB_FLOOR), index=df.index)
    return (s / s.groupby([df["trip_id"], df["point_index"]]).transform("sum")).values


def score_frame(df, raw, fold=None):
    out = df[KEYS + ["arc_id"]].copy()
    if fold is not None:
        out.insert(0, "fold", fold)
    out["raw_score"] = raw
    out["candidate_probability"] = normalised(df, raw)
    return out


def evaluate(df_eval, p):
    """df_eval = rows of label_usable == 1 points; p = raw positive-class probabilities."""
    t1, n_pts = top1(df_eval, p)
    m = df_eval["training_group_usable"].values == 1
    res = candidate_metrics(df_eval.loc[m, "target"], p[m])
    res.update(top1=t1, eligible_points=n_pts, candidate_rows=int(m.sum()))
    return res


# --------------------------------------------------------------------------- data
def load(data_dir, subsample, seed):
    d = Path(data_dir)
    schema = json.load(open(d / "feature_schema.json"))
    feats = schema["feature_columns"]
    tr = pd.read_csv(d / "train_candidates.csv.gz", dtype={"trip_id": str})
    st = pd.read_csv(d / "student_test_candidates.csv.gz", dtype={"trip_id": str})
    tr["fold_id"] = tr["fold_id"].astype(int)
    if subsample < 1.0:  # debugging only: keep whole point groups
        rng = np.random.default_rng(seed)
        for name in ("tr", "st"):
            df = tr if name == "tr" else st
            gid = df.groupby(KEYS).ngroup()
            keep = rng.random(gid.max() + 1) < subsample
            df = df[keep[gid.values]].reset_index(drop=True)
            if name == "tr":
                tr = df
            else:
                st = df
    return tr, st, feats, schema


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Member 1's output/handoff folder")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subsample", type=float, default=1.0, help="DEBUG ONLY: fraction of point groups")
    ap.add_argument("--models", nargs="*", default=None, help="subset, e.g. logreg hist_gb")
    ap.add_argument("--no-save-models", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    tr, st, feats, schema = load(args.data_dir, args.subsample, args.seed)
    print(f"train rows {len(tr):,}  student_test rows {len(st):,}  features {len(feats)}")

    models = build_models(args.seed)
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}

    rng = np.random.default_rng(args.seed)
    fold_rows, summary, mlp_curve = [], [], None
    n_st_trips = st["trip_id"].nunique()

    for name, (family, est_proto, cfg) in models.items():
        print(f"\n=== {name} ({family}) ===")
        oof_parts, fold_metrics = [], []
        for f in sorted(tr["fold_id"].unique()):
            train = tr[(tr.fold_id != f) & (tr.training_group_usable == 1)]
            val = tr[(tr.fold_id == f) & (tr.label_usable == 1)]
            est = sklearn.clone(est_proto)
            t0 = time.perf_counter()
            est.fit(train[feats], train["target"])
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            p = est.predict_proba(val[feats])[:, 1]
            pred_s = time.perf_counter() - t0
            res = evaluate(val, p)
            # overfitting evidence: macro F1 on (a sample of) the training rows themselves
            samp = train.sample(min(100_000, len(train)), random_state=args.seed)
            tr_f1 = candidate_metrics(samp["target"], est.predict_proba(samp[feats])[:, 1])["macro_f1"]
            res.update(model=name, fold=int(f), train_macro_f1=tr_f1, fit_time_s=fit_s,
                       predict_time_s=pred_s, val_trips=val["trip_id"].nunique())
            fold_metrics.append(res)
            oof_parts.append(score_frame(val, p, fold=int(f)))
            print(f"  fold {f}: top1={res['top1']:.4f} macroF1={res['macro_f1']:.4f} "
                  f"logloss={res['log_loss']:.4f} trainF1={tr_f1:.4f} fit={fit_s:.1f}s")
        fm = pd.DataFrame(fold_metrics)
        fold_rows.append(fm)
        pd.concat(oof_parts).to_csv(out / f"oof_scores_{name}.csv.gz", index=False)

        # ---- refit on all 80 train trips, evaluate on 10 student_test trips
        full = tr[tr.training_group_usable == 1]
        est = sklearn.clone(est_proto)
        t0 = time.perf_counter()
        est.fit(full[feats], full["target"])
        full_fit_s = time.perf_counter() - t0
        st_eval = st[st.label_usable == 1]
        t0 = time.perf_counter()
        p_st = est.predict_proba(st_eval[feats])[:, 1]
        inf_s = time.perf_counter() - t0
        r = evaluate(st_eval, p_st)
        score_frame(st_eval, p_st).assign(model_version=name).to_csv(
            out / f"student_scores_{name}.csv.gz", index=False)
        if not args.no_save_models:
            joblib.dump({"model": est, "features": feats, "name": name, "config": cfg,
                         "seed": args.seed}, out / "models" / f"{name}.joblib")
        if name == "mlp":
            clf = est[-1] if hasattr(est, "steps") else est
            mlp_curve = pd.DataFrame({"epoch": np.arange(1, len(clf.loss_curve_) + 1),
                                      "train_loss": clf.loss_curve_})
        summary.append({
            "model": name, "family": family, "key_config": json.dumps(cfg),
            "cv_top1_mean": fm.top1.mean(), "cv_top1_std": fm.top1.std(),
            "cv_macro_f1_mean": fm.macro_f1.mean(), "cv_macro_f1_std": fm.macro_f1.std(),
            "cv_log_loss_mean": fm.log_loss.mean(),
            "cv_train_macro_f1_mean": fm.train_macro_f1.mean(),
            "cv_fit_time_s_mean": fm.fit_time_s.mean(),
            "full_fit_time_s": full_fit_s,
            "student_top1": r["top1"], "student_macro_f1": r["macro_f1"],
            "student_precision_pos": r["precision_pos"], "student_recall_pos": r["recall_pos"],
            "student_log_loss": r["log_loss"],
            "student_infer_s_total": inf_s, "student_infer_s_per_trip": inf_s / n_st_trips,
            "student_eligible_points": r["eligible_points"],
            "route_continuity": "N/A here - computed by Member 3 after Viterbi",
        })
        print(f"  student_test: top1={r['top1']:.4f} macroF1={r['macro_f1']:.4f} "
              f"logloss={r['log_loss']:.4f} infer/trip={inf_s / n_st_trips:.3f}s")

    comp = pd.DataFrame(summary)
    comp.to_csv(out / "comparison.csv", index=False)
    pd.concat(fold_rows).to_csv(out / "fold_results.csv", index=False)
    if mlp_curve is not None:
        mlp_curve.to_csv(out / "mlp_loss_curve.csv", index=False)

    rank = comp.sort_values(["cv_top1_mean", "cv_macro_f1_mean", "student_infer_s_per_trip"],
                            ascending=[False, False, True])
    print("\nRanking under the written selection rule:")
    print(rank[["model", "cv_top1_mean", "cv_macro_f1_mean", "student_top1"]].to_string(index=False))

    json.dump({
        "seed": args.seed, "subsample": args.subsample, "selection_rule": SELECTION_RULE,
        "features": feats, "fit_filter": schema.get("filter_before_fitting"),
        "label_status": schema.get("label_status"),
        "threshold": THRESHOLD, "prob_floor": PROB_FLOOR,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
        "models": {k: dict(family=v[0], config=v[2]) for k, v in models.items()},
    }, open(out / "run_config.json", "w"), indent=2)
    print(f"\nSaved results to {out.resolve()}")


if __name__ == "__main__":
    main()
