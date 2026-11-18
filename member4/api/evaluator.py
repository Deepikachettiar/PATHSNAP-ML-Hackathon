"""
Server-side scorer. The ONLY module in the serving path that reads .route files.
Runs after predictions are final and returns AGGREGATES ONLY (no per-point correctness, no labels).

SCAFFOLD CONVENTION (replace with the organisers' convention if one is supplied at the event):
  eligible_points      = points with status "matched"
  route_arc_hit_rate   = share of eligible points whose predicted arc appears anywhere in the reference route
  This is NOT Top-1 accuracy against an aligned reference. Document the real denominator once agreed.
"""
import numpy as np
from .data_access import load_reference


def score_trip(root: str, trip_id: str, prediction: dict) -> dict:
    try:
        ref = load_reference(root, trip_id)
    except FileNotFoundError:
        return {"reference_available": False, "convention": "route_arc_membership"}
    ref_set = set(int(a) for a in ref)
    pts = prediction["points"]
    eligible = [p for p in pts if p["status"] == "matched"]
    hits = sum(1 for p in eligible if p["arc_id"] in ref_set)
    n = len(pts)
    return {
        "reference_available": True,
        "convention": "route_arc_membership: predicted arc appears in supplied route",
        "points_total": n,
        "eligible_points": len(eligible),
        "unmatched_points": n - len(eligible),
        "route_arc_hit_rate": round(hits / len(eligible), 4) if eligible else None,
        "decoder_breaks": len(prediction["breaks"]),
        "reference_route_arcs": len(ref),
    }
