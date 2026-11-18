"""
No server needed. Runs the prediction pipeline directly on every trip in the configured partition AND
proves prediction works with the reference (.route) files inaccessible (guide p.9).
Usage:  python -m scripts.offline_check
"""
import json, shutil, tempfile, os
from pathlib import Path
from api.config import load_config
from api.data_access import list_trip_ids
from api.pipeline import Predictor
from api.evaluator import score_trip

FORBIDDEN = {"reference", "route", "label", "labels", "reference_arc_id", "correct", "is_correct"}


def main():
    cfg = load_config()
    root = cfg["data_root"]
    ids = list_trip_ids(root)
    print(f"mode={cfg['mode']} root={root} trips={len(ids)}")
    pred = Predictor(cfg)
    print("engine:", pred.engine.name)
    # copy WITHOUT .route files
    tmp = Path(tempfile.mkdtemp())
    for t in ids:
        (tmp / t).mkdir()
        for e in ("track", "nodes", "arcs"):
            shutil.copy(Path(root) / t / f"{t}.{e}", tmp / t / f"{t}.{e}")
    pred_noref = Predictor(cfg)
    for t in ids:
        a = pred.predict(root, t)
        b = pred_noref.predict(str(tmp), t)
        same = a["points"] == b["points"]
        assert all(len(p) and p["status"] for p in a["points"]), "missing status"
        assert not (FORBIDDEN & set(a.keys())) and not any(FORBIDDEN & set(p) for p in a["points"]), "label leak"
        m = score_trip(root, t, a)
        print(f"{t}: points={a['point_count']} segs={len(a['route_segments'])} breaks={len(a['breaks'])} "
              f"ms={a['runtime_ms']} identical_without_route={same} metrics={m.get('route_arc_hit_rate')}")
        assert same
    shutil.rmtree(tmp)
    print("OK")

if __name__ == "__main__":
    main()
