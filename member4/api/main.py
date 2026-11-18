"""
FastAPI backend (Laptop 1).
Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from .config import REPO_ROOT, load_config
from .data_access import list_trip_ids
from .evaluator import score_trip
from .pipeline import Predictor

(REPO_ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap")
_audit = REPO_ROOT / "logs" / "predict_audit.jsonl"

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()                       # fail fast on bad config
    STATE["cfg"] = cfg
    STATE["predictor"] = Predictor(cfg)       # model/assets loaded ONCE here
    log.info("mode=%s root=%s engine=%s model=%s trips=%d", cfg["mode"], cfg["data_root"],
             STATE["predictor"].engine.name, cfg["model_version"], len(list_trip_ids(cfg["data_root"])))
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)


class PredictRequest(BaseModel):
    trip_id: str                              # string always: leading zeros matter


class ReplayRequest(BaseModel):
    trip_id: str
    anchor_point_index: int | None = None


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    log.info("REQ %s %s from %s -> %s (%.0f ms)", request.method, request.url.path,
             request.client.host if request.client else "?", resp.status_code,
             (time.perf_counter() - t0) * 1000)
    return resp


@app.get("/health")
def health():
    cfg = STATE["cfg"]
    return {"status": "ok", "mode": cfg["mode"], "model_version": cfg["model_version"],
            "engine": STATE["predictor"].engine.name}


@app.get("/trips")
def trips():
    ids = list_trip_ids(STATE["cfg"]["data_root"])   # read from disk, never guessed
    if not ids:
        raise HTTPException(503, f"No trips found in {STATE['cfg']['data_root']} (mode={STATE['cfg']['mode']}). "
                                 "Check data_roots in configs/config.yaml.")
    return {"mode": STATE["cfg"]["mode"], "trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    cfg, predictor = STATE["cfg"], STATE["predictor"]
    root = cfg["data_root"]
    valid = list_trip_ids(root)
    if req.trip_id not in valid:                      # whitelist: blocks arbitrary paths
        raise HTTPException(404, "Unknown trip_id. Call GET /trips for valid IDs.")
    trip_id = valid[valid.index(req.trip_id)]         # server-side value, not client input
    status, result = "ok", None
    t0 = time.perf_counter()
    try:
        result = predictor.predict(root, trip_id)           # no reference data touched
        result["metrics"] = score_trip(root, trip_id, result)  # after predictions are final
        result["runtime_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result
    except HTTPException:
        raise
    except Exception as e:
        status = f"error: {type(e).__name__}"
        log.exception("predict failed")
        raise HTTPException(500, f"Prediction failed for trip {trip_id}.")
    finally:
        m = (result or {}).get("metrics") or {}
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "trip_id": trip_id,
               "model_version": cfg["model_version"], "mode": cfg["mode"], "status": status,
               "ms": round((time.perf_counter() - t0) * 1000, 1),
               "points": (result or {}).get("point_count"), "eligible": m.get("eligible_points")}
        with open(_audit, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        log.info("PREDICT %s", rec)


@app.post("/replay")
def replay(req: ReplayRequest):
    """Return the live decoder's prefix replay for the selected trip."""
    valid = list_trip_ids(STATE["cfg"]["data_root"])
    if req.trip_id not in valid:
        raise HTTPException(404, f"Unknown trip_id '{req.trip_id}'.")
    result = STATE["predictor"].predict(STATE["cfg"]["data_root"], req.trip_id)
    steps = result.get("replay", [])
    if req.anchor_point_index is not None:
        steps = [step for step in steps if step["anchor"] == req.anchor_point_index]
    return {"trip_id": req.trip_id, "steps": steps, "roads": result.get("replay_roads", {}),
            "changed": len({step["anchor_arc"] for step in steps}) > 1}
