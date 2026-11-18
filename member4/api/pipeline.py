"""
Prediction pipeline used by POST /predict.

Two engines:
  * RealEngine     - chains the teammates' modules (page 10 contract):
                       data_prep.candidates.generate -> models.scorer.score -> decoder.viterbi.decode
                     ADAPT the three call sites below to their actual signatures.
  * ScaffoldEngine - nearest-road snapping (distance only). Lets you build and test the API/UI before
                     the real model exists. Responses carry engine="scaffold" and the UI shows a warning.
                     Do NOT report scaffold numbers as model results.

This module never reads .route files (labels). See api/evaluator.py for that.
"""
from __future__ import annotations
import importlib
import time
from collections import OrderedDict

import numpy as np
from scipy.spatial import cKDTree

from .data_access import LocalProjection, load_inputs


# --------------------------------------------------------------------------- helpers
def _point_seg(px, py, ax, ay, bx, by):
    """Distance from points to segments (vectorised) and the closest point + fraction t."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), cx, cy, t, np.sqrt(L2)


class TripGraph:
    """Per-trip road graph in metres. Arc ID = row index in .arcs, node ID = row index in .nodes."""

    def __init__(self, inputs: dict):
        nodes, arcs = inputs["nodes"], inputs["arcs"]
        bad = (arcs < 0) | (arcs >= len(nodes))
        self.valid_arc = ~bad.any(axis=1)
        arcs_safe = np.where(bad, 0, arcs)
        self.proj = LocalProjection(float(nodes[:, 0].mean()), float(nodes[:, 1].mean()))
        nx_, ny_ = self.proj.fwd(nodes[:, 0], nodes[:, 1])
        self.nx, self.ny = nx_, ny_
        self.nodes_ll = nodes
        self.arcs = arcs_safe
        self.ax, self.ay = nx_[arcs_safe[:, 0]], ny_[arcs_safe[:, 0]]
        self.bx, self.by = nx_[arcs_safe[:, 1]], ny_[arcs_safe[:, 1]]
        mid = np.column_stack([(self.ax + self.bx) / 2, (self.ay + self.by) / 2])
        mid[~self.valid_arc] = 1e12  # push invalid arcs out of every query
        self.tree = cKDTree(mid)

    def arc_lonlat(self, arc_ids):
        a = self.arcs[arc_ids]
        return self.nodes_ll[a[:, 0]], self.nodes_ll[a[:, 1]]


# --------------------------------------------------------------------------- engines
class ScaffoldEngine:
    name = "scaffold"

    def __init__(self, cfg: dict):
        self.c = cfg.get("scaffold", {})

    def run(self, inputs: dict, graph: TripGraph) -> dict:
        track = inputs["track"]
        lon, lat, sec = track[:, 0], track[:, 1], track[:, 2]
        n = len(track)
        finite = np.isfinite(track).all(axis=1) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
        px, py = graph.proj.fwd(np.where(finite, lon, 0), np.where(finite, lat, 0))
        k = int(self.c.get("knn_arcs", 12))
        radius = float(self.c.get("radius_m", 60.0))
        k = min(k, len(graph.arcs))
        _, idx = graph.tree.query(np.column_stack([px, py]), k=k)
        idx = idx.reshape(n, -1)
        d, cx, cy, t, L = _point_seg(px[:, None], py[:, None], graph.ax[idx], graph.ay[idx],
                                     graph.bx[idx], graph.by[idx])
        best = np.argmin(d, axis=1)           # ties -> lowest column; idx sorted by midpoint distance
        r = np.arange(n)
        arc = idx[r, best]
        dist = d[r, best]
        mx, my = cx[r, best], cy[r, best]
        offset = t[r, best] * L[r, best]
        matched = finite & (dist <= radius)
        mlon, mlat = graph.proj.inv(mx, my)

        points, breaks, segments, cur = [], [], [], []
        prev = None
        for i in range(n):
            if not finite[i]:
                status, reason = "invalid", "invalid_gps_row"
            elif not matched[i]:
                status, reason = "unmatched", "no_candidate_within_radius"
            else:
                status, reason = "matched", None
            points.append({
                "point_index": i, "lon": _f(lon[i]), "lat": _f(lat[i]), "timestamp_s": _f(sec[i]),
                "status": status,
                "arc_id": int(arc[i]) if status == "matched" else None,
                "matched_lon": _f(mlon[i]) if status == "matched" else None,
                "matched_lat": _f(mlat[i]) if status == "matched" else None,
                "offset_m": _f(offset[i]) if status == "matched" else None,
                "distance_m": _f(dist[i]) if status == "matched" else None,
            })
            if status != "matched":
                if cur:
                    segments.append(cur); cur = []
                breaks.append({"point_index": i, "reason": reason})
                prev = None
                continue
            if prev is not None:
                gap_m = float(np.hypot(mx[i] - mx[prev], my[i] - my[prev]))
                gap_s = float(sec[i] - sec[prev])
                if gap_m > self.c.get("break_gap_m", 400.0) or gap_s > self.c.get("break_gap_s", 120.0):
                    segments.append(cur); cur = []
                    breaks.append({"point_index": i, "reason": "gap_too_large"})
            cur.append([_f(mlon[i]), _f(mlat[i])])   # GeoJSON order: lon, lat
            prev = i
        if cur:
            segments.append(cur)
        segments = [s for s in segments if len(s) >= 2]
        return {"points": points, "breaks": breaks, "route_segments": segments}


class RealEngine:
    name = "real"

    def __init__(self, cfg: dict):
        # Import lazily so a missing teammate module gives a clear message instead of crashing the server.
        self.generate = importlib.import_module("data_prep.candidates").generate
        self.score = importlib.import_module("models.scorer").score
        self.decode = importlib.import_module("decoder.viterbi").decode

    def run(self, inputs: dict, graph: TripGraph) -> dict:
        # TODO(adapt): match the actual signatures agreed on page 10 of the guide.
        cands = self.generate(inputs["track"], graph)
        scores = self.score(cands)
        out = self.decode(inputs["track"], cands, graph, scores)
        # Expected keys: points (list of dicts like the scaffold), breaks, route_segments (lon,lat lists)
        return out


def _f(x):
    x = float(x)
    return x if np.isfinite(x) else None


# --------------------------------------------------------------------------- predictor
class Predictor:
    def __init__(self, cfg: dict, cache_size: int = 8):
        self.cfg = cfg
        self.model_version = cfg.get("model_version", "unknown")
        self.engine = self._make_engine(cfg)
        self._graphs: "OrderedDict[str, tuple]" = OrderedDict()
        self._cache_size = cache_size

    @staticmethod
    def _make_engine(cfg):
        want = cfg.get("engine", "auto")
        if want in ("auto", "real"):
            try:
                return RealEngine(cfg)
            except Exception as e:  # ImportError or anything in teammate code
                if want == "real":
                    raise RuntimeError(f"Real engine requested but unavailable: {e}") from e
        return ScaffoldEngine(cfg)

    def _graph(self, root, trip_id):
        key = trip_id
        if key in self._graphs:
            self._graphs.move_to_end(key)
            return self._graphs[key]
        inputs = load_inputs(root, trip_id)
        item = (inputs, TripGraph(inputs))
        self._graphs[key] = item
        if len(self._graphs) > self._cache_size:
            self._graphs.popitem(last=False)
        return item

    def road_context(self, inputs, graph) -> list:
        rc = self.cfg.get("road_context", {})
        track = inputs["track"]
        ok = np.isfinite(track[:, :2]).all(axis=1)
        gx, gy = graph.proj.fwd(track[ok, 0], track[ok, 1])
        near = cKDTree(np.column_stack([gx, gy]))
        mids = graph.tree.data
        dist, _ = near.query(mids, distance_upper_bound=float(rc.get("radius_m", 120.0)))
        sel = np.flatnonzero(np.isfinite(dist) & graph.valid_arc)[: int(rc.get("max_arcs", 6000))]
        a, b = graph.arc_lonlat(sel)
        return [{"arc_id": int(i), "coords": [[float(p[0]), float(p[1])], [float(q[0]), float(q[1])]]}
                for i, p, q in zip(sel, a, b)]

    def predict(self, root: str, trip_id: str) -> dict:
        t0 = time.perf_counter()
        inputs, graph = self._graph(root, trip_id)
        out = self.engine.run(inputs, graph)
        pts = out["points"]
        return {
            "trip_id": trip_id,
            "model_version": self.model_version,
            "engine": self.engine.name,
            "point_count": len(pts),
            "points": pts,
            "breaks": out["breaks"],
            "route_segments": out["route_segments"],
            "road_context": self.road_context(inputs, graph),
            "runtime_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
