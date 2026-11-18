"""Adapt Member 3's directed decoder to the existing API response contract."""
import importlib.util
import math
from pathlib import Path


source = Path(__file__).resolve().parents[1] / "vendor/member3/run.py"
spec = importlib.util.spec_from_file_location("member3_decoder", source)
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)


def decode(track, candidates, graph, scores):
    groups = [{} for _ in track]
    lookup = {}
    for row, probability in zip(candidates.itertuples(index=False), scores):
        key = (row.point_index, row.arc_id)
        lookup[key] = row
        groups[row.point_index][row.arc_id] = (
            (row.projected_lon, row.projected_lat), math.log(max(float(probability), 1e-300)))
    result = upstream.decode("live", groups, track.tolist(), graph.nodes_ll.tolist(), graph.arcs.tolist())
    points, breaks = [], []
    for i, selected in enumerate(result["points"]):
        arc = selected["arc_id"]
        row = lookup.get((i, arc))
        points.append(dict(point_index=i, lon=float(track[i, 0]), lat=float(track[i, 1]),
            timestamp_s=float(track[i, 2]), status="matched" if row is not None else "unmatched",
            arc_id=arc, baseline_arc_id=selected["baseline"],
            matched_lon=float(row.projected_lon) if row else None,
            matched_lat=float(row.projected_lat) if row else None,
            offset_m=float(row.distance_from_arc_start_m) if row else None,
            distance_m=float(row.distance_m) if row else None,
            decoder_status=selected["status"]))
        if selected["status"] not in ("start", "connected"):
            breaks.append(dict(point_index=i, reason=selected["status"]))
    return dict(points=points, breaks=breaks, route_segments=[line for line in result["lines"] if len(line) > 1],
                replay=result["replay"], replay_roads=result["anchor_roads"], sequence=result["sequence"])
