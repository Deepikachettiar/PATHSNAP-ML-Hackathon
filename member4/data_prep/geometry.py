"""Validate PathSnap inputs and produce directed-road candidates without route labels."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pyproj
import scipy
import shapely
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely import STRtree, linestrings, points


def read_array(path, columns, integer=False):
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != columns or not np.isfinite(data).all():
        raise ValueError(f"Invalid shape or nonfinite values: {path}")
    if integer:
        if not np.equal(data, np.floor(data)).all():
            raise ValueError(f"Noninteger IDs: {path}")
        data = data.astype(np.int64)
    return data


def validate_lonlat(a, path):
    if ((np.abs(a[:, 0]) > 180) | (np.abs(a[:, 1]) > 90)).any():
        raise ValueError(f"Coordinates outside WGS84 bounds: {path}")


def metric_crs(lonlat):
    lon, lat = np.median(lonlat, axis=0)
    if not -80 <= lat <= 84:
        raise ValueError("UTM unavailable at this latitude")
    zone = min(60, max(1, int((lon + 180) // 6) + 1))
    # Conventional exceptions in Norway and Svalbard.
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone = 32
    if 72 <= lat < 84 and 0 <= lon < 42:
        zone = 31 if lon < 9 else 33 if lon < 21 else 35 if lon < 33 else 37
    return (32600 if lat >= 0 else 32700) + zone


def project_to_arcs(point, starts, vectors, lengths_squared):
    fraction = np.divide(
        np.einsum("ij,ij->i", point - starts, vectors), lengths_squared,
        out=np.zeros(len(starts)), where=lengths_squared > 0,
    ).clip(0, 1)
    projected = starts + fraction[:, None] * vectors
    distance = np.linalg.norm(projected - point, axis=1)
    return distance, fraction, projected


def select_candidates(point, tree, starts, vectors, lengths_squared, radius, top_k):
    geometry = points(point)
    ids = tree.query(geometry, predicate="dwithin", distance=radius)
    fallback = len(ids) == 0
    if fallback:
        # Return all exactly tied nearest arcs first; stable arc_id tie-breaking below.
        ids = np.asarray(tree.query_nearest(geometry, all_matches=True)).ravel()
    distance, fraction, projected = project_to_arcs(
        point, starts[ids], vectors[ids], lengths_squared[ids]
    )
    order = np.lexsort((ids, distance))[:1 if fallback else top_k]
    return ids[order], distance[order], fraction[order], projected[order], fallback, len(ids)


def build(data_root, output, radius=100.0, top_k=8):
    if radius <= 0 or top_k < 1:
        raise ValueError("radius and top_k must be positive")
    output.mkdir(parents=True, exist_ok=True)
    (output / "graphs").mkdir(exist_ok=True)
    inventory = []
    counts = {}
    main_fields = ["trip_id", "point_index", "arc_id", "distance_m", "projected_x_m",
                   "projected_y_m", "projected_lon", "projected_lat", "crs_epsg",
                   "split", "candidate_rank", "outside_radius"]
    feature_fields = ["trip_id", "point_index", "arc_id", "arc_length_m", "arc_fraction",
                      "distance_from_arc_start_m", "distance_to_arc_end_m", "arc_bearing_deg",
                      "gps_heading_deg", "heading_difference_deg", "dt_s", "gps_step_m",
                      "gps_speed_m_s", "from_out_degree", "to_in_degree"]
    with (output / "candidates.csv").open("w", newline="") as mf, gzip.open(
        output / "candidate_features.csv.gz", "wt", newline=""
    ) as ff:
        writer, features = csv.writer(mf), csv.writer(ff)
        writer.writerow(main_fields)
        features.writerow(feature_fields)
        for split in ("train", "student_test"):
            trips = sorted(p for p in (data_root / split).iterdir() if p.is_dir())
            counts[split] = len(trips)
            for directory in trips:
                trip = directory.name
                paths = {ext: directory / f"{trip}.{ext}" for ext in ("track", "nodes", "arcs", "route")}
                track = read_array(paths["track"], 3)
                nodes = read_array(paths["nodes"], 2)
                arcs = read_array(paths["arcs"], 2, integer=True)
                validate_lonlat(track[:, :2], paths["track"])
                validate_lonlat(nodes, paths["nodes"])
                if not len(arcs) or arcs.min() < 0 or arcs.max() >= len(nodes):
                    raise ValueError(f"Invalid node references: {paths['arcs']}")
                dt = np.diff(track[:, 2], prepend=np.nan)
                if np.any(dt[1:] <= 0):
                    raise ValueError(f"Non-increasing timestamps: {paths['track']}")
                epsg = metric_crs(nodes)
                forward = Transformer.from_crs(4326, epsg, always_xy=True)
                backward = Transformer.from_crs(epsg, 4326, always_xy=True)
                xy = np.column_stack(forward.transform(nodes[:, 0], nodes[:, 1]))
                gps = np.column_stack(forward.transform(track[:, 0], track[:, 1]))
                starts, ends = xy[arcs[:, 0]], xy[arcs[:, 1]]
                vectors = ends - starts
                squared = np.einsum("ij,ij->i", vectors, vectors)
                lengths = np.sqrt(squared)
                bearing = np.degrees(np.arctan2(vectors[:, 0], vectors[:, 1])) % 360
                tree = STRtree(linestrings(np.stack([starts, ends], axis=1)))
                graph = csr_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(len(nodes), len(nodes)))
                weak, labels = connected_components(graph, directed=True, connection="weak")
                strong = connected_components(graph, directed=True, connection="strong", return_labels=False)
                out_degree = np.bincount(arcs[:, 0], minlength=len(nodes))
                in_degree = np.bincount(arcs[:, 1], minlength=len(nodes))
                order = np.argsort(arcs[:, 0], kind="stable")
                offsets = np.r_[0, np.cumsum(out_degree)]
                np.savez_compressed(output / "graphs" / f"{trip}.npz",
                    node_lonlat=nodes, node_xy_m=xy, arc_from=arcs[:, 0], arc_to=arcs[:, 1],
                    arc_length_m=lengths, outgoing_offsets=offsets, outgoing_arc_ids=order,
                    weak_component=labels, crs_epsg=np.array(epsg))
                delta = np.diff(gps, axis=0, prepend=np.full((1, 2), np.nan))
                step = np.linalg.norm(delta, axis=1)
                heading = np.degrees(np.arctan2(delta[:, 0], delta[:, 1])) % 360
                heading[~np.isfinite(step) | (step < 1)] = np.nan
                candidate_count, nearest, fallback_points, truncated = [], [], [], 0
                rows = 0
                for i, point in enumerate(gps):
                    ids, distance, fraction, projected, fallback, before_cap = select_candidates(
                        point, tree, starts, vectors, squared, radius, top_k)
                    candidate_count.append(len(ids))
                    nearest.append(float(distance[0]))
                    if fallback:
                        fallback_points.append(i)
                    truncated += int(not fallback and before_cap > top_k)
                    lon, lat = backward.transform(projected[:, 0], projected[:, 1])
                    for rank, arc in enumerate(ids):
                        writer.writerow([trip, i, int(arc), f"{distance[rank]:.6f}",
                            f"{projected[rank, 0]:.6f}", f"{projected[rank, 1]:.6f}",
                            f"{lon[rank]:.9f}", f"{lat[rank]:.9f}", epsg, split, rank + 1, int(fallback)])
                        angular = abs((heading[i] - bearing[arc] + 180) % 360 - 180)
                        values = [lengths[arc], fraction[rank], fraction[rank] * lengths[arc],
                            (1 - fraction[rank]) * lengths[arc], bearing[arc] if lengths[arc] > 0 else np.nan,
                            heading[i], angular if lengths[arc] > 0 else np.nan, dt[i], step[i],
                            step[i] / dt[i], out_degree[arcs[arc, 0]], in_degree[arcs[arc, 1]]]
                        features.writerow([trip, i, int(arc)] + [f"{v:.6f}" if np.isfinite(v) else "" for v in values])
                        rows += 1
                # Reference routes are inspected only after prediction inputs and candidates are built.
                route_stats = {"route_available": paths["route"].exists()}
                if paths["route"].exists():
                    route = read_array(paths["route"], 1, integer=True).ravel()
                    if route.min() < 0 or route.max() >= len(arcs):
                        raise ValueError(f"Invalid route arc references: {paths['route']}")
                    route_stats.update(route_arc_count=len(route), route_discontinuities=int(
                        np.sum(arcs[route[:-1], 1] != arcs[route[1:], 0])))
                item = dict(trip_id=trip, split=split, gps_points=len(track), nodes=len(nodes), arcs=len(arcs),
                    crs_epsg=epsg, candidate_rows=rows, fallback_points=len(fallback_points),
                    fallback_point_indices=fallback_points, points_capped_at_k=truncated,
                    zero_length_arcs=int(np.sum(lengths == 0)), self_loops=int(np.sum(arcs[:, 0] == arcs[:, 1])),
                    duplicate_directed_arcs=len(arcs) - len(np.unique(arcs, axis=0)),
                    weak_components=int(weak), strong_components=int(strong),
                    isolated_nodes=int(np.sum(in_degree + out_degree == 0)),
                    largest_weak_component_nodes=int(np.bincount(labels).max()),
                    min_dt_s=float(dt[1:].min()), max_dt_s=float(dt[1:].max()),
                    non_1s_intervals=int(np.sum(dt[1:] != 1)), gaps_over_5s=int(np.sum(dt[1:] > 5)),
                    stationary_steps_under_1m=int(np.sum(step[1:] < 1)),
                    steps_over_70m_s=int(np.sum(step[1:] / dt[1:] > 70)),
                    nearest_distance_p50_m=float(np.median(nearest)), nearest_distance_p95_m=float(np.percentile(nearest, 95)),
                    max_nearest_distance_m=max(nearest),
                    track_bounds_lonlat=[*track[:, :2].min(axis=0), *track[:, :2].max(axis=0)],
                    network_bounds_lonlat=[*nodes.min(axis=0), *nodes.max(axis=0)],
                    file_sizes={k: p.stat().st_size for k, p in paths.items() if p.exists()},
                    sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items() if p.exists()},
                    **route_stats)
                inventory.append(item)
                print(f"{split}/{trip}: {len(track)} points -> {rows} candidates; {len(fallback_points)} fallbacks", flush=True)
    summary = dict(trips_by_split=counts, gps_points=sum(x["gps_points"] for x in inventory),
        candidate_rows=sum(x["candidate_rows"] for x in inventory),
        fallback_points=sum(x["fallback_points"] for x in inventory),
        points_capped_at_k=sum(x["points_capped_at_k"] for x in inventory),
        radius_m=radius, top_k=top_k, point_index_base=0, arc_id_base=0,
        invalid_rows=0, versions={"numpy": np.__version__, "pyproj": pyproj.__version__,
                                "scipy": scipy.__version__, "shapely": shapely.__version__})
    (output / "validation_report.json").write_text(json.dumps(dict(summary=summary, trips=inventory), indent=2) + "\n")
    flat = [{k: v for k, v in x.items() if not isinstance(v, (list, dict))} for x in inventory]
    with (output / "trip_inventory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Student folder (2)"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--radius-m", type=float, default=100)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    build(args.data_root, args.output, args.radius_m, args.top_k)
