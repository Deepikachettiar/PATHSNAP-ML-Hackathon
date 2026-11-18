"""Offline, route-only sequence alignment for approximate supervised reference labels.

This module must never be called by the inference pipeline.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer

from build_candidates import project_to_arcs


def align_sequence(gps, timestamps, starts, ends, arc_ids, components,
                   beam=32, distance_scale=20., progress_scale=50., gap_s=60., spatial_limit_m=150.):
    """Align to ordered route occurrences using a spatial shortlist and dynamic programming.

    Repeated arc occurrences remain distinct states. Forward occurrence order and
    route connectivity are enforced within each alignment block. GPS gaps and
    an empty feasible transition set start a new block, explicitly flagged.
    """
    n, m = len(gps), len(arc_ids)
    width = min(beam, m)
    vector = ends - starts
    squared = np.sum(vector * vector, axis=1)
    lengths = np.sqrt(squared)
    cumulative = np.r_[0., np.cumsum(lengths)[:-1]]
    states = np.empty((n, width), dtype=np.int32)
    parents = np.full((n, width), -1, dtype=np.int16)
    best = np.empty(n, dtype=np.int16)
    reset = np.zeros(n, dtype=bool)
    reset[0] = True
    previous_cost = previous_state = previous_position = None
    previous_outlier = False
    for i, point in enumerate(gps):
        distances, fraction, _ = project_to_arcs(point, starts, vector, squared)
        # lexsort resolves exact ties using chronological route occurrence.
        ids = np.lexsort((np.arange(m), distances))[:width]
        states[i] = ids
        position = cumulative[ids] + fraction[ids] * lengths[ids]
        emission = (distances[ids] / distance_scale) ** 2
        outlier = bool(distances[ids[0]] > spatial_limit_m)
        if not outlier:
            emission[distances[ids] > spatial_limit_m] = np.inf
        if i == 0 or timestamps[i] - timestamps[i - 1] > gap_s or outlier or previous_outlier:
            cost = emission.copy()
            reset[i] = True
        else:
            delta = position[:, None] - previous_position[None, :]
            step = np.linalg.norm(gps[i] - gps[i - 1])
            transition = np.abs(delta - step) / progress_scale
            valid = ((ids[:, None] >= previous_state[None, :]) & (delta >= -20.) &
                     (components[ids, None] == components[previous_state][None, :]))
            transition[~valid] = np.inf
            matrix = transition + previous_cost[None, :]
            parent = np.argmin(matrix, axis=1)
            cost = emission + matrix[np.arange(width), parent]
            if not np.isfinite(cost).any():
                cost = emission.copy()
                reset[i] = True
            else:
                parents[i] = parent
        best[i] = np.argmin(cost)
        previous_cost = cost - np.min(cost)
        previous_state, previous_position = ids, position
        previous_outlier = outlier
    selected = np.empty(n, dtype=np.int32)
    state = int(best[-1])
    for i in range(n - 1, -1, -1):
        selected[i] = states[i, state]
        state = int(parents[i, state]) if parents[i, state] >= 0 else (int(best[i - 1]) if i else 0)
    return selected, reset


def run(data_root=Path("Student folder (2)"), output=Path("output"), beam=32):
    destination = output / "reference_alignment"
    destination.mkdir(exist_ok=True)
    validation = json.loads((output / "validation_report.json").read_text())
    fields = ["trip_id", "point_index", "split", "reference_arc_id", "route_occurrence_index",
              "reference_distance_m", "reference_projected_lon", "reference_projected_lat",
              "route_progress_m", "local_distance_margin_m", "alignment_block", "alignment_reset",
              "label_usable", "exclusion_reason"]
    summaries = []
    with (destination / "point_labels.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for meta in validation["trips"]:
            trip, split = meta["trip_id"], meta["split"]
            directory = data_root / split / trip
            track = np.loadtxt(directory / f"{trip}.track", ndmin=2)
            route = np.loadtxt(directory / f"{trip}.route", dtype=np.int64, ndmin=1)
            graph = np.load(output / "graphs" / f"{trip}.npz", allow_pickle=False)
            xy = graph["node_xy_m"]
            arc_from, arc_to = graph["arc_from"][route], graph["arc_to"][route]
            starts, ends = xy[arc_from], xy[arc_to]
            component = np.r_[0, np.cumsum(arc_to[:-1] != arc_from[1:])]
            tf = Transformer.from_crs(4326, meta["crs_epsg"], always_xy=True)
            inverse = Transformer.from_crs(meta["crs_epsg"], 4326, always_xy=True)
            gps = np.column_stack(tf.transform(track[:, 0], track[:, 1]))
            selected, reset = align_sequence(gps, track[:, 2], starts, ends, route, component, beam=beam)
            block = np.cumsum(reset) - 1
            vector = ends - starts
            squared = np.sum(vector * vector, axis=1)
            cumulative = np.r_[0., np.cumsum(np.sqrt(squared))[:-1]]
            boundary = reset.copy()
            boundary[:-1] |= reset[1:]
            # The first point is not a reset failure, though a later boundary may still exclude it.
            boundary[0] = bool(len(reset) > 1 and reset[1])
            usable = 0
            distances_selected = []
            for i, occurrence in enumerate(selected):
                d, fraction, q = project_to_arcs(gps[i], starts, vector, squared)
                other = d[route != route[occurrence]]
                margin = float(other.min() - d[occurrence]) if len(other) else float("inf")
                reasons = []
                if d[occurrence] > 50:
                    reasons.append("distance_over_50m")
                if margin < 3:
                    reasons.append("ambiguous_or_non_nearest")
                if boundary[i]:
                    reasons.append("alignment_block_boundary")
                if squared[occurrence] == 0:
                    reasons.append("zero_length_reference")
                good = not reasons
                usable += int(good)
                lon, lat = inverse.transform(*q[occurrence])
                writer.writerow([trip, i, split, int(route[occurrence]), int(occurrence),
                    f"{d[occurrence]:.6f}", f"{lon:.9f}", f"{lat:.9f}",
                    f"{cumulative[occurrence] + fraction[occurrence] * np.sqrt(squared[occurrence]):.6f}",
                    f"{margin:.6f}" if np.isfinite(margin) else "", int(block[i]), int(reset[i]),
                    int(good), ";".join(reasons)])
                distances_selected.append(float(d[occurrence]))
            summaries.append(dict(trip_id=trip, split=split, points=len(track), usable_labels=usable,
                alignment_blocks=int(block[-1] + 1), route_discontinuities=int(component[-1]),
                reference_distance_p95_m=float(np.percentile(distances_selected, 95)),
                reference_distance_max_m=max(distances_selected)))
            print(f"{split}/{trip}: {usable}/{len(track)} conservative labels, {block[-1] + 1} alignment blocks", flush=True)
    summary = dict(method="route-only spatial shortlist plus ordered dynamic programming",
        approximate_labels=True, beam=beam, distance_scale_m=20, progress_scale_m=50,
        gap_reset_s=60, allowed_backward_progress_on_same_occurrence_m=20,
        spatial_limit_m=150,
        usable_distance_limit_m=50, usable_competing_arc_margin_m=3,
        points=sum(t["points"] for t in summaries), usable_labels=sum(t["usable_labels"] for t in summaries),
        trips=summaries)
    (destination / "alignment_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Student folder (2)"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--beam", type=int, default=32)
    args = parser.parse_args()
    if args.beam < 1 or args.beam > 32767:
        parser.error("beam must be between 1 and 32767")
    run(args.data_root, args.output, args.beam)
