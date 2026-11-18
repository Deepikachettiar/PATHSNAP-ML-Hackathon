"""Live candidate features using Member 1's UTM geometry and candidate policy."""
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import STRtree, linestrings

from .geometry import metric_crs, select_candidates


def generate(track, graph):
    nodes, arcs = graph.nodes_ll, graph.arcs
    epsg = metric_crs(nodes)
    forward = Transformer.from_crs(4326, epsg, always_xy=True)
    inverse = Transformer.from_crs(epsg, 4326, always_xy=True)
    xy = np.column_stack(forward.transform(nodes[:, 0], nodes[:, 1]))
    gps = np.column_stack(forward.transform(track[:, 0], track[:, 1]))
    starts, ends = xy[arcs[:, 0]], xy[arcs[:, 1]]
    vectors = ends - starts
    squared = np.sum(vectors * vectors, axis=1)
    lengths = np.sqrt(squared)
    bearing = np.degrees(np.arctan2(vectors[:, 0], vectors[:, 1])) % 360
    tree = STRtree(linestrings(np.stack([starts, ends], axis=1)))
    delta = np.diff(gps, axis=0, prepend=np.full((1, 2), np.nan))
    step = np.linalg.norm(delta, axis=1)
    dt = np.diff(track[:, 2], prepend=np.nan)
    heading = np.degrees(np.arctan2(delta[:, 0], delta[:, 1])) % 360
    heading[~np.isfinite(step) | (step < 1)] = np.nan
    out_degree = np.bincount(arcs[:, 0], minlength=len(nodes))
    in_degree = np.bincount(arcs[:, 1], minlength=len(nodes))
    rows = []
    for i, point in enumerate(gps):
        ids, distances, fractions, snapped, fallback, _ = select_candidates(
            point, tree, starts, vectors, squared, 100., 8)
        lon, lat = inverse.transform(snapped[:, 0], snapped[:, 1])
        for rank, arc in enumerate(ids):
            fraction, length = fractions[rank], lengths[arc]
            rows.append(dict(point_index=i, arc_id=int(arc), distance_m=distances[rank],
                projected_lon=lon[rank], projected_lat=lat[rank], candidate_rank=rank + 1,
                outside_radius=int(fallback), arc_length_m=length, arc_fraction=fraction,
                distance_from_arc_start_m=fraction * length, distance_to_arc_end_m=(1 - fraction) * length,
                arc_bearing_deg=bearing[arc] if length else np.nan, gps_heading_deg=heading[i],
                heading_difference_deg=abs((heading[i] - bearing[arc] + 180) % 360 - 180) if length else np.nan,
                dt_s=dt[i], gps_step_m=step[i], gps_speed_m_s=step[i] / dt[i],
                from_out_degree=out_degree[arcs[arc, 0]], to_in_degree=in_degree[arcs[arc, 1]]))
    return pd.DataFrame(rows)
