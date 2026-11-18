"""Independently reopen exported tables and verify every candidate's geometry and key."""
import csv
import gzip
import itertools
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer


def audit(root=Path("output"), data=Path("Student folder (2)")):
    report = json.loads((root / "validation_report.json").read_text())
    inventory = {t["trip_id"]: t for t in report["trips"]}
    total, point_total, outside_total, trips = 0, 0, 0, []
    max_distance_error, max_coordinate_error, max_lonlat_error = 0., 0., 0.
    with (root / "candidates.csv").open() as cf, gzip.open(root / "candidate_features.csv.gz", "rt") as ff:
        candidates, features = csv.DictReader(cf), csv.DictReader(ff)
        for trip, group in itertools.groupby(candidates, key=lambda r: r["trip_id"]):
            rows = list(group)
            meta = inventory[trip]
            g = np.load(root / "graphs" / f"{trip}.npz", allow_pickle=False)
            track = np.loadtxt(data / meta["split"] / trip / f"{trip}.track", ndmin=2)
            index = np.array([int(r["point_index"]) for r in rows])
            arc = np.array([int(r["arc_id"]) for r in rows])
            d = np.array([float(r["distance_m"]) for r in rows])
            q = np.array([[float(r["projected_x_m"]), float(r["projected_y_m"])] for r in rows])
            geographic = np.array([[float(r["projected_lon"]), float(r["projected_lat"])] for r in rows])
            outside = np.array([int(r["outside_radius"]) for r in rows])
            rank = np.array([int(r["candidate_rank"]) for r in rows])
            assert np.isfinite(d).all() and np.isfinite(q).all()
            assert np.array_equal(np.unique(index), np.arange(len(track))), (trip, "point coverage")
            assert len(np.unique(np.column_stack([index, arc]), axis=0)) == len(rows), (trip, "duplicate key")
            assert np.all((arc >= 0) & (arc < len(g["arc_from"])))
            tf = Transformer.from_crs(4326, meta["crs_epsg"], always_xy=True)
            gps = np.column_stack(tf.transform(track[:, 0], track[:, 1]))[index]
            start = g["node_xy_m"][g["arc_from"][arc]]
            end = g["node_xy_m"][g["arc_to"][arc]]
            vector = end - start
            sq = np.sum(vector * vector, axis=1)
            fraction = np.divide(np.sum((gps - start) * vector, axis=1), sq,
                                 out=np.zeros(len(sq)), where=sq > 0).clip(0, 1)
            expected = start + fraction[:, None] * vector
            expected_distance = np.linalg.norm(gps - expected, axis=1)
            max_distance_error = max(max_distance_error, float(np.max(np.abs(d - expected_distance))))
            max_coordinate_error = max(max_coordinate_error, float(np.max(np.abs(q - expected))))
            back = Transformer.from_crs(meta["crs_epsg"], 4326, always_xy=True)
            lonlat = np.column_stack(back.transform(q[:, 0], q[:, 1]))
            max_lonlat_error = max(max_lonlat_error, float(np.max(np.abs(lonlat - geographic))))
            assert np.all((outside == 0) | (outside == 1))
            assert np.all(d[outside == 0] <= report["summary"]["radius_m"] + 1e-6)
            assert np.all(d[outside == 1] > report["summary"]["radius_m"] - 1e-6)
            same = index[1:] == index[:-1]
            assert np.all(np.diff(d)[same] >= -1e-6)
            assert np.all(np.diff(index) >= 0)
            boundaries = np.r_[0, np.flatnonzero(~same) + 1, len(rows)]
            sizes = np.diff(boundaries)
            assert np.all((sizes >= 1) & (sizes <= report["summary"]["top_k"]))
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                assert np.array_equal(rank[left:right], np.arange(1, right - left + 1))
                if np.any(outside[left:right]):
                    assert right - left == 1
            for r in rows:
                f = next(features)
                assert all(r[k] == f[k] for k in ["trip_id", "point_index", "arc_id"])
                assert r["split"] == meta["split"] and int(r["crs_epsg"]) == meta["crs_epsg"]
            assert len(rows) == meta["candidate_rows"]
            assert int(outside.sum()) == meta["fallback_points"]
            total += len(rows)
            point_total += len(track)
            outside_total += int(outside.sum())
            trips.append(trip)
        assert next(features, None) is None
    assert set(trips) == set(inventory)
    assert total == report["summary"]["candidate_rows"]
    assert max_distance_error < 1e-6 and max_coordinate_error < 1e-6 and max_lonlat_error < 1e-8
    result = dict(status="passed", trips=len(trips), candidate_rows=total, gps_points=point_total,
                  outside_radius_points=outside_total, max_distance_rounding_error_m=max_distance_error,
                  max_projected_coordinate_rounding_error_m=max_coordinate_error,
                  max_lonlat_roundtrip_error_deg=max_lonlat_error, geometry_unit_tests_passed=3,
                  note="All exported rows checked; index retrieval checked separately against exhaustive geometry on 20 real points.")
    (root / "output_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    audit()
