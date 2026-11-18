"""Geometry and spatial-index checks against known answers and exhaustive GEOS distances."""
import sys
import unittest
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely import STRtree, distance, linestrings, points

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_candidates import metric_crs, project_to_arcs, read_array, select_candidates


class GeometryTests(unittest.TestCase):
    def test_clamped_projection_and_degenerate_arc(self):
        starts = np.array([[0., 0.], [10., 0.], [0., 0.]])
        vectors = np.array([[10., 0.], [0., 10.], [0., 0.]])
        d, f, q = project_to_arcs(np.array([12., 4.]), starts, vectors, np.array([100., 100., 0.]))
        np.testing.assert_allclose(q, [[10, 0], [10, 4], [0, 0]])
        np.testing.assert_allclose(d, [np.sqrt(20), 2, np.sqrt(160)])
        np.testing.assert_allclose(f, [1, .4, 0])

    def test_reverse_arcs_retained_and_fallback(self):
        start = np.array([[0., 0.], [10., 0.], [0., 20.]])
        end = np.array([[10., 0.], [0., 0.], [10., 20.]])
        v = end - start
        sq = (v * v).sum(1)
        tree = STRtree(linestrings(np.stack([start, end], axis=1)))
        result = select_candidates(np.array([5., 1.]), tree, start, v, sq, 2, 8)
        np.testing.assert_array_equal(result[0], [0, 1])
        self.assertFalse(result[4])
        result = select_candidates(np.array([5., -100.]), tree, start, v, sq, 2, 8)
        np.testing.assert_array_equal(result[0], [0])
        self.assertTrue(result[4])

    def test_real_trips_against_all_arc_distances(self):
        root = Path(__file__).resolve().parents[1] / "Student folder (2)"
        for trip in [root / "train" / "00000001", root / "student_test" / "00000000"]:
            nodes = read_array(trip / f"{trip.name}.nodes", 2)
            arcs = read_array(trip / f"{trip.name}.arcs", 2, True)
            track = read_array(trip / f"{trip.name}.track", 3)
            tf = Transformer.from_crs(4326, metric_crs(nodes), always_xy=True)
            xy = np.column_stack(tf.transform(nodes[:, 0], nodes[:, 1]))
            start, end = xy[arcs[:, 0]], xy[arcs[:, 1]]
            v = end - start
            sq = (v * v).sum(1)
            geometry = linestrings(np.stack([start, end], axis=1))
            tree = STRtree(geometry)
            for i in np.linspace(0, len(track) - 1, 10, dtype=int):
                p = np.array(tf.transform(*track[i, :2]))
                all_d = distance(geometry, points(p))
                eligible = np.flatnonzero(all_d <= 100)
                fallback = len(eligible) == 0
                if fallback:
                    eligible = np.flatnonzero(all_d == all_d.min())
                expected = eligible[np.lexsort((eligible, all_d[eligible]))][:1 if fallback else 8]
                result = select_candidates(p, tree, start, v, sq, 100, 8)
                # Floating-point ties on collinear/reversed roads may order differently;
                # compare distances and membership except for sub-nanometre ties at cutoff.
                np.testing.assert_allclose(result[1], all_d[expected], atol=1e-7)
                self.assertTrue(np.all(all_d[result[0]] <= all_d[expected[-1]] + 1e-7))
                np.testing.assert_allclose(distance(points(result[3]), geometry[result[0]]), 0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
