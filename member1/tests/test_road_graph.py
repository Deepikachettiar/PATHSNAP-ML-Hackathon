import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_graph import RoadGraph


class RoadGraphTests(unittest.TestCase):
    def test_parallel_edges_choose_minimum_and_direction_is_preserved(self):
        g = RoadGraph(3, [0, 0, 1], [1, 1, 2], [3, 5, 7])
        self.assertEqual(g.shortest_distance_m(0, 2), 10)
        self.assertTrue(np.isinf(g.shortest_distance_m(2, 0)))
        self.assertEqual(len(g.arc_from), 3)

    def test_zero_weight_edges_and_candidate_distances(self):
        g = RoadGraph(4, [0, 1, 2], [1, 2, 3], [10, 0, 20])
        self.assertEqual(g.shortest_distance_m(0, 3), 30)
        self.assertEqual(g.candidate_distance_m(0, .5, 2, .25), 10)
        self.assertAlmostEqual(g.candidate_distance_m(2, .1, 2, .4), 6)

    def test_backward_move_requires_directed_cycle(self):
        g = RoadGraph(2, [0, 1], [1, 0], [10, 10])
        self.assertEqual(g.candidate_distance_m(0, .8, 0, .2), 14)
        g = RoadGraph(2, [0], [1], [10])
        self.assertTrue(np.isinf(g.candidate_distance_m(0, .8, 0, .2)))


if __name__ == "__main__":
    unittest.main()
