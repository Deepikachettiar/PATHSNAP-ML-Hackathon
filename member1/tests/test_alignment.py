import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from align_reference import align_sequence


class AlignmentTests(unittest.TestCase):
    def test_known_forward_route(self):
        start = np.array([[0., 0.], [100., 0.], [200., 0.]])
        end = start + [100., 0.]
        gps = np.array([[10., 1.], [80., -2.], [140., 2.], [260., 1.]])
        selected, reset = align_sequence(gps, np.arange(4), start, end,
                                        np.array([9, 3, 7]), np.zeros(3, dtype=int))
        np.testing.assert_array_equal(selected, [0, 0, 1, 2])
        np.testing.assert_array_equal(reset, [True, False, False, False])

    def test_large_gap_allows_new_alignment_block(self):
        start = np.array([[0., 0.], [100., 0.]])
        end = start + [100., 0.]
        selected, reset = align_sequence(np.array([[180., 0.], [10., 0.]]), np.array([0., 100.]),
                                        start, end, np.array([0, 1]), np.zeros(2, dtype=int))
        np.testing.assert_array_equal(selected, [1, 0])
        np.testing.assert_array_equal(reset, [True, True])

    def test_disconnected_reference_does_not_form_transition(self):
        start = np.array([[0., 0.], [1000., 0.]])
        end = start + [100., 0.]
        selected, reset = align_sequence(np.array([[10., 0.], [1010., 0.]]), np.array([0., 1.]),
                                        start, end, np.array([0, 1]), np.array([0, 1]), beam=2)
        np.testing.assert_array_equal(selected, [0, 1])
        np.testing.assert_array_equal(reset, [True, True])


if __name__ == "__main__":
    unittest.main()
