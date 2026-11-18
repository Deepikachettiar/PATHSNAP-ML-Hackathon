"""Directed shortest-path helpers for consuming the preserved per-trip road graphs."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class RoadGraph:
    def __init__(self, node_count, arc_from, arc_to, arc_length_m):
        self.arc_from = np.asarray(arc_from, dtype=np.int64)
        self.arc_to = np.asarray(arc_to, dtype=np.int64)
        self.arc_length_m = np.asarray(arc_length_m, dtype=float)
        order = np.lexsort((self.arc_to, self.arc_from))
        u, v, length = self.arc_from[order], self.arc_to[order], self.arc_length_m[order]
        first = np.r_[0, np.flatnonzero((u[1:] != u[:-1]) | (v[1:] != v[:-1])) + 1]
        # scipy sums duplicate sparse entries by default. Distances require the minimum,
        # while the original per-arc arrays continue to preserve every candidate arc ID.
        self.adjacency = csr_matrix((np.minimum.reduceat(length, first), (u[first], v[first])),
                                    shape=(node_count, node_count))

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as arrays:
            return cls(len(arrays["node_lonlat"]), arrays["arc_from"], arrays["arc_to"], arrays["arc_length_m"])

    @lru_cache(maxsize=32)
    def _from_node(self, source):
        return dijkstra(self.adjacency, directed=True, indices=int(source))

    def shortest_distance_m(self, source, target):
        """Return directed node-to-node distance; infinity means unreachable."""
        return float(self._from_node(int(source))[int(target)])

    def candidate_distance_m(self, from_arc, from_fraction, to_arc, to_fraction):
        """Distance along directed roads between two candidate positions.

        Fractions are measured from arc start. A backward move on the same arc
        requires an actual directed cycle; it is not assigned negative distance.
        """
        if not (0 <= from_fraction <= 1 and 0 <= to_fraction <= 1):
            raise ValueError("Arc fractions must lie in [0, 1]")
        if from_arc == to_arc and to_fraction >= from_fraction:
            return float((to_fraction - from_fraction) * self.arc_length_m[from_arc])
        middle = self.shortest_distance_m(self.arc_to[from_arc], self.arc_from[to_arc])
        return float((1 - from_fraction) * self.arc_length_m[from_arc] + middle +
                     to_fraction * self.arc_length_m[to_arc])
