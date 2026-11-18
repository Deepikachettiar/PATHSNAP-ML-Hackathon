"""
Reading trip files.

SEPARATION RULE (guide p.8-9): the prediction path may only call `load_inputs`
(.track, .nodes, .arcs). Only api/evaluator.py may call `load_reference` (.route).
Trip IDs are always strings so leading zeros survive.
"""
from pathlib import Path
import numpy as np
import pandas as pd

R_EARTH = 6371008.8
_INPUT_EXT = ("track", "nodes", "arcs")


def _trip_dir(root: str, trip_id: str) -> Path:
    # trip_id must already be validated against list_trip_ids(); never build paths from raw client input.
    return Path(root) / trip_id


def list_trip_ids(root: str) -> list[str]:
    """Trip IDs actually present on disk (needs the 3 input files). Sorted for stable order."""
    rootp = Path(root)
    if not rootp.is_dir():
        return []
    ids = []
    for d in sorted(p for p in rootp.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        if all((d / f"{d.name}.{e}").is_file() for e in _INPUT_EXT):
            ids.append(d.name)
    return ids


def _read_table(path: Path, ncols: int, dtype) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype)
    if df.shape[1] < ncols:
        raise ValueError(f"{path.name}: expected {ncols} columns, found {df.shape[1]}")
    return df.iloc[:, :ncols].to_numpy()


def load_inputs(root: str, trip_id: str) -> dict:
    d = _trip_dir(root, trip_id)
    track = _read_table(d / f"{trip_id}.track", 3, np.float64)   # lon, lat, seconds
    nodes = _read_table(d / f"{trip_id}.nodes", 2, np.float64)   # lon, lat  (row = node id)
    arcs = _read_table(d / f"{trip_id}.arcs", 2, np.int64)       # from, to  (row = arc id)
    return {"track": track, "nodes": nodes, "arcs": arcs}


def load_reference(root: str, trip_id: str) -> np.ndarray:
    """Chronological arc IDs. ONLY for api/evaluator.py and dev-map scripts."""
    path = _trip_dir(root, trip_id) / f"{trip_id}.route"
    if not path.is_file():
        raise FileNotFoundError(path.name)
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=np.int64).iloc[:, 0].to_numpy()


class LocalProjection:
    """Local metric projection (equirectangular) centred on the trip. Fixed per trip."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self._cx = R_EARTH * np.cos(np.radians(lat0)) * np.pi / 180.0
        self._cy = R_EARTH * np.pi / 180.0

    def fwd(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self._cx, (np.asarray(lat) - self.lat0) * self._cy

    def inv(self, x, y):
        return self.lon0 + np.asarray(x) / self._cx, self.lat0 + np.asarray(y) / self._cy
