"""Load Member 2's fitted preprocessing/classifier pipeline once per API process."""
from pathlib import Path
import joblib
import numpy as np


class Scorer:
    def __init__(self, config):
        path = Path(config["model_path"])
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.name = bundle["name"]

    def score(self, candidates):
        raw = self.model.predict_proba(candidates[self.features])[:, 1]
        weight = candidates.point_index.map(
            candidates.assign(weight=np.maximum(raw, 1e-6)).groupby("point_index").weight.sum())
        return np.maximum(raw, 1e-6) / weight.to_numpy()
