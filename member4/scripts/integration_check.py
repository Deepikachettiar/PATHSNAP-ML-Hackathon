"""Run all development trips through the real FastAPI pipeline and verify replay."""
import json
import shutil
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from api.main import app, STATE


def main():
    results = []
    with TestClient(app) as client:
        assert client.get("/health").json()["engine"] == "real"
        trips = client.get("/trips").json()["trips"]
        assert len(trips) == 10
        for trip in trips:
            response = client.post("/predict", json={"trip_id": trip})
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["engine"] == "real"
            assert len(data["points"]) == data["point_count"]
            assert [p["point_index"] for p in data["points"]] == list(range(data["point_count"]))
            assert all(p["arc_id"] is not None for p in data["points"])
            assert not any("reference_arc_id" in p for p in data["points"])
            replay = client.post("/replay", json={"trip_id": trip})
            assert replay.status_code == 200, replay.text
            repeated = client.post("/predict", json={"trip_id": trip}).json()
            assert repeated["points"] == data["points"]
            results.append(dict(trip_id=trip, points=data["point_count"], breaks=len(data["breaks"]),
                                replay_steps=len(replay.json()["steps"]), runtime_ms=data["runtime_ms"]))
            print(results[-1], flush=True)
        assert client.post("/predict", json={"trip_id": "../../etc/passwd"}).status_code == 404
        assert client.post("/predict", json={"trip_id": "99999999"}).status_code == 404
        # A fresh root with input files only verifies that prediction has no route-label dependency.
        trip = "00000052"
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / trip
            dest.mkdir()
            for extension in ["track", "nodes", "arcs"]:
                source = Path(STATE["cfg"]["data_root"]) / trip / f"{trip}.{extension}"
                shutil.copy2(source, dest / source.name)
            prediction = STATE["predictor"].predict(directory, trip)
            assert prediction["point_count"] == 262
    report = dict(status="passed", engine="real", model="extra_trees", trips=results,
                  points=sum(r["points"] for r in results), prediction_without_reference_passed=True)
    target = Path(__file__).resolve().parents[1] / "results/integration_check.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
