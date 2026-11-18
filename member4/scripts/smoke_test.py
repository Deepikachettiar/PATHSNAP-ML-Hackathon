"""
Live API test (run from Laptop 2).  Usage:  python -m scripts.smoke_test http://SERVER_IP:8000
Checks: health, trips, every trip predicts, one status per point, valid coordinates, no label leakage,
repeat request is identical, unknown ID is rejected, path-traversal ID is rejected.
"""
import sys, requests

FORBIDDEN = {"reference", "route", "label", "labels", "reference_arc_id", "correct", "is_correct"}
base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
fails = 0

def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg)
    fails += (not ok)

h = requests.get(base + "/health", timeout=10).json()
check(h.get("status") == "ok", f"/health {h}")
trips = requests.get(base + "/trips", timeout=10).json()["trips"]
check(len(trips) > 0, f"/trips returned {len(trips)} ids, first={trips[:3]}")
check(all(isinstance(t, str) and len(t) == 8 for t in trips), "trip IDs are 8-char strings (leading zeros kept)")
for t in trips:
    r = requests.post(base + "/predict", json={"trip_id": t}, timeout=180)
    if r.status_code != 200:
        check(False, f"{t}: HTTP {r.status_code} {r.text[:120]}"); continue
    j = r.json()
    pts = j["points"]
    ok = (j["trip_id"] == t and len(pts) == j["point_count"]
          and [p["point_index"] for p in pts] == list(range(len(pts)))
          and all(p["status"] for p in pts)
          and all(p["lat"] is None or (-90 <= p["lat"] <= 90 and -180 <= p["lon"] <= 180) for p in pts)
          and not (FORBIDDEN & set(j)) and not any(FORBIDDEN & set(p) for p in pts))
    r2 = requests.post(base + "/predict", json={"trip_id": t}, timeout=180).json()
    same = r2["points"] == pts and r2["route_segments"] == j["route_segments"]
    check(ok and same, f"{t}: {len(pts)} pts, {len(j['route_segments'])} segs, {j['runtime_ms']} ms, repeat-identical={same}")
check(requests.post(base + "/predict", json={"trip_id": "99999999"}, timeout=10).status_code == 404, "unknown ID -> 404")
check(requests.post(base + "/predict", json={"trip_id": "../../etc/passwd"}, timeout=10).status_code == 404, "path traversal -> 404")
print("ALL PASSED" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
