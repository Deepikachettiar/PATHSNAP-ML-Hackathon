"""
Development-only Folium map for the REPORT (section 6): GPS + road network + reference route + prediction.
Reads .route, so it must never be imported by the API/UI.
Usage:  python -m scripts.make_dev_map 00000056   ->  maps/dev_map_00000056.html
"""
import sys
import folium
from api.config import load_config, REPO_ROOT
from api.data_access import load_reference
from api.pipeline import Predictor

trip = sys.argv[1]
cfg = load_config()
pred = Predictor(cfg)
res = pred.predict(cfg["data_root"], trip)
inputs, graph = pred._graph(cfg["data_root"], trip)
ref = load_reference(cfg["data_root"], trip)
ll = lambda p: [p[1], p[0]]
pts = [p for p in res["points"] if p["lat"] is not None]
lats, lons = [p["lat"] for p in pts], [p["lon"] for p in pts]
m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=15, tiles="cartodbpositron")
m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

g = folium.FeatureGroup("Road network (grey)")
for r in res["road_context"]:
    folium.PolyLine([ll(c) for c in r["coords"]], color="#9aa0a6", weight=1.5).add_to(g)
g.add_to(m)
g = folium.FeatureGroup("Reference route (green)")
a, b = graph.arc_lonlat(ref[(ref >= 0) & (ref < len(graph.arcs))])
for p, q in zip(a, b):
    folium.PolyLine([ll(p), ll(q)], color="#188038", weight=6, opacity=0.8).add_to(g)
g.add_to(m)
g = folium.FeatureGroup("Raw GPS (blue)")
for p in pts:
    folium.CircleMarker([p["lat"], p["lon"]], radius=2, color="#1a73e8", fill=True).add_to(g)
g.add_to(m)
g = folium.FeatureGroup(f"Predicted route (red, engine={res['engine']})")
for s in res["route_segments"]:
    folium.PolyLine([ll(c) for c in s], color="#d93025", weight=3).add_to(g)
g.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.get_root().html.add_child(folium.Element(
    f'<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;border:1px solid #888;'
    f'font:12px sans-serif"><b>Trip {trip} (partition: {cfg["mode"]})</b><br><span style="color:#1a73e8">●</span> raw GPS<br>'
    '<span style="color:#188038">━</span> reference route<br><span style="color:#d93025">━</span> predicted route<br>'
    '<span style="color:#9aa0a6">━</span> road network</div>'))
out = REPO_ROOT / "maps" / f"dev_map_{trip}.html"
m.save(str(out))
print("saved", out)
