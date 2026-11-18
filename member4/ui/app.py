"""
Evaluator UI (Laptop 2).  Run:  streamlit run ui/app.py
Set the backend with the sidebar box or env PATHSNAP_API, e.g. http://192.168.1.20:8000
"""
import os
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: GPS-to-road matcher")

base = st.sidebar.text_input("Backend URL", os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000")).rstrip("/")
TIMEOUT = 120


def api(method, path, **kw):
    try:
        r = requests.request(method, base + path, timeout=TIMEOUT, **kw)
    except requests.exceptions.ConnectTimeout:
        return None, f"Timed out connecting to {base}. Check the IP, port, firewall and that both laptops share a network."
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach {base}. Is the backend running with --host 0.0.0.0, and is the URL correct?"
    except requests.exceptions.ReadTimeout:
        return None, "The backend took too long to answer. Try again."
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {e}"
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        return None, f"Backend returned {r.status_code}: {detail}"
    return r.json(), None


health, err = api("GET", "/health")
if err:
    st.error(err)
    st.stop()
st.sidebar.success(f"Connected: mode={health['mode']}, model={health['model_version']}")
if health.get("engine") == "scaffold":
    st.warning("SCAFFOLD engine (nearest-road snapping). These are not model results. Do not report them.")

data, err = api("GET", "/trips")
if err:
    st.error(err)
    st.stop()
trip_id = st.selectbox("Trip", data["trips"])
if st.button("Run", type="primary"):
    with st.spinner(f"Matching trip {trip_id}..."):
        res, err = api("POST", "/predict", json={"trip_id": trip_id})
    if err:
        st.error(err)
        st.session_state.pop("result", None)
    else:
        st.session_state["result"] = res
        st.session_state.pop("replay", None)

res = st.session_state.get("result")
if not res:
    st.info("Pick a trip and press Run.")
    st.stop()

# ---- metrics
m = res.get("metrics", {})
c = st.columns(5)
c[0].metric("Trip", res["trip_id"])
c[1].metric("Points", res["point_count"])
c[2].metric("Eligible", m.get("eligible_points", "n/a"))
hit = m.get("route_arc_hit_rate")
c[3].metric("Route-arc hit rate", f"{hit:.3f}" if hit is not None else "n/a")
c[4].metric("Runtime (ms)", res["runtime_ms"])
st.caption(f"model {res['model_version']} | engine {res['engine']} | convention: {m.get('convention', 'n/a')} | "
           f"breaks: {len(res['breaks'])}")

# ---- map (API gives [lon, lat]; Folium wants [lat, lon])
pts = [p for p in res["points"] if p["lat"] is not None and p["lon"] is not None]
if not pts:
    st.warning("This trip has no valid GPS coordinates to display.")
    st.stop()
ll = lambda lon, lat: [lat, lon]
lats = [p["lat"] for p in pts]
lons = [p["lon"] for p in pts]
fmap = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=15, tiles="OpenStreetMap", prefer_canvas=True)
fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

g_road = folium.FeatureGroup("Road context (grey)", show=True)
for r in res["road_context"]:
    folium.PolyLine([ll(*xy) for xy in r["coords"]], color="#9aa0a6", weight=2, opacity=0.7).add_to(g_road)
g_gps = folium.FeatureGroup("Raw GPS (blue)", show=True)
for p in pts:
    folium.CircleMarker(ll(p["lon"], p["lat"]), radius=2, color="#1a73e8", fill=True, fill_opacity=0.8,
                        tooltip=f"#{p['point_index']} {p['status']} arc={p['arc_id']}").add_to(g_gps)
g_route = folium.FeatureGroup("Decoded route (red)", show=True)
for seg in res["route_segments"]:                       # separate line per chain: no line across breaks
    folium.PolyLine([ll(*xy) for xy in seg], color="#d93025", weight=4, opacity=0.9).add_to(g_route)
g_break = folium.FeatureGroup("Break / unmatched points (orange)", show=True)
byidx = {p["point_index"]: p for p in pts}
for b in res["breaks"]:
    p = byidx.get(b["point_index"])
    if p:
        folium.CircleMarker(ll(p["lon"], p["lat"]), radius=5, color="#f9ab00", fill=True,
                            tooltip=f"#{b['point_index']}: {b['reason']}").add_to(g_break)
for g in (g_road, g_gps, g_route, g_break):
    g.add_to(fmap)
folium.LayerControl(collapsed=False).add_to(fmap)
fmap.get_root().html.add_child(folium.Element(
    '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;color:#111;padding:8px 10px;'
    'border:1px solid #888;font:12px sans-serif"><b>Legend</b><br>'
    '<span style="color:#1a73e8">●</span> raw GPS<br><span style="color:#d93025">━</span> decoded route<br>'
    '<span style="color:#9aa0a6">━</span> road network<br><span style="color:#f9ab00">●</span> break/unmatched</div>'))
st_folium(fmap, height=620, use_container_width=True, key=f"trip-map-{res['trip_id']}", returned_objects=[])

# ---- replay (optional; needs Member 3)
with st.expander("Replay: how later GPS changed an earlier road choice"):
    if st.button("Load replay"):
        rp, e = api("POST", "/replay", json={"trip_id": res["trip_id"]})
        if e:
            st.info(e)
        else:
            st.session_state["replay"] = rp
    rp = st.session_state.get("replay")
    if rp:
        steps = rp.get("prefixes") or rp.get("steps") or []
        if steps:
            i = st.slider("Replay step", 0, len(steps) - 1, len(steps) - 1) if len(steps) > 1 else 0
            step = steps[i]
            st.write(f"GPS through point {step['prefix_end']} selects arc {step['anchor_arc']} for earlier point {step['anchor']}.")
            view = folium.Map(location=ll(*step["lonlat"]), zoom_start=17)
            for arc, coords in rp.get("roads", {}).items():
                selected = int(arc) == step["anchor_arc"]
                folium.PolyLine([ll(*xy) for xy in coords], color="red" if selected else "grey",
                                weight=6 if selected else 3, tooltip=f"Arc {arc}").add_to(view)
            folium.Marker(ll(*step["lonlat"]), tooltip="Earlier matched position").add_to(view)
            folium.CircleMarker(ll(*step["gps"]), radius=6, color="blue", tooltip="Latest GPS evidence").add_to(view)
            st_folium(view, height=380, key=f"replay-{res['trip_id']}", returned_objects=[])
            if not rp.get("changed"):
                st.caption("This replay window keeps the same road choice as more observations arrive.")
        else:
            st.write(rp)
