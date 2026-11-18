"""PathSnap evaluator UI - runs on LAPTOP 2.

Run from the repo root:   streamlit run ui/app.py
Type Laptop 1's address into the sidebar (on Laptop 1: `ipconfig getifaddr en0`).

The UI only talks to the API: it never loads data or the model, and it never receives ground truth -
only the predicted route and the accuracy numbers the API computed server-side.
"""
import folium
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap – Route Reconstruction")

api = st.sidebar.text_input("API URL (Laptop 1)", "http://localhost:8000").rstrip("/")

try:
    health = requests.get(f"{api}/health", timeout=5).json()
    trips = requests.get(f"{api}/trips", timeout=5).json()["trips"]
except Exception as e:
    st.error(f"Cannot reach the API at {api}. Is it running on Laptop 1 with --host 0.0.0.0? ({e})")
    st.stop()

st.sidebar.success(f"Connected · model: {health['model']} · HMM: {'on' if health['hmm'] else 'off'}")
if not trips:
    st.warning("The API has no trips in its evaluation folder.")
    st.stop()

trip_id = st.selectbox(f"Choose one of the {len(trips)} evaluation trips", trips)

if st.button("Reconstruct route", type="primary"):
    with st.spinner(f"API is reconstructing trip {trip_id}..."):
        resp = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=300)
    if resp.status_code != 200:
        st.error(f"API error {resp.status_code}: {resp.text}")
        st.stop()
    st.session_state["result"] = resp.json()

res = st.session_state.get("result")
if res:
    m = res["metrics"]
    pct = lambda k: f"{m[k]:.1%}" if m.get(k) is not None else "–"
    st.subheader(f"Trip {res['trip_id']} · {m['decoder']} · model {res['model']}")
    c = st.columns(5)
    c[0].metric("Top-1 accuracy (HMM)", pct("top1_hmm"))
    c[1].metric("Top-1 accuracy (classifier only)", pct("top1_classifier"))
    c[2].metric("Route continuity", pct("continuity_hmm") if "continuity_hmm" in m else pct("continuity_classifier"))
    c[3].metric("GPS points", m["n_points"])
    c[4].metric("Inference time", f"{m['inference_time_s']:.2f} s")
    if "note" in m:
        st.info(m["note"])
    with st.expander("All metrics returned by the API"):
        st.json(m)

    gps = res["gps_points"]
    fmap = folium.Map(tiles="OpenStreetMap")
    roads = folium.FeatureGroup("Road network (nearby)")
    for seg in res["road_network_segments"]:
        folium.PolyLine(seg, color="#999999", weight=2, opacity=0.5).add_to(roads)
    raw = folium.FeatureGroup("Raw GPS points")
    for lat, lon in gps:
        folium.CircleMarker([lat, lon], radius=2, color="#1f77b4", fill=True).add_to(raw)
    pred = folium.FeatureGroup("Predicted route")
    for seg in res["predicted_route_segments"]:
        folium.PolyLine(seg, color="#d62728", weight=5, opacity=0.85).add_to(pred)
    for layer in (roads, raw, pred):
        layer.add_to(fmap)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    st.caption("Grey = road network · Blue dots = raw GPS · Red = predicted route · Green / black pin = start / end")
    components.html(fmap.get_root().render(), height=650)
