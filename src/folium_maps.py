"""Folium maps: raw GPS + road network + reference route (+ optional predictions).

Run:  python src/folium_maps.py            -> results/maps/*.html
Folium expects [lat, lon]; raw files are (lon, lat) so we swap explicitly.
"""
import sys
from pathlib import Path

import folium
import numpy as np
from folium.plugins import MiniMap

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, load_trip, point_segment

OUT = RESULTS / "maps"
OUT.mkdir(parents=True, exist_ok=True)

LEGEND = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
 padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font-size: 13px;">
<b>{title}</b><br>
<span style="color:#1f77b4">&#9679;</span> raw GPS points<br>
<span style="color:#2ca02c">&#9644;</span> reference route<br>
<span style="color:#999">&#9644;</span> road network (arcs within {buf} m of track)<br>
{extra}
</div>"""


def ll(lonlat):
    return [[float(p[1]), float(p[0])] for p in lonlat]


def trip_map(partition, trip_id, buf=150, pred_arcs=None, pred_label="predicted (classifier Top-1)",
             point_step=1, fname=None):
    tr = load_trip(partition, trip_id)
    lon, lat = tr.track[:, 0], tr.track[:, 1]
    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=14, tiles="OpenStreetMap",
                   control_scale=True)
    # network subset: arcs whose segment is within `buf` m of any (subsampled) GPS point
    a = tr.arcs
    A_, B_ = tr.nodes_xy[a[:, 0]], tr.nodes_xy[a[:, 1]]
    xy = tr.track_xy
    lo, hi = xy.min(0) - buf, xy.max(0) + buf
    box = ((np.minimum(A_, B_) <= hi).all(1) & (np.maximum(A_, B_) >= lo).all(1))
    cand = np.flatnonzero(box)
    near = np.zeros(len(cand), bool)
    sub = xy[::5]
    for s in range(0, len(cand), 2000):
        c = cand[s:s + 2000]
        d, _ = point_segment(sub[:, None, 0], sub[:, None, 1], A_[None, c, 0], A_[None, c, 1], B_[None, c, 0], B_[None, c, 1])
        near[s:s + 2000] = d.min(0) <= buf
    net = folium.FeatureGroup(name=f"Road network (≤{buf} m of track)")
    seen = set()
    for i in cand[near]:
        u, v = a[i]
        k = (min(u, v), max(u, v))
        if k in seen:
            continue
        seen.add(k)
        folium.PolyLine(ll(tr.nodes[[u, v]]), color="#999999", weight=1.5, opacity=.7,
                        tooltip=f"arc {i} ({u}->{v})").add_to(net)
    net.add_to(m)

    rt = folium.FeatureGroup(name="Reference route (.route)")
    for i in tr.route:
        folium.PolyLine(ll(tr.nodes[a[i]]), color="#2ca02c", weight=5, opacity=.8, tooltip=f"route arc {i}").add_to(rt)
    rt.add_to(m)

    gps = folium.FeatureGroup(name="Raw GPS points (.track)")
    for j in range(0, len(lon), point_step):
        folium.CircleMarker([float(lat[j]), float(lon[j])], radius=2, color="#1f77b4", fill=True, fill_opacity=.8,
                            weight=0, tooltip=f"pt {j}, t={tr.track[j, 2]:.0f}s").add_to(gps)
    gps.add_to(m)

    extra = ""
    if pred_arcs is not None:
        pr = folium.FeatureGroup(name=pred_label)
        for i in np.unique(pred_arcs[pred_arcs >= 0]):
            folium.PolyLine(ll(tr.nodes[a[i]]), color="#d62728", weight=3, opacity=.9, dash_array="6",
                            tooltip=f"pred arc {i}").add_to(pr)
        pr.add_to(m)
        extra = f'<span style="color:#d62728">&#9644;</span> {pred_label}<br>'
    folium.Marker([float(lat[0]), float(lon[0])], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([float(lat[-1]), float(lon[-1])], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(
        title=f"PathSnap – {partition} trip {trip_id}", buf=buf, extra=extra)))
    fname = fname or OUT / f"{partition}_{trip_id}.html"
    m.save(str(fname))
    return fname


if __name__ == "__main__":
    print(trip_map("train", "00000001"))
    print(trip_map("train", "00000054", point_step=10, fname=OUT / "train_00000054_anomaly.html"))
    print(trip_map("student_test", "00000007"))
