"""Generate measured exploration figures and layered Folium maps."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pathsnap-matplotlib")
import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


def make_map(trip, meta, candidates, labels, output, data, html=True):
    directory = data / meta["split"] / trip
    track = np.loadtxt(directory / f"{trip}.track", ndmin=2)
    route = np.loadtxt(directory / f"{trip}.route", dtype=int, ndmin=1)
    graph = np.load(output / "graphs" / f"{trip}.npz", allow_pickle=False)
    nodes = graph["node_lonlat"]
    segments = np.stack([nodes[graph["arc_from"]], nodes[graph["arc_to"]]], axis=1)
    # Limit the road layer to a corridor about the observed trip for legibility and file size.
    low, high = track[:, :2].min(0) - .005, track[:, :2].max(0) + .005
    visible = np.all(segments.max(1) >= low, axis=1) & np.all(segments.min(1) <= high, axis=1)
    visible_ids = np.flatnonzero(visible)
    if not html:
        make_static_road(trip, meta, track, segments, route, candidates, output)
        return
    road_features = [dict(type="Feature", geometry=dict(type="LineString", coordinates=segments[a].tolist()),
                          properties=dict(arc_id=int(a))) for a in visible_ids]
    m = folium.Map(location=[float(track[:, 1].mean()), float(track[:, 0].mean())],
                   tiles="OpenStreetMap", prefer_canvas=True)
    folium.GeoJson(dict(type="FeatureCollection", features=road_features), name="Road network",
        style_function=lambda _: dict(color="#88929b", weight=1, opacity=.4),
        tooltip=folium.GeoJsonTooltip(fields=["arc_id"])).add_to(m)
    reference = folium.FeatureGroup(name="Supplied reference route", show=True).add_to(m)
    for occurrence, arc in enumerate(route):
        folium.PolyLine(segments[arc][:, ::-1].tolist(), color="#228449", weight=4, opacity=.7,
                        tooltip=f"Reference occurrence {occurrence}; arc {arc}").add_to(reference)
    observations = folium.FeatureGroup(name="Raw GPS observations", show=True).add_to(m)
    nearest_layer = folium.FeatureGroup(name="Nearest candidate (diagnostic baseline)", show=False).add_to(m)
    aligned_layer = folium.FeatureGroup(name="Offline reference-aligned points", show=False).add_to(m)
    lookup = candidates.set_index("point_index")
    label_lookup = labels.set_index("point_index")
    # Draw points, not connecting lines across potentially large gaps or disconnected arcs.
    raw_features, candidate_features, alignment_features = [], [], []
    def point_feature(lon, lat, **properties):
        return dict(type="Feature", geometry=dict(type="Point", coordinates=[float(lon), float(lat)]), properties=properties)
    for i, row in enumerate(track):
        raw_features.append(point_feature(row[0], row[1], point_index=i, timestamp_s=float(row[2])))
        c = lookup.loc[i]
        candidate_features.append(point_feature(c["projected_lon"], c["projected_lat"], point_index=i,
            arc_id=int(c["arc_id"]), distance_m=float(c["distance_m"]), outside_radius=int(c["outside_radius"])))
        label = label_lookup.loc[i]
        alignment_features.append(point_feature(label["reference_projected_lon"], label["reference_projected_lat"],
            point_index=i, reference_arc_id=int(label["reference_arc_id"]), label_usable=int(label["label_usable"])))
    for layer, features, color, fields in [
        (observations, raw_features, "#176dc1", ["point_index", "timestamp_s"]),
        (nearest_layer, candidate_features, "#d76b19", ["point_index", "arc_id", "distance_m", "outside_radius"]),
        (aligned_layer, alignment_features, "#773bb0", ["point_index", "reference_arc_id", "label_usable"])]:
        folium.GeoJson(dict(type="FeatureCollection", features=features),
            marker=folium.CircleMarker(radius=2, weight=0, fill=True, fill_color=color, fill_opacity=.8),
            style_function=(lambda feature: dict(fillColor="#773bb0" if feature["properties"]["label_usable"] else "#c84a55"))
                if layer is aligned_layer else None,
            tooltip=folium.GeoJsonTooltip(fields=fields)).add_to(layer)
    title = f"{meta['split']} / {trip} | GPS and road alignment"
    legend = f'''<div style="position:fixed;top:12px;left:55px;z-index:9999;background:white;padding:12px;
      border:1px solid #ccd3da;border-radius:6px;font:13px Arial;max-width:360px">
      <b>{title}</b><br><span style="color:#176dc1">●</span> Raw GPS &nbsp;
      <span style="color:#228449">━</span> Supplied route<br>
      <span style="color:#d76b19">●</span> Nearest-candidate baseline (toggle layer)<br>
      <span style="color:#773bb0">●</span> Usable approximate label &nbsp;
      <span style="color:#c84a55">●</span> Uncertain label<br>
      Offline labels use the reference route. The baseline is not a trained model.
      </div>'''
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(track[:, 1].min()), float(track[:, 0].min())],
                  [float(track[:, 1].max()), float(track[:, 0].max())]])
    m.save(str(output / "maps" / f"{meta['split']}_{trip}.html"))
    make_static_road(trip, meta, track, segments, route, candidates, output)


def make_static_road(trip, meta, track, segments, route, candidates, output):
    # Overview plus local detail makes near-overlapping GPS/road layers inspectable.
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), width_ratios=[1, 1.5], layout="constrained")
    low, high = track[:, :2].min(0) - .005, track[:, :2].max(0) + .005
    center = track[len(track) // 2, :2]
    half = np.array([.008 / np.cos(np.deg2rad(center[1])), .008])
    windows = [(low, high), (center - half, center + half)]
    for ax, (lower, upper), subtitle in zip(axes, windows, ["Whole-trip overview", "Local detail near middle observation"]):
        visible = np.all(segments.max(1) >= lower, axis=1) & np.all(segments.min(1) <= upper, axis=1)
        ax.add_collection(LineCollection(segments[visible], colors="#bbc2ca", linewidths=.7, alpha=.6))
        ax.add_collection(LineCollection(segments[route], colors="#228449", linewidths=2.5, alpha=.8, label="Supplied route"))
        ax.scatter(track[:, 0], track[:, 1], s=10, c="#176dc1", label="GPS", zorder=3)
        ax.scatter(candidates["projected_lon"], candidates["projected_lat"], s=6, c="#d76b19",
                   label="Nearest candidate baseline", alpha=.7, zorder=4)
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), xlabel="Longitude (degrees)",
               ylabel="Latitude (degrees)", title=subtitle)
        ax.set_aspect(1 / np.cos(np.deg2rad(center[1])))
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.15)
    axes[1].legend(loc="upper left", fontsize=8, markerscale=1.5)
    fig.suptitle(f"{meta['split']} / {trip} | GPS and road alignment", fontsize=15)
    fig.savefig(output / "figures" / f"{meta['split']}_{trip}_roads.png", dpi=160)
    plt.close(fig)


def run(output=Path("output"), data=Path("Student folder (2)"), maps=True):
    for folder in ["maps", "figures"]:
        (output / folder).mkdir(exist_ok=True)
    report = json.loads((output / "validation_report.json").read_text())
    inventory = pd.DataFrame(report["trips"])
    labels = pd.read_csv(output / "reference_alignment" / "point_labels.csv", dtype={"trip_id": str})
    selected = {"00000001", "00000054", "00000000"}
    nearest_frames = []
    for chunk in pd.read_csv(output / "candidates.csv", dtype={"trip_id": str}, chunksize=100000):
        nearest_frames.append(chunk[chunk.candidate_rank == 1])
    nearest = pd.concat(nearest_frames, ignore_index=True)
    intervals = {s: [] for s in ["train", "student_test"]}
    for trip in report["trips"]:
        path = data / trip["split"] / trip["trip_id"] / f"{trip['trip_id']}.track"
        t = np.loadtxt(path, ndmin=2)
        intervals[trip["split"]].append(np.diff(t[:, 2]))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    colors = {"train": "#176dc1", "student_test": "#d76b19"}
    for split, color in colors.items():
        rows = inventory[inventory.split == split]
        axes[0, 0].hist(rows.gps_points, bins=np.geomspace(200, 25000, 18), histtype="step", label=split, color=color, linewidth=2)
        dt = np.concatenate(intervals[split])
        axes[0, 1].hist(dt, bins=np.geomspace(1, 200000, 30), histtype="step", color=color, label=split, linewidth=2)
        distances = np.sort(nearest.loc[nearest.split == split, "distance_m"].to_numpy())
        axes[0, 2].plot(distances, np.arange(1, len(distances) + 1) / len(distances), color=color, label=split)
        axes[1, 0].scatter(rows.nodes, rows.arcs, color=color, label=split, s=23, alpha=.7)
        axes[1, 1].scatter(rows.weak_components, rows.largest_weak_component_nodes / rows.nodes,
                           color=color, label=split, s=23, alpha=.7)
        coverage = labels[labels.split == split].groupby("trip_id").label_usable.mean()
        axes[1, 2].hist(coverage, bins=np.linspace(0, 1, 16), histtype="step", color=color, label=split, linewidth=2)
    settings = [dict(title="GPS observations per trip", xlabel="Point count", ylabel="Trips", xscale="log"),
        dict(title="Sampling intervals (all steps)", xlabel="Interval (seconds)", ylabel="Steps", xscale="log", yscale="log"),
        dict(title="Nearest-road distance distribution", xlabel="Distance (m)", ylabel="Cumulative fraction"),
        dict(title="Road network sizes", xlabel="Nodes", ylabel="Directed arcs"),
        dict(title="Road network connectivity", xlabel="Weak component count", ylabel="Largest component node fraction"),
        dict(title="Conservative label coverage by trip", xlabel="Usable approximate-label fraction", ylabel="Trips")]
    for ax, setting in zip(axes.flat, settings):
        ax.set(**setting); ax.grid(alpha=.15); ax.legend(fontsize=8)
    axes[1, 0].ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    fig.suptitle("PathSnap dataset exploration | 80 training and 10 student-test trips", fontsize=16)
    fig.savefig(output / "figures" / "dataset_exploration.png", dpi=160)
    plt.close(fig)
    for trip in report["trips"]:
        if trip["trip_id"] in selected:
            make_map(trip["trip_id"], trip, nearest[nearest.trip_id == trip["trip_id"]],
                     labels[labels.trip_id == trip["trip_id"]], output, data, html=maps)
            print(f"Map and road figure: {trip['split']}/{trip['trip_id']}", flush=True)
    missing = []
    for chunk in pd.read_csv(output / "candidate_features.csv.gz", chunksize=100000):
        missing.append(chunk.isna().sum())
    total_missing = sum(missing)
    pd.DataFrame(dict(column=total_missing.index, missing_count=total_missing.values,
        rows=report["summary"]["candidate_rows"], missing_fraction=total_missing.values / report["summary"]["candidate_rows"])
        ).to_csv(output / "feature_missingness.csv", index=False)
    print("Exploration figures, missing-value table, and three Folium maps generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-maps", action="store_true", help="Regenerate figures and missingness only")
    args = parser.parse_args()
    run(maps=not args.skip_maps)
