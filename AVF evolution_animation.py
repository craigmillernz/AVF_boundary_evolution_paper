# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 09:30:57 2025

@author: craigm
"""

from shapely.geometry import Point, MultiPoint, Polygon
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import geopandas as gpd
from pathlib import Path

# SET GLOABL FONT PARAMETERS
font = {"family": "Arial", "weight": "normal", "size": 8}
plt.rc("font", **font)

data_file = Path(__file__).resolve().parent / "data" / "AVF_main_vents.csv"
vents = pd.read_csv(data_file, comment="#")

vents = vents.sort_values(by="Hopkins2020_Age")

coastline = data_file = (
    Path(__file__).resolve().parent
    / "data"
    / "Coastlines_and_Islands_Polygons_50K.shp"
)
coast = gpd.read_file(coastline)

# Ensure sorted by age (oldest → youngest)
vents = vents.sort_values("Hopkins2020_Age", ascending=False).reset_index(
    drop=True
)
vents.dropna(subset="Hopkins2020_Age", inplace=True)


# Compute time differences between successive eruptions
# We'll scale them to milliseconds for the animation
# positive differences
time_diff = vents["Hopkins2020_Age"].diff(-1).fillna(0).abs()
# Scale factor: adjust so the largest interval isn't too slow
scale = 0.5  # frames per ka
frame_counts = np.ceil(time_diff / scale).astype(int)


# %%
# Animated version

frames_expanded = []
for idx, count in enumerate(frame_counts):
    frames_expanded.extend([idx] * max(1, count))
frames_expanded.append(len(vents) - 1)

# --- Compute corresponding ages for countdown label ---
ages_expanded = []
for i, count in enumerate(frame_counts):
    start_age = vents.loc[i, "Hopkins2020_Age"]
    end_age = vents.loc[i + 1, "Hopkins2020_Age"] if i + 1 < len(vents) else 0
    # Linearly interpolate age values across repeated frames
    ages_segment = np.linspace(
        start_age, end_age, max(1, count), endpoint=False
    )
    ages_expanded.extend(ages_segment)
ages_expanded.append(0)  # youngest vent final frame


# --- Setup plot ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")


# --- Animation update function ---
def update(frame_idx):
    eruption_idx = frames_expanded[frame_idx]
    current_age = ages_expanded[frame_idx]

    # ax.clear()
    ax.cla()
    ax.set_aspect("equal")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")

    # plot coast
    coast.plot(ax=ax, linewidth=1, zorder=0, color="gray", alpha=0.2)

    # Add all points up to current frame
    pts = [
        Point(vents.loc[i, "easting"], vents.loc[i, "northing"])
        for i in range(eruption_idx + 1)
    ]

    # Compute convex hull
    hull = MultiPoint(pts).convex_hull if len(pts) >= 3 else None

    buffer_dist = 5000
    hull_buffered = hull.buffer(buffer_dist) if len(pts) >= 3 else None

    # Color logic
    colors = [
        "blue" if age > 60 else "black"
        for age in vents.loc[:eruption_idx, "Hopkins2020_Age"]
    ]

    # Plot eruption sites
    xs = vents.loc[:eruption_idx, "easting"]
    ys = vents.loc[:eruption_idx, "northing"]

    ax.scatter(
        xs,
        ys,
        c=colors,
        marker="^",
        zorder=100,
        label="Eruption sites, black < 60 ka",
    )

    # Highlight newest point
    ax.scatter(
        vents.loc[eruption_idx, "easting"],
        vents.loc[eruption_idx, "northing"],
        c="red",
        s=80,
        marker="^",
        zorder=200,
        label=f"New site ({vents.loc[eruption_idx, 'name']} {
            vents.loc[eruption_idx, 'Hopkins2020_Age']
        } ka)",
    )

    # Plot convex hull
    if hull and isinstance(hull, Polygon):
        xh, yh = hull.exterior.xy
        ax.fill(xh, yh, color="pink", alpha=0.4)
        ax.plot(xh, yh, "k-", lw=1.5, label="AVF outline")

        xhb, yhb = hull_buffered.exterior.xy
        # ax.fill(xh, yh, color="lightblue", alpha=0.4)
        ax.plot(
            xhb,
            yhb,
            "k--",
            lw=1.5,
            label=f"AVF outline: buffer = {buffer_dist}m",
        )

    # Countdown label (top-left)
    ax.text(
        0.02,
        0.95,
        f"Time: {current_age:.1f} ka",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    ax.legend(loc="lower right")
    ax.set_title(f"Eruption {eruption_idx + 1}/{len(vents)}: AVF evolution")
    ax.set_xlim(1745000, 1780000)
    ax.set_ylim(5896500, 5933500)
    ax.set_title("Auckland Volcanic Field Evolution\n Miller et al")


# --- Create animation ---
anim = FuncAnimation(
    fig, update, frames=len(frames_expanded), interval=100, repeat=False
)

# --- Save as GIF
anim.save(
    Path(__file__).resolve().parent / "Auckland_volcanic_field_evolution.gif",
    writer="pillow",
    fps=4,
)

plt.show()
