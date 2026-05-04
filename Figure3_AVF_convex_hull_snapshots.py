# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 15:36:03 2025

@author: craigm
"""

from shapely.geometry import Point, MultiPoint, Polygon
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import geopandas as gpd
from pathlib import Path

# SET GLOABL FONT PARAMETERS
font = {"family": "Arial", "weight": "normal", "size": 8}
plt.rc("font", **font)

data_file = Path(__file__).resolve().parent / "data" / "AVF_main_vents.csv"
vents = pd.read_csv(data_file, comment="#")

vents = vents.sort_values(by="Hopkins2020_Age")

# Load distance to convex hull data
distance_file = "D:/Dropbox/AVF/data/AVF_distance_to_convex_hull.csv"
distances = pd.read_csv(distance_file)
# Create a dictionary of vent names that are outside the hull (positive signed_distance_m)
outside_vents_set = set(
    distances[distances["signed_distance_m"] > 0]["name"].dropna()
)

coastline = "D:/Dropbox/GIS/NZTM/Coastlines_and_Islands_Polygons_50K.shp"
# Example DataFrame (oldest → youngest)

coast = gpd.read_file(coastline)

# Ensure sorted by age (oldest → youngest)
vents = vents.sort_values("Hopkins2020_Age", ascending=False).reset_index(
    drop=True
)
# vents.dropna(subset="Hopkins2020_Age", inplace=True)

# Get the first 3 vents (needed for initial convex hull)
first_three_vents = set(vents.iloc[:3]["name"])

center_point = (1761475, 5914911)
width = 16481
height = 28884
angle = -2.6

# %%
fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax = axs[0, 0]
age = 125
vent_subset = vents[vents["Hopkins2020_Age"] > age]

# Separate vents inside and outside the hull
inside_subset = vent_subset[~vent_subset["name"].isin(outside_vents_set)]
outside_subset = vent_subset[vent_subset["name"].isin(outside_vents_set)]

# Further separate first 3 vents
inside_first_three = inside_subset[
    inside_subset["name"].isin(first_three_vents)
]
inside_others = inside_subset[~inside_subset["name"].isin(first_three_vents)]
outside_first_three = outside_subset[
    outside_subset["name"].isin(first_three_vents)
]
outside_others = outside_subset[
    ~outside_subset["name"].isin(first_three_vents)
]

past_points = []
for i, row in vent_subset.iterrows():
    pt = Point(row["easting"], row["northing"])
    past_points.append(pt)

# Compute hull
hull = None
hull = MultiPoint(past_points).convex_hull
# Compute hull

# Plot inside vents in black
for i, row in inside_others.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="black",
        marker="^",
        zorder=100,
        s=60,
        edgecolors="steelblue",
        linewidths=1.0,
    )

# Plot inside first 3 vents with no fill
for i, row in inside_first_three.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="none",
        marker="^",
        zorder=100,
        s=60,
        edgecolors="steelblue",
        linewidths=1.5,
    )

# Plot outside vents in red
for i, row in outside_others.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="red",
        marker="^",
        zorder=101,
        s=80,
        edgecolors="darkred",
        linewidths=1.2,
    )

# Plot outside first 3 vents with no fill
for i, row in outside_first_three.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="none",
        marker="^",
        zorder=101,
        s=80,
        edgecolors="darkred",
        linewidths=1.5,
    )

# plot coast
coast.plot(ax=ax, linewidth=1, zorder=0, color="gray", alpha=0.2)

# Plot convex hull
if hull and isinstance(hull, Polygon):
    xh, yh = hull.exterior.xy
    ax.fill(xh, yh, color="pink", alpha=0.4)
    ax.plot(xh, yh, "r-", lw=1.5, label="AVF outline")

ax.scatter(
    vent_subset["easting"][-1:],
    vent_subset["northing"][-1:],
    s=200,
    edgecolors="k",
    facecolors="none",
)

ax.text(x=1746000, y=5931000, s=f"A: {age} ka", fontsize=14)
ax.set_xlim(1745000, 1780000)
ax.set_ylim(5896500, 5933500)
ax.set_ylabel("Northing NZTM [m]")
print(f"Hull area {hull.area / 1000 / 1000} km^2")
# %%
ax = axs[0, 1]
age1 = 63
vent_subset = vents[(vents["Hopkins2020_Age"] > age1)]
vent_subset2 = vents[
    (vents["Hopkins2020_Age"] > age1) & (vents["Hopkins2020_Age"] < age)
]

# Separate vents inside and outside the hull
inside_subset = vent_subset[~vent_subset["name"].isin(outside_vents_set)]
outside_subset = vent_subset[vent_subset["name"].isin(outside_vents_set)]

# Further separate first 3 vents
inside_first_three = inside_subset[
    inside_subset["name"].isin(first_three_vents)
]
inside_others = inside_subset[~inside_subset["name"].isin(first_three_vents)]
outside_first_three = outside_subset[
    outside_subset["name"].isin(first_three_vents)
]
outside_others = outside_subset[
    ~outside_subset["name"].isin(first_three_vents)
]

past_points = []
for i, row in vent_subset.iterrows():
    pt = Point(row["easting"], row["northing"])
    past_points.append(pt)

# Compute hull
hull = None
hull = MultiPoint(past_points).convex_hull
# Compute hull

# Plot inside vents in black
for i, row in inside_others.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="black",
        marker="^",
        zorder=100,
        s=60,
        edgecolors="steelblue",
        linewidths=1.0,
    )

# Plot inside first 3 vents with no fill
for i, row in inside_first_three.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="none",
        marker="^",
        zorder=100,
        s=60,
        edgecolors="steelblue",
        linewidths=1.5,
    )

# Plot outside vents in red
for i, row in outside_others.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="red",
        marker="^",
        zorder=101,
        s=80,
        edgecolors="darkred",
        linewidths=1.2,
    )

# Plot outside first 3 vents with no fill
for i, row in outside_first_three.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="none",
        marker="^",
        zorder=101,
        s=80,
        edgecolors="darkred",
        linewidths=1.5,
    )

# plot coast
coast.plot(ax=ax, linewidth=1, zorder=0, color="gray", alpha=0.2)

# Plot convex hull
if hull and isinstance(hull, Polygon):
    xh, yh = hull.exterior.xy
    ax.fill(xh, yh, color="pink", alpha=0.4)
    ax.plot(xh, yh, "r-", lw=1.5, label="AVF outline")

ax.text(x=1746000, y=5931000, s=f"B: {age1} ka", fontsize=14)
ax.set_xlim(1745000, 1780000)
ax.set_ylim(5896500, 5933500)
print(f"Hull area {hull.area / 1000 / 1000} km^2")
# %%
ax = axs[1, 0]
age2 = 15
vent_subset = vents[(vents["Hopkins2020_Age"] > age2)]
vent_subset3 = vents[
    (vents["Hopkins2020_Age"] > age2) & (vents["Hopkins2020_Age"] < age1)
]

# Separate vents inside and outside the hull
inside_subset = vent_subset[~vent_subset["name"].isin(outside_vents_set)]
outside_subset = vent_subset[vent_subset["name"].isin(outside_vents_set)]

# Further separate first 3 vents
inside_first_three = inside_subset[
    inside_subset["name"].isin(first_three_vents)
]
inside_others = inside_subset[~inside_subset["name"].isin(first_three_vents)]
outside_first_three = outside_subset[
    outside_subset["name"].isin(first_three_vents)
]
outside_others = outside_subset[
    ~outside_subset["name"].isin(first_three_vents)
]

past_points = []
for i, row in vent_subset.iterrows():
    pt = Point(row["easting"], row["northing"])
    past_points.append(pt)

# Compute hull
hull = None
hull = MultiPoint(past_points).convex_hull
# Compute hull

# Plot inside vents in black
for i, row in inside_others.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="black",
        marker="^",
        zorder=100,
        s=60,
        edgecolors="steelblue",
        linewidths=1.0,
    )

# Plot inside first 3 vents with no fill
for i, row in inside_first_three.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="none",
        marker="^",
        zorder=100,
        s=60,
        edgecolors="steelblue",
        linewidths=1.5,
    )

# Plot outside vents in red
for i, row in outside_others.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="red",
        marker="^",
        zorder=101,
        s=80,
        edgecolors="darkred",
        linewidths=1.2,
    )

# Plot outside first 3 vents with no fill
for i, row in outside_first_three.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="none",
        marker="^",
        zorder=101,
        s=80,
        edgecolors="darkred",
        linewidths=1.5,
    )

# plot coast
coast.plot(ax=ax, linewidth=1, zorder=0, color="gray", alpha=0.2)

# Plot convex hull
if hull and isinstance(hull, Polygon):
    xh, yh = hull.exterior.xy
    ax.fill(xh, yh, color="pink", alpha=0.4)
    ax.plot(xh, yh, "r-", lw=1.5, label="AVF outline")

ax.text(x=1746000, y=5931000, s=f"C: {age2} ka ", fontsize=14)
ax.set_xlim(1745000, 1780000)
ax.set_ylim(5896500, 5933500)
ax.set_ylabel("Northing NZTM [m]")
ax.set_xlabel("Easting NZTM [m]")
print(f"Hull area {hull.area / 1000 / 1000} km^2")
# %%
ax = axs[1, 1]
age2 = 15
vent_subset = vents

# Separate vents inside and outside the hull
inside_subset = vent_subset[~vent_subset["name"].isin(outside_vents_set)]
outside_subset = vent_subset[vent_subset["name"].isin(outside_vents_set)]

# Further separate first 3 vents
inside_first_three = inside_subset[
    inside_subset["name"].isin(first_three_vents)
]
inside_others = inside_subset[~inside_subset["name"].isin(first_three_vents)]
outside_first_three = outside_subset[
    outside_subset["name"].isin(first_three_vents)
]
outside_others = outside_subset[
    ~outside_subset["name"].isin(first_three_vents)
]

past_points = []
for i, row in vents.iterrows():
    pt = Point(row["easting"], row["northing"])
    past_points.append(pt)

# Compute hull
hull = None
hull = MultiPoint(past_points).convex_hull
# Compute hull

# Plot inside vents in black
for i, row in inside_others.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="black",
        marker="^",
        zorder=100,
        s=60,
        edgecolors="steelblue",
        linewidths=1.0,
    )

# Plot inside first 3 vents with no fill
for i, row in inside_first_three.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="none",
        marker="^",
        zorder=100,
        s=60,
        edgecolors="steelblue",
        linewidths=1.5,
    )

# Plot outside vents in red
for i, row in outside_others.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="red",
        marker="^",
        zorder=101,
        s=80,
        edgecolors="darkred",
        linewidths=1.2,
    )

# Plot outside first 3 vents with no fill
for i, row in outside_first_three.iterrows():
    ax.scatter(
        row["easting"],
        row["northing"],
        c="none",
        marker="^",
        zorder=101,
        s=80,
        edgecolors="darkred",
        linewidths=1.5,
    )

ellipse = patches.Ellipse(
    xy=center_point,
    width=width,
    height=height,
    angle=angle,
    color="skyblue",  # fill color
    alpha=1,  # transparency
    linestyle="-",
    fill=None,
    lw=1.5,
)

ax.add_patch(ellipse)

# plot coast
coast.plot(ax=ax, linewidth=1, zorder=0, color="gray", alpha=0.2)

# Plot convex hull
if hull and isinstance(hull, Polygon):
    xh, yh = hull.exterior.xy
    ax.fill(xh, yh, color="pink", alpha=0.4)
    ax.plot(xh, yh, "r-", lw=1.5, label="AVF outline")

vents = pd.read_csv(data_file, comment="#")
vents = vents.sort_values(by="Hopkins2020_Age")

ax.scatter(
    vents.iloc[51].easting,
    vents.iloc[51].northing,
    c="red",
    marker="^",
    zorder=100,
    s=80,
    edgecolors="darkred",
    linewidths=1,
)

ax.scatter(
    vents.iloc[52].easting,
    vents.iloc[52].northing,
    c="black",
    marker="^",
    zorder=100,
    s=60,
    edgecolors="steelblue",
    linewidths=1,
)

ax.scatter(
    vents.iloc[53].easting,
    vents.iloc[53].northing,
    c="black",
    marker="^",
    zorder=100,
    s=60,
    edgecolors="steelblue",
    linewidths=1,
)

ax.text(x=1746000, y=5931000, s="D: all", fontsize=14)
ax.set_xlim(1745000, 1780000)
ax.set_ylim(5896500, 5933500)
ax.set_xlabel("Easting NZTM [m]")

print(f"Hull area {hull.area / 1000 / 1000} km^2")

plt.subplots_adjust(wspace=0.01, hspace=0.04)

plt.savefig(
    Path(__file__).resolve().parent / "Figure3_convex_hull_snapshots.png",
    dpi=300,
)
